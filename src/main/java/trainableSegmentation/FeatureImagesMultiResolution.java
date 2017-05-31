package trainableSegmentation;

/**
 *
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * Authors: Ignacio Arganda-Carreras (iarganda@mit.edu), Verena Kaynig (verena.kaynig@inf.ethz.ch),
 *          Albert Cardona (acardona@ini.phys.ethz.ch)
 */

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Prefs;
import ij.measure.Calibration;
import ij.plugin.Binner;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This class stores the feature stacks of a set of input slices.
 * It can be used so for 2D stacks or as the container of 3D features (by
 * using a feature stack per section). 
 *
 * @author Ignacio Arganda-Carreras (iarganda@mit.edu)
 *
 */
public class FeatureImagesMultiResolution implements FeatureImages
{
    /** original input image */
    private ImagePlus originalImage = null;

    /** the feature images */
    private ArrayList<ArrayList<ArrayList<ImagePlus>>> multiResolutionFeatureImages;

    private ImagePlus[][] multiResolutionFeatureImageArray = null;

    /** flag to specify the use of color features */
    private final boolean colorFeatures = false;

    /** image width */
    private int width = 0;
    /** image height */
    private int height = 0;

    /** minmum sigma/radius used in the filters */
    private float minimumSigma = 1;
    /** maximum sigma/radius used in the filters */
    private float maximumSigma = 8;

    /** Gaussian filter flag index */
    public static final int GAUSSIAN 				=  0;
    /** Hessian filter flag index */
    public static final int HESSIAN 				=  1;
    /** Derivatives filter flag index */
    public static final int DERIVATIVES				=  2;
    /** Laplacian filter flag index */
    public static final int LAPLACIAN				=  3;
    /** structure tensor filter flag index */
    public static final int STRUCTURE				=  4;
    /** edge filter flag index */
    public static final int EDGES					=  5;
    /** difference of Gaussian filter flag index */
    public static final int DOG						=  6;
    /** Minimum flag index */
    public static final int MINIMUM					=  7;
    /** Maximum flag index */
    public static final int MAXIMUM					=  8;
    /** Mean flag index */
    public static final int MEAN					=  9;
    /** Median flag index */
    public static final int MEDIAN					=  10;
    /** Variance flag index */
    public static final int VARIANCE				=  11;


    /** Features only available if the ImageScience library is present. */
    public static final boolean[] IMAGESCIENCE_FEATURES = {
            false, // Gaussian_blur
            true,  // Hessian
            true,  // Derivatives
            true,  // Laplacian
            true,  // Structure
            true,  // Edges
            false,  // Difference of Gaussian
            false, // Minimum
            false, // Maximum
            false, // Mean
            false, // Median
            false  // Variance
    };

    /** flags of filters to be used */
    private boolean[] enableFeatures = new boolean[]{
            false, 	/* Gaussian_blur */
            true, 	/* Hessian */
            false, 	/* Derivatives */
            false, 	/* Laplacian */
            false,	/* Structure */
            false,	/* Edges */
            false,	/* Difference of Gaussian */
            false,	/* Minimum */
            false,	/* Maximum */
            false,	/* Mean */
            false,	/* Median */
            false	/* Variance */
    };

    // TODO: possibliy need multiple of above in case there is more than one image
    // in the hyperstack

    /** index of the feature stack that is used as reference (to read attribute, etc.).
     * -1 if not defined yet. */
    private int referenceStackIndex = -1;

    private ExecutorService exe;

    /** names of available filters */
    public static final String[] availableFeatures
            = new String[]{	"Hessian", "Structure"};

    /**
     * Initialize a feature stack list of a specific size
     *
     * @param num number of elements in the list
     * @param minimumSigma minimum sigma value (usually filter radius)
     * @param maximumSigma maximum sigma value (usually filter radius)
     * @param useNeighbors flag to use neighbor features
     * @param membraneSize expected membrane thickness
     * @param membranePatchSize membrane patch size
     * @param enabledFeatures array of flags to enable features
     */
    public FeatureImagesMultiResolution()
    {
    }

    public FeatureImagesMultiResolution( ImagePlus image )
    {

        width = image.getWidth();
        height = image.getHeight();
        originalImage = image;
        originalImage.setTitle("O");

    }

    public boolean saveStackAsTiff( String filePath )
    {
        IJ.showMessage("not implemented");
        return true;
    }

    /**
     * Merge input channels if they are more than 1
     * @param channels results channels
     * @return result image
     */
    ArrayList< ImagePlus > mergeResultChannels(final ArrayList<ImagePlus>[] channels)
    {
        if(channels.length > 1)
        {
            ArrayList< ImagePlus > mergedList = new ArrayList<ImagePlus> ();

            for(int i=0; i<channels[0].size(); i++)
            {

                ImageStack mergedColorStack = mergeStacks(channels[0].get(i).getImageStack(), channels[1].get(i).getImageStack(), channels[2].get(i).getImageStack());

                ImagePlus merged = new ImagePlus(channels[0].get(i).getTitle(), mergedColorStack);

                for(int n = 1; n <= merged.getImageStackSize(); n++)
                    merged.getImageStack().setSliceLabel(channels[0].get(i).getImageStack().getSliceLabel(n), n);
                mergedList.add( merged );
            }

            return mergedList;
        }
        else
            return channels[0];
    }

    /**
     * Merge three image stack into a color stack (no scaling)
     *
     * @param redChannel image stack representing the red channel
     * @param greenChannel image stack representing the green channel
     * @param blueChannel image stack representing the blue channel
     * @return RGB merged stack
     */
    ImageStack mergeStacks(ImageStack redChannel, ImageStack greenChannel, ImageStack blueChannel)
    {
        final ImageStack colorStack = new ImageStack( redChannel.getWidth(), redChannel.getHeight());

        for(int n=1; n<=redChannel.getSize(); n++)
        {
            final ByteProcessor red = (ByteProcessor) redChannel.getProcessor(n).convertToByte(false);
            final ByteProcessor green = (ByteProcessor) greenChannel.getProcessor(n).convertToByte(false);
            final ByteProcessor blue = (ByteProcessor) blueChannel.getProcessor(n).convertToByte(false);

            final ColorProcessor cp = new ColorProcessor(redChannel.getWidth(), redChannel.getHeight());
            cp.setRGB((byte[]) red.getPixels(), (byte[]) green.getPixels(), (byte[]) blue.getPixels() );

            colorStack.addSlice(redChannel.getSliceLabel(n), cp);
        }

        return colorStack;
    }

    /**
     * Extract channels from input image if it is RGB
     * @param originalImage input image
     * @return array of channels
     */
    ImagePlus[] extractChannels(final ImagePlus originalImage)
    {
        final int width = originalImage.getWidth();
        final int height = originalImage.getHeight();
        ImagePlus[] channels;
        if( originalImage.getType() == ImagePlus.COLOR_RGB )
        {
            final ImageStack isRed = new ImageStack ( width, height );
            final ImageStack isGreen = new ImageStack ( width, height );
            final ImageStack isBlue = new ImageStack ( width, height );

            for(int n = 1; n<= originalImage.getImageStackSize(); n++)
            {
                final ByteProcessor redBp = new ByteProcessor(width, height);
                final ByteProcessor greenBp = new ByteProcessor(width, height);
                final ByteProcessor blueBp = new ByteProcessor(width, height);

                final byte[] redPixels = (byte[]) redBp.getPixels();
                final byte[] greenPixels = (byte[]) greenBp.getPixels();
                final byte[] bluePixels = (byte[]) blueBp.getPixels();


                ((ColorProcessor)(originalImage.getImageStack().getProcessor( n ).duplicate())).getRGB(redPixels, greenPixels, bluePixels);

                isRed.addSlice( null, redBp.convertToFloat() );
                isGreen.addSlice( null, greenBp.convertToFloat() );
                isBlue.addSlice( null, blueBp.convertToFloat() );

            }

            channels = new ImagePlus[]{new ImagePlus("red", isRed),
                    new ImagePlus("green", isGreen),
                    new ImagePlus("blue", isBlue )};
        }
        else
        {
            channels = new ImagePlus[1];
            final ImageStack is = new ImageStack ( width, height );
            for(int i=1; i<=originalImage.getImageStackSize(); i++)
                is.addSlice("original-slice-" + i, originalImage.getImageStack().getProcessor(i).convertToFloat() );
            channels[0] = new ImagePlus(originalImage.getTitle(), is );
        }

        for(int i=0; i<channels.length; i++)
            channels[i].setCalibration(originalImage.getCalibration());

        return channels;
    }

    /**
     * Get Hessian features (to be submitted in an ExecutorService)
     *
     * @param originalImage input image
     * @param sigma isotropic smoothing scale
     * @return filter Hessian filter images
     */
    public Callable<ArrayList<ImagePlus>> getHessian(
            final ImagePlus originalImage,
            final double sigma,
            final boolean absolute)
    {
        if (Thread.currentThread().isInterrupted())
            return null;

        return new Callable <ArrayList <ImagePlus> >()
        {
            public ArrayList< ImagePlus >call()
            {

                // Get channel(s) to process
                ImagePlus[] channels = extractChannels(originalImage);

                ArrayList<ImagePlus>[] results = new ArrayList[ channels.length ];

                for(int ch=0; ch < channels.length; ch++)
                {
                    results[ ch ] = new ArrayList<ImagePlus>();

                    // pad image on the back and the front
                    final ImagePlus channel = channels [ ch ].duplicate();
                    channel.getImageStack().addSlice("pad-back", channels[ch].getImageStack().getProcessor( channels[ ch ].getImageStackSize()));
                    channel.getImageStack().addSlice("pad-front", channels[ch].getImageStack().getProcessor( 1 ), 1);

                    final ArrayList<ImagePlus> result = ImageScience.computeHessianImages(sigma, absolute, channel);
                    final ImageStack smallest = result.get(0).getImageStack();
                    final ImageStack middle   = result.get(1).getImageStack();
                    final ImageStack largest  = result.get(2).getImageStack();
                    // remove pad
                    smallest.deleteLastSlice();
                    smallest.deleteSlice(1);
                    middle.deleteLastSlice();
                    middle.deleteSlice(1);
                    largest.deleteLastSlice();
                    largest.deleteSlice(1);

                    results[ ch ].add(new ImagePlus("HL_" + originalImage.getTitle(), smallest));
                    results[ ch ].add(new ImagePlus("HM_" + originalImage.getTitle(), middle));
                    results[ ch ].add(new ImagePlus("HS_" + originalImage.getTitle(), largest ) );
                }

                return mergeResultChannels(results);
            }
        };
    }


    public void setFeatureSlice(int slice, int frame, double[][][] featureSlice)
    {
        int nf = getNumFeatures();

        for ( int f = 0; f < nf; f++ )
        {
            ImagePlus imp = multiResolutionFeatureImageArray[ frame - 1 ][ f ];
            Calibration calibration = imp.getCalibration();
            double calX = calibration.pixelWidth;
            double calY = calibration.pixelHeight;
            double calZ = calibration.pixelDepth;
            int nxFeatureImage = imp.getWidth();
            int nyFeatureImage = imp.getHeight();

            // deal with the fact that the feature image might
            // not have all pixels
            // due to the binning
            int nx = (int)(nxFeatureImage * calX);
            int ny = (int)(nyFeatureImage * calY);

            int sliceCal =  (int) ( (slice - 1) / calZ ) + 1;

            if ( sliceCal == imp.getNSlices()+1 )
            {
                // this can happen due to the binning
                sliceCal = imp.getNSlices();
            }

            if ( imp.getBitDepth() == 8 )
            {
                byte[] pixels = (byte[]) imp.getStack().getProcessor(sliceCal).getPixels();

                for ( int y = 0; y < ny; y++ )
                {
                    int offsetY = (int) ( (double)y/calY) * nxFeatureImage;
                    for ( int x = 0; x < nx; x++ )
                    {
                        if ( offsetY + (int)((double)x/calX) >= pixels.length)
                        {
                            IJ.log("");
                        }
                        featureSlice[x][y][f] = (double) pixels[offsetY + (int)((double)x/calX)];
                    }
                }
            }

            if ( imp.getBitDepth() == 16 )
            {
                short[] pixels = (short[]) imp.getStack().getProcessor(sliceCal).getPixels();

                for ( int y = 0; y < ny; y++ )
                {
                    int offsetY = (int) ( (double)y/calY) * nxFeatureImage;
                    for ( int x = 0; x < nx; x++ )
                    {
                        featureSlice[x][y][f] = (double) pixels[offsetY + (int)((double)x/calX)];
                    }
                }
            }


            if ( imp.getBitDepth() == 32 )
            {
                float[] pixels = (float[]) imp.getStack().getProcessor(sliceCal).getPixels();
                int widthFeatureImage = imp.getWidth();
                for ( int y = 0; y < ny; y++ )
                {
                    int offsetY = (int) ( (double)y/calY) * widthFeatureImage;
                    for ( int x = 0; x < nx; x++ )
                    {
                        featureSlice[x][y][f] = pixels[offsetY + (int)((double)x/calX)];
                    }
                }
            }


        }


        // set last value to class value
        /*
        int nx = getWidth();
        int ny = getHeight();
        for ( int y = 0; y < ny; y++ )
        {
            for ( int x = 0; x < nx; x++ )
            {
                featureSlice[x][y][nf] = classNum;
            }
        }*/

    }

    /**
     * Get structure tensor features (to be submitted in an ExecutorService).
     * It computes, for all pixels in the input image, the eigenvalues of the so-called structure tensor.
     *
     * @param originalImage input image
     * @param sigma isotropic smoothing scale
     * @param integrationScale integration scale (standard deviation of the Gaussian
     * 		kernel used for smoothing the elements of the structure tensor, must be larger than zero)
     * @return filter structure tensor filter image
     */
    public Callable<ArrayList<ImagePlus>> getStructure(
            final ImagePlus originalImage,
            final double sigma,
            final double integrationScale)
    {
        if (Thread.currentThread().isInterrupted())
            return null;

        return new Callable<ArrayList< ImagePlus >>()
        {
            public ArrayList< ImagePlus > call()
            {

                // Get channel(s) to process
                ImagePlus[] channels = extractChannels(originalImage);

                ArrayList<ImagePlus>[] results = new ArrayList[ channels.length ];

                for(int ch=0; ch < channels.length; ch++)
                {
                    results[ ch ] = new ArrayList<ImagePlus>();

                    // pad image on the back and the front
                    final ImagePlus channel = channels [ ch ].duplicate();
                    channel.getImageStack().addSlice("pad-back", channels[ch].getImageStack().getProcessor( channels[ ch ].getImageStackSize()));
                    channel.getImageStack().addSlice("pad-front", channels[ch].getImageStack().getProcessor( 1 ), 1);

                    final ArrayList<ImagePlus> result = ImageScience.computeEigenimages(sigma, integrationScale,
                            channel);
                    final ImageStack largest    = result.get(0).getImageStack();
                    final ImageStack middle     = result.get(1).getImageStack();
                    final ImageStack smallest   = result.get(2).getImageStack();

                    // remove pad
                    smallest.deleteLastSlice();
                    smallest.deleteSlice(1);
                    middle.deleteLastSlice();
                    middle.deleteSlice(1);
                    largest.deleteLastSlice();
                    largest.deleteSlice(1);

                    results[ ch ].add( new ImagePlus("SL_" + originalImage.getTitle(), largest ) );
                    results[ ch ].add( new ImagePlus("SM_" + originalImage.getTitle(), middle ) );
                    results[ ch ].add( new ImagePlus("SS_" + originalImage.getTitle(), smallest ) );

                }

                return mergeResultChannels(results);
            }
        };
    }

    /**
     * Create instance (feature vector) of a specific coordinate
     *
     * @param x x- axis coordinate
     * @param y y- axis coordinate
     * @param classValue class value to be assigned
     * @return corresponding instance
     */
    public DenseInstance createInstance(
            int x,
            int y,
            int z,
            int t,
            int classValue )
    {

        final double[] values = new double[ getNumFeatures() + 1 ];

        // get the feature values at the x, y, z location
        for ( int i = 0; i < getNumFeatures(); i++ )
        {
            values[ i ] = getFeatureValue(x, y, z, t, i);
        }

        // Assign class
        values[values.length-1] = (double) classValue;

        return new DenseInstance(1.0, values);
    }


    /**
     * Set values to an instance (feature vector) of a specific coordinate.
     * The input instance needs to have a data set assigned.
     *
     * @param x x- axis coordinate
     * @param y y- axis coordinate
     * @param sliceNum z- axis coordinate
     * @param classValue class value to be assigned
     * @param ins instance to be filled
     * @param auxArray auxiliary array to store feature values
     */
    public void setInstance(
            int x,
            int y,
            int z,
            int t,
            int classValue,
            final ReusableDenseInstance ins,
            final double[] auxArray )
    {
        // get the feature values at the x, y, z location
        for ( int i = 0; i < getNumFeatures(); i++ )
        {
            auxArray[ i ] = getFeatureValue(x, y, z, t, i);
        }

        // Assign class
        auxArray[ auxArray.length - 1 ] = (double) classValue;

        // Set attribute values to input instance
        ins.setValues(1.0, auxArray);
        return;
    }


    public void setMinimumSigma( double sigma )
    {

    }

    public void setMaximumSigma( double sigma )
    {

    }

    /**
     * Create instances for whole stack
     *
     * @param classes list of classes names
     *
     * @return whole stack set of instances
     */
    public Instances createInstances(ArrayList<String> classes)
    {
        // TODO: what is this doing? feels like it is only for 2-D?

        IJ.showMessage("NOT IMPLEMENTED: createInstances(ArrayList<String> classes)");

        if (Thread.currentThread().isInterrupted())
            return null;

        /*
        if( oldColorFormat )
            IJ.log( "Using old color format...");
        */

        // get feature names
        ArrayList<Attribute> attributes = getFeatureNamesAsAttributes();

        attributes.add(new Attribute("class", classes));

        Instances data = new Instances("segment", attributes, width*height);

        for (int y=0; y<width; y++)
        {
            if (Thread.currentThread().isInterrupted())
                return null;
            IJ.showProgress(y, width);
            for (int x=0; x<height; x++)
            {
                data.add(createInstance(x, y, 0, 0, 0));
            }
        }
        // Set the index of the class attribute
        data.setClassIndex( attributes.size() - 1 );
        IJ.showProgress(1.0);
        return data;
    }

    /**
     * Get value of the feature
     *
     * @param x x- axis coordinate
     * @param y y- axis coordinate
     * @param z z- slice number, one-based
     * @param t t- frame number, one-based
     * @param i Feature number
     * @return value of the feature
     */
    public double getFeatureValue( int x, int y, int z, int t, int iFeature )
    {
        long tStart;
        ArrayList<Long> tTotal = new ArrayList<Long>();

        tStart = System.nanoTime();
        ImagePlus imp = multiResolutionFeatureImageArray[ t - 1 ][ iFeature ];
        Calibration calibration = imp.getCalibration();
        tTotal.add( System.nanoTime() - tStart );

        tStart = System.nanoTime();
        int zSlice =  (int) ( (z - 1) / calibration.pixelDepth) + 1;
        tTotal.add( System.nanoTime() - tStart );

        /*
        if ( (zSlice < 1) || zSlice > (imp.getNSlices()+1) )
        {
            IJ.log("accessing wrong image plane");
        }*/


        tStart = System.nanoTime();
        if ( zSlice == imp.getNSlices()+1 )
        {
            // this can happen due to the binning
            zSlice = imp.getNSlices();
        }
        tTotal.add( System.nanoTime() - tStart );


        tStart = System.nanoTime();
        ImageProcessor ip = imp.getStack().getProcessor(zSlice);
        tTotal.add( System.nanoTime() - tStart );

        /*
        int xCal = (int) (x / calibration.pixelWidth);
        int yCal = (int) (y / calibration.pixelHeight);

        if ( ( xCal > ip.getWidth() ) || ( yCal > ip.getHeight() ) )
        {
            IJ.log("accessing wrong image pixles");
        }
        */

        tStart = System.nanoTime();
        double value = ip.getPixelValue( (int) (x / calibration.pixelWidth) , (int) (y / calibration.pixelHeight) );
        tTotal.add( System.nanoTime() - tStart );

        //double value = ip.getInterpolatedValue(
        //        (double)x / calibration.pixelWidth ,
        //        (double)y / calibration.pixelHeight);


        return value;
    }






    /**
     * Get the number of features
     *
     * @return number of features
     */
    public int getNumFeatures()
    {
        return multiResolutionFeatureImageArray[0].length;
    }




    /**
     * Update features with current list in a multi-thread fashion
     *
     * @return true if the features are correctly updated
     */
    public boolean updateFeaturesMT()
    {
        int t = 0;

        if (Thread.currentThread().isInterrupted() )
            return false;

        long start = System.currentTimeMillis();

        exe = Executors.newFixedThreadPool( Prefs.getThreads() );

        // don't duplicate and don't convert the input image to float
        // in order to save memory
        /*
        ImageStack is = new ImageStack ( width, height );

        if( colorFeatures )
        {
            for(int i=1; i<=originalImage.getImageStackSize(); i++)
                is.addSlice("original-slice-" + i, originalImage.getImageStack().getProcessor(i) );
        }
        else
        {
            for(int i=1; i<=originalImage.getImageStackSize(); i++)
                is.addSlice("original-slice-" + i, originalImage.getImageStack().getProcessor(i).convertToFloat() );
        }*/


        multiResolutionFeatureImages = new ArrayList<ArrayList<ArrayList<ImagePlus>>>();

        for ( int frame = 1 ; frame <= originalImage.getNFrames() ; frame++ )
        {
            multiResolutionFeatureImages.add(new ArrayList<ArrayList<ImagePlus>>());
        }


        // Set a calibration that can be changed during the binning
        Calibration calibration = new Calibration();
        calibration.pixelDepth = 1;
        calibration.pixelWidth = 1;
        calibration.pixelHeight = 1;
        calibration.setUnit("um");
        originalImage.setCalibration(calibration);

        // Add original image to first resolution layer
        ArrayList<ImagePlus> singleResolutionFeatureImages = new ArrayList<ImagePlus>();
        singleResolutionFeatureImages.add( originalImage );
        multiResolutionFeatureImages.get(t).add( singleResolutionFeatureImages );

        int currentIndex = 0;
        IJ.showStatus("Updating features...");

        try{

            int numScales = 3; // 3,9,27,81

            // loop through resolutions
            for ( int iScale = 0; iScale < numScales; iScale++ )
            {
                IJ.log("Resolution level "+(iScale+1));

                ArrayList<ImagePlus> featureImagesPreviousResolution =
                        multiResolutionFeatureImages.get( t ).get( iScale );

                final ArrayList<ImagePlus> featureImagesThisResolution = new ArrayList<>();

                // determine binning for this resolution layer
                int[] binning = new int[3];

                binning[0] = 3; // x-binning
                binning[1] = 3; // y-binning

                if ( originalImage.getNSlices() > 1 )
                {
                    // don't bin in z if there are much less slices
                    int previousResolutionWidth = featureImagesPreviousResolution.get(0).getWidth();
                    int previousResolutionDepth = featureImagesPreviousResolution.get(0).getNSlices();
                    if ( previousResolutionWidth > 3 * previousResolutionDepth )
                    {
                        binning[2] = 1; // no z-binning
                    }
                    else
                    {
                        binning[2] = 3; // z-binning
                    }
                }
                else
                {
                    binning[2] = 1; // no z-binning
                }

                // add average binning of raw data as 1st feature image
                // in order to compute features in the next resolution layer
                ImagePlus avgBinnedOrigPrevResolution = featureImagesPreviousResolution.get(0);
                featureImagesThisResolution.add( bin(avgBinnedOrigPrevResolution, binning, new Binner().AVERAGE) );

                // max-pooling and min-pooling of all (non-structure eigenvalue) images of the previous resolution layer
                // in order to represent distances to image features at different scales
                for ( final ImagePlus featureImagePreviousResolution : featureImagesPreviousResolution )
                {

                    String title = featureImagePreviousResolution.getTitle();

                    // don't overdo it...
                    if ( ! ( title.contains("Max_Max_") || title.contains("Min_Min_") ) )
                    {
                        // don't do Max_Min as this seems not so useful
                        if ( title.contains("Max_") )
                        {
                            featureImagesThisResolution.add(
                                    bin( featureImagePreviousResolution, binning, new Binner().MAX) );
                        }

                        // don't do Min_Max as this seems not so useful
                        if ( title.contains("Min_") )
                        {
                            // Structure tensor eigenvalues are positive => no min-pooling
                            if (!title.contains("_S"))
                            {
                                featureImagesThisResolution.add(
                                        bin( featureImagePreviousResolution, binning, new Binner().MIN) );
                            }
                        }

                    }
                }

                // compute new features

                // temporarily remove calibration while computing the features
                Calibration calibration1 = avgBinnedOrigPrevResolution.getCalibration().copy();
                removeCalibration( avgBinnedOrigPrevResolution );

                if (Thread.currentThread().isInterrupted())
                    return false;

                final ArrayList<Future<ArrayList<ImagePlus>>> futures = new ArrayList<Future<ArrayList<ImagePlus>>>();

                // Hessian
                boolean absoluteValues = false;
                futures.add( exe.submit( getHessian( avgBinnedOrigPrevResolution, 1, absoluteValues )) );

                // Structure tensor
                // https://en.wikipedia.org/wiki/Structure_tensor
                // https://imagescience.org/meijering/software/featurej/structure/
                float integrationScale = 1; // TODO: is that correct?
                futures.add( exe.submit( getStructure( avgBinnedOrigPrevResolution, 1, integrationScale)) );

                // put calibration back
                avgBinnedOrigPrevResolution.setCalibration(calibration1);

                // Wait for jobs to be done
                for (Future<ArrayList<ImagePlus>> f : futures)
                {
                    final ArrayList<ImagePlus> newFeatureImages = f.get();
                    IJ.showStatus("Updating features at resolution scale " + iScale);

                    for (final ImagePlus newFeatureImage : newFeatureImages)
                    {
                        // add back the calibration
                        newFeatureImage.setCalibration( calibration1.copy() );

                        // Bin the new convolution results of this resolution layer
                        featureImagesThisResolution.add( bin( newFeatureImage, binning, new Binner().MAX) );
                        featureImagesThisResolution.add( bin( newFeatureImage, binning, new Binner().MIN) );

                    }
                }

                multiResolutionFeatureImages.get(t).add(featureImagesThisResolution);

            }

            // Put the feature images into an array for simpler access via indexing

            int numFeatures = 0;
            for ( ArrayList<ImagePlus> singleResolutionFeatureImages2 : multiResolutionFeatureImages.get(t) )
            {
                numFeatures += singleResolutionFeatureImages2.size();
            }

            IJ.log( "Number of features: " + numFeatures );

            multiResolutionFeatureImageArray =
                    new ImagePlus[ originalImage.getNFrames() ][ numFeatures ];

            int iFeature = 0;
            int iScale = 0;
            for ( ArrayList<ImagePlus> featureStacks : multiResolutionFeatureImages.get(t))
            {
                IJ.log("##### Level " + (iScale++) );
                for ( ImagePlus featureImage : featureStacks )
                {
                    //featureImage.show();
                    IJ.log(featureImage.getTitle()+": "
                            +featureImage.getWidth()+" "
                            +featureImage.getHeight()+" "
                            +featureImage.getNSlices()+" "
                            +featureImage.getCalibration().pixelDepth);
                    multiResolutionFeatureImageArray[ t ][ iFeature++ ] = featureImage;
                }
            }

        }
        catch(InterruptedException ie)
        {
            IJ.log("The features udpate was interrupted by the user.");
            return false;
        }
        catch(Exception ex)
        {
            IJ.log("Error when updating feature stack.");
            ex.printStackTrace();
            return false;
        }
        finally{
            exe.shutdownNow();
        }

        long end = System.currentTimeMillis();

        IJ.log("Updating the feature images took: " +
                (end - start) + "ms.");
        IJ.showProgress(1.0);
        IJ.showStatus("Features stack is updated now!");
        reportMemoryStatus();
        return true;
    }

    public void reportMemoryStatus()
    {

        double memory = 0;
        double memoryInputImage = 0;
        boolean inputImage = true;

        for (int t = 0; t < multiResolutionFeatureImages.size(); t++)
        {
            for (ArrayList<ImagePlus> featureImages : multiResolutionFeatureImages.get(t))
            {
                //IJ.log("##### Scale " + (iScale++) );
                for (ImagePlus featureImage : featureImages)
                {
                    int width = featureImage.getWidth();
                    int height = featureImage.getHeight();
                    int depth = featureImage.getNSlices();
                    int bitDepth = featureImage.getBitDepth();

                    if (inputImage)
                    {
                        // the first of the feature images is the input image
                        memoryInputImage = width * height * depth * bitDepth;
                        inputImage = false;
                    }
                    else
                    {
                        memory += width * height * depth * bitDepth;
                    }
                }
            }
        }
        memory /= 8000000000L; // convert from bit to GB
        memoryInputImage /= 8000000000L;

        IJ.log("Size of input image [GB]: " + memoryInputImage);
        IJ.log("Size of feature images [GB]: " + memory);
        IJ.log("Size ratio: Feature images / input image = " + memory/memoryInputImage );

    }

    public void removeCalibration ( ImagePlus imp )
    {
        Calibration calibration = new Calibration();
        calibration.pixelDepth = 1;
        calibration.pixelWidth = 1;
        calibration.pixelHeight = 1;
        calibration.setUnit("pixel");
        imp.setCalibration( calibration );
    }

    public ImagePlus bin(ImagePlus imp, int[] binning, int method)
    {
        String title = new String(imp.getTitle());
        Binner binner = new Binner();

        Calibration saveCalibration = imp.getCalibration().copy(); // this is due to a bug in the binner

        ImagePlus impBinned = binner.shrink( imp, binning[0], binning[1], binning[2], method );
        imp.setCalibration(saveCalibration);

        if ( method == binner.MAX )
        {
            impBinned.setTitle("Max_" + title);
        }
        else if ( method == binner.AVERAGE )
        {
            impBinned.setTitle("Avg_" + title);
        }
        else if ( method == binner.MIN )
        {
            impBinned.setTitle("Min_" + title);
        }

        return ( impBinned );
    }

    public ArrayList<Attribute> getFeatureNamesAsAttributes()
    {
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (int i = 0; i < getNumFeatures(); i++)
        {
            String attString = getLabel(i);
            attributes.add(new Attribute(attString));
        }
        return attributes;
    }

    /**
     * Reset the reference index (used when the are
     * changes in the features)
     */
    public void resetReference()
    {
        this.referenceStackIndex = -1;
    }

    /**
     * Set the reference index (used when the are
     * changes in the features)
     */
    public void setReference( int index )
    {
        this.referenceStackIndex = index;
    }

    /**
     * Shut down the executor service
     */
    public void shutDownNow()
    {
        // TODO: implement
    }

    /**
     * Check if the array has not been yet initialized
     *
     * @return true if the array has been initialized
     */
    public boolean isEmpty()
    {
        if ( multiResolutionFeatureImageArray[0] == null )
            return true;

        if ( multiResolutionFeatureImageArray[0].length > 1 )
            return false;

        return true;
    }


    /**
     * Get a specific label of the reference stack
     * @param index slice index (&gt;=1)
     * @return label name
     */
    public String getLabel(int index)
    {
        if(referenceStackIndex == -1)
            return null;

        String featureName = "Feature " + index;

        return featureName;
    }

    /**
     * Get the features enabled for the reference stack
     * @return features to be calculated on each stack
     */
    public boolean[] getEnabledFeatures()
    {
        return new boolean[] {true, true};
    }

    /**
     * Set the features enabled for the reference stack
     * @param newFeatures boolean flags for the features to use
     */
    public void setEnabledFeatures(boolean[] newFeatures)
    {
        /*
        this.enabledFeatures = newFeatures;
        if(referenceStackIndex != -1)
            featureStackArray[referenceStackIndex].setEnabledFeatures(newFeatures);
            */
    }

    public int getReferenceSliceIndex()
    {
        return referenceStackIndex;
    }

    public int getWidth()
    {
        return originalImage.getWidth();
    }

    public int getHeight()
    {
        return originalImage.getHeight();
    }

    public int getDepth()
    {
        return originalImage.getNSlices();
    }

    public int getSize() {
        return getNumFeatures();
    }


}

	
