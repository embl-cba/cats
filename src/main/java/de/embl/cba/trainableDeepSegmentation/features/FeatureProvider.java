package de.embl.cba.trainableDeepSegmentation.features;

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

import de.embl.cba.trainableDeepSegmentation.settings.FeatureSettings;
import de.embl.cba.utils.logging.Logger;
import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.StackProcessor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import de.embl.cba.trainableDeepSegmentation.results.ResultImage;
import de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils;
import de.embl.cba.trainableDeepSegmentation.utils.ThreadUtils;

import java.util.*;
import java.util.concurrent.*;

/**
 * This class stores the feature stacks of a set of input slices.
 * It can be used so for 2D stacks or as the container of 3D features (by
 * using a feature stack per section). 
 *
 * @author Ignacio Arganda-Carreras (iarganda@mit.edu)
 *
 */
public class FeatureProvider
{

    private ImagePlus inputImage;

    private Map< String, ImagePlus > featureImages = new LinkedHashMap<>();

    public static final Integer FG_DIST_BG_IMAGE = -1;
    private final String CONV_DEPTH = "CD";

    private int X = 0, Y = 1, C = 2, Z = 3, T = 4;
    private int[] XYZ = new int[]{ X, Y, Z};
    private int[] XYZT = new int[]{ X, Y, Z, T};

    private int[] featureImageBorderSizes;

    public DeepSegmentation deepSegmentation = null;

    private String showFeatureImageTitle = "";

    /** flags of filters to be used */
    private boolean[] enabledFeatures = new boolean[]{
            true, 	/* Hessian */
            true,	/* Structure */
            false,	/* Minimum */
            false,	/* Maximum */
            true,	/* Mean */
            };

    /** names of available filters */
    public static final String[] availableFeatures = new String[]{	"Hessian", "Structure", "Minimum", "Maximum", "Mean" };

    /** index of the feature stack that is used as reference (to read attribute, etc.).
     * -1 if not defined yet. */
    private int referenceStackIndex = -1;

    private Logger logger;

    public ResultImage getResultImageFgDistBg()
    {
        return resultImageFgDistBg;
    }

    public void setResultImageFgDistBg( ResultImage resultImageFgDistBg )
    {
        this.resultImageFgDistBg = resultImageFgDistBg;
    }

    public void setShowFeatureImageTitle( String showFeatureImageTitle )
    {
        this.showFeatureImageTitle = showFeatureImageTitle;
    }


    public Set<Integer> getFeatureSliceCacheKeys()
    {
        return featureSliceCache.keySet();
    }

    private ResultImage resultImageFgDistBg;

    private FinalInterval interval = null;

    public int getCacheSize()
    {
        return cacheSize;
    }

    public void setCacheSize( int cacheSize )
    {
        this.cacheSize = cacheSize;
    }

    public int cacheSize = 0;

    FeatureSettings featureSettings;

    final LinkedHashMap< Integer, double[][][] > featureSliceCache = new LinkedHashMap< Integer, double[][][]>() {
        @Override
        protected boolean removeEldestEntry(final Map.Entry eldest) {
            return size() > cacheSize;
        }
    };


    public FeatureProvider( DeepSegmentation deepSegmentation )
    {
        this.deepSegmentation = deepSegmentation;
        this.inputImage = deepSegmentation.getInputImage();
        this.resultImageFgDistBg = deepSegmentation.getResultImageBgFg();
        this.logger = deepSegmentation.getLogger();
        this.featureSettings = deepSegmentation.featureSettings.copy();

        // TODO: isn't below a job for the feature-provider?
        featureImageBorderSizes = deepSegmentation.getFeatureBorderSizes();
    }

    public void setInterval( FinalInterval interval )
    {
        this.interval = interval;
    }

    public FinalInterval getInterval( )
    {
        return ( interval );
    }

    /**
     * Merge input channels if they are more than 1
     * @param channels results channels
     * @return resultImagePlus image
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

                for( int n = 1; n <= merged.getImageStackSize(); n++ )
                {
                    merged.getImageStack().setSliceLabel( channels[ 0 ].get( i ).getImageStack().getSliceLabel( n ), n );
                }
                mergedList.add( merged );
            }

            return mergedList;
        }
        else
        {
            return channels[ 0 ];
        }
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


    private String getFilterBaseName( String filterName, double sigma )
    {
        if ( sigma == 1 )
        {
            return filterName + "_"; // for backward compatibility
        }
        else
        {
            return filterName + (int)(sigma) + "_";
        }

    }


    public static final String HESSIAN = "He";
    public static final String STRUCTURE = "St";
    public static final String GAUSSIAN = "Ga";


    /**
     * Get Hessian features (to be submitted in an ExecutorService)
     *
     * @param originalImage input image
     * @param sigma isotropic smoothing scale
     * @return filter Hessian filter images
     */
    public Callable<ArrayList<ImagePlus>> getHessianOrStructure(
            final String filterName,
            final ImagePlus originalImage,
            final double sigma )
    {
        if (Thread.currentThread().isInterrupted())
            return null;

        return new Callable <ArrayList <ImagePlus> >()
        {
            public ArrayList< ImagePlus >call()
            {

                // Get channel(s) to process
                ImagePlus[] channels = extractChannels( originalImage );

                String filterNameIncludingSmoothing = getFilterBaseName( filterName, sigma );

                ArrayList<ImagePlus>[] results = new ArrayList[ channels.length ];

                for( int ch = 0; ch < channels.length; ch++ )
                {
                    results[ch] = new ArrayList< >();

                    final ImagePlus channelImage = channels[ch].duplicate();

                    if ( channelImage.getNSlices() > 1 )
                    {
                        // pad 3-D image on the back and the front
                        channelImage.getImageStack().addSlice("pad-back", channels[ch].getImageStack().getProcessor(channels[ch].getImageStackSize()));
                        channelImage.getImageStack().addSlice("pad-front", channels[ch].getImageStack().getProcessor(1), 1);
                    }

                    final ArrayList<ImagePlus> result;

                    if ( filterName.equals( HESSIAN ) )
                    {
                        result = ImageScience.computeHessianImages( sigma, true, channelImage );
                    }
                    else if ( filterName.equals( STRUCTURE ) )
                    {
                        result = ImageScience.computeEigenimages( sigma, sigma, channelImage );
                    }
                    else
                    {
                        result = null;
                    }

                    final ImageStack largest = result.get( 0 ).getImageStack();
                    final ImageStack middle = result.get( 1 ).getImageStack();

                    if ( channelImage.getNSlices() > 1 )
                    {
                        // remove pad
                        largest.deleteLastSlice();
                        largest.deleteSlice( 1 );
                        middle.deleteLastSlice();
                        middle.deleteSlice( 1 );
                    }

                    results[ ch ].add(new ImagePlus("L" + filterNameIncludingSmoothing + originalImage.getTitle(), largest));

                    if ( result.size() == 2 )
                    {
                        // TODO: change this to S at some point (issue: backwards compatibility with stored features)
                        results[ ch ].add(new ImagePlus("M" + filterNameIncludingSmoothing + originalImage.getTitle(), middle));
                    }
                    else // 3D
                    {
                        results[ ch ].add(new ImagePlus("M" + filterNameIncludingSmoothing + originalImage.getTitle(), middle));
                        final ImageStack smallest = result.get(2).getImageStack();
                        // remove pad
                        smallest.deleteLastSlice();
                        smallest.deleteSlice(1);
                        results[ ch ].add(new ImagePlus("S"+ filterNameIncludingSmoothing + originalImage.getTitle(), smallest));
                    }

                }

                return mergeResultChannels( results );
            }
        };
    }


    /**
     * Get Gaussian features (to be submitted to an ExecutorService). Sigma values will be adjusted
     * based on original image calibration.
     *
     * @param originalImage input image
     * @param sigma filter radius
     * @return filter Gaussian filtered image
     */
    public Callable<ArrayList< ImagePlus >> getGaussian(
            final ImagePlus originalImage,
            final double sigma )
    {
        if (Thread.currentThread().isInterrupted())
            return null;

        return new Callable<ArrayList< ImagePlus >>()
        {
            public ArrayList< ImagePlus > call()
            {

                String filterBaseName = getFilterBaseName( GAUSSIAN, sigma );

                // Get channel(s) to process
                ImagePlus[] channels = extractChannels(originalImage);

                ArrayList<ImagePlus>[] results = new ArrayList[ channels.length ];

                for(int ch=0; ch < channels.length; ch++)
                {
                    results[ ch ] = new ArrayList<ImagePlus>();

                    final ImagePlus im = channels [ ch ].duplicate();
                    final Img<FloatType > resultImage = ImagePlusAdapter.wrap( im );

                    // first extend the image with mirror
                    RandomAccessible< FloatType > extendedInputImage = Views.extendMirrorSingle( resultImage );

                    // adjust sigma based on voxel size
                    final double[] isoSigma = new double[ extendedInputImage.numDimensions() ];
                    Calibration calibration = originalImage.getCalibration();
                    isoSigma[ 0 ] = sigma / calibration.pixelWidth;
                    isoSigma[ 1 ] = sigma / calibration.pixelHeight;

                    if ( isoSigma.length == 3 ) isoSigma[ 2 ] = sigma / calibration.pixelDepth;

                    try
                    {
                        Gauss3.gauss( isoSigma, extendedInputImage, resultImage );
                    }
                    catch (IncompatibleTypeException e) {
                        IJ.log( "Error when calculating Gaussian feature." );
                        e.printStackTrace();
                        return null;
                    }

                    final ImagePlus ip = ImageJFunctions.wrapFloat( resultImage, filterBaseName + originalImage.getTitle());

                    results[ch].add( ip );
                }

                return mergeResultChannels(results);
            }
        };
    }

    /**
     * Also contains empty (0) classID
     * @param xGlobal
     * @param yGlobal
     * @return
     */
    public double[] getValuesFromFeatureSlice(
            int xGlobal,
            int yGlobal,
            double[][][] featureSlice )
    {
        int x = xGlobal - (int) interval.min( X );
        int y = yGlobal - (int) interval.min( Y );

        return ( featureSlice[x][y] );

    }


    public void setFeatureValuesAndClassIndex(
            double[] values,
            int xGlobal,
            int yGlobal,
            double[][][] featureSlice,
            int classNum )
    {

        int x = xGlobal - (int) interval.min( X );
        int y = yGlobal - (int) interval.min( Y );

        int nf = getNumActiveFeatures();

        System.arraycopy( featureSlice[x][y], 0, values, 0, nf );
        values[ nf ] = classNum;

    }

    public double[][][] getReusableFeatureSlice()
    {
        double[][][] featureSlice = new double
                [(int) interval.dimension(X)]
                [(int) interval.dimension(Y)]
                [ getNumActiveFeatures() + 1 ]; // one extra for class ID

        return ( featureSlice );
    }

    public synchronized double[][][] getCachedFeatureSlice( int z )
    {
        if ( featureSliceCache.containsKey( z ) )
        {
            return ( featureSliceCache.get( z ) );
        }
        else
        {
            return null;
        }
    }

    /**
     * set all feature values for one z-slice
     * coordinates are relative to within the set interval
     * @param zGlobal
     * @param featureSlice
     */
    public boolean setFeatureSlicesValues( final int zGlobal,
                                           double[][][] featureSlice,
                                           int numThreads )
    {

        // compute new feature slice

        if ( (zGlobal > interval.max(Z)) || (zGlobal < interval.min(Z)) )
        {
            logger.error("No features available for slice " + zGlobal);
            return false;
        }

        int xs = 0;
        int xe = (int) (interval.dimension( X ) - 1);
        int ys = 0;
        int ye = (int) (interval.dimension( Y ) - 1);
        int z = zGlobal - (int) interval.min( Z );

        ArrayList< String > featureNames = getActiveFeatureNames();

        // The feature images in featureImages
        // are larger than the requested interval, because of border
        // issues during feature computation.
        // Here we skip to the values outside the borders and
        // only put the values into the feature
        // slice that were properly computed, and are in fact within
        // the requested interval
        xs += featureImageBorderSizes[ X ];
        xe += featureImageBorderSizes[ X ];
        ys += featureImageBorderSizes[ Y ];
        ye += featureImageBorderSizes[ Y ];
        z += featureImageBorderSizes[ Z ];

        ExecutorService exe = Executors.newFixedThreadPool( numThreads );
        ArrayList<Future> futures = new ArrayList<>();

        for ( int f = 0; f < featureNames.size(); ++f )
        {
            futures.add(
                exe.submit(
                    setFeatureSliceValue(
                            featureSlice,
                            f, featureNames.get( f ),
                            xs, xe, ys, ye, z)
                )
            );
        }

        ThreadUtils.joinThreads( futures, logger );
        exe.shutdown();

        // cache featureSlice
        if ( cacheSize > 0 )
        {
            featureSliceCache.put( zGlobal, featureSlice );
        }

        return ( true );
    }


    private Runnable setFeatureSliceValue( double[][][] featureSlice,
                                           int f, String feature,
                                           int xs, int xe, int ys, int ye, int z )
    {
        return () ->
        {
            if ( deepSegmentation.stopCurrentTasks ) return;

            double v000,v100,vA00,v010,v110,vA10,v001,v101,vA01,v011,v111,vA11;
            double vAA0,vAA1,vAAA;
            double xCal, yCal, zCal, xHalfWidth, yHalfWidth, zHalfWidth;
            double xTmp, xBaseDist, xBaseDist2;
            double zTmp, zBaseDist, zBaseDist2;
            double yTmp, yBaseDist, yBaseDist2;

            int xBase, yBase, zBase, yBaseOffset, yAboveOffset, xAbove, x, y;
            int nxFeatureImage;
            Calibration calibration = null;

            // getInstancesAndMetadata feature values as doubles
            float[] pixelsBase = null;
            float[] pixelsAbove = null;


            ImagePlus imp = featureImages.get( feature );
            if ( imp == null )
            {
                logger.error( "Feature not found among current feature images: " + feature );
                return;
            }


            calibration = imp.getCalibration();
            xCal = calibration.pixelWidth;
            yCal = calibration.pixelHeight;
            zCal = calibration.pixelDepth;
            xHalfWidth = (xCal - 1) / 2 ;
            yHalfWidth = (yCal - 1) / 2 ;
            zHalfWidth = (zCal - 1) / 2 ;

            nxFeatureImage = imp.getWidth();

            /*

            # Example: binning 3, value 3

            orig: 0,1,2,3,4,5,6,7,8
            bin:  .,0,.|.,1,.|.,2,.
            value at orig 3 should be computed from bin 0 and 1
            halfwidth = (3-1)/2 = 1
            tmp = (3 - halfwidth) / 3 = 2 / 3
            base = (int) tmp = 0
            baseDist = tmp - base = 2 / 3 - 0 = 2 / 3
            baseDist2 = 1 - 2 / 3 = 1 / 3
            ..this makes sense, because
            3 is 2 away from 1, which is the center of run 0
            3 is 1 away from 4, which is the center of run 1

            # Example: binning 3, value 4

            orig: 0,1,2,3,4,5,6,7,8
            bin:  .,0,.|.,1,.|.,2,.
            value at orig 4 should be computed from bin 1, because it is its center
            halfwidth = (3-1)/2 = 1
            tmp = (4 - halfwidth) / 3 = 3 / 3 = 1
            base = (int) tmp = 1
            baseDist = tmp - base = 1 - 1 = 0
            baseDist2 = 1 - 0 = 1
            ..this means that run 2 will getInstancesAndMetadata a weight of 0 (i.e., baseDist)
            ..and run 1 will getInstancesAndMetadata a weight of 1

            # Example: binning=cal=2, value 3

            orig: 0,1,2,3,4,5,6,7,8
            bin:  .0.|.1.|.2.|.3.|
            value at orig 3 should be computed from bin 1 and bin 2
            halfwidth = (cal-1)/2 = (2-1)/2 = 0.5
            tmp = (3 - halfwidth) / cal = (3-0.5)/2 = 2.5/2 = 1.25
            base = (int) tmp = 1  (=> above will be 2)
            baseDist = tmp - base = 1.25 - 1 = 0.25
            baseDist2 = 1 - 0.25 = 0.75
            ..center of run 1 is 2.5 in orig
            ..center of run 2 is 4.5 in orig
            ..distance of 3 from 2.5 is 0.5
            ..distance of 3 from 4.5 is 1.5
            ..dividing 0.5 and 1.5 by the calibration (=binning) 2 yields the scaled distances 0.25 and 0.75
            ..=> makes sense
            */


            zTmp = ( (z - zHalfWidth) / zCal );
            zBase = (int) zTmp;
            zBaseDist = ( zTmp - zBase ) ;
            zBaseDist2 = 1 - zBaseDist ;

            if (imp.getBitDepth() == 8)
            {
                pixelsBase = getBytesAsFloats((byte[]) (imp.getStack().getProcessor(zBase + 1).getPixels()));
                if ( zBaseDist > 0 )
                    pixelsAbove = getBytesAsFloats((byte[]) (imp.getStack().getProcessor(zBase + 1 + 1 ).getPixels()));
            }
            else if (imp.getBitDepth() == 16)
            {
                pixelsBase = getShortsAsFloats((short[]) (imp.getStack().getProcessor(zBase + 1 ).getPixels()));
                if ( zBaseDist > 0 )
                    pixelsAbove = getShortsAsFloats((short[]) (imp.getStack().getProcessor(zBase + 1 + 1).getPixels()));
            }
            else if (imp.getBitDepth() == 32)
            {
                pixelsBase = (float[]) (imp.getStack().getProcessor(zBase + 1).getPixels());
                if ( zBaseDist > 0 )
                    pixelsAbove = (float[]) (imp.getStack().getProcessor(zBase + 1 + 1).getPixels());
            }

            for (y = ys; y <= ye; ++y)
            {
                yTmp = (y - yHalfWidth ) / yCal;
                yBase = (int) yTmp;
                yBaseDist = (yTmp - yBase);
                yBaseDist2 = 1 - yBaseDist;

                yBaseOffset = yBase * nxFeatureImage;
                yAboveOffset = yBaseOffset + nxFeatureImage;

                for (x = xs; x <= xe; ++x)
                {
                    xTmp = (x - xHalfWidth ) / xCal ;
                    xBase = (int) xTmp;
                    xBaseDist = ( xTmp - xBase );
                    xBaseDist2 = ( 1 - xBaseDist );
                    xAbove = xBase + 1;

                    /*
                    v000 = pixelsBase[yBaseOffset + xBase];
                    v100 = pixelsBase[yBaseOffset + xAbove];
                    vA00 = xBaseDist2 * v000 + xBaseDist * v100;

                    v010 = pixelsBase[yAboveOffset + xBase];
                    v110 = pixelsBase[yAboveOffset + xAbove];
                    vA10 = xBaseDist2 * v010 + xBaseDist * v110;
                    */

                    vA00 = xBaseDist2 * pixelsBase[yBaseOffset + xBase] + xBaseDist * pixelsBase[yBaseOffset + xAbove];
                    vA10 = xBaseDist2 * pixelsBase[yAboveOffset + xBase] + xBaseDist * pixelsBase[yAboveOffset + xAbove];
                    vAA0 = yBaseDist2 * vA00 + yBaseDist * vA10;

                    if ( zBaseDist > 0 )
                    {
                        /*
                        v001 = pixelsAbove[yBaseOffset + xBase];
                        v101 = pixelsAbove[yBaseOffset + xAbove];
                        vA01 = xBaseDist2 * v001 + xBaseDist * v101;

                        v011 = pixelsAbove[yAboveOffset + xBase];
                        v111 = pixelsAbove[yAboveOffset + xAbove];
                        vA11 = xBaseDist2 * v011 + xBaseDist * v111;
                        */

                        vA01 = xBaseDist2 * pixelsAbove[yBaseOffset + xBase] + xBaseDist * pixelsAbove[yBaseOffset + xAbove];
                        vA11 = xBaseDist2 * pixelsAbove[yAboveOffset + xBase] + xBaseDist * pixelsAbove[yAboveOffset + xAbove];
                        vAA1 = yBaseDist2 * vA01 + yBaseDist * vA11;
                    }
                    else
                    {
                        vAA1 = 0;
                    }

                    vAAA = zBaseDist2 * vAA0 + zBaseDist * vAA1;

                    featureSlice[ x - xs ][ y - ys ][ f ] = vAAA;

                }
            }
        };

    }

    public ImagePlus interpolateFast( ImagePlus imp )
    {

        Calibration calibration = imp.getCalibration();
        double xCal = calibration.pixelWidth;
        double yCal = calibration.pixelHeight;
        double zCal = calibration.pixelDepth;
        double xHalfWidth = (xCal - 1) / 2 ;
        double yHalfWidth = (yCal - 1) / 2 ;
        double zHalfWidth = (zCal - 1) / 2 ;

        int nxFeatureImage = imp.getWidth();
        int nyFeatureImage = imp.getHeight();
        int nzFeatureImage = imp.getNSlices();

        int nx = (int) (nxFeatureImage * xCal);
        int ny = (int) (nyFeatureImage * yCal);
        int nz = (int) (nzFeatureImage * zCal);

        double v000,v100,vA00,v010,v110,vA10,v001,v101,vA01,v011,v111,vA11;
        double vAA0,vAA1,vAAA;

        double xTmp, xBaseDist, xBaseDist2, zBaseDist, zBaseDist2, zTmp, yTmp;

        int zBase, xBase, yBase, yBaseOffset, yAboveOffset, xAbove;
        int ys, ye, xs, xe, zs, ze;

        // getInstancesAndMetadata feature values as doubles
        float[] pixelsBase = null;
        float[] pixelsAbove = null;


        ImageStack stack = ImageStack.create(nx, ny, nz, 32);

        long start = System.currentTimeMillis();

        zs = (int) Math.ceil(zCal * 0.5);
        ze = nz - (int) Math.ceil(zCal * 0.5);
        for (int z = zs; z < ze; z++)
        {
            zTmp = ( (z - zHalfWidth ) / zCal );
            zBase = (int) zTmp ;
            zBaseDist = zTmp - zBase ;
            zBaseDist2 = 1 - zBaseDist ;

            if (imp.getBitDepth() == 8)
            {
                pixelsBase = getBytesAsFloats((byte[]) (imp.getStack().getProcessor(zBase + 1).getPixels()));
                pixelsAbove = getBytesAsFloats((byte[]) (imp.getStack().getProcessor(zBase + 1 + 1).getPixels()));
            }
            else if (imp.getBitDepth() == 16)
            {
                pixelsBase = getShortsAsFloats((short[]) (imp.getStack().getProcessor(zBase + 1).getPixels()));
                pixelsAbove = getShortsAsFloats((short[]) (imp.getStack().getProcessor(zBase + 1 + 1).getPixels()));
            }
            else if (imp.getBitDepth() == 32)
            {
                pixelsBase = (float[]) (imp.getStack().getProcessor(zBase + 1).getPixels());
                pixelsAbove = (float[]) (imp.getStack().getProcessor(zBase + 1 + 1).getPixels());
            }

            ys = (int) Math.ceil(yCal * 0.5);
            ye = ny - ys;
            for (int y = ys; y < ye; ++y)
            {
                yTmp = (y - yHalfWidth ) / yCal ;
                yBase = (int) yTmp;
                double yBaseDist = ( yTmp - yBase );
                double yBaseDist2 = 1 - yBaseDist;

                yBaseOffset = yBase * nxFeatureImage;
                yAboveOffset = yBaseOffset + nxFeatureImage;

                xs = (int) Math.ceil(xCal * 0.5);
                xe = nx - xs;
                for (int x = xs; x < xe; ++x)
                {
                    xTmp = ( x - xHalfWidth ) / xCal;
                    xBase = (int) xTmp;
                    xBaseDist = ( xTmp - xBase );
                    xBaseDist2 = ( 1 - xBaseDist );

                    xAbove = xBase + 1;

                    v000 = pixelsBase[yBaseOffset + xBase];
                    v100 = pixelsBase[yBaseOffset + xAbove];
                    vA00 = xBaseDist2 * v000 + xBaseDist * v100;

                    v010 = pixelsBase[yAboveOffset + xBase];
                    v110 = pixelsBase[yAboveOffset + xAbove];
                    vA10 = xBaseDist2 * v010 + xBaseDist * v110;

                    v001 = pixelsAbove[yBaseOffset + xBase];
                    v101 = pixelsAbove[yBaseOffset + xAbove];
                    vA01 = xBaseDist2 * v001 + xBaseDist * v101;

                    v011 = pixelsAbove[yAboveOffset + xBase];
                    v111 = pixelsAbove[yAboveOffset + xAbove];
                    vA11 = xBaseDist2 * v011 + xBaseDist * v111;

                    vAA0 = yBaseDist2 * vA00 + yBaseDist * vA10;
                    vAA1 = yBaseDist2 * vA01 + yBaseDist * vA11;

                    vAAA = zBaseDist2 * vAA0 + zBaseDist * vAA1;

                    stack.setVoxel(x, y, z, vAAA);
                    /*
                    if ( vAAA < 0 )
                    {
                        int g = 1;
                    }*/
                }
            }
        }

        //IJ.log("Interpolation took " + (System.currentTimeMillis()-start) + "ms");

        String title = imp.getTitle()
                + " " + xCal
                + " " + yCal
                + " " + zCal;

        ImagePlus impOut = new ImagePlus(title, stack);
        return impOut;

    }

    public void interpolate( ImagePlus imp )
    {
        Calibration calibration = imp.getCalibration();

        double xCal = calibration.pixelWidth;
        double yCal = calibration.pixelHeight;
        double zCal = calibration.pixelDepth;

        double xHalfWidth = (xCal - 1) / 2 ;
        double yHalfWidth = (yCal - 1) / 2 ;
        double zHalfWidth = (zCal - 1) / 2 ;

        int nxFeatureImage = imp.getWidth();
        int nyFeatureImage = imp.getHeight();
        int nzFeatureImage = imp.getNSlices();

        int nx = (int) (nxFeatureImage * xCal);
        int ny = (int) (nyFeatureImage * yCal);
        int nz = (int) (nzFeatureImage * zCal);

        ImageStack stack = ImageStack.create(nx, ny, nz, 32);

        long start = System.currentTimeMillis();

        int zs = (int) Math.ceil(zCal * 0.5);
        int ze = nz - (int) Math.ceil(zCal * 0.5);
        for (int z = zs; z < ze; z++)
        {
            int zBase = (int) ( (z - zHalfWidth ) / zCal );
            double zBaseDist = z - ( zBase * zCal + zHalfWidth );
            int xBase, yBase, yBaseOffset, yAboveOffset, xAbove;
            int ys, ye, xs, xe;

            // getInstancesAndMetadata feature values as doubles
            float[] pixelsBase = null;
            float[] pixelsAbove = null;

            if (imp.getBitDepth() == 8)
            {
                pixelsBase = getBytesAsFloats((byte[]) (imp.getStack().getProcessor(zBase + 1).getPixels()));
                pixelsAbove = getBytesAsFloats((byte[]) (imp.getStack().getProcessor(zBase + 1 + 1).getPixels()));
            }
            else if (imp.getBitDepth() == 16)
            {
                pixelsBase = getShortsAsFloats((short[]) (imp.getStack().getProcessor(zBase + 1).getPixels()));
                pixelsAbove = getShortsAsFloats((short[]) (imp.getStack().getProcessor(zBase + 1 + 1).getPixels()));
            }
            else if (imp.getBitDepth() == 32)
            {
                pixelsBase = (float[]) (imp.getStack().getProcessor(zBase + 1).getPixels());
                pixelsAbove = (float[]) (imp.getStack().getProcessor(zBase + 1 + 1).getPixels());
            }


            ys = (int) Math.ceil(yCal * 0.5);
            ye = ny - (int) Math.ceil(yCal * 0.5);
            for (int y = ys; y < ye; ++y)
            {
                yBase = (int) ( (y - yHalfWidth ) / yCal );
                double yBaseDist = y - ( yBase * yCal + yHalfWidth );

                yBaseOffset = yBase * nxFeatureImage;
                yAboveOffset = yBaseOffset + nxFeatureImage;

                xs = (int) Math.ceil(xCal * 0.5);
                xe = nx - (int) Math.ceil(xCal * 0.5);
                for (int x = xs; x < xe; ++x)
                {
                    xBase = (int) ( (x - xHalfWidth ) / xCal );
                    double xBaseDist = x - ( xBase * xCal + xHalfWidth );

                    xAbove = xBase + 1;

                    double v000 = pixelsBase[yBaseOffset + xBase];
                    double v100 = pixelsBase[yBaseOffset + xAbove];
                    double vA00 = (xCal - xBaseDist) / xCal * v000 + xBaseDist / xCal * v100;

                    double v010 = pixelsBase[yAboveOffset + xBase];
                    double v110 = pixelsBase[yAboveOffset + xAbove];
                    double vA10 = (xCal - xBaseDist) / xCal * v010 + xBaseDist / xCal * v110;

                    double v001 = pixelsAbove[yBaseOffset + xBase];
                    double v101 = pixelsAbove[yBaseOffset + xAbove];
                    double vA01 = (xCal - xBaseDist) / xCal * v001 + xBaseDist / xCal * v101;

                    double v011 = pixelsAbove[yAboveOffset + xBase];
                    double v111 = pixelsAbove[yAboveOffset + xAbove];
                    double vA11 = (xCal - xBaseDist) / xCal * v011 + xBaseDist / xCal * v111;

                    double vAA0 = (yCal - yBaseDist) / yCal * vA00 + yBaseDist / yCal * vA10;
                    double vAA1 = (yCal - yBaseDist) / yCal * vA01 + yBaseDist / yCal * vA11;

                    double vAAA = (zCal - zBaseDist) / zCal * vAA0 + zBaseDist / zCal * vAA1;

                    stack.setVoxel(x, y, z, vAAA);
                    /*
                    if ( vAAA < 0 )
                    {
                        int g = 1;
                    }*/
                }
            }
        }

        IJ.log("Interpolation took " + (System.currentTimeMillis()-start) + "ms");
        ImagePlus impOut = new ImagePlus("interpolate", stack);
        impOut.show();

    }

    private float[] getBytesAsFloats(byte[] array)
    {
        float[] pixels = new float[array.length];
        for (int i = 0; i < array.length; ++i)
        {
            pixels[i] = (float) (array[i] & 0xFF);
        }
        return pixels;
    }

    private float[] getShortsAsFloats(short[] array)
    {
        float[] pixels = new float[array.length];
        for (int i = 0; i < array.length; ++i)
        {
            pixels[i] = (float) (array[i] & 0xFFFF);
        }
        return pixels;
    }



    public void sqrt(ImagePlus imp)
    {
        ImageStack stack = imp.getStack();

        for ( int z = 1; z <= stack.getSize(); z++ )
        {
            float[] pixels = (float[]) stack.getProcessor(z).getPixels();
            for ( int i = 0; i < pixels.length; i++ )
            {
                pixels[i] = (float) Math.sqrt( pixels[i] );
            }
        }

    }

    /**
     * Get the number of features
     *
     * @return number of features
     */
    public int getNumActiveFeatures()
    {

        if ( featureListSubset == null )
        {
            return getNumAllFeatures();
        }
        else
        {
            return featureListSubset.size();
        }

    }

    public int getNumAllFeatures()
    {
        if ( featureImages == null )
        {
            logger.error("Something went wrong during the feature computation; " +
                    "probably a memoryMB issue. Please try increasing your RAM " +
                    "and/or got to [FeatureSettings] and reduce the " +
                    "'Maximum resolution level'");
            System.gc();
        }

        return featureImages.size();

    }


    public FinalInterval addBordersXYZ( Interval interval, int[] borders ) {

        int n = interval.numDimensions();
        long[] min = new long[n];
        long[] max = new long[n];
        interval.min(min);
        interval.max(max);

        for( int d : XYZ ) {
            min[d] -= borders[d];
            max[d] += borders[d];
        }

        return new FinalInterval(min, max);
    }

    private boolean isLogging = false;

    public void isLogging( boolean isLogging )
    {
        this.isLogging = isLogging;
    }

    ArrayList< String > featureListSubset = null;

    public synchronized void setFeatureListSubset( ArrayList< String > featureListSubset )
    {
        this.featureListSubset = featureListSubset;
    }

    public boolean isFeatureNeeded( String featureImageTitle )
    {
        if ( featureListSubset == null )
        {
            return true;
        }

        if ( featureListSubset.contains( featureImageTitle ) )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    public boolean isFeatureOrChildrenNeeded( String featureImageTitle )
    {
        if ( featureListSubset == null ) return true;

        for ( String feature : featureListSubset )
        {
            if ( feature.contains( featureImageTitle ) )
            {
                return ( true );
            }
        }

        return false;

    }

    public boolean computeFeatures( int numThreads )
    {

        long start = System.currentTimeMillis();

        for ( int channel : featureSettings.activeChannels )
        {
            boolean success = computeFeatureImages( channel, numThreads, 100 );

            if ( ! success )
            {
                return false;
            }
        }

        logReport( numThreads, start );


        return true;
    }

    private void logReport( int numThreads, long start )
    {
        if ( isLogging )
        {
            logger.info("Anisotropy setting: " +  featureSettings.anisotropy );
            logger.info("Number of active features: " + getNumActiveFeatures() );
            logger.info("Features computed in [ms]: " + (System.currentTimeMillis() - start) +
                    ", using " + numThreads + " threads");
        }
    }


    /**
     * Get data cube from data taking into account and out-of-bounds strategy.
     *
     * @param imp image to read voxel values from
     * @param region5D region to extract
     * @param outOfBoundsStrategy out-of-bounds strategy to use
     * @return data cube with correct voxel values for out-of-bounds positions.
     */
    public < T extends NumericType< T > & NativeType< T >> ImagePlus getDataCube( FinalInterval interval, int channel, String outOfBoundsStrategy )
    {
        assert interval.min( T ) == interval.max( T );

        interval = IntervalUtils.fixDimension( interval, C, channel );

        FinalInterval requestedInterval3D = getFinalIntervalForXYZ( interval );

        FinalInterval existingInterval3D = new FinalInterval( new long[]{ inputImage.getWidth(), inputImage.getHeight(), inputImage.getNSlices() } );

        FinalInterval intersect3D = Intervals.intersect( existingInterval3D, requestedInterval3D );

        if ( Intervals.equals( intersect3D, requestedInterval3D ) )
        {
            return createWithinBoundsDataCube( interval, channel );
        }
        else
        {
            return createDataCubeIncludingOutOfBoundsPixels( interval, channel, requestedInterval3D, intersect3D );

        }

    }

    private < T extends NumericType< T > & NativeType< T > > ImagePlus createDataCubeIncludingOutOfBoundsPixels( FinalInterval interval, int channel, FinalInterval requestedInterval3D, FinalInterval intersect3D )
    {
        // there are out of bounds pixels
        // read data cube within bounds

        long[] minIntersect5D = new long[5];
        long[] maxIntersect5D = new long[5];

        interval.min( minIntersect5D );
        interval.max( maxIntersect5D );

        for ( int i = 0; i < 3; ++i )
        {
            minIntersect5D[ XYZ[i] ] = intersect3D.min( i );
            maxIntersect5D[ XYZ[i] ] = intersect3D.max( i );
        }

        FinalInterval intersect5D = new FinalInterval( minIntersect5D, maxIntersect5D );

        ImagePlus impWithinBounds;
        if ( channel == FG_DIST_BG_IMAGE )
        {
            impWithinBounds = resultImageFgDistBg.getDataCubeCopy( intersect5D );
        }
        else
        {
            impWithinBounds = de.embl.cba.bigDataTools.utils.Utils.getDataCube(
                    inputImage, IntervalUtils.convertIntervalToRegion5D( intersect5D ),
                    new int[]{ -1, -1 }, 1 );
        }

        // - copy impWithinBounds into a larger imp
        // that has the originally requested size
        // and pad with the chosen outOfBoundsStrategy
        // wrap it into an ImgLib image (no copying)
        final Img< T > image = ImagePlusAdapter.wrap( impWithinBounds );

        // create an infinite view where all values outside of the Interval are
        // the mirrored content, the mirror is the last pixel
        RandomAccessible< T > randomAccessible = Views.extendMirrorSingle( image );

        // in order to visualize them, we have to define a new interval
        // on them which can be displayed
        long[] min = new long[ image.numDimensions() ];
        long[] max = new long[ image.numDimensions() ];


        for ( int d = 0; d < image.numDimensions(); ++d )
        {
            min[d] = requestedInterval3D.min(d) - intersect3D.min(d);
            max[d] = min[d] + requestedInterval3D.dimension(d);
        }

        ImagePlus imagePlusIncludingOutOfBoundsRegions = createImagePlus( randomAccessible, min, max );

        ImagePlus inRAM = imagePlusIncludingOutOfBoundsRegions.duplicate();  // force into RAM, because it seems too slow otherwise

        return inRAM;
    }

    private < T extends NumericType< T > & NativeType< T > > ImagePlus createImagePlus( RandomAccessible< T > infinite, long[] min, long[] max )
    {
        FinalInterval intervalIncludingOutOfBoundsRegions = new FinalInterval( min, max );
        ImagePlus imagePlusIncludingOutOfBoundsRegions = ImageJFunctions.wrap( Views.interval( infinite, intervalIncludingOutOfBoundsRegions ), "" );
        setImagePlusDimensions( intervalIncludingOutOfBoundsRegions, imagePlusIncludingOutOfBoundsRegions );
        return imagePlusIncludingOutOfBoundsRegions;
    }

    private void setImagePlusDimensions( FinalInterval intervalIncludingOutOfBoundsPixels, ImagePlus impWithMirror )
    {
        int numSlicesIncludingOutOfBoundsSlices = 0;

        if ( intervalIncludingOutOfBoundsPixels.numDimensions() == 3 )
        {
            numSlicesIncludingOutOfBoundsSlices = (int) intervalIncludingOutOfBoundsPixels.dimension(2);
        }
        else if ( intervalIncludingOutOfBoundsPixels.numDimensions() == 2 )
        {
            numSlicesIncludingOutOfBoundsSlices = 1;
        }


        impWithMirror.setDimensions(1, numSlicesIncludingOutOfBoundsSlices, 1 );
    }

    private ImagePlus createWithinBoundsDataCube( FinalInterval interval, int channel )
    {
        // everything is within bounds
        if ( channel == FG_DIST_BG_IMAGE )
        {
            return ( resultImageFgDistBg.getDataCubeCopy(interval) );
        }
        else
        {
            return de.embl.cba.bigDataTools.utils.Utils.getDataCube( inputImage,
                    IntervalUtils.convertIntervalToRegion5D( interval ),
                    new int[]{ -1, -1 }, 1 );
        }
    }

    private FinalInterval getFinalIntervalForXYZ( FinalInterval interval )
    {
        return new FinalInterval(
                    new long[]{ interval.min( X ), interval.min( Y ), interval.min( Z ) },
                    new long[]{ interval.max( X ), interval.max( Y ), interval.max( Z ) } );
    }


    public static int getNumResolutionLevels( ArrayList< Integer > binFactors )
    {
        int maxBinLevel = 0;
        for ( int b : binFactors )
        {
            if ( b > 0 )
            {
                maxBinLevel++;
            }
        }
        return maxBinLevel;
    }

    /**
     * Update features with current list in a multi-thread fashion
     *
     * @return true if the features are correctly updated
     */
    public boolean computeFeatureImages( int channel, int numThreads, int maximumMultithreadedLevel )
    {
        int numResolutionLevels = getNumResolutionLevels( featureSettings.binFactors );

        double adaptiveAnisotropy = featureSettings.anisotropy;

        ImagePlus processedOriginalImage = getProcessedOriginalImage( channel );

        setUnitMicrometerCalibration( processedOriginalImage );

        ArrayList < ArrayList < ImagePlus > > multiResolutionFeatureImages = new ArrayList<>();

        try
        {
            for ( int resolutionLayer = 0; resolutionLayer < numResolutionLevels; ++resolutionLayer )
            {
                if ( deepSegmentation.stopCurrentTasks || Thread.currentThread().isInterrupted() ) return ( false );

                long start = System.currentTimeMillis();

                final ArrayList<ImagePlus> featureImagesThisResolution = new ArrayList<>();

                adaptiveAnisotropy = downsampleImagesForCurrentResolutionLayer( numThreads, numResolutionLevels, adaptiveAnisotropy, processedOriginalImage, multiResolutionFeatureImages, resolutionLayer, featureImagesThisResolution );

                computeAndAddFeatureImagesForCurrentResolutionLayer( numThreads, maximumMultithreadedLevel, adaptiveAnisotropy, multiResolutionFeatureImages, resolutionLayer, featureImagesThisResolution );

                logProgress( numThreads, start, resolutionLayer );
            }

            putIntoFeatureImagesMap( multiResolutionFeatureImages );

        }
        catch (InterruptedException ie)
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

        return true;
    }

    private void logProgress( int numThreads, long start, int level )
    {
        if ( isLogging )
        {
            logger.info("Image features at level " + level + " computed in [ms]: " + (System.currentTimeMillis() - start) + ", using " + numThreads + " threads." );
        }
    }

    private void computeAndAddFeatureImagesForCurrentResolutionLayer( int numThreads, int maximumMultithreadedLevel, double adaptiveAnisotropy, ArrayList< ArrayList< ImagePlus > > multiResolutionFeatureImages, int level, ArrayList< ImagePlus > featureImagesThisResolution ) throws Exception
    {

        ExecutorService exe;

        boolean hessianAbsoluteValues = false;

        // Multi-threaded
        exe = Executors.newFixedThreadPool( numThreads );
        ArrayList<Future<ArrayList<ImagePlus>> > featureFutures = new ArrayList<>();

        // Single threaded
        ArrayList<ArrayList<ImagePlus>> featureImagesList = new ArrayList<>();

        Calibration calibrationThisResolution = featureImagesThisResolution.get(0).getCalibration().copy();

        Calibration calibrationForFeatureComputation = getCalibrationForFeatureComputation( adaptiveAnisotropy );

        for ( ImagePlus featureImage : featureImagesThisResolution )
        {
            // temporarily change calibration while computing
            // hessian and structure
            // in order to account for anisotropy in xy vs z
            featureImage.setCalibration( calibrationForFeatureComputation );

            for ( int smoothingScale : featureSettings.smoothingScales )
            {
                if ( ! featureImage.getTitle().contains( CONV_DEPTH + deepSegmentation.featureSettings.maxDeepConvLevel ) )
                {
                    if ( level <= maximumMultithreadedLevel ) // multi-threaded
                    {
                        if ( isFeatureOrChildrenNeeded( getFilterBaseName( HESSIAN, smoothingScale ) + featureImage.getTitle() ) )
                            featureFutures.add( exe.submit( getHessianOrStructure( HESSIAN, featureImage, smoothingScale ) ) );

                        if ( isFeatureOrChildrenNeeded( getFilterBaseName( STRUCTURE, smoothingScale )  + featureImage.getTitle() ) )
                            featureFutures.add( exe.submit( getHessianOrStructure( STRUCTURE, featureImage, smoothingScale ) ) );

                        if ( featureSettings.commputeGaussian )
                        {
                            if ( isFeatureOrChildrenNeeded( getFilterBaseName( GAUSSIAN, smoothingScale )  + featureImage.getTitle() ) )
                            {
                                featureFutures.add( exe.submit( getGaussian( featureImage, smoothingScale ) ) );
                            }
                        }
                    }
                    else // single-threaded
                    {
                        if ( isFeatureOrChildrenNeeded( getFilterBaseName( HESSIAN, smoothingScale )  + featureImage.getTitle() ) )
                            featureImagesList.add( getHessianOrStructure( HESSIAN, featureImage, smoothingScale ).call() );

                        if ( isFeatureOrChildrenNeeded( getFilterBaseName( STRUCTURE, smoothingScale ) + featureImage.getTitle() ) )
                            featureImagesList.add( getHessianOrStructure( STRUCTURE, featureImage, smoothingScale ).call() );

                        if ( featureSettings.commputeGaussian )
                        {
                            if ( isFeatureOrChildrenNeeded( getFilterBaseName( GAUSSIAN, smoothingScale )  + featureImage.getTitle() ) )
                            {
                                featureImagesList.add( getGaussian( featureImage, smoothingScale ).call() );
                            }
                        }
                    }
                }
            }
        }

        collectFeatureImages( maximumMultithreadedLevel, level, exe, featureFutures, featureImagesList );

        addFeatureImagesToMultiResolutionFeatureImageList( multiResolutionFeatureImages, featureImagesThisResolution, calibrationThisResolution, featureImagesList );

    }

    private Calibration getCalibrationForFeatureComputation( double adaptiveAnisotropy )
    {
        Calibration calibrationForFeatureComputation = new Calibration();
        calibrationForFeatureComputation.pixelWidth = 1.0;
        calibrationForFeatureComputation.pixelHeight = 1.0;
        calibrationForFeatureComputation.pixelDepth = 1.0 * adaptiveAnisotropy;
        calibrationForFeatureComputation.setUnit("micrometer");
        return calibrationForFeatureComputation;
    }

    private double downsampleImagesForCurrentResolutionLayer( int numThreads, int numLevels, double adaptiveAnisotropy, ImagePlus inputImage, ArrayList< ArrayList< ImagePlus > > multiResolutionFeatureImages, int level, ArrayList< ImagePlus > featureImagesThisResolution ) throws InterruptedException, ExecutionException
    {
        DownSampler downSampler = new DownSampler( featureSettings.downSamplingMethod );

        final ArrayList<ImagePlus> featureImagesPreviousResolution;

        if ( level == 0 )
        {
            featureImagesPreviousResolution = new ArrayList<>(  );
            featureImagesPreviousResolution.add( inputImage );
        }
        else
        {
            featureImagesPreviousResolution = multiResolutionFeatureImages.get( level - 1 );
        }

        int[] binning = getBinning( featureImagesPreviousResolution.get(0), adaptiveAnisotropy, featureSettings.binFactors.get( level ) );

        // add binning information to image title
        String binningTitle = "Bin" + binning[0] + "x" + binning[1] + "x" + binning[2];

        // adapt settingAnisotropy, which could have changed during
        // the (anisotropic) binning
        adaptiveAnisotropy /= 1.0 * binning[0] / binning[2];

        ExecutorService exe = Executors.newFixedThreadPool( numThreads );
        ArrayList<Future<ImagePlus> > futuresBinning = new ArrayList<>();

        for ( ImagePlus featureImage : featureImagesPreviousResolution )
        {
            if ( level == numLevels - 1 )
            {
                /*
                don't bin but smooth at the last resolution level
                to better preserve the spatial information.
                */

                // TODO:
                // - check again for boundary effects!

                // TODO:
                // - don't compute this feature if not needed

                // TODO:
                // - maybe change below to gaussian smoothing for better
                // derivative computation further down

                /*
                 currently, below an average filter is computed.
                 for computing image derivatives this might not be ideal
                 because the difference of two shifted means only
                 reflects difference between the two pixels at the edge
                 of the mean filter, which could be quite noisy.
                 on the other hand, the hessian and structure themselves
                 gaussian-smooth a bit before computing the derivatives
                 such that it actually might be ok.
                 */

                int[] radii = new int[ 3 ];

                for( int i = 0; i < 3; ++i )
                {
                    radii[i] = (int) Math.ceil ( ( binning[i] - 1 ) / 2.0 );
                }

                futuresBinning.add( exe.submit( filter3d( featureImage, radii ) ) );
            }
            else
            {
                if ( isFeatureOrChildrenNeeded( binningTitle + "_" + featureImage.getTitle() ) )
                {
                    futuresBinning.add( exe.submit( downSampler.run( featureImage, binning, binningTitle ) ) );
                }
            }
        }

        for ( Future<ImagePlus> f : futuresBinning )
        {
            // getInstancesAndMetadata feature images
            featureImagesThisResolution.add( f.get() );
        }

        futuresBinning = null;
        exe.shutdown();
        System.gc();
        return adaptiveAnisotropy;
    }

    private void collectFeatureImages( int maximumMultithreadedLevel, int level, ExecutorService exe, ArrayList< Future< ArrayList< ImagePlus > > > futuresFeatures, ArrayList< ArrayList< ImagePlus > > featureImagesList ) throws InterruptedException, ExecutionException
    {
        if ( level <= maximumMultithreadedLevel )
        {
            for (Future<ArrayList<ImagePlus>> f : futuresFeatures)
            {
                featureImagesList.add( f.get() );
            }
        }
        futuresFeatures = null;
        exe.shutdown();
        System.gc();
    }

    private void addFeatureImagesToMultiResolutionFeatureImageList( ArrayList< ArrayList< ImagePlus > > multiResolutionFeatureImages, ArrayList< ImagePlus > featureImagesThisResolution, Calibration calibrationThisResolution, ArrayList< ArrayList< ImagePlus > > featureImagesList )
    {
        for (ArrayList<ImagePlus> featureImages : featureImagesList )
        {
            for ( ImagePlus featureImage : featureImages )
            {
                // add feature images to this resolution layer
                putDeepConvFeatLevelIntoTitle( featureImage );
                featureImagesThisResolution.add( featureImage );
            }
        }

        // (re-)set calibrations of all images at this resolution
        for (ImagePlus featureImage : featureImagesThisResolution)
        {
            featureImage.setCalibration( calibrationThisResolution.copy() );
        }

        // and add everything to the multi-resolution array
        multiResolutionFeatureImages.add( featureImagesThisResolution );


    }

    private void putIntoFeatureImagesMap( ArrayList< ArrayList< ImagePlus > > multiResolutionFeatureImages )
    {
        for ( ArrayList<ImagePlus> featureImages : multiResolutionFeatureImages )
        {
            for ( ImagePlus featureImage : featureImages )
            {
                if ( featureImage.getTitle().equals( showFeatureImageTitle ) )
                {
                    //ImagePlus imp = interpolateFast(featureImage);
                    featureImage.show();

                }

                if ( isFeatureNeeded( featureImage.getTitle() ) )
                {
                    this.featureImages.put(  featureImage.getTitle(), featureImage );
                }
            }
        }
    }

    private void setUnitMicrometerCalibration( ImagePlus inputImage )
    {
        // Set a calibration that can be changed during the binning
        Calibration calibration = new Calibration();
        calibration.pixelDepth = 1;
        calibration.pixelWidth = 1;
        calibration.pixelHeight = 1;
        calibration.setUnit("um");
        inputImage.setCalibration( calibration );
    }

    private ImagePlus getProcessedOriginalImage( int channel )
    {
        FinalInterval expandedInterval = addBordersXYZ( interval, featureImageBorderSizes );

        ImagePlus processedInputImage = getDataCube( expandedInterval, channel, "mirror" );

        // pre-processing
        if ( deepSegmentation.featureSettings.log2 && channel != FG_DIST_BG_IMAGE )
        {
            // subtract background
            IJ.run( processedInputImage, "Subtract...", "value=" + deepSegmentation.featureSettings.imageBackground + " stack" );
            // make sure there are no zeros, because the log will give -Infinity
            IJ.run( processedInputImage, "Add...", "value=1 stack" );

            // log transformation to go to multiplicative math
            IJ.run( processedInputImage, "32-bit", "" );
            IJ.run( processedInputImage, "Log", "stack" );
        }

        processedInputImage.setTitle( "Orig_ch" + channel );
        return processedInputImage;
    }

    private void putDeepConvFeatLevelIntoTitle(ImagePlus imp)
    {
        String title = imp.getTitle();

        if ( title.contains(CONV_DEPTH+"3") ) { logger.warning("Too deep convolution!"); return; };
        if ( title.contains(CONV_DEPTH+"2") ) { imp.setTitle(CONV_DEPTH+"3_" + title); return; }
        if ( title.contains(CONV_DEPTH+"1") ) { imp.setTitle(CONV_DEPTH+"2_" + title); return; }
        imp.setTitle(CONV_DEPTH+"1_" + title);

        return;

    }

    public Callable<ImagePlus> filter3d( ImagePlus inputImage, int[] radii )
    {
        return () -> {

            if ( radii[0] == 0 && radii[1] == 0 && radii[2] == 0 )
            {
                return inputImage;
            }

            ImageStack result = ImageStack.create( inputImage.getWidth(), inputImage.getHeight(), inputImage.getNSlices(), 32 );
            StackProcessor stackProcessor = new StackProcessor( inputImage.getStack() );

            stackProcessor.filter3D( result,
                    (float) radii[0], (float) radii[1], (float) radii[2],
                    0, result.size(),
                    StackProcessor.FILTER_MEAN );

            String title = inputImage.getTitle();
            // TODO: change from "_" to "x" (issue backward compatibility)
            ImagePlus filteredImage = new ImagePlus(
                            String.format("Mean%d", (radii[0]*2)+1)
                            + String.format("_%d", (radii[1]*2)+1)
                            + String.format("_%d", (radii[2]*2)+1)
                            + "_" + title,
                    result );

            filteredImage.setCalibration( inputImage.getCalibration().copy() );
            return ( filteredImage );
        };
    }


    public Callable<ImagePlus> fastFilter3d(ImagePlus imp, float[] radii)
    {
        return () -> {


            /*
            ImageFloat wrap = ;
            ImageFloat HandMedian = FastFilters3D.filterFloatImage(
                    new ImageFloat( imp ),
                    FastFilters3D., median_radius_xy,median_radius_xy, median_radius_z, ncpu,false);
            ImagePlus medianImage=HandMedian.getWholeImageCopy();


            ImageStack resultImagePlus = ImageStack.create( imp.getWidth(), imp.getHeight(), imp.getNSlices(), 32 );
            StackProcessor stackProcessor = new StackProcessor( imp.getStack() );

            stackProcessor.filter3D( resultImagePlus,
                    (float) radii[0], (float) radii[1], (float) radii[2],
                    0, resultImagePlus.size(),
                    StackProcessor.FILTER_MEAN );

            String title = imp.getTitle();
            ImagePlus impResult = new ImagePlus(
                    String.format("Mean%d", (radii[0]*2)+1)
                            + String.format("_%d", (radii[1]*2)+1)
                            + String.format("_%d", (radii[2]*2)+1)
                            + "_" + title,
                    resultImagePlus );

            impResult.setUnitMicrometerCalibration( imp.getCalibration().copy() );
            return ( impResult );
            */
            return null;
        };
    }

    private int[] getBinning(ImagePlus imp,
                             double anisotropy,
                             int binFactor)
    {
        int[] binning = new int[3];

        binning[0] = binFactor; // x-binning
        binning[1] = binFactor; // y-binning

        if ( imp.getNSlices() == 1 )
        {
            binning[2] = 1;  // 2D
        }
        else
        {
            // potentially run less in z
            binning[2] = (int) Math.ceil( binFactor / anisotropy );
            if( binning[2]==0 ) binning[2] = 1;
        }

        return ( binning );

    }

    public ArrayList<String> getAllFeatureNames()
    {
        return new ArrayList<>( featureImages.keySet() );
    }

    public ArrayList<String> getActiveFeatureNames()
    {
        if ( featureListSubset == null )
        {
            return new ArrayList<>( featureImages.keySet() );
        }
        else
        {
            return featureListSubset;
        }

    }


}

	
