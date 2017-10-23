package trainableDeepSegmentation;

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

import bigDataTools.Region5D;
import bigDataTools.logging.IJLazySwingLogger;
import bigDataTools.logging.Logger;
import com.sun.org.apache.regexp.internal.RE;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;
import ij.plugin.Binner;
import ij.plugin.Duplicator;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.StackProcessor;
import javafx.geometry.Point3D;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import trainableDeepSegmentation.filters.HessianImgLib2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
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
    /** original input image */
    private ImagePlus inputImage = null;

    /** the feature images */
    private ArrayList < ImagePlus > multiResolutionFeatureImageArray = new ArrayList<>();

    /** names of feature images */
    private ArrayList<String> featureNames = new ArrayList<>();

    /** image width */
    private int width = 0;
    /** image height */
    private int height = 0;

    private ArrayList<Integer> activeChannels;

    private final String CONV_DEPTH = "CD";

    /** Hessian filter flag index */
    public static final int HESSIAN 				=  0;
    /** structure tensor filter flag index */
    public static final int STRUCTURE 				=  1;
    /** Minimum flag index */
    public static final int MINIMUM					=  2;
    /** Maximum flag index */
    public static final int MAXIMUM					=  3;
    /** Mean flag index */
    public static final int AVERAGE                 =  4;
    /** Median flag index */

    private int X = 0, Y = 1, C = 2, Z = 3, T = 4;
    private int[] XYZ = new int[]{ X, Y, Z};
    private int[] XYZT = new int[]{ X, Y, Z, T};

    private int[] featureImageBorderSizes = new int[5];

    public WekaSegmentation wekaSegmentation = null;

    /** flags of filters to be used */
    private boolean[] enabledFeatures = new boolean[]{
            true, 	/* Hessian */
            true,	/* Structure */
            false,	/* Minimum */
            false,	/* Maximum */
            true,	/* Mean */
            };

    /** names of available filters */
    public static final String[] availableFeatures
            = new String[]{	"Hessian", "Structure", "Minimum", "Maximum", "Mean" };

    /** index of the feature stack that is used as reference (to read attribute, etc.).
     * -1 if not defined yet. */
    private int referenceStackIndex = -1;

    private Logger logger;

    private FinalInterval interval = null;

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
    public FeatureProvider()
    {
    }

    public FeatureProvider(ImagePlus imp )
    {
        setInputImage(imp);
    }

    public void setInputImage(ImagePlus imp )
    {
        width = imp.getWidth();
        height = imp.getHeight();
        inputImage = imp;
        inputImage.setTitle("Orig");
    }

    public void setActiveChannels( ArrayList<Integer> activeChannels )
    {
        this.activeChannels = activeChannels;
    }

    public void setInterval( FinalInterval interval )
    {
        this.interval = interval;
    }

    public FinalInterval getInterval( )
    {
        return ( interval );
    }

    public long getIntervalWidth ( int d )
    {
        long width = interval.max( d ) - interval.min( d ) + 1;
        return ( width );
    }

    public void setWekaSegmentation( WekaSegmentation wekaSegmentation )
    {
        this.wekaSegmentation = wekaSegmentation;
        setLogger( wekaSegmentation.getLogger() );
    }

    public void setLogger( Logger logger )
    {
        this.logger = logger;
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

                String filterBaseName = "He"; //+(int)(sigma);

                ArrayList<ImagePlus>[] results = new ArrayList[ channels.length ];

                for(int ch=0; ch < channels.length; ch++)
                {
                    results[ch] = new ArrayList<ImagePlus>();

                    final ImagePlus channel = channels[ch].duplicate();

                    if (channel.getNSlices() > 1)
                    {
                        // pad 3-D image on the back and the front
                        channel.getImageStack().addSlice("pad-back", channels[ch].getImageStack().getProcessor(channels[ch].getImageStackSize()));
                        channel.getImageStack().addSlice("pad-front", channels[ch].getImageStack().getProcessor(1), 1);
                    }
                    final ArrayList<ImagePlus> result = ImageScience.computeHessianImages(sigma, absolute, channel);
                    final ImageStack largest = result.get(0).getImageStack();
                    final ImageStack middle = result.get(1).getImageStack();

                    if (channel.getNSlices() > 1) // 3D
                    {
                        // remove pad
                        largest.deleteLastSlice();
                        largest.deleteSlice(1);
                        middle.deleteLastSlice();
                        middle.deleteSlice(1);
                    }
                    results[ ch ].add(new ImagePlus("L" + filterBaseName + "_" + originalImage.getTitle(), largest));
                    results[ ch ].add(new ImagePlus("M" + filterBaseName + "_" + originalImage.getTitle(), middle));

                    if ( result.size() == 3 ) // 3D
                    {
                        final ImageStack smallest = result.get(2).getImageStack();
                        smallest.deleteLastSlice();
                        smallest.deleteSlice(1);
                        results[ ch ].add(new ImagePlus("S"+ filterBaseName + "_" + originalImage.getTitle(), smallest));
                    }

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

        int nf = getNumFeatures();

        System.arraycopy( featureSlice[x][y], 0, values, 0, nf );
        values[ nf ] = classNum;

    }

    public double[][][] getReusableFeatureSlice()
    {
        // allocate memory for featureSlice
        double[][][] featureSlice = new double
                [(int) interval.dimension(X)]
                [(int) interval.dimension(Y)]
                [getNumFeatures() + 1]; // last one for class ID

        return ( featureSlice );
    }

    /**
     * set all feature values for one z-slice
     * coordinates are relative to within the set interval
     * @param zGlobal
     * @param featureSlice
     */
    public boolean setFeatureSliceValues(final int zGlobal, double[][][] featureSlice )
    {


        if ( (zGlobal > interval.max(Z)) || (zGlobal < interval.min(Z)) )
        {
            logger.error("No features have been computed for slice " + zGlobal);
            return false;
        }


        int xs = 0;
        int xe = (int) (interval.dimension( X ) - 1);
        int ys = 0;
        int ye = (int) (interval.dimension( Y ) - 1);
        int z = zGlobal - (int) interval.min( Z );
        int nf = getNumFeatures();


        // The feature images in multiResolutionFeatureImageArray
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

        double v000,v100,vA00,v010,v110,vA10,v001,v101,vA01,v011,v111,vA11;
        double vAA0,vAA1,vAAA;
        double xCal, yCal, zCal, xHalfWidth, yHalfWidth, zHalfWidth;
        double xTmp, xBaseDist, xBaseDist2;
        double zTmp, zBaseDist, zBaseDist2;
        double yTmp, yBaseDist, yBaseDist2;

        int xBase, yBase, zBase, yBaseOffset, yAboveOffset, xAbove, x, y;
        int nxFeatureImage;
        Calibration calibration = null;
        ImagePlus imp = null;

        // get feature values as doubles
        float[] pixelsBase = null;
        float[] pixelsAbove = null;

        for ( int f = 0; f < nf; f++ )
        {
            //if ( ! featureList.get( f + c*nf ).isActive )
            //{
            //    continue; // don't upsample if not needed
            //}

            imp = multiResolutionFeatureImageArray.get(f);
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
            3 is 2 away from 1, which is the center of bin 0
            3 is 1 away from 4, which is the center of bin 1

            # Example: binning 3, value 4

            orig: 0,1,2,3,4,5,6,7,8
            bin:  .,0,.|.,1,.|.,2,.
            value at orig 4 should be computed from bin 1, because it is its center
            halfwidth = (3-1)/2 = 1
            tmp = (4 - halfwidth) / 3 = 3 / 3 = 1
            base = (int) tmp = 1
            baseDist = tmp - base = 1 - 1 = 0
            baseDist2 = 1 - 0 = 1
            ..this means that bin 2 will get a weight of 0 (i.e., baseDist)
            ..and bin 1 will get a weight of 1

            # Example: binning=cal=2, value 3

            orig: 0,1,2,3,4,5,6,7,8
            bin:  .0.|.1.|.2.|.3.|
            value at orig 3 should be computed from bin 1 and bin 2
            halfwidth = (cal-1)/2 = (2-1)/2 = 0.5
            tmp = (3 - halfwidth) / cal = (3-0.5)/2 = 2.5/2 = 1.25
            base = (int) tmp = 1  (=> above will be 2)
            baseDist = tmp - base = 1.25 - 1 = 0.25
            baseDist2 = 1 - 0.25 = 0.75
            ..center of bin 1 is 2.5 in orig
            ..center of bin 2 is 4.5 in orig
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
                    pixelsAbove = getBytesAsFloats((byte[]) (imp.getStack().getProcessor(zBase + 1 + 1).getPixels()));
            }
            else if (imp.getBitDepth() == 16)
            {
                pixelsBase = getShortsAsFloats((short[]) (imp.getStack().getProcessor(zBase + 1).getPixels()));
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

                    featureSlice[x-xs][y-ys][f] = vAAA;

                }
            }
        }

        return ( true );
    }


    public ImagePlus interpolateFast(ImagePlus imp)
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

        // get feature values as doubles
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

    public void interpolate(ImagePlus imp)
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

            // get feature values as doubles
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

        return new Callable<ArrayList<ImagePlus>>() {
            public ArrayList<ImagePlus> call()
            {
                ArrayList<ImagePlus> results = new ArrayList<>();

                String filterBaseName = "St";//+(int)(sigma);

                ImagePlus tmp = originalImage.duplicate();

                if (tmp.getNSlices() > 1) // 3-D
                {
                    // pad 3-D image on the back and the front
                    tmp.getImageStack().addSlice("pad-back", tmp.getImageStack().getProcessor(tmp.getImageStackSize()));
                    tmp.getImageStack().addSlice("pad-front", tmp.getImageStack().getProcessor(1), 1);
                }

                final ArrayList<ImagePlus> result = ImageScience.computeEigenimages(sigma, integrationScale, tmp);
                final ImageStack largest = result.get(0).getImageStack();
                final ImageStack middle = result.get(1).getImageStack();

                // remove pad
                if (tmp.getNSlices() > 1) // 3-D
                {
                    largest.deleteLastSlice();
                    largest.deleteSlice(1);
                    middle.deleteLastSlice();
                    middle.deleteSlice(1);
                }
                results.add(new ImagePlus("L"+filterBaseName+"_" + originalImage.getTitle(), largest));
                results.add(new ImagePlus("M"+filterBaseName+"_" + originalImage.getTitle(), middle));

                if (result.size() == 3) // 3D
                {
                    final ImageStack smallest = result.get(2).getImageStack();
                    smallest.deleteLastSlice();
                    smallest.deleteSlice(1);
                    results.add(new ImagePlus("S"+filterBaseName+"_" + originalImage.getTitle(), smallest));
                }

                // remove the square as it over-pronounces strong edges and might
                // (or might not?) confuse the classifier
                for (ImagePlus imp : results)
                {
                    sqrt(imp);
                }

                return (results);
            }
        };
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
    public int getNumFeatures()
    {
        if ( multiResolutionFeatureImageArray == null )
        {
            logger.error("Something went wrong during the feature computation; " +
                    "probably a memory issue. Please try increasing your RAM " +
                    "and/or got to [Settings] and reduce the " +
                    "'Maximum resolution level'");
            System.gc();
        }
        return multiResolutionFeatureImageArray.size();
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

    public void setLogging( boolean isLogging )
    {
        this.isLogging = isLogging;
    }

    public boolean computeFeatures( int numThreads, boolean computeAllFeatures )
    {

        for ( int c = (int) interval.min( C ); c <= interval.max( C ); ++c )
        {
            if ( activeChannels.contains( c ) )
            {
                FinalInterval intervalOneChannel = wekaSegmentation.fixDimension(interval, C, c);

                boolean success = computeFeatureImages(
                        intervalOneChannel,
                        numThreads,
                        computeAllFeatures);

                if ( ! success )
                {
                    return false;
                }
            }
        }

        return true;
    }


    /**
     * Get data cube from data taking into account and out-of-bounds strategy.
     *
     * @param imp image to read voxel values from
     * @param region5D region to extract
     * @param outOfBoundsStrategy out-of-bounds strategy to use
     * @return data cube with correct voxel values for out-of-bounds positions.
     */
    public < T extends NumericType< T > & NativeType< T >> ImagePlus getDataCube(
            ImagePlus imp,
            FinalInterval interval,
            String outOfBoundsStrategy )
    {

        FinalInterval interval3D = new FinalInterval(
                new long[]{ interval.min( X ), interval.min( Y ), interval.min( Z ) },
                new long[]{ interval.max( X ), interval.max( Y ), interval.max( Z ) } );
        FinalInterval impInterval3D = new FinalInterval( new long[]{ imp.getWidth(), imp.getHeight(), imp.getNSlices() } );
        FinalInterval intersect3D = Intervals.intersect( impInterval3D, interval3D );

        if ( Intervals.equals( intersect3D, interval3D ) )
        {
            // everything is within bounds
            return bigDataTools.utils.Utils.getDataCube(imp,
                    wekaSegmentation.convertIntervalToRegion5D( interval ),
                    new int[]{-1, -1}, 1);
        }


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
        ImagePlus impWithinBounds = bigDataTools.utils.Utils.getDataCube(
                imp, wekaSegmentation.convertIntervalToRegion5D( intersect5D ),
                new int[]{-1,-1}, 1 );

        // - copy impWithinBounds into a larger imp
        // that has the originally requested size
        // and pad with the chosen outOfBoundsStrategy
        // wrap it into an ImgLib image (no copying)
        final Img< T > image = ImagePlusAdapter.wrap( impWithinBounds );

        // create an infinite view where all values outside of the Interval are
        // the mirrored content, the mirror is the last pixel
        RandomAccessible< T > infinite = Views.extendMirrorSingle( image );

        // in order to visualize them, we have to define a new interval
        // on them which can be displayed
        long[] min = new long[ image.numDimensions() ];
        long[] max = new long[ image.numDimensions() ];


        for ( int d = 0; d < image.numDimensions(); ++d )
        {
            min[d] = interval3D.min(d) - intersect3D.min(d);
            max[d] = min[d] + interval3D.dimension(d);
        }

        // define the Interval on the infinite random accessible
        FinalInterval interval2 = new FinalInterval( min, max );
        ImagePlus impWithMirror = ImageJFunctions.wrap( Views.interval( infinite, interval2 ), "" );
        impWithMirror.setDimensions(1, (int) interval2.dimension(2), 1 );

        //impWithMirror.show();

        // force into RAM, because seems too slow otherwise
        ImagePlus impWithMirrorDuplicate = impWithMirror.duplicate();

        return ( impWithMirrorDuplicate );
    }


    /**
     * Update features with current list in a multi-thread fashion
     *
     * @return true if the features are correctly updated
     */
    public boolean computeFeatureImages(
            FinalInterval interval,
            int numThreads,
            boolean computeAllFeatures )
    {
        // TODO:
        // - bit of a mess which variables are passed on via
        // wekaSegmentation object and which not
        double anisotropy = wekaSegmentation.settings.anisotropy;
        String channelName = "ch" + interval.min( C );

        // get the larger part of original image that is
        // needed to compute features for requested interval.
        featureImageBorderSizes = wekaSegmentation.getFeatureBorderSizes();
        FinalInterval expandedInterval = addBordersXYZ( interval,
                featureImageBorderSizes );


        long start = System.currentTimeMillis();

        ImagePlus inputImageCrop = getDataCube( inputImage, expandedInterval, "mirror" );

        if ( isLogging )
        {
            logger.info("Image data cube loaded in [ms]: " +
                    (System.currentTimeMillis() - start));
        }



        // Set a calibration that can be changed during the binning
        Calibration calibration = new Calibration();
        calibration.pixelDepth = 1;
        calibration.pixelWidth = 1;
        calibration.pixelHeight = 1;
        calibration.setUnit("um");
        inputImageCrop.setCalibration( calibration );
        inputImageCrop.setTitle( inputImage.getTitle() + "_" + channelName );

        // ResolutionLevelList of ImageList
        ArrayList < ArrayList < ImagePlus > > multiResolutionFeatureImages = new ArrayList<>();

        try
        {

            //
            // BINNING
            //

            ArrayList< int[] > binnings = new ArrayList<>();

            for (int level = 0; level <= wekaSegmentation.settings.maxResolutionLevel; level++)
            {

                start = System.currentTimeMillis();

                if ( wekaSegmentation.stopCurrentThreads || Thread.currentThread().isInterrupted() )
                {
                    return ( false );
                }


                final ArrayList<ImagePlus> featureImagesThisResolution = new ArrayList<>();
                final ArrayList<ImagePlus> featureImagesPreviousResolution;

                if ( level == 0 )
                {

                    featureImagesThisResolution.add( inputImageCrop );
                    binnings.add( new int[]{1, 1, 1} );

                }
                else
                {
                    // bin images of previous resolution
                    // (no multi-threading because this is very fast)

                    featureImagesPreviousResolution = multiResolutionFeatureImages.get(level - 1);

                    // compute how much should be binned;
                    // if the data is anisotropic
                    // we bin anisotropic as well
                    int[] binning = getBinning(
                            featureImagesPreviousResolution.get(0),
                            anisotropy,
                            wekaSegmentation.settings.downSamplingFactor);

                    int[] combinedBinning = new int[3];
                    for ( int i = 0; i < 3; i++ )
                    {
                        combinedBinning[i] = binnings.get( level - 1 )[i] * binning[i];
                    }
                    binnings.add( combinedBinning ); // remember for feature name construction

                    // add binning information to image title
                    String binningTitle = "Bin" +
                            binning[0] + "x" +
                            binning[1] + "x" +
                            binning[2];

                    // adapt settingAnisotropy, which could have changed during
                    // the (anisotropic) binning
                    anisotropy /= 1.0 * binning[0] / binning[2];

                    for (ImagePlus featureImage : featureImagesPreviousResolution)
                    {
                        if ( level == wekaSegmentation.settings.maxResolutionLevel )
                        {
                            /*
                            don't bin but smooth last scale to better preserve
                            spatial information.
                            smoothing radius of 3 is the largest that will not
                            cause boundary effects,
                            because the ignored border during classification is
                            3 pixel at settings.maxResolutionLevel - 1
                            (i.e. 1 pixel at settings.maxResolutionLevel)
                            */

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

                            int filterRadius = 2;
                            featureImagesThisResolution.add( filter3d( featureImage,
                                    filterRadius ) );

                            //featureImagesThisResolution.add( bin(featureImage, binning, "AVERAGE").call());
                        }
                        else
                        {

                            if ( computeAllFeatures || wekaSegmentation.isFeatureOrChildrenNeeded(
                                    binningTitle + "_" +
                                            featureImage.getTitle()) )
                            {
                                featureImagesThisResolution.add(
                                        bin( featureImage,
                                                binning,
                                                binningTitle, "AVERAGE").call() );
                            }
                        }
                    }

                }

                Calibration calibrationThisResolution =
                        featureImagesThisResolution.get(0).getCalibration().copy();

                //
                // Compute features
                //

                double smoothingScale = 1.0;
                double integrationScale = smoothingScale;

                if ( level == wekaSegmentation.settings.maxResolutionLevel )
                {
                    // for the last scale we do not bin but only smooth (s.a.),
                    // thus we need to look for features over a wider range
                    integrationScale *= 2.0;
                    // smoothingScale does not need to be increased, because the images are smoothed already
                }

                boolean hessianAbsoluteValues = false;

                // Multi-threaded
                ExecutorService exe = Executors.newFixedThreadPool(numThreads);
                ArrayList<Future<ArrayList<ImagePlus>>> futures = new ArrayList<>();

                // Single threaded
                ArrayList<ArrayList<ImagePlus>> featureImagesList = new ArrayList<>();

                // Generate calibration for feature computation
                // to account for current settingAnisotropy
                Calibration calibrationFeatureComp = new Calibration();
                calibrationFeatureComp.pixelWidth = 1.0;
                calibrationFeatureComp.pixelHeight = 1.0;
                calibrationFeatureComp.pixelDepth = 1.0 * anisotropy;
                calibrationFeatureComp.setUnit("um");

                for ( ImagePlus featureImage : featureImagesThisResolution )
                {

                    // temporarily change calibration while computing
                    // hessian and structure
                    // in order to account for the settingAnisotropy in xz vs z
                    featureImage.setCalibration( calibrationFeatureComp );

                    /*
                     The next if statement serves to only compute
                     features of features up to a certain level; the
                     reason for this is to keep the number of features
                     at bay.
                    */
                    if ( ! featureImage.getTitle().contains(
                            CONV_DEPTH + wekaSegmentation.settings.maxDeepConvolutionLevel) )
                    {
                        if (level <= 1) // multi-threaded
                        {
                            if ( computeAllFeatures ||  wekaSegmentation.isFeatureOrChildrenNeeded(
                                    "He_" + featureImage.getTitle()) )
                                futures.add(exe.submit(getHessian(featureImage, smoothingScale, hessianAbsoluteValues)));

                            if ( computeAllFeatures ||  wekaSegmentation.isFeatureOrChildrenNeeded(
                                    "St_" + featureImage.getTitle()) )
                                futures.add( exe.submit( getStructure(featureImage, smoothingScale, integrationScale)));
                        }
                        else // single-threaded
                        {
                            if ( computeAllFeatures ||  wekaSegmentation.isFeatureOrChildrenNeeded(
                                    "He_" + featureImage.getTitle()) )
                                featureImagesList.add( getHessian(featureImage, smoothingScale, hessianAbsoluteValues).call());

                            if ( computeAllFeatures ||  wekaSegmentation.isFeatureOrChildrenNeeded(
                                    "St_" + featureImage.getTitle()) )
                                featureImagesList.add( getStructure(featureImage, smoothingScale, integrationScale).call());
                        }
                    }


                }

                if ( level <= 1 ) // multi-threaded
                {
                    for (Future<ArrayList<ImagePlus>> f : futures)
                    {
                        // get feature images
                        featureImagesList.add( f.get() );
                        wekaSegmentation.totalThreadsExecuted.addAndGet(1);
                    }
                }
                futures = null;
                exe.shutdown();


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

                if ( isLogging )
                {
                    logger.info("Image features at level " + level +
                            " computed in [ms]: " +
                            (System.currentTimeMillis() - start));
                }


            }

            // put feature images into simple array for easier access
            int numFeatures = 0;
            for ( ArrayList<ImagePlus> singleResolutionFeatureImages2 : multiResolutionFeatureImages )
                numFeatures += singleResolutionFeatureImages2.size();


            int iFeature = 0;
            for ( ArrayList<ImagePlus> featureImages : multiResolutionFeatureImages )
            {
                for ( ImagePlus featureImage : featureImages )
                {
                    /*
                    if (  showFeatureImages && ( featuresToShow != null ) )
                    {
                        if ( featuresToShow.contains( iFeature ) )
                        {
                            //ImagePlus imp = interpolateFast(featureImage);
                            //imp.show();
                            featureImage.show();
                        }
                    }*/

                    if ( computeAllFeatures || wekaSegmentation.isFeatureNeeded( featureImage.getTitle() ) )
                    {
                        multiResolutionFeatureImageArray.add ( featureImage );
                        featureNames.add( featureImage.getTitle() );
                    }
                }
            }



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


    private void putDeepConvFeatLevelIntoTitle(ImagePlus imp)
    {
        String title = imp.getTitle();

        if ( title.contains(CONV_DEPTH+"3") ) { logger.warning("Too deep convolution!"); return; };
        if ( title.contains(CONV_DEPTH+"2") ) { imp.setTitle(CONV_DEPTH+"3_" + title); return; }
        if ( title.contains(CONV_DEPTH+"1") ) { imp.setTitle(CONV_DEPTH+"2_" + title); return; }
        imp.setTitle(CONV_DEPTH+"1_" + title);

        return;

    }

    public ImagePlus filter3d(ImagePlus imp, int r)
    {
        ImageStack result = ImageStack.create(imp.getWidth(), imp.getHeight(), imp.getNSlices(), 32);
        StackProcessor stackProcessor = new StackProcessor( imp.getStack() );
        int rz = r;
        stackProcessor.filter3D( result, (float) r, (float) r, (float) rz,
                0, result.size(), StackProcessor.FILTER_MEAN );
        String title = imp.getTitle();
        ImagePlus impResult = new ImagePlus("Mean"+r+"x"+r+"x"+r+"_" + title, result);
        impResult.setCalibration( imp.getCalibration().copy() );
        return ( impResult );
    }

    public ImagePlus getStackCT (ImagePlus imp, int c, int t)
    {
        Duplicator duplicator = new Duplicator();
        ImagePlus impCT = duplicator.run(imp, c+1, c+1, 1, imp.getNSlices(), t+1, t+1);
        return ( impCT );
    }

    public void getHessianImgLib2( ImagePlus imp, int numThreads )
    {
        HessianImgLib2 hessianImgLib2 = new HessianImgLib2();
        try
        {
            hessianImgLib2.run( imp, 1);
        } catch (IncompatibleTypeException e)
        {
            e.printStackTrace();
        } catch (InterruptedException e)
        {
            e.printStackTrace();
        } catch (ExecutionException e)
        {
            e.printStackTrace();
        }
    }

    public boolean is3D()
    {
        if ( inputImage.getNSlices() > 1 )
            return true;
        else
            return false;
    }

    private int[] getBinning(ImagePlus imp, double anisotropy, int scalingFactor)
    {
        // compute binning for this resolution layer
        int[] binning = new int[3];

        binning[0] = scalingFactor; // x-binning
        binning[1] = scalingFactor; // y-binning

        if ( imp.getNSlices() == 1 )
        {
            binning[2] = 1;  // 2D
        }
        else
        {
            // potentially bin less in z
            binning[2] = (int) (scalingFactor / anisotropy); // z-binning
            if( binning[2]==0 ) binning[2] = 1;
        }

        return ( binning );

    }

    public void removeCalibration ( ImagePlus imp )
    {
        Calibration calibration = new Calibration();
        calibration.pixelDepth = 1;
        calibration.pixelWidth = 1;
        calibration.pixelHeight = 1;
        calibration.setUnit("pixel");
        imp.setCalibration(calibration);
    }

    public Callable<ImagePlus> bin(ImagePlus imp_, int[] binning_, String binningTitle, String method)
    {
        return () -> {

            ImagePlus imp = imp_;
            int[] binning = binning_;
            String title = new String( imp.getTitle() );
            Binner binner = new Binner();

            Calibration saveCalibration = imp.getCalibration().copy(); // this is due to a bug in the binner

            ImagePlus impBinned = null;

            switch( method )
            {
                case "OPEN":
                    impBinned = binner.shrink(imp, binning[0], binning[1], binning[2], binner.MIN);
                    //impBinned = binner.shrink(imp, binning[0], binning[1], binning[2], binner.AVERAGE);
                    //IJ.run(impBinned, "Minimum 3D...", "x=1 y=1 z=1");
                    //IJ.run(impBinned, "Maximum 3D...", "x=1 y=1 z=1");
                    impBinned.setTitle("Open_" + title);
                    break;
                case "CLOSE":
                    impBinned = binner.shrink(imp, binning[0], binning[1], binning[2], binner.MAX);
                    //impBinned = binner.shrink(imp, binning[0], binning[1], binning[2], binner.AVERAGE);
                    //IJ.run(impBinned, "Maximum 3D...", "x=1 y=1 z=1");
                    //IJ.run(impBinned, "Minimum 3D...", "x=1 y=1 z=1");
                    impBinned.setTitle("Close_" + title);
                    break;
                case "AVERAGE":
                    impBinned = binner.shrink(imp, binning[0], binning[1], binning[2], binner.AVERAGE);
                    impBinned.setTitle(binningTitle + "_" + title);
                    break;
                case "MIN":
                    impBinned = binner.shrink(imp, binning[0], binning[1], binning[2], binner.MIN);
                    impBinned.setTitle(binningTitle + "_Min_" + title);
                    break;
                case "MAX":
                    impBinned = binner.shrink(imp, binning[0], binning[1], binning[2], binner.MAX);
                    impBinned.setTitle(binningTitle + "_Max_" + title);
                    break;
                default:
                    IJ.showMessage("Error while binning; method not supported :"+method);
                    break;
            }

            // reset calibration of input image
            // necessary due to a bug in the binner
            imp.setCalibration( saveCalibration );

            return ( impBinned );

        };
    }


    public ArrayList<String> getFeatureNames()
    {
        return featureNames;
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
        if ( multiResolutionFeatureImageArray == null )
            return true;

        if ( multiResolutionFeatureImageArray.size() > 1 )
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
        //if(referenceStackIndex == -1)
        //   return null;

        String featureName = "feat" + index;

        return featureName;
    }

    /**
     * Get the features enabled for the reference stack
     * @return features to be calculated on each stack
     */
    public boolean[] getEnabledFeatures()
    {
        return enabledFeatures;
    }

    /**
     * Set the features enabled for the reference stack
     * @param newFeatures boolean flags for the features to use
     */
    public void setEnabledFeatures(boolean[] enabledFeatures)
    {
        this.enabledFeatures = enabledFeatures;
    }

    public int getReferenceSliceIndex()
    {
        return referenceStackIndex;
    }

    public int getWidth()
    {


        return inputImage.getWidth();
    }

    public int getHeight()
    {
        return inputImage.getHeight();
    }

    public int getDepth()
    {
        return inputImage.getNSlices();
    }

    public int getSize() {
        return getNumFeatures();
    }


}

	
