package de.embl.cba.cats.features;

import ij.IJ;
import ij.ImagePlus;
import ij.measure.Calibration;
import ij.plugin.Binner;
import imagescience.image.Image;
import imagescience.transform.Scale;
import ij.plugin.Scaler;

import java.util.concurrent.Callable;

public class DownSampler
{

    String method;

    public static final String BIN_AVERAGE = "Bin_average";

    public static final String BIN_MAXIMUM = "Bin_maximum";

    public static final String TRANSFORMJ_SCALE_LINEAR = "TransformJ_linear";

    public static final String TRANSFORMJ_SCALE_CUBIC = "TransformJ_cubic";

    public static final String IMAGEJ_SCALE = "ImageJ_scale";

    public DownSampler( String method )
    {
        this.method = method;
    }

    public DownSampler( int method )
    {
        this.method = getString( method );
    }

    public static int getID( String method )
    {
        switch ( method )
        {
            case BIN_AVERAGE: return 0;
            case BIN_MAXIMUM: return 1;
            case TRANSFORMJ_SCALE_LINEAR: return 2;
            case TRANSFORMJ_SCALE_CUBIC: return 3;
            case IMAGEJ_SCALE: return 4;
            default: return -1;
        }
    }

    public static String getString( int method )
    {
        switch ( method )
        {
            case 0: return BIN_AVERAGE;
            case 1: return BIN_MAXIMUM;
            case 2: return TRANSFORMJ_SCALE_LINEAR;
            case 3: return TRANSFORMJ_SCALE_CUBIC;
            case 4: return IMAGEJ_SCALE;
            default: return "";
        }
    }


    public Callable<ImagePlus> run( ImagePlus imp, int[] binning, String binningTitle )
    {
        return () -> {

            String title = new String( imp.getTitle() );
            Binner binner = new Binner();

            Calibration saveCalibration = imp.getCalibration().copy(); // this is due to a bug in the binner

            ImagePlus binnedImagePlus = null;

            switch( method )
            {
                case BIN_AVERAGE:
                    binnedImagePlus = binner.shrink(imp, binning[0], binning[1], binning[2], binner.AVERAGE );
                    binnedImagePlus.setTitle(binningTitle + "_" + title);
                    break;
                case BIN_MAXIMUM:
                    binnedImagePlus = binner.shrink(imp, binning[0], binning[1], binning[2], binner.MAX );
                    binnedImagePlus.setTitle(binningTitle + "_Max_" + title);
                    break;
                case TRANSFORMJ_SCALE_LINEAR:
                    binnedImagePlus = getScaledWithTransformJ( imp, binning, Scale.LINEAR );
                    binnedImagePlus.setTitle(binningTitle + "_" + title);
                    break;
                case TRANSFORMJ_SCALE_CUBIC:
                    binnedImagePlus = getScaledWithTransformJ( imp, binning, Scale.CUBIC );
                    binnedImagePlus.setTitle(binningTitle + "_" + title);
                    break;
                case IMAGEJ_SCALE:
                    Scaler scaler = new Scaler();
                    //scale.run(  )
                    break;
                case "OPEN":
                    binnedImagePlus = binner.shrink( imp, binning[0], binning[1], binning[2], binner.MIN );
                    //impBinned = binner.shrink(imp, binning[0], binning[1], binning[2], binner.AVERAGE);
                    //IJ.run(impBinned, "Minimum 3D...", "x=1 y=1 z=1");
                    //IJ.run(impBinned, "Maximum 3D...", "x=1 y=1 z=1");
                    binnedImagePlus.setTitle("Open_" + title);
                    break;
                case "CLOSE":
                    binnedImagePlus = binner.shrink( imp, binning[0], binning[1], binning[2], binner.MAX );
                    //impBinned = binner.shrink(imp, binning[0], binning[1], binning[2], binner.AVERAGE);
                    //IJ.run(impBinned, "Maximum 3D...", "x=1 y=1 z=1");
                    //IJ.run(impBinned, "Minimum 3D...", "x=1 y=1 z=1");
                    binnedImagePlus.setTitle("Close_" + title);
                    break;
                default:
                    IJ.showMessage("Error while binning; method not supported :"+method);
                    break;
            }

            // reset calibration of input image
            // necessary due to a bug in the binner
            imp.setCalibration( saveCalibration );

            return ( binnedImagePlus );

        };
    }

    private static void setNewCalibrationAfterBinning( ImagePlus imp, ImagePlus impBinned, int[] binnning )
    {
        // Set a calibration that can be changed during the binning
        Calibration calibrationAfterBinning = imp.getCalibration().copy();
        calibrationAfterBinning.pixelWidth *= binnning[ 0 ];
        calibrationAfterBinning.pixelHeight *= binnning[ 1 ];
        calibrationAfterBinning.pixelDepth *= binnning[ 2 ] ;
        impBinned.setCalibration( calibrationAfterBinning );
    }

    private ImagePlus getScaledWithTransformJ( ImagePlus imp, int[] binning, int method )
    {
        ImagePlus binnedImagePlus;
        Scale scale = new Scale();

        double[] scaling = new double[ 3 ];

        for( int i = 0; i < 3; ++i )
        {
            scaling[ i ] = 1.0 / binning[ i ];
        }

        Image image = Image.wrap( imp );
        Image binnedImage = scale.run( image, scaling[ 0 ], scaling[ 1 ], scaling[ 2 ], 1.0, 1.0, method );
        binnedImagePlus = binnedImage.imageplus();

        setNewCalibrationAfterBinning( imp, binnedImagePlus, binning );

        return binnedImagePlus;
    }

}
