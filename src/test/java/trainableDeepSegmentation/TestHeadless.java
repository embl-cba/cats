package trainableDeepSegmentation;

import bigDataTools.logging.Logger;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.measure.Calibration;

import java.util.ArrayList;

/**
 * Created by tischi on 04/10/17.
 */
public class TestHeadless {


    public static void main( final String[] args )
    {
        new ImageJ();

        WekaSegmentation wekaSegmentation = new WekaSegmentation();
        Logger logger = wekaSegmentation.getLogger();

        ImagePlus trainingImage = IJ.openImage("/Users/tischi/Desktop/mri-stack-big.tif");
        wekaSegmentation.setTrainingImage( trainingImage );

        Calibration calibration = trainingImage.getCalibration();
        wekaSegmentation.settings.anisotropy = 1.0 * calibration.pixelDepth / calibration.pixelWidth;

        if( calibration.pixelWidth != calibration.pixelHeight )
        {
            logger.error("Image calibration in x and y is not the same; currently cannot take this into " +
                    "account; but you can still use this plugin, may work anyway...");
        }

        ArrayList< Integer > channelsToConsider = new ArrayList<>();
        for ( int c = 0; c < trainingImage.getNChannels(); c++ )
        {
            channelsToConsider.add(c); // zero-based
        }
        wekaSegmentation.settings.activeChannels = channelsToConsider;



    }


}
