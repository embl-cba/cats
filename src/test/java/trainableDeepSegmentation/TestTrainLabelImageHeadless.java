package trainableDeepSegmentation;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.Prefs;
import ij.io.FileSaver;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.iterator.ZeroMinIntervalIterator;

import java.io.File;
import java.util.ArrayList;

import static trainableDeepSegmentation.IntervalUtils.*;

public class TestTrainLabelImageHeadless {

    final static String INPUT_IMAGE_PATH = "/Users/tischi/Desktop/segmentation-challenges/brainiac2-mit-edu-SNEMI3D/combined-clahe.tif";
    final static String LABEL_IMAGE_PATH = "/Users/tischi/Desktop/segmentation-challenges/brainiac2-mit-edu-SNEMI3D/train" +
            "-labels/doNotUse100-train-labels-2pixLargerBorders.tif";
    final static String INSTANCES_DIRECTORY = "/Users/tischi/Desktop/segmentation-challenges/brainiac2-mit-edu-SNEMI3D/instances-test";

    public static void main( final String[] args )
    {
        new ImageJ();

        // Load image data
        //
        ImagePlus inputImage = IJ.openImage( INPUT_IMAGE_PATH );
        ImagePlus labelImage = IJ.openImage( LABEL_IMAGE_PATH );

        // Set up weka
        //
        WekaSegmentation ws = new WekaSegmentation( );
        ws.setNumThreads( 16 );

        ws.setInputImage( inputImage );
        ws.setLabelImage( labelImage );
        ws.setResultImageRAM( );

        ws.settings.log2 = false;
        ws.settings.binFactors[0] = 1;
        ws.settings.binFactors[1] = 2;
        ws.settings.binFactors[2] = 3;
        ws.settings.binFactors[3] = 3;
        ws.settings.anisotropy = 5;
        ws.settings.activeChannels.add( 0 );

        ws.classifierBatchSizePercent = "100";
        ws.classifierNumTrees = 100; // => ~ 33 oob trees for each instance
        ws.classifierFractionFeaturesPerNode = 0.1;

        //FinalInterval interval = IntervalUtils.getInterval( inputImage );
        //IntervalUtils.getIntervalByReplacingValues(
        //       interval, IntervalUtils.Z, 1, 96 );

        long[] min = new long[ 5 ];
        long[] max = new long[ 5 ];
        min[ X ] = 430; max[ X ] = 530;
        min[ Y ] = 250; max[ Y ] = 450;
        min[ Z ] = 0; max[ Z ] = 95;
        min[ C ] = 0; max[ C ] = 0;
        min[ T ] = 0; max[ T ] = 0;

        FinalInterval interval = new FinalInterval( min, max );

        ArrayList< Double > classWeights = new ArrayList<>(  );
        classWeights.add( 1.05 );
        classWeights.add( 1.0 );

        ws.trainFromLabelImage( "labelImageTraining",
                WekaSegmentation.START_NEW_INSTANCES,
                100,
                4,
                1,
                7,
               20,
                1000,
                classWeights,
                INSTANCES_DIRECTORY,
                interval);

        ImagePlus result = ws.getResultImage().getImagePlus();
        result.show();
        IJ.run(result, "Enhance Contrast", "saturated=0.35");

        // Analyse objects
        //
        //ImagePlus labelMask = ws.analyzeObjects( MIN_NUM_VOXELS );
        //labelMask.show();

        // Save image
        //String outFilePath = ARG_OUTPUT_DIRECTORY + File.separator + "R"+ regExp + "--labelMask.tif";
        //WekaSegmentation.logger.info("\n# Saving file " + outFilePath + "...");
        //FileSaver fileSaver = new FileSaver( labelMask );
        //fileSaver.saveAsTiff( outFilePath );
        //WekaSegmentation.logger.info("...done.");


    }
}
