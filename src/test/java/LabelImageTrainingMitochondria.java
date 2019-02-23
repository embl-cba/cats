import de.embl.cba.cats.CATS;
import de.embl.cba.cats.utils.IntervalUtils;
import ij.IJ;
import ij.ImagePlus;
import net.imglib2.FinalInterval;
import net.imglib2.util.Intervals;

import java.io.File;
import java.util.ArrayList;

import static de.embl.cba.cats.utils.IntervalUtils.T;

public class LabelImageTrainingMitochondria
{

    final static String ROOT_PATH = "/Users/tischer/Documents/segmentation-challenges/mitochondria/";

    final static String INPUT_IMAGE_PATH = ROOT_PATH + "mitochondria-train-test-label-composite-crop.zip";

    final static String INSTANCES_PATH = ROOT_PATH + "training/Instances-5491.ARFF";

    final static String LOGGING_DIRECTORY = ROOT_PATH + "log";
    final static String OUTPUT_DIRECTORY = ROOT_PATH + "output";

    public static void main( final String[] args )
    {

        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        // Load image data
        //
        final ImagePlus inputImage = IJ.openImage( INPUT_IMAGE_PATH );

        // Set up segmentation
        //
        final CATS cats = new CATS( );
        cats.setInputImage( inputImage );
        cats.setResultImageRAM( );
        //ws.loadInstancesAndMetadata( INSTANCES_PATH );

        cats.setNumThreads( 4 );

        cats.featureSettings.log2 = false;

        int level = 0;
        cats.featureSettings.binFactors.set(level++ , 2);
        cats.featureSettings.binFactors.set(level++ , 2);
        cats.featureSettings.binFactors.set(level++ , 3);
        cats.featureSettings.binFactors.set(level++ , -1);
        cats.featureSettings.binFactors.set(level++ , -1);

        cats.featureSettings.anisotropy = 1.0D;
        cats.featureSettings.activeChannels.add( 0 );


        String loggingDirectory = LOGGING_DIRECTORY;
        String outputDirectory = OUTPUT_DIRECTORY;

        for ( int b : cats.featureSettings.binFactors )
        {
            if ( b == -1 ) break;

            loggingDirectory += "_"+b;
            outputDirectory += "_"+b;
        }

        createIfNotExists( outputDirectory );

        cats.setAndCreateLogDirAbsolute( loggingDirectory );

        cats.classifierBatchSizePercent = "66";
        cats.classifierNumTrees = 10;
        cats.classifierFractionFeaturesPerNode = 0.1;

        FinalInterval inputImageInterval = IntervalUtils.getIntervalWithChannelsDimensionAsSingleton( inputImage );

        FinalInterval trainInterval = getTrainingInterval( inputImageInterval );

        FinalInterval applyInterval = inputImageInterval; // everything

        ArrayList< Double > classWeights = new ArrayList<>(  );
        classWeights.add( 1.0D );
        classWeights.add( 1.0D );

        int maxNumIterations = 100;
        int zChunkSize = 16;
        int nxyTiles = 2;
        int localRadius = 7;
        int maxNumInstanceSetsPerTileAndPlane = 20;
        int maxNumInstances = 4000;
        int numTrainingTrees = 20;
        int numClassificationTrees = 20;
        int minNumVoxels = 1000;
        int minNumInstancesBeforeNewTraining = 100; // to speed it up a bit

        cats.trainFromLabelImage(
                "mitochondria",
                CATS.START_NEW_INSTANCES,
                maxNumIterations,
                zChunkSize,
                nxyTiles,
                localRadius,
                maxNumInstanceSetsPerTileAndPlane,
                maxNumInstances,
                minNumInstancesBeforeNewTraining,
                numTrainingTrees,
                numClassificationTrees,
                minNumVoxels,
                classWeights,
                outputDirectory,
                trainInterval,
                applyInterval);

        cats.getInputImage().show();

        //System.exit( 0 );
        //IJ.run("Quit");

    }

    private static FinalInterval getTrainingInterval( FinalInterval inputImageInterval )
    {
        long[] min = Intervals.minAsLongArray( inputImageInterval );
        long[] max = Intervals.maxAsLongArray( inputImageInterval );
        min[ T ] = 0;
        max[ T ] = 0;
        return new FinalInterval( min, max );
    }

    public static void createIfNotExists( String dir )
    {
        File file = new File( dir );
        if ( !file.exists() )
        {
            file.mkdir();
        }

    }

}
