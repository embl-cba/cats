package trainableDeepSegmentation;

import trainableDeepSegmentation.*;


import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.imglib2.FinalInterval;
import net.imglib2.util.Intervals;

import java.io.File;
import java.util.ArrayList;

import static trainableDeepSegmentation.IntervalUtils.*;

public class TrainLabel {

    final static String ROOT_PATH = "/Users/tischi/Documents/segmentation-challenges--data/brainiac/";
    // final static String ROOT_PATH = "/g/cba/tischer/segmentation-challenges--data/brainiac/";

    final static String INPUT_IMAGE_PATH = ROOT_PATH + "combined-clahe-nz99.tif";
    final static String LABEL_IMAGE_PATH = ROOT_PATH + "train-labels-border2-nz99.tif";

    final static String LOGGING_DIRECTORY = ROOT_PATH + "log";
    final static String OUTPUT_DIRECTORY = ROOT_PATH + "output";

    public static void main( final String[] args )
    {

        new ImageJ();

        // Load image data
        //
        final ImagePlus inputImage = IJ.openImage( INPUT_IMAGE_PATH );
        final ImagePlus labelImage = IJ.openImage( LABEL_IMAGE_PATH );
        IJ.run(labelImage, "Divide...", "value=255.0000 stack");

        // Set up segmentation
        //
        final WekaSegmentation ws = new WekaSegmentation( );
        ws.setInputImage( inputImage );
        ws.setLabelImage( labelImage );
        ws.setResultImageRAM( );

        ws.setNumThreads( 16 );

        ws.settings.log2 = false;

        int level = 0;
        ws.settings.binFactors[level++] = 2;
        ws.settings.binFactors[level++] = 2;
        ws.settings.binFactors[level++] = 2;
        ws.settings.binFactors[level++] = 2;
        ws.settings.binFactors[level++] = -1;

        String loggingDirectory = LOGGING_DIRECTORY;
        String outputDirectory = OUTPUT_DIRECTORY;

        for ( int b : ws.settings.binFactors )
        {
            if ( b == -1 ) break;

            loggingDirectory += "_"+b;
            outputDirectory += "_"+b;
        }

        createIfNotExists( outputDirectory );

        ws.setAndCreateLogDirAbsolute( loggingDirectory );
        ws.settings.anisotropy = 5;
        ws.settings.activeChannels.add( 0 );

        ws.classifierBatchSizePercent = "100";
        ws.classifierNumTrees = 100;
        ws.classifierFractionFeaturesPerNode = 0.1;

        FinalInterval inputImageInterval = IntervalUtils.getInterval( inputImage );

        long[] min = Intervals.minAsLongArray( inputImageInterval );
        long[] max = Intervals.maxAsLongArray( inputImageInterval );
        min[ T ] = 0; max[ T ] = 0; // only labeled part
        FinalInterval trainInterval = new FinalInterval( min, max );

        FinalInterval applyInterval = inputImageInterval; // everything

        ArrayList< Double > classWeights = new ArrayList<>(  );
        classWeights.add( 1.0D );
        classWeights.add( 1.0D );

        int maxNumIterations = 100000;
        int zChunkSize = 16;
        int nxyTiles = 3;
        int localRadius = 7;
        int maxNumInstanceSetsPerTilePlane = 20;
        int maxNumInstances = 5000000;
        int numTrainingTrees = 75;
        int numClassificationTrees = 500;
        int minNumVoxels = 1000;

        ws.trainFromLabelImage(
                "brainiac",
                WekaSegmentation.START_NEW_INSTANCES,
                maxNumIterations,
                zChunkSize,
                nxyTiles,
                localRadius,
                maxNumInstanceSetsPerTilePlane,
                maxNumInstances,
                numTrainingTrees,
                numClassificationTrees,
                minNumVoxels,
                classWeights,
                outputDirectory,
                trainInterval,
                applyInterval);


        ws.getInputImage().show();
        ws.getLabelImage().show();

        System.exit( 0 );
        IJ.run("Quit");

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
