package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.bigDataTools.dataStreamingTools.DataStreamingTools;
import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierOnSlurmCommand;
import de.embl.cba.trainableDeepSegmentation.utils.GetPasswordFromUI;
import ij.ImagePlus;
import org.scijava.module.Module;
import org.scijava.module.ModuleService;
import org.scijava.plugin.Parameter;
import org.scijava.script.ScriptService;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class ClassifyOnSlurmShotaroOneCell
{
    @Parameter
    private ModuleService moduleService;

    public static void main( final String[] args )
    {

        String userName = "tischer";
        //String password = ""; // TODO

        String fileName = "6.1min-crop.tif";
        String directory = "/Volumes/cba/tischer/projects/shotaro-otsuka-fib-sem-mitosis-segmentation--data/10x10x10-crop-test";
        String probabilitiesDirectory = "";
        String classifierPath = "";
        int numJobs = 10;
        int numThreadsPerJob = 16;

        // Open image
        //
        DataStreamingTools dst = new DataStreamingTools();
        ImagePlus inputImage = dst.openFromDirectory(
                directory,
                "None",
                ".*--C.*",
                "Resolution 0/Data",
                null,
                3,
                true,
                false );

        // Initialise deep segmentation
        //
        DeepSegmentation deepSegmentation = new DeepSegmentation( );
        deepSegmentation.setInputImage( inputImage );
        deepSegmentation.setResultImageDisk( probabilitiesDirectory );

        // Run cluster classification
        //
        Map< String, Object > parameters = new HashMap<>();

        parameters.put( ApplyClassifierOnSlurmCommand.USER_NAME, new File( classifierPath ) );
        parameters.put( ApplyClassifierOnSlurmCommand.PASSWORD, new File( classifierPath ) );
        parameters.put( ApplyClassifierOnSlurmCommand.CLASSIFIER_FILE, new File( classifierPath ) );
        parameters.put( ApplyClassifierOnSlurmCommand.NUM_JOBS, numJobs );
        parameters.put( ApplyClassifierOnSlurmCommand.NUM_WORKERS, numThreadsPerJob );

        deepSegmentation.applyClassifierOnSlurm( parameters );

    }


}
