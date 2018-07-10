package de.embl.cba.cats;

import de.embl.cba.bigDataTools.dataStreamingTools.DataStreamingTools;
import de.embl.cba.cats.commands.ApplyClassifierOnSlurmCommand;
import ij.ImagePlus;
import org.scijava.module.ModuleService;
import org.scijava.plugin.Parameter;

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
        CATS CATS = new CATS( );
        CATS.setInputImage( inputImage );
        CATS.setResultImageDisk( probabilitiesDirectory );

        // Run cluster classification
        //
        Map< String, Object > parameters = new HashMap<>();

        parameters.put( ApplyClassifierOnSlurmCommand.USER_NAME, new File( classifierPath ) );
        parameters.put( ApplyClassifierOnSlurmCommand.PASSWORD, new File( classifierPath ) );
        parameters.put( ApplyClassifierOnSlurmCommand.CLASSIFIER_FILE, new File( classifierPath ) );
        parameters.put( ApplyClassifierOnSlurmCommand.NUM_JOBS, numJobs );
        parameters.put( ApplyClassifierOnSlurmCommand.NUM_WORKERS, numThreadsPerJob );

        CATS.applyClassifierOnSlurm( parameters );

    }


}
