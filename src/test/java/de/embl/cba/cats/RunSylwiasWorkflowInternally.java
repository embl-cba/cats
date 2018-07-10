package de.embl.cba.cats;

import net.imagej.ImageJ;
import de.embl.cba.cats.commands.AnalyzeObjectsCommand;
import de.embl.cba.cats.commands.ApplyClassifierCommand;
import de.embl.cba.cats.utils.IOUtils;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class RunSylwiasWorkflowInternally
{

    public static void main(final String... args) throws Exception {

        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        String outputRootDirectory = TestingUtils.TEST_RESOURCES;
        String dataSetPattern = ".*--W00016--P00004--.*";

        String dataSetName = IOUtils.createDataSetNameFromPattern( dataSetPattern );
        String outputDirectory = outputRootDirectory + File.separator + dataSetName;

        Map< String, Object > parameters = new HashMap<>(  );

        parameters.clear();
        parameters.put( IOUtils.INPUT_IMAGE_FILE, new File( TestingUtils.TEST_RESOURCES + "/image-sequence/" + dataSetPattern ) );
        parameters.put( ApplyClassifierCommand.CLASSIFIER_FILE, new File( TestingUtils.TEST_RESOURCES + "/transmission-cells-3d.classifier" ) );
        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( IOUtils.OUTPUT_MODALITY, IOUtils.SAVE_AS_TIFF_STACKS );
        ij.command().run( ApplyClassifierCommand.class, false, parameters ).get();

        parameters.clear();
        parameters.put( AnalyzeObjectsCommand.INPUT_IMAGE_FILE, new File( outputDirectory + "/foreground.tif" ) );
        parameters.put( AnalyzeObjectsCommand.LOWER_THRESHOLD, 1 );
        parameters.put( AnalyzeObjectsCommand.UPPER_THRESHOLD, 255 );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_MODALITY, IOUtils.SHOW_RESULTS_TABLE );
        parameters.put( AnalyzeObjectsCommand.QUIT_AFTER_RUN, true );
        ij.command().run( AnalyzeObjectsCommand.class, false, parameters ).get();


    }


}
