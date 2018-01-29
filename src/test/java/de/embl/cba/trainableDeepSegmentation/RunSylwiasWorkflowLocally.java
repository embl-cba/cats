package de.embl.cba.trainableDeepSegmentation;

import net.imagej.ImageJ;
import de.embl.cba.trainableDeepSegmentation.commands.AnalyzeObjectsCommand;
import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierCommand;
import de.embl.cba.trainableDeepSegmentation.commands.IOUtils;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class RunSylwiasWorkflowLocally
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
        parameters.put( ApplyClassifierCommand.INPUT_IMAGE_PATH, new File( TestingUtils.TEST_RESOURCES + "/image-sequence/" + dataSetPattern ) );
        parameters.put( ApplyClassifierCommand.CLASSIFIER_PATH, new File( TestingUtils.TEST_RESOURCES + "/transmission-cells-3d.classifier" ) );
        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( ApplyClassifierCommand.OUTPUT_MODALITY, ApplyClassifierCommand.SAVE_AS_TIFF_FILES );
        ij.command().run( ApplyClassifierCommand.class, false, parameters ).get();

        parameters.clear();
        parameters.put( AnalyzeObjectsCommand.INPUT_IMAGE_PATH, new File( outputDirectory + "/foreground.tif" ) );
        parameters.put( AnalyzeObjectsCommand.LOWER_THRESHOLD, 1 );
        parameters.put( AnalyzeObjectsCommand.UPPER_THRESHOLD, 255 );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_MODALITY, AnalyzeObjectsCommand.SHOW );
        parameters.put( AnalyzeObjectsCommand.QUIT_AFTER_RUN, true );
        ij.command().run( AnalyzeObjectsCommand.class, false, parameters ).get();


    }


}
