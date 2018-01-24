package trainableDeepSegmentation;

import net.imagej.ImageJ;
import trainableDeepSegmentation.ij2plugins.AnalyzeObjectsCommand;
import trainableDeepSegmentation.ij2plugins.ApplyClassifierCommand;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class RunAnalyzeObjectsCommand
{
    // Main
    public static void main(final String... args) throws Exception {


        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        Map< String, Object > parameters = new HashMap<>(  );
        parameters.put( AnalyzeObjectsCommand.INPUT_IMAGE_FILE, new File( TestingUtils.TEST_RESOURCES + "/foreground.tif" ) );
        parameters.put( AnalyzeObjectsCommand.LOWER_THRESHOLD, 1 );
        parameters.put( AnalyzeObjectsCommand.UPPER_THRESHOLD, 255 );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_FOLDER, new File( TestingUtils.TEST_RESOURCES ) );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_MODALITY, AnalyzeObjectsCommand.SHOW );
        ij.command().run( AnalyzeObjectsCommand.class, false, parameters );


    }
}
