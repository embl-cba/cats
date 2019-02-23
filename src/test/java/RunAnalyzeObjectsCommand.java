import de.embl.cba.cats.utils.IOUtils;
import net.imagej.ImageJ;
import de.embl.cba.cats.ui.AnalyzeObjectsCommand;

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
        parameters.put( AnalyzeObjectsCommand.INPUT_IMAGE_FILE, new File( "/image-sequence--classified/foreground.tif" ) );
        parameters.put( AnalyzeObjectsCommand.LOWER_THRESHOLD, 1 );
        parameters.put( AnalyzeObjectsCommand.UPPER_THRESHOLD, 255 );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_DIRECTORY, new File( "/image-sequence--classified/" ) );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_MODALITY, IOUtils.SAVE_RESULTS_TABLE );
        ij.command().run( AnalyzeObjectsCommand.class, false, parameters );

    }
}
