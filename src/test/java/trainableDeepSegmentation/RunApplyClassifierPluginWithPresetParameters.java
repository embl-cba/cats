package trainableDeepSegmentation;

import net.imagej.ImageJ;
import trainableDeepSegmentation.ij2plugins.ApplyClassifier;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class RunApplyClassifierPluginWithPresetParameters
{

    // Main
    public static void main(final String... args) throws Exception {


        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        /*Map< String, Object > parameters = new HashMap<>(  );
        parameters.put( "inputImageFile", new File( TestingUtils.TEST_RESOURCES + "transmission-cells-3d.zip" ) );
        parameters.put( "classifierFile", new File( TestingUtils.TEST_RESOURCES + "transmission-cells-3d.classifier" ) );
        parameters.put( "outputFolder", new File( TestingUtils.TEST_RESOURCES ) );*/


        Map< String, Object > parameters = new HashMap<>(  );
        parameters.put( "inputImageFile", new File( TestingUtils.TEST_RESOURCES + "/image-sequence/.*--W00016--P00004--.*" ) );
        parameters.put( "classifierFile", new File( TestingUtils.TEST_RESOURCES + "/transmission-cells-3d.classifier" ) );
        parameters.put( "outputFolder", new File( TestingUtils.TEST_RESOURCES ) );
        parameters.put( "outputModality", ApplyClassifier.SAVE_AS_TIFF_FILES );

        ij.command().run( ApplyClassifier.class, false, parameters );

    }


}
