package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import net.imagej.ImageJ;
import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierCommand;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class RunApplyClassifierCommandOld
{

    public static void main(final String... args) throws Exception {


        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        /*Map< String, Object > parameters = new HashMap<>(  );
        parameters.put( "inputDirectory", new File( TestingUtils.TEST_RESOURCES + "transmission-cells-3d.zip" ) );
        parameters.put( "classifierFile", new File( TestingUtils.TEST_RESOURCES + "transmission-cells-3d.classifier" ) );
        parameters.put( "outputDirectory", new File( TestingUtils.TEST_RESOURCES ) );*/


        Map< String, Object > parameters = new HashMap<>(  );
        parameters.put( "inputDirectory", new File( TestingUtils.TEST_RESOURCES + "/image-sequence/.*--W00016--P00004--.*" ) );
        parameters.put( "classifierFile", new File( TestingUtils.TEST_RESOURCES + "/transmission-cells-3d.classifier" ) );
        parameters.put( "outputDirectory", new File( TestingUtils.TEST_RESOURCES ) );
        parameters.put( "outputModality", IOUtils.SAVE_AS_TIFF_STACKS );

        ij.command().run( ApplyClassifierCommand.class, false, parameters );

    }


}
