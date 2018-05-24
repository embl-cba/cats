package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.trainableDeepSegmentation.commands.AnalyzeObjectsCommand;
import de.embl.cba.trainableDeepSegmentation.ui.ContextAwareTrainableSegmentationPlugin;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.ImageJ;

import static de.embl.cba.trainableDeepSegmentation.TestUtils.TEST_RESOURCES;

public class CatsUI
{

    public static void main(final String... args) throws Exception
    {

        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage(TEST_RESOURCES + "3d-objects.zip" );
        inputImagePlus.show();

        ij.command().run( ContextAwareTrainableSegmentationPlugin.class, true );

    }


}
