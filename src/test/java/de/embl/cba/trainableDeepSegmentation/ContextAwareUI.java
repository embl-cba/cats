package de.embl.cba.trainableDeepSegmentation;

import ij.IJ;
import ij.ImagePlus;

import static de.embl.cba.trainableDeepSegmentation.TestUtils.TEST_RESOURCES;

public class ContextAwareUI
{

    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage(TEST_RESOURCES + "context-aware-v6-scale1.5-noise.tif" );
        inputImagePlus.show();

        IJ.wait(100);

        de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");
    }
}
