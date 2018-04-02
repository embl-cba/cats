package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin;
import ij.IJ;
import ij.ImagePlus;

import static de.embl.cba.trainableDeepSegmentation.TestUtils.TEST_RESOURCES;

public class MriStack2Channels
{


    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage(TEST_RESOURCES + "mri-stack-2channels.tif" );
        inputImagePlus.show();

        IJ.wait(100);

        DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");

    }

}