package de.embl.cba.cats;

import ij.IJ;
import ij.ImagePlus;

public class MitochondriaUI
{

    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage("/Users/tischer/Documents/segmentation-challenges/mitochondria-train-test-label-composite-crop.zip" );
        inputImagePlus.show();

        IJ.wait(100);

        de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");
    }

}
