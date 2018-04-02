package de.embl.cba.trainableDeepSegmentation;

import ij.IJ;
import ij.ImagePlus;

import static de.embl.cba.trainableDeepSegmentation.TestUtils.TEST_RESOURCES;

public class SimoneReber
{
    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage( "/Volumes/cba/tischer/projects/simone-reber-cell-volume-and-spindle-size--data/concatenated_undifferentiated_tubulin.tif" );
        //ImagePlus inputImagePlus = IJ.openImage( "/Volumes/cba/tischer/projects/simone-reber-cell-volume-and-spindle-size--data/concatenated_differentiated_tubulin.tif" );

        inputImagePlus.show();

        IJ.wait(100);

        de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");
    }
}
