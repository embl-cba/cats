package de.embl.cba.cats;

import ij.IJ;
import ij.ImagePlus;

public class FIBSEMCellClusterUI
{
    public final static String TEST_RESOURCES = "/Volumes/cba/tischer/projects/em-automated-segmentation--data/";

    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openVirtual(TEST_RESOURCES + "fib-sem--cell--8x8x8nm.tif" );
        inputImagePlus.show();

        IJ.wait(100);

        de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");
    }
}
