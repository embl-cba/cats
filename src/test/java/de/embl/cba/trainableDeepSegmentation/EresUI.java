package de.embl.cba.trainableDeepSegmentation;

import ij.IJ;
import ij.ImagePlus;

import static de.embl.cba.trainableDeepSegmentation.TestUtils.TEST_RESOURCES;

public class EresUI
{

    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage( "/Users/tischer/Documents/segmentation-challenges--data/eres-em/eres-004.tif" );
        inputImagePlus.show();

        IJ.wait( 100 );

        de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin deepSegmentationIJ1Plugin = new de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin();
        deepSegmentationIJ1Plugin.run( "" );
    }
}
