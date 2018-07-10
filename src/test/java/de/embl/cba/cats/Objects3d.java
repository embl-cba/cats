package de.embl.cba.cats;

import de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin;
import ij.IJ;
import ij.ImagePlus;

import static de.embl.cba.cats.TestUtils.TEST_RESOURCES;

public class Objects3d
{

    public static void main( final String[] args )
    {

        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        // Open Image
        //
        ImagePlus inputImagePlus = IJ.openImage(TEST_RESOURCES + "3d-objects.zip" );
        inputImagePlus.show();

        CATS CATS = new CATS();
        CATS.setInputImage( inputImagePlus );
        CATS.setResultImageDisk( TEST_RESOURCES + "3d-objects-probabilities" );
        CATS.loadInstancesAndMetadata( TEST_RESOURCES + "3d-objects-instances/3d-objects.ARFF"  );

        DeepSegmentationIJ1Plugin plugin = new DeepSegmentationIJ1Plugin();
        plugin.initialise( CATS, false );
        //plugin.segmentObjects();
        //plugin.reviewObjects();

    }

}
