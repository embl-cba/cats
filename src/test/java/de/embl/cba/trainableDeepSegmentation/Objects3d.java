package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.trainableDeepSegmentation.postprocessing.ObjectSegmentation;
import de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin;
import ij.IJ;
import ij.ImagePlus;

import static de.embl.cba.trainableDeepSegmentation.TestUtils.TEST_RESOURCES;

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

        DeepSegmentation deepSegmentation = new DeepSegmentation();
        deepSegmentation.setInputImage( inputImagePlus );
        deepSegmentation.setResultImageDisk( TEST_RESOURCES + "3d-objects-probabilities" );
        deepSegmentation.loadInstancesAndMetadata( TEST_RESOURCES + "3d-objects-instances/3d-objects.ARFF"  );

        DeepSegmentationIJ1Plugin plugin = new DeepSegmentationIJ1Plugin();
        plugin.initialise( deepSegmentation, false );
        //plugin.segmentObjects();
        //plugin.reviewObjects();

    }

}
