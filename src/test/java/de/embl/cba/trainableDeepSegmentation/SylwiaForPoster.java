package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import ij.ImagePlus;

import java.io.File;

public class SylwiaForPoster
{

    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus imp = IOUtils.openImageWithIJOpenImage( new File("/Volumes/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/for_poster/data/dense.tif" ));
        imp.show();

        de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");
    }




}
