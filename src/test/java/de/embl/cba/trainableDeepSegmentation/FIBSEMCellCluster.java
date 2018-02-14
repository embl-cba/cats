package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierAsSlurmJobsCommand;
import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierCommand;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils;
import ij.IJ;
import ij.ImagePlus;
import net.imglib2.FinalInterval;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static de.embl.cba.trainableDeepSegmentation.FIBSEMCell.TEST_RESOURCES;

public class FIBSEMCellCluster
{
    public final static String TEST_RESOURCES = "/Volumes/cba/tischer/projects/em-automated-segmentation--data/";

    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();


        ImagePlus inputImagePlus = IJ.openVirtual(TEST_RESOURCES + "fib-sem--cell--8x8x8nm.tif" );
        inputImagePlus.show();

        IJ.wait(100);

        de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");
    }
}
