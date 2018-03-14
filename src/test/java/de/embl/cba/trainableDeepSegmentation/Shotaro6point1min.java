package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.bigDataTools.dataStreamingTools.DataStreamingTools;
import ij.IJ;
import ij.ImageJ;
import net.imglib2.FinalInterval;

import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.*;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.C;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.T;

public class Shotaro6point1min
{

    public static void main(final String... args) throws Exception {

        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        // Open Image
        //
        DataStreamingTools dst = new DataStreamingTools();
        dst.openFromDirectory(
                "/Volumes/cba/tischer/projects/shotaro-otsuka-fib-sem-mitosis-segmentation--data/10x10x10",
                "None",
                "6.1.*",
                "None",
                null,
                3,
                true,
                false);
        IJ.wait( 1000 );

        de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run( "" );


    }


}
