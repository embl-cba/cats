import de.embl.cba.bigDataTools.dataStreamingTools.DataStreamingTools;
import ij.IJ;

public class AnnaSteyerInterphase
{


    public static void main(final String... args) throws Exception {

        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        // Open Image
        //
        DataStreamingTools dst = new DataStreamingTools();
        dst.openFromDirectory(
                "/Volumes/cba/tischer/projects/anna-steyer-fib-sem-interphase-cell-segmentation--data/data",
                "None",
                ".*",
                "None",
                null,
                3,
                true,
                false);
        IJ.wait( 5000 );

        de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run( "" );


    }
}
