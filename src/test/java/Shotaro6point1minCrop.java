import de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin;
import ij.IJ;
import ij.ImagePlus;

public class Shotaro6point1minCrop
{

    public static void main(final String... args) throws Exception
    {

        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage( "/Volumes/cba/tischer/projects/shotaro-otsuka-fib-sem-mitosis-segmentation--data/10x10x10-crop-test/6.1min-crop.tif" );
        inputImagePlus.show();

        IJ.wait( 100 );

        DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run( "" );

    }
}
