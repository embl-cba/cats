import ij.IJ;
import ij.ImagePlus;

public class Cells3Ch3D
{
    public final static String TEST_RESOURCES = "/Users/tischer/Documents/fiji-plugin-CATS/src/test/resources/";

    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage("hela-cells-3ch-2d.zip" );
        inputImagePlus.show();

        IJ.wait(100);

        de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");
    }
}