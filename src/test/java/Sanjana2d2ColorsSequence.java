import de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin;
import ij.IJ;
import ij.ImagePlus;



public class Sanjana2d2ColorsSequence
{

    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage("2d-2c-series.tif" );
        inputImagePlus.show();

        IJ.wait(100);

        DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");

    }

}
