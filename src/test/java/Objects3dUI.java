import ij.IJ;
import ij.ImagePlus;
import de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin;


public class Objects3dUI
{
    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage( Objects3dUI.class.getResource( "3d-objects.zip" ).getFile() );
        inputImagePlus.show();

        IJ.wait(100);

        DeepSegmentationIJ1Plugin weka_segmentation = new DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");
    }

}
