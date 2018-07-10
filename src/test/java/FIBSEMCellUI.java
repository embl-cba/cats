import ij.IJ;
import ij.ImagePlus;


public class FIBSEMCellUI
{
    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage( FIBSEMCellUI.class.getResource( "fib-sem--cell--8x8x8nm.zip" ).getFile() );
        inputImagePlus.show();

        IJ.wait(100);

        de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");
    }

}
