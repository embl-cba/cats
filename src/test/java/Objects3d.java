import de.embl.cba.cats.CATS;
import de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin;
import ij.IJ;
import ij.ImagePlus;


public class Objects3d
{

    public static void main( final String[] args )
    {

        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        // Open Image
        //
        ImagePlus inputImagePlus = IJ.openImage("3d-objects.zip" );
        inputImagePlus.show();

        CATS cats = new CATS();
        cats.setInputImage( inputImagePlus );
        cats.setResultImageDisk( "3d-objects-probabilities" );
        cats.loadInstancesAndMetadata( "3d-objects-instances/3d-objects.ARFF" );

        DeepSegmentationIJ1Plugin plugin = new DeepSegmentationIJ1Plugin();
        plugin.initialise( cats, false );
        //plugin.segmentObjects();
        //plugin.reviewObjects();

    }

}
