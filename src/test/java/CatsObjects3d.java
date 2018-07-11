import de.embl.cba.cats.CATS;
import de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin;
import ij.IJ;
import ij.ImagePlus;


public class CatsObjects3d
{

    public static void main( final String[] args )
    {

        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        // Open Image
        //
        ImagePlus inputImagePlus = IJ.openImage( CatsObjects3d.class.getResource( "3d-objects.zip" ).getFile() );
        inputImagePlus.show();

        CATS cats = new CATS();
        cats.setInputImage( inputImagePlus );
        cats.setResultImageDisk( CatsObjects3d.class.getResource( "3d-objects-probabilities" ).getFile() );
        cats.loadInstancesAndMetadata( CatsObjects3d.class.getResource( "3d-objects-instances/3d-objects.ARFF" ).getFile() );
        cats.segmentObjects();
        cats.reviewObjects();
        cats.getObjectReview().getObjectsInRoiManagerAsImage().show();
        //plugin.segmentObjects();
        //plugin.reviewObjects();

    }

}
