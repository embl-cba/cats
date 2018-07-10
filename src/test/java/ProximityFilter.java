import de.embl.cba.cats.postprocessing.ProximityFilter3D;
import ij.IJ;
import ij.ImagePlus;
import inra.ijpb.segment.Threshold;

public class ProximityFilter
{
    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus mri = IJ.openImage( "mri-stack.zip" );
        mri.show();

        IJ.wait(100);

        // Create binary mask of reference stack
        //
        ImagePlus mriBinary = Threshold.threshold( mri, 100, 255 );
        mriBinary.show();

        // Create dilated reference stack
        //
        ImagePlus proximityFiltered = ProximityFilter3D.filter( mriBinary, mri, 1 );
        proximityFiltered.setTitle( "proximity_filtered" );
        proximityFiltered.show();
    }
}
