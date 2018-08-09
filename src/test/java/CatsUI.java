import de.embl.cba.cats.ui.CATSCommand;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.ImageJ;

public class CatsUI
{

    public static void main( final String... args ) throws Exception
    {

        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

//        ImagePlus inputImagePlus = IJ.openImage( CatsUI.class.getResource( "3d-objects-2channels.zip" ).getFile() );
        ImagePlus inputImagePlus = IJ.openImage( "/Users/tischer/Documents/andrea-callegari-stitching--data/MolDev/MolDev-001-scale0.5.tif" );
//        ImagePlus inputImagePlus = IJ.openImage( "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/mitosis-3d-2ch-movie.zip" );

        inputImagePlus.show();

        ij.command().run( CATSCommand.class, true );

    }


}
