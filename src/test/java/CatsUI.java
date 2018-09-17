import de.embl.cba.cats.ui.CATSCommand;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.FolderOpener;
import net.imagej.ImageJ;
import org.python.core.imp;

public class CatsUI
{

    public static void main( final String... args ) throws Exception
    {

        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

//        ImagePlus imp = IJ.openImage( CatsUI.class.getResource( "3d-objects-2channels.zip" ).getFile() );
//        ImagePlus imp = IJ.openImage( "/Users/tischer/Documents/andrea-callegari-stitching--data/MolDev/MolDev-001-scale0.5.tif" );
//	 	ImagePlus imp = IJ.openImage( "/Users/tischer/Documents/ashna-alladin-organoid-segmentation--data/seg-train-test.tif" );
//
	 	ImagePlus imp = IJ.openVirtual( "/Volumes/cba/exchange/paolo/hela-interphase01/input-data/hela-interphase01-iso10nm-8bit.tif" );

//        ImagePlus imp = FolderOpener.open("/Users/tischer/Documents/andrea-callegari-stitching--data/MolDev/2018-08-10-raw-test--processed/", "virtual");
//		imp.setDimensions( 1, 1, imp.getNSlices() );

		imp.show();

        ij.command().run( CATSCommand.class, true );

    }


}
