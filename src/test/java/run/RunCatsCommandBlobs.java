package run;

import de.embl.cba.cats.ui.CATSCommand;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.ImageJ;

public class RunCatsCommandBlobs
{
    public static void main( final String... args )
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ImagePlus imp = IJ.openImage(
               "/Users/tischer/Documents/fiji-plugin-cats/src/test/resources/blobs/input/blobs_00.tif" );

		imp.show();

        ij.command().run( CATSCommand.class, true );
    }
}
