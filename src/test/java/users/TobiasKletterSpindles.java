package users;

import de.embl.cba.cats.ui.CATSCommand;
import ij.CompositeImage;
import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
import net.imagej.ImageJ;

public class TobiasKletterSpindles
{
	public static void main( final String... args )
	{
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		ImagePlus imp = IJ.openImage(
				"/Users/tischer/Documents/spindle-feedback-kletter-knime/CATS/Isotropic_spindles_for_CATS.tif" );

		imp.show();
		IJ.run("Make Composite", "");
		imp = IJ.getImage();
		( ( CompositeImage ) imp ).setC( 2 );
		IJ.run(imp, "Blue", "");

		Prefs.setThreads( 4 );

		ij.command().run( CATSCommand.class, true );
	}
}
