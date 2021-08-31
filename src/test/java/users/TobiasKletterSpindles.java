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

//		ImagePlus imp = IJ.openImage(
//				"/Users/tischer/Documents/spindle-feedback-kletter-knime/CATS/Confocal/3D_Iso0.25um_8bit.tif" );

		ImagePlus imp = IJ.openImage(
				"/Users/tischer/Documents/tobias-kletter/CATS/SpinningDisc/3D_Iso0.25um_16bit.tif" );

//		ImagePlus imp = IJ.openImage(
//				"/Users/tischer/Documents/spindle-feedback-kletter-knime/CATS/SpinningDisc/3D_BrightOtherDNA_iso0.25um_16bit.zip" );

		imp.show();
		IJ.run("Make Composite", "");
		imp = IJ.getImage();
		( ( CompositeImage ) imp ).setC( 2 );
		IJ.run(imp, "Grays", "");

		Prefs.setThreads( 4 );

		ij.command().run( CATSCommand.class, true );
	}
}
