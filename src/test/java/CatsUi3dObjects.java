import de.embl.cba.cats.ui.CATSPlugin;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.ImageJ;



public class CatsUi3dObjects
{
	public static void main(final String... args) throws Exception
	{

		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		ImagePlus inputImagePlus = IJ.openImage("3d-objects.zip" );
		inputImagePlus.show();

		ij.command().run( CATSPlugin.class, true );

	}

}
