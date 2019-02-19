import de.embl.cba.cats.commands.CATSCommand;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.ImageJ;



public class CatsUi3dObjects
{
	public static void main(final String... args) throws Exception
	{

		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		ImagePlus inputImagePlus = IJ.openImage( CatsObjects3d.class.getResource(  "3d-objects.zip" ).getFile() );
		inputImagePlus.show();

		ij.command().run( CATSCommand.class, true );

	}

}
