import de.embl.cba.cats.commands.BatchClassificationCommand;
import de.embl.cba.cats.ui.CATSCommand;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.ImageJ;

public class BatchClassificationCommandTest
{

	public static void main( final String... args ) throws Exception
	{

		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		ij.command().run( BatchClassificationCommand.class, true );

	}

}
