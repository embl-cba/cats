import de.embl.cba.cats.commands.BatchClassificationCommand;
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
