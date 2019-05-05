package command;

import de.embl.cba.cats.ui.BatchClassificationCommand;
import net.imagej.ImageJ;

public class RunBatchClassificationCommand
{
	public static void main( final String... args )
	{
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		ij.command().run( BatchClassificationCommand.class, true );
	}
}
