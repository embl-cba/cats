package headless;

import de.embl.cba.cats.ui.BatchClassificationCommand;
import net.imagej.ImageJ;

import java.io.File;

public class RunHeadlessBatchClassificationCommand
{
	public static void main( String[] args )
	{

		final ImageJ imageJ = new ImageJ();
		final BatchClassificationCommand command = new BatchClassificationCommand();

		command.classifierFile = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/blobs/classifier/blobs_00.classifier");
		command.commandService = imageJ.command();
		command.filenameRegExp = ".*";
		command.inputDirectory = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/blobs/input");
		command.logService = imageJ.log();
		command.outputDirectory = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/blobs/output");

		command.run();

	}
}
