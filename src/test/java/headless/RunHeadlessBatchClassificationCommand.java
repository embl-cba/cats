package headless;

import de.embl.cba.cats.ui.BatchClassificationCommand;
import ij.IJ;
import ij.Prefs;

import java.io.File;

public class RunHeadlessBatchClassificationCommand
{
	public static void main( String[] args )
	{
		final BatchClassificationCommand command = new BatchClassificationCommand();

		command.classifierFile = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/blobs/classifier/blobs_00.classifier");
		command.filenameRegExp = ".*";
		command.inputDirectory = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/blobs/input");
		command.outputDirectory = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/blobs/output");

		command.numThreads = Prefs.getThreads();
		command.memoryMB = (int) ( IJ.maxMemory() / ( 1024 * 1024 ) );
		command.run();

	}
}
