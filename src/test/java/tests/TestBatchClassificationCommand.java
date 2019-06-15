package tests;

import de.embl.cba.cats.results.ResultExportSettings;
import de.embl.cba.cats.ui.BatchClassificationCommand;
import ij.IJ;
import ij.Prefs;
import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertTrue;

public class TestBatchClassificationCommand
{
	@Test
	public void testProbabilitiesExport( )
	{
		final BatchClassificationCommand command = new BatchClassificationCommand();

		command.classifierFile = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/test-data/blobs/classifier/blobs_00.classifier");
		command.filenameRegExp = ".*";
		command.inputDirectory = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/test-data/blobs/input");
		command.outputDirectory = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/test-data/blobs/output");

		command.exportType = ResultExportSettings.SAVE_AS_CLASS_PROBABILITY_TIFF_STACKS;

		command.numThreads = Prefs.getThreads();
		command.memoryMB = (int) ( IJ.maxMemory() / ( 1024 * 1024 ) );

		final File testOutputFile = new File( command.outputDirectory + File.separator + "blobs_00--background.tif" );

		if ( testOutputFile.exists() )
		{
			System.out.println( "Deleting test file: " + testOutputFile );
			testOutputFile.delete();
		}

		command.run();

		assertTrue( testOutputFile.exists() );
	}


	@Test
	public void testLabelMaskExport( )
	{
		final BatchClassificationCommand command = new BatchClassificationCommand();

		command.classifierFile = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/test-data/blobs/classifier/blobs_00.classifier");
		command.filenameRegExp = ".*";
		command.inputDirectory = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/test-data/blobs/input");
		command.outputDirectory = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/test-data/blobs/output");

		command.exportType = ResultExportSettings.SAVE_AS_CLASS_LABEL_MASK_TIFF_STACKS;

		command.numThreads = Prefs.getThreads();
		command.memoryMB = (int) ( IJ.maxMemory() / ( 1024 * 1024 ) );

		final File testOutputFile = new File(
				command.outputDirectory + File.separator + "blobs_00--labelMask.tif" );

		if ( testOutputFile.exists() )
		{
			System.out.println( "Deleting test file: " + testOutputFile );
			testOutputFile.delete();
		}

		command.run();

		assertTrue( testOutputFile.exists() );
	}

	public static void main( String[] args )
	{
		new TestBatchClassificationCommand().testLabelMaskExport();
		//new TestBatchClassificationCommand().testProbabilitiesExport();
	}


}
