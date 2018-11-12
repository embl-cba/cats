import de.embl.cba.cats.CATS;
import de.embl.cba.cats.results.ResultExportSettings;
import de.embl.cba.cats.results.ResultImage;
import ij.IJ;

import java.io.File;

public class TestMinimalAPI
{
	public static void main( String[] args )
	{
		// user input
		final String inputImagePath = "";
		final String classifierPath = "";
		final String outputDirectory = "";
		final String outputFileNamesPrefix = "";
		
		// create instance
		final CATS cats = new CATS();

		// load the image to be classified
		cats.setInputImage( IJ.openImage( inputImagePath ) );
		cats.setResultImageRAM();

		// load classifier (to be trained and saved using the UI)
		cats.loadClassifier( classifierPath);

		// apply classifier
		cats.applyClassifierWithTiling();

		// configure results export
		final ResultExportSettings resultExportSettings = new ResultExportSettings();
		resultExportSettings.inputImagePlus = cats.getInputImage();
		resultExportSettings.exportType = ResultExportSettings.TIFF_STACKS;
		resultExportSettings.directory = outputDirectory;
		resultExportSettings.exportNamesPrefix = outputFileNamesPrefix + "-";
		resultExportSettings.classNames = cats.getClassNames();

		// save results
		final ResultImage resultImage = cats.getResultImage();
		resultImage.exportResults( resultExportSettings );
	}
}
