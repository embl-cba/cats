import de.embl.cba.cats.CATS;
import de.embl.cba.cats.results.ResultExportSettings;
import de.embl.cba.cats.results.ResultImage;
import ij.IJ;

public class APIExample01
{
	public static void main( String[] args )
	{
		// project dependent input parameters
		final String inputImagePath = "";
		final String classifierPath = "";
		final String outputDirectory = "";
		final String outputFileNamesPrefix = "";

		//
		// project independent code
		//

		// create instance
		final CATS cats = new CATS();

		// load and set the image to be classified
		cats.setInputImage( IJ.openImage( inputImagePath ) );

		// configure CATS such that the result (classified) image is allocated in RAM
		// for big image data this is not possible and thus there is the other option:
		// cats.setResultImageDisk( directory );
		cats.setResultImageRAM();

		// load classifier (to be trained and saved before using the UI)
		cats.loadClassifier( classifierPath );

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
