import de.embl.cba.cats.CATS;
import de.embl.cba.cats.results.ResultExportSettings;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

import java.io.File;

public class ResultsExport2D
{

	public static void main( String[] args )
	{


		ImagePlus inputImagePlus = IJ.openImage( ResultsExport2D.class.getResource( "boat2d/boat2d.zip" ).getFile() );

		CATS cats = new CATS();
		cats.setInputImage( inputImagePlus );
		cats.setResultImageRAM( );
		cats.loadClassifier( ResultsExport2D.class.getResource("boat2d/boat2d.classifier" ).getFile() );
		cats.applyClassifierWithTiling();


		ResultExportSettings resultExportSettings = new ResultExportSettings();

		//
		// Show in ImageJ
		//
		resultExportSettings.exportType = ResultExportSettings.SHOW_IN_IMAGEJ;
		resultExportSettings.classNames = cats.getClassNames();
		resultExportSettings.inputImagePlus = cats.getInputImage();
		resultExportSettings.resultImage = cats.getResultImage();
		resultExportSettings.timePointsFirstLast = new int[]{ 0, 0 };

		new ImageJ();
		cats.getResultImage().exportResults( resultExportSettings );


		//
		// Save as Tiff
		//
		resultExportSettings.exportType = ResultExportSettings.TIFF_STACKS;
		resultExportSettings.directory = ResultsExport2D.class.getResource("boat2d" ).getPath();
		resultExportSettings.classNames = cats.getClassNames();
		resultExportSettings.inputImagePlus = cats.getInputImage();
		resultExportSettings.resultImage = cats.getResultImage();
		resultExportSettings.timePointsFirstLast = new int[]{ 0, 0 };

		cats.getResultImage().exportResults( resultExportSettings );

		IJ.open( resultExportSettings.directory + File.separator + "background.tif" );
		IJ.open( resultExportSettings.directory + File.separator + "foreground.tif" );


	}
}
