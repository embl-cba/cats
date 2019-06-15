import de.embl.cba.cats.CATS;
import de.embl.cba.cats.results.ResultExportSettings;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

import java.io.File;
import java.util.ArrayList;

public class ResultsExport2D
{

	public static void main( String[] args )
	{
		new ImageJ();

		ImagePlus inputImagePlus = IJ.openImage(
				ResultsExport2D.class.getResource( "boat2d/boat2d.zip" ).getFile() );

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


		cats.getResultImage().exportResults( resultExportSettings );


		//
		// Save as Tiff
		//
		resultExportSettings.exportType = ResultExportSettings.SAVE_AS_CLASS_PROBABILITY_TIFF_STACKS;
		resultExportSettings.directory = ResultsExport2D.class.getResource("boat2d" ).getPath();
		resultExportSettings.classNames = cats.getClassNames();
		resultExportSettings.inputImagePlus = cats.getInputImage();
		resultExportSettings.resultImage = cats.getResultImage();
		resultExportSettings.timePointsFirstLast = new int[]{ 0, 0 };

		cats.getResultImage().exportResults( resultExportSettings );

		IJ.open( resultExportSettings.directory + File.separator + "background.tif" );
		IJ.open( resultExportSettings.directory + File.separator + "foreground.tif" );


		//
		// Get as ImagePlus
		//
		resultExportSettings.exportType = ResultExportSettings.GET_AS_IMAGEPLUS_ARRAYLIST;
		resultExportSettings.classNames = cats.getClassNames();
		resultExportSettings.inputImagePlus = cats.getInputImage();
		resultExportSettings.resultImage = cats.getResultImage();
		resultExportSettings.timePointsFirstLast = new int[]{ 0, 0 };

		final ArrayList< ImagePlus > classImps = cats.getResultImage().exportResults( resultExportSettings );

		for ( ImagePlus classImp : classImps )
		{
			classImp.show();
		}



	}
}
