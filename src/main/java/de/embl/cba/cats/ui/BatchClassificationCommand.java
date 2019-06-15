package de.embl.cba.cats.ui;

import de.embl.cba.cats.CATS;
import de.embl.cba.cats.results.ResultExportSettings;
import de.embl.cba.cats.results.ResultImage;
import de.embl.cba.cats.utils.IOUtils;
import de.embl.cba.cats.utils.Utils;
import de.embl.cba.utils.logging.IJLazySwingLogger;
import ij.IJ;
import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.util.List;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>CATS>Apply Classifier on Multiple Images" )
public class BatchClassificationCommand implements Command
{
	@Parameter
	public LogService logService;

	@Parameter (label = "Input directory containing the images", style = "directory" )
	public File inputDirectory;

	@Parameter (label = "Filename regular expression" )
	public String filenameRegExp = ".*.tif";

	@Parameter (label = "Classifier" )
	public File classifierFile;

	@Parameter (label = "Output directory", style = "directory" )
	public File outputDirectory;

	@Parameter (label = "Number of threads" )
	public int numThreads = Runtime.getRuntime().availableProcessors();

	@Parameter (label = "Memory [MB]" )
	public long memoryMB = IJ.maxMemory() / ( 1024 * 1024 );

	@Parameter (label = "Export modality", choices = {
			ResultExportSettings.SAVE_AS_CLASS_PROBABILITY_TIFF_STACKS, ResultExportSettings.SAVE_AS_CLASS_LABEL_MASK_TIFF_STACKS } )
	public String exportType = ResultExportSettings.SAVE_AS_CLASS_PROBABILITY_TIFF_STACKS;

	IJLazySwingLogger logger = new IJLazySwingLogger();

	public void run()
	{
		logger.setLogService( logService );

		List< File > filepaths =
				IOUtils.getFiles( inputDirectory.getAbsolutePath(), filenameRegExp );

		classifyImagesAndSaveResults( classifierFile, filepaths, outputDirectory );
	}


	private void classifyImagesAndSaveResults(
			File classifierPath,
			List< File > filePaths,
			File outputDirectory  )
	{

		for ( File filePath : filePaths )
		{
			logger.info( "Working on: " + filePath  );
			classifyImageAndSaveResult( classifierPath, filePath, outputDirectory  );
		}

	}

	private void classifyImageAndSaveResult(
			File classifierPath,
			File inputImagePath,
			File outputDirectory )
	{
		// create instance
		final CATS cats = new CATS();

		// load and set the image to be classified
		cats.setInputImage( IJ.openImage( inputImagePath.getAbsolutePath() ) );

		// configure CATS such that the result (classified) image is allocated in RAM
		// for big image data this is not possible and thus there is the other option:
		// cats.setResultImageDisk( directory );
		cats.setResultImageRAM();

		cats.setNumThreads( numThreads );
		cats.setMaxMemoryBytes( memoryMB * 1024 * 1024 );

		// load classifier (to be trained and saved before using the UI)
		cats.loadClassifier( classifierPath );

		// apply classifier
		cats.applyClassifierWithTiling();

		// configure results export
		final ResultExportSettings resultExportSettings = new ResultExportSettings();
		resultExportSettings.inputImagePlus = cats.getInputImage();
		resultExportSettings.exportType = exportType;
		resultExportSettings.directory = outputDirectory.getAbsolutePath();
		resultExportSettings.exportNamesPrefix = Utils.removeExtension( inputImagePath.getName() ) + "--" ;
		resultExportSettings.classNames = cats.getClassNames();

		final ResultImage resultImage = cats.getResultImage();
		resultImage.exportResults( resultExportSettings );
	}

	public static void main( final String... args )
	{
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		ij.command().run( BatchClassificationCommand.class, true );
	}

}

