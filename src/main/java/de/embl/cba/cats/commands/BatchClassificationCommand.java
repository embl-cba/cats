package de.embl.cba.cats.commands;

import de.embl.cba.cluster.ImageJCommandsSubmitter;
import de.embl.cba.cluster.JobFuture;
import de.embl.cba.cluster.SlurmJobMonitor;
import de.embl.cba.cats.utils.IOUtils;
import de.embl.cba.utils.logging.IJLazySwingLogger;
import ij.Prefs;
import org.scijava.command.Command;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static de.embl.cba.cats.utils.Utils.getSimpleString;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Development>Batch Classification" )
public class BatchClassificationCommand implements Command
{
	@Parameter
	public LogService logService;

	@Parameter
	public CommandService commandService;

	@Parameter (label = "Data directory", style = "directory" )
	public File inputDirectory;

	@Parameter (label = "Filename regular expression" )
	public String filenameRegExp = ".*.tif";

	@Parameter (label = "Classifier" )
	public File classifierFile;

	@Parameter (label = "Minimum number of voxels per object" )
	public int minNumVoxels = 100;

	public int memoryMB = (int) ( Runtime.getRuntime().maxMemory() / 1000 / 1000 );

	public int numWorkers = (int) Runtime.getRuntime().availableProcessors();

	IJLazySwingLogger logger = new IJLazySwingLogger();

	private static String masterRegExp = "(?<treatment>.+)--W(?<well>\\d+)--P(?<position>\\d+)--Z(?<slice>\\d+)--T(?<timePoint>\\d+)--(?<channel>.+)\\.tif";

	private static String[] datasetGroups = {"treatment","well","position"};


	public void run()
	{

		logger.setLogService( logService );

		List< File > filepaths = IOUtils.getFiles( inputDirectory.getAbsolutePath(), filenameRegExp );

		executeClassificationTasks( classifierFile, filepaths );

	}


	private void executeClassificationTasks( File classifierPath, List< File > filePaths )
	{

		for ( File filePath : filePaths )
		{
			logger.info( "Working on: " + filePath  );
			executeClassificationTask( classifierPath, filePath );
		}

	}

	private void executeClassificationTask( File classifierPath, File inputImagePath )
	{

		logger.info( "Classifying pixels..." );

		Map< String, Object > parameters = new HashMap<>();

		//
		// Pixel classification
		//

		Path outputDirectory = Paths.get ( inputImagePath.getParent() + "--analysis" + "/" ); //+ "DataSet--" + simpleDataSetName;
		String dataSetID = getSimpleString( inputImagePath.getName() );

		parameters.clear();
		parameters.put( ApplyClassifierCommand.DATASET_ID, dataSetID );

		parameters.put( IOUtils.INPUT_MODALITY, IOUtils.OPEN_USING_IMAGEJ1 );
		parameters.put( IOUtils.INPUT_IMAGE_FILE,  inputImagePath );
		parameters.put( ApplyClassifierCommand.CLASSIFIER_FILE, classifierPath );
		parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, outputDirectory.toFile() );
		parameters.put( IOUtils.OUTPUT_MODALITY, IOUtils.SAVE_AS_TIFF_STACKS );
		parameters.put( ApplyClassifierCommand.NUM_WORKERS, numWorkers );
		parameters.put( ApplyClassifierCommand.MEMORY_MB, memoryMB );
		parameters.put( ApplyClassifierCommand.CLASSIFICATION_INTERVAL, ApplyClassifierCommand.WHOLE_IMAGE );
		parameters.put( ApplyClassifierCommand.QUIT_AFTER_RUN, false );
		parameters.put( ApplyClassifierCommand.SAVE_RESULTS_TABLE, false );

		parameters.put( "inputImageVSSDirectory", "" );
		parameters.put( "inputImageVSSScheme", "" );
		parameters.put( "inputImageVSSPattern", "" );
		parameters.put( "inputImageVSSHdf5DataSetName", "" );

		commandService.run( ApplyClassifierCommand.class, true, parameters );

		//
		// Object analysis
		//

		/*
		inputImagePath = Paths.get( outputDirectory + "/" + dataSetID + "--foreground.tif" );

		parameters.clear();
		parameters.put( AnalyzeObjectsCommand.DATASET_ID, dataSetID );
		parameters.put( AnalyzeObjectsCommand.INPUT_IMAGE_FILE, PathMapper.asEMBLClusterMounted( inputImagePath ) );
		parameters.put( AnalyzeObjectsCommand.LOWER_THRESHOLD, 1 );
		parameters.put( AnalyzeObjectsCommand.UPPER_THRESHOLD, 255 );
		parameters.put( AnalyzeObjectsCommand.MIN_NUM_VOXELS, minNumVoxels );

		parameters.put( AnalyzeObjectsCommand.OUTPUT_DIRECTORY, PathMapper.asEMBLClusterMounted( outputDirectory ) );
		parameters.put( AnalyzeObjectsCommand.OUTPUT_MODALITY, IOUtils.SAVE );
		parameters.put( AnalyzeObjectsCommand.QUIT_AFTER_RUN, true );

		commandsSubmitter.addIJCommandWithParameters( AnalyzeObjectsCommand.PLUGIN_NAME , parameters );
		*/

	}

}

