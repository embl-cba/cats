package de.embl.cba.cats.commands;

import de.embl.cba.cluster.ImageJCommandsSubmitter;
import de.embl.cba.cluster.JobFuture;
import de.embl.cba.cluster.JobSettings;
import de.embl.cba.cluster.SlurmJobMonitor;
import de.embl.cba.cats.utils.CommandUtils;
import de.embl.cba.cats.utils.IOUtils;
import de.embl.cba.utils.fileutils.PathMapper;
import de.embl.cba.utils.logging.IJLazySwingLogger;
import net.imagej.ops.image.ImageNamespace;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.TextWidget;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static de.embl.cba.cats.utils.Utils.getSimpleString;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Development>Batch Classification On Cluster" )
public class BatchClassificationOnSlurmCommand implements Command
{
    @Parameter
    public LogService logService;

    @Parameter(label = "Username" )
    private String username;

    @Parameter(label = "Password", style = TextWidget.PASSWORD_STYLE, persist = false )
    private String password;

    @Parameter (label = "Data directory", style = "directory" )
    public File inputDirectory;

    @Parameter (label = "Job directory", style = "directory" )
    public File jobDirectory;

    @Parameter (label = "Classifier" )
    public File classifierFile;

    @Parameter (label = "Minimum number of voxels per object" )
    public int minNumVoxels;

    @Parameter (label = "Memory [MB]"  )
    public int memoryMB = 32000;

    @Parameter (label = "Number of threads per job" )
    public int numWorkers = 12;

    @Parameter (label = "Time per job in minutes" )
    public int timePerJobInMinutes = 120;

    /*
    @Parameter( label = "ImageJ executable (must be linux and cluster accessible)", required = false )
    public File imageJFile;
    public static final String IMAGEJ_FILE = "imageJFile";
    */

    static String masterRegExp = "(?<treatment>.+)--W(?<well>\\d+)--P(?<position>\\d+)--Z(?<slice>\\d+)--T(?<timePoint>\\d+)--(?<channel>.+)\\.tif";

    static String[] datasetGroups = {"treatment","well","position"};

    IJLazySwingLogger logger = new IJLazySwingLogger();

    public void run()
    {

        logger.setLogService( logService );

        List< Path > dataSetPatterns = getDataSetPatterns();

        ArrayList< JobFuture > jobFutures = submitJobsOnSlurm(
                ImageJCommandsSubmitter.IMAGEJ_EXECTUABLE_ALMF_CLUSTER_HEADLESS,
                jobDirectory.toPath() ,
                classifierFile.toPath(),
                dataSetPatterns );

        SlurmJobMonitor slurmJobMonitor = new SlurmJobMonitor( logger );
        slurmJobMonitor.monitorJobProgress( jobFutures, 60, 5 );

    }


    private List< Path > getDataSetPatterns()
    {
        logger.info( "Extracting data sets from directory: " + inputDirectory );
        logger.info( "Using regular expression: " + masterRegExp );
        List< Path > dataSetPatterns = IOUtils.getDataSetPatterns( inputDirectory.toString(), masterRegExp, datasetGroups );
        for ( Path dataSetPattern : dataSetPatterns )
        {
            logger.info( "Data set pattern: " + dataSetPattern );
        }
        return dataSetPatterns;
    }

    private void monitorJobProgress( ArrayList< JobFuture > jobFutures )
    {

        ArrayList< JobFuture > doneJobs = new ArrayList<>();

        while ( doneJobs.size() < jobFutures.size() )
        {
            for ( JobFuture jobFuture : jobFutures )
            {
                if ( jobFuture.isStarted() )
                {

                    String currentOutput = jobFuture.getOutput();

                    if  ( ! doneJobs.contains( jobFuture ) )
                    {
                        String[] currentOutputLines = currentOutput.split( "\n" );
                        String lastLine = currentOutputLines[ currentOutputLines.length - 1 ];
                        logger.info( "Current last line of job output: " + lastLine );
                    }

                    if ( jobFuture.isDone() )
                    {

                        logger.info("Final and full job output:" );
                        logger.info( currentOutput );

                        doneJobs.add( jobFuture );

                        if ( doneJobs.size() == jobFutures.size() )
                        {
                            break;
                        }

                    }
                }
                else
                {
                    logger.info( "Job " + jobFuture.getJobID() + " has not yet started." );
                }

                /*
                try
                {
                    HashMap< String, Object > output = jobFuture.getBinned();
                } catch ( InterruptedException e )
                {
                    e.printStackTrace();
                } catch ( ExecutionException e )
                {
                    e.printStackTrace();
                }
                */

                try { Thread.sleep( 5000 ); } catch ( InterruptedException e ) { e.printStackTrace(); }
            }
        }

        logger.info( "All jobs finished." );

    }

    private ArrayList< JobFuture > submitJobsOnSlurm( String imageJ, Path jobDirectory, Path classifierPath, List< Path > dataSetPatterns )
    {

        ImageJCommandsSubmitter commandsSubmitter = getImageJCommandsSubmitter( imageJ, jobDirectory );

        JobSettings jobSettings = getJobSettings();

        ArrayList< JobFuture > jobFutures = new ArrayList<>( );

        for ( Path dataSetPattern : dataSetPatterns )
        {
            commandsSubmitter.clearCommands();
            setCommandAndParameterStrings( commandsSubmitter, dataSetPattern, classifierPath );
            jobFutures.add( commandsSubmitter.submitCommands( jobSettings ) );
        }

        return jobFutures;
    }

    private JobSettings getJobSettings()
    {
        JobSettings jobSettings = new JobSettings();
        jobSettings.queue = JobSettings.DEFAULT_QUEUE;
        jobSettings.numWorkersPerNode = numWorkers;
        jobSettings.timePerJobInMinutes = timePerJobInMinutes;
        jobSettings.memoryPerJobInMegaByte = memoryMB;
        return jobSettings;
    }

    private ImageJCommandsSubmitter getImageJCommandsSubmitter( String imageJ, Path jobDirectory )
    {
        return new ImageJCommandsSubmitter(
                    ImageJCommandsSubmitter.EXECUTION_SYSTEM_EMBL_SLURM,
                    PathMapper.asEMBLClusterMounted( jobDirectory.toString() ),
                    imageJ,
                    username,
                    password );
    }


    private void setCommandAndParameterStrings( ImageJCommandsSubmitter commandsSubmitter, Path inputImagePath, Path classifierPath )
    {

        Map< String, Object > parameters = new HashMap<>();

        //
        // Pixel classification
        //

        Path outputDirectory = Paths.get ( inputImagePath.getParent() + "--analysis" + "/" ); //+ "DataSet--" + simpleDataSetName;
        String dataSetID = getSimpleString( inputImagePath.getFileName().toString() );

        parameters.clear();
        parameters.put( ApplyClassifierCommand.DATASET_ID, dataSetID );

        parameters.put( IOUtils.INPUT_MODALITY, IOUtils.OPEN_USING_IMAGEJ1_IMAGE_SEQUENCE );
        parameters.put( IOUtils.INPUT_IMAGE_FILE, PathMapper.asEMBLClusterMounted( inputImagePath ) );
        parameters.put( ApplyClassifierCommand.CLASSIFIER_FILE, PathMapper.asEMBLClusterMounted( classifierPath ) );
        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, PathMapper.asEMBLClusterMounted( outputDirectory ) );
        parameters.put( IOUtils.OUTPUT_MODALITY, IOUtils.SAVE_AS_TIFF_STACKS );
        parameters.put( ApplyClassifierCommand.NUM_WORKERS, numWorkers );
        parameters.put( ApplyClassifierCommand.MEMORY_MB, memoryMB );
        parameters.put( ApplyClassifierCommand.CLASSIFICATION_INTERVAL, ApplyClassifierCommand.WHOLE_IMAGE );
        parameters.put( ApplyClassifierCommand.QUIT_AFTER_RUN, true );
        parameters.put( ApplyClassifierCommand.SAVE_RESULTS_TABLE, true );

        parameters.put( "inputImageVSSDirectory", "" );
        parameters.put( "inputImageVSSScheme", "" );
        parameters.put( "inputImageVSSPattern", "" );
        parameters.put( "inputImageVSSHdf5DataSetName", "" );

        commandsSubmitter.addIJCommandWithParameters( ApplyClassifierCommand.PLUGIN_NAME , parameters );

        //
        // Object analysis
        //

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

    }

}
