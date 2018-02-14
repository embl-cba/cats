package de.embl.cba.trainableDeepSegmentation.commands;

import de.embl.cba.cluster.ImageJCommandsSubmitter;
import de.embl.cba.cluster.JobFuture;
import de.embl.cba.cluster.SlurmQueue;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import de.embl.cba.utils.logging.IJLazySwingLogger;
import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.TextWidget;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static de.embl.cba.trainableDeepSegmentation.utils.IOUtils.clusterMounted;
import static de.embl.cba.trainableDeepSegmentation.utils.Utils.getSimpleString;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Development>Batch Classification On Cluster" )
public class RunSylwiasWorkflowOnSlurm implements Command
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

    @Parameter (label = "Queue", required = false, choices = { SlurmQueue.DEFAULT_QUEUE, SlurmQueue.ONE_DAY_QUEUE, SlurmQueue.ONE_WEEK_QUEUE, SlurmQueue.BIGMEM_QUEUE, SlurmQueue.GPU_QUEUE } )
    public String queue = SlurmQueue.DEFAULT_QUEUE;
    public static final String SLURM_QUEUE = "queue";

    @Parameter (label = "Memory [MB]"  )
    public int memoryMB = 32000;

    @Parameter (label = "Number of workers" )
    public int numWorkers = 16;

    @Parameter (label = "ImageJ" )
    public String imageJ = ImageJCommandsSubmitter.IMAGEJ_EXECTUABLE_ALMF_CLUSTER_XVFB;


    static String masterRegExp="(?<treatment>.+)--W(?<well>\\d+)--P(?<position>\\d+)--Z(?<slice>\\d+)--T(?<timePoint>\\d+)--(?<channel>.+)\\.tif";

    static String[] datasetGroups={"treatment","well","position"};

    IJLazySwingLogger logger = new IJLazySwingLogger();

    public void run()
    {

        logger.setLogService( logService );

        List< Path > dataSetPatterns = getDataSetPatterns();

        ArrayList< JobFuture > jobFutures = submitJobsOnSlurm(
                imageJ,
                clusterMounted( jobDirectory.toPath() ) ,
                clusterMounted( classifierFile.toPath() ),
                clusterMounted( dataSetPatterns ) );

        monitorJobProgress( jobFutures );

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
                    HashMap< String, Object > output = jobFuture.get();
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

        ImageJCommandsSubmitter commandsSubmitter = new ImageJCommandsSubmitter(
                ImageJCommandsSubmitter.EXECUTION_SYSTEM_EMBL_SLURM,
                jobDirectory.toString(),
                ImageJCommandsSubmitter.IMAGEJ_EXECTUABLE_ALMF_CLUSTER_XVFB,
                username, password );

        ArrayList< JobFuture > jobFutures = new ArrayList<>( );

        for ( Path dataSetPattern : dataSetPatterns )
        {
            commandsSubmitter.clearCommands();
            setCommandAndParameterStrings( commandsSubmitter, dataSetPattern, classifierPath );
            jobFutures.add( commandsSubmitter.submitCommands( memoryMB, numWorkers, queue ) );
        }

        return jobFutures;
    }


    private void setCommandAndParameterStrings( ImageJCommandsSubmitter commandsSubmitter, Path inputImagePath, Path classifierPath )
    {

        String outputDirectory = inputImagePath.getParent() + "--analysis" + "/"; //+ "DataSet--" + simpleDataSetName;
        String dataSetID = getSimpleString( inputImagePath.getFileName().toString() );

        commandsSubmitter.addLinuxCommand( "hostname" );
        commandsSubmitter.addLinuxCommand( "lscpu" );
        commandsSubmitter.addLinuxCommand( "free -m" );
        commandsSubmitter.addLinuxCommand( "START_TIME=$SECONDS" );

        Map< String, Object > parameters = new HashMap<>();

        parameters.clear();
        parameters.put( ApplyClassifierCommand.DATASET_ID, dataSetID );

        parameters.put( IOUtils.INPUT_MODALITY, IOUtils.OPEN_USING_IMAGEJ1_IMAGE_SEQUENCE );
        parameters.put( IOUtils.INPUT_IMAGE_PATH, inputImagePath );
        parameters.put( ApplyClassifierCommand.CLASSIFIER_FILE, classifierPath );
        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( IOUtils.OUTPUT_MODALITY, IOUtils.SAVE_AS_TIFF_STACKS );
        parameters.put( ApplyClassifierCommand.NUM_WORKERS, numWorkers );
        parameters.put( ApplyClassifierCommand.MEMORY_MB, memoryMB );
        parameters.put( ApplyClassifierCommand.CLASSIFICATION_INTERVAL, ApplyClassifierCommand.WHOLE_IMAGE );
        parameters.put( ApplyClassifierCommand.QUIT_AFTER_RUN, true );
        parameters.put( ApplyClassifierCommand.SAVE_RESULTS_TABLE, true );

        commandsSubmitter.addIJCommandWithParameters( ApplyClassifierCommand.PLUGIN_NAME , parameters );


        parameters.clear();
        parameters.put( AnalyzeObjectsCommand.DATASET_ID, dataSetID );
        parameters.put( AnalyzeObjectsCommand.INPUT_IMAGE_PATH, new File( outputDirectory + "/" + dataSetID + "--foreground.tif" ) );
        parameters.put( AnalyzeObjectsCommand.LOWER_THRESHOLD, 1 );
        parameters.put( AnalyzeObjectsCommand.UPPER_THRESHOLD, 255 );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_MODALITY, IOUtils.SHOW_RESULTS_TABLE );
        parameters.put( AnalyzeObjectsCommand.QUIT_AFTER_RUN, true );

        commandsSubmitter.addIJCommandWithParameters( AnalyzeObjectsCommand.PLUGIN_NAME , parameters );


        commandsSubmitter.addLinuxCommand( "ELAPSED_TIME=$(($SECONDS - $START_TIME))" );

        commandsSubmitter.addLinuxCommand( "echo \"Elapsed time [s]:\"" );
        commandsSubmitter.addLinuxCommand( "echo $ELAPSED_TIME" );


    }


    public static void main(final String... args) throws Exception
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ij.command().run( RunSylwiasWorkflowOnSlurm.class, true );

    }
}
