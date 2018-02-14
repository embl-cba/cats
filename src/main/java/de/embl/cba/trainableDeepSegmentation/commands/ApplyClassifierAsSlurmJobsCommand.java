package de.embl.cba.trainableDeepSegmentation.commands;

import de.embl.cba.cluster.ImageJCommandsSubmitter;
import de.embl.cba.cluster.JobFuture;
import de.embl.cba.cluster.SlurmQueue;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils;
import de.embl.cba.trainableDeepSegmentation.utils.SlurmUtils;
import net.imagej.ImageJ;
import net.imglib2.FinalInterval;
import org.scijava.command.Command;
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

import static de.embl.cba.trainableDeepSegmentation.DeepSegmentation.logger;
import static de.embl.cba.trainableDeepSegmentation.utils.IOUtils.clusterMounted;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.T;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.XYZT;
import static de.embl.cba.trainableDeepSegmentation.utils.Utils.getSimpleString;


@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Development>Apply Classifier Test" )
public class ApplyClassifierAsSlurmJobsCommand implements Command
{

    @Parameter( label = "Username" )
    private String username = "tischer";

    @Parameter( label = "Password", style = TextWidget.PASSWORD_STYLE, persist = false )
    private String password;

    @Parameter( label = "Queue", choices = { SlurmQueue.DEFAULT_QUEUE, SlurmQueue.ONE_DAY_QUEUE, SlurmQueue.ONE_WEEK_QUEUE, SlurmQueue.BIGMEM_QUEUE, SlurmQueue.GPU_QUEUE } )
    public String queue = SlurmQueue.DEFAULT_QUEUE;

    @Parameter( label = "Number of jobs" )
    public int numJobs = 20;

    @Parameter( label = "Number of CPUs per job" )
    public int numWorkers;
    public static final String WORKERS = "numWorkers";

    @Parameter( label = "Classifier file (must be cluster accessible)" )
    public File classifierFile;

    @Parameter()
    public String inputModality;

    @Parameter()
    public File inputImagePath; // TODO: rename to File

    @Parameter()
    public File outputDirectory;

    @Parameter()
    public FinalInterval interval;
    public static final String INTERVAL = "interval";

    @Parameter( required = false )
    public String inputImageVSSDirectory;

    @Parameter( required = false )
    public String inputImageVSSScheme;

    @Parameter( required = false )
    public String inputImageVSSPattern;

    @Parameter( required = false )
    public String inputImageVSSHdf5DataSetName;

    public String imageJ = ImageJCommandsSubmitter.IMAGEJ_EXECTUABLE_ALMF_CLUSTER_XVFB;


    public static int memoryFactor = 10;

    public boolean quitAfterRun = true;

    FinalInterval inputImageIntervalXYZT;

    public void run()
    {

        Path jobDirectory = Paths.get( "/g/cba/cluster/" + username );

        List< Path > dataSets = new ArrayList<>();
        dataSets.add( inputImagePath.toPath() );

        ArrayList< JobFuture > jobFutures = submitJobsOnSlurm( imageJ,
                clusterMounted( jobDirectory ) ,
                clusterMounted( classifierFile.toPath() ),
                clusterMounted( dataSets ) );

        SlurmUtils.monitorJobProgress( jobFutures, logger );

    }


    private ArrayList< JobFuture > submitJobsOnSlurm( String imageJ, Path jobDirectory, Path classifierPath, List< Path > dataSets )
    {

        ImageJCommandsSubmitter commandsSubmitter = new ImageJCommandsSubmitter(
                ImageJCommandsSubmitter.EXECUTION_SYSTEM_EMBL_SLURM,
                jobDirectory.toString(),
                imageJ,
                username, password );

        ArrayList< JobFuture > jobFutures = new ArrayList<>( );

        ArrayList< FinalInterval > tiles = IntervalUtils.createTiles( interval, interval, numJobs,true, null );

        for ( Path dataSet : dataSets )
        {
            for ( FinalInterval tile : tiles )
            {
                commandsSubmitter.clearCommands();
                setCommandAndParameterStrings( commandsSubmitter, dataSet, classifierPath, tile );
                jobFutures.add( commandsSubmitter.submitCommands( getApproximatelyNeededMemoryMB( tile ), numWorkers, queue ) );
            }
        }

        return jobFutures;
    }

    private int getApproximatelyNeededMemoryMB( FinalInterval tile )
    {
        long memoryB = IntervalUtils.getApproximateNeededBytes( tile, memoryFactor );
        int memoryMB = (int) ( 1.0 * memoryB / 1000000L );
        if ( memoryMB < 32000 ) memoryMB = 32000;
        return memoryMB;
    }


    private void setCommandAndParameterStrings(
            ImageJCommandsSubmitter commandsSubmitter,
            Path inputImagePath,
            Path classifierPath,
            FinalInterval tile )
    {

        String dataSetID = getSimpleString( inputImagePath.getFileName().toString() );
        if ( dataSetID.equals( "" ) )
        {
            dataSetID = "dataSet";
        }

        // TODO: put to fiji-slurm
        commandsSubmitter.addLinuxCommand( "hostname" );
        commandsSubmitter.addLinuxCommand( "lscpu" );
        commandsSubmitter.addLinuxCommand( "free -m" );
        commandsSubmitter.addLinuxCommand( "START_TIME=$SECONDS" );

        Map< String, Object > parameters = new HashMap<>();

        String intervalXYZT = "";
        for ( int d : XYZT )
        {
            intervalXYZT += tile.min( d ) + "," + tile.max( d );
            if (d != T) intervalXYZT += ",";
        }

        parameters.clear();
        parameters.put( ApplyClassifierCommand.DATASET_ID, dataSetID );

        parameters.put( IOUtils.INPUT_MODALITY, inputModality );
        parameters.put( IOUtils.INPUT_IMAGE_PATH, inputImagePath );

        parameters.put( IOUtils.INPUT_IMAGE_VSS_DIRECTORY, clusterMounted( inputImageVSSDirectory ) );
        parameters.put( IOUtils.INPUT_IMAGE_VSS_PATTERN, inputImageVSSPattern );
        parameters.put( IOUtils.INPUT_IMAGE_VSS_SCHEME, inputImageVSSScheme );
        parameters.put( IOUtils.INPUT_IMAGE_VSS_HDF5_DATA_SET_NAME, inputImageVSSHdf5DataSetName );

        parameters.put( ApplyClassifierCommand.CLASSIFICATION_INTERVAL, intervalXYZT );

        parameters.put( ApplyClassifierCommand.CLASSIFIER_FILE, classifierPath );

        parameters.put( IOUtils.OUTPUT_MODALITY, IOUtils.SAVE_AS_TIFF_SLICES );
        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, clusterMounted( outputDirectory ) );

        parameters.put( ApplyClassifierCommand.NUM_WORKERS, numWorkers );
        parameters.put( ApplyClassifierCommand.MEMORY_MB, getApproximatelyNeededMemoryMB( tile ) );

        parameters.put( ApplyClassifierCommand.QUIT_AFTER_RUN, true );


        commandsSubmitter.addIJCommandWithParameters( ApplyClassifierCommand.PLUGIN_NAME , parameters );

        commandsSubmitter.addLinuxCommand( "ELAPSED_TIME=$(($SECONDS - $START_TIME))" );

        commandsSubmitter.addLinuxCommand( "echo \"Elapsed time [s]:\"" );
        commandsSubmitter.addLinuxCommand( "echo $ELAPSED_TIME" );

    }


    public static void main(final String... args) throws Exception
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ij.command().run( ApplyClassifierAsSlurmJobsCommand.class, true );

    }
}
