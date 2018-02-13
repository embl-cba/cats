package de.embl.cba.trainableDeepSegmentation.commands;

import de.embl.cba.cluster.ImageJCommandsSubmitter;
import de.embl.cba.cluster.JobFuture;
import de.embl.cba.cluster.SlurmQueue;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import de.embl.cba.trainableDeepSegmentation.utils.SlurmUtils;
import net.imagej.ImageJ;
import net.imglib2.FinalInterval;
import org.scijava.command.Command;
import org.scijava.command.DynamicCommand;
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
import static de.embl.cba.trainableDeepSegmentation.utils.Utils.getSimpleString;


@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Development>Apply Classifier Test" )
public class ApplyClassifierAsSlurmJobsCommand extends DynamicCommand
{

    @Parameter(label = "Username" )
    private String username = "tischer";

    @Parameter(label = "Password", style = TextWidget.PASSWORD_STYLE, persist = false )
    private String password;

    @Parameter ( label = "Queue", choices = { SlurmQueue.DEFAULT_QUEUE, SlurmQueue.ONE_DAY_QUEUE, SlurmQueue.ONE_WEEK_QUEUE, SlurmQueue.BIGMEM_QUEUE, SlurmQueue.GPU_QUEUE } )
    public String queue = SlurmQueue.DEFAULT_QUEUE;

    @Parameter( label = "Memory per job [MB]" )
    public int memoryMB = 32000;
    public static final String MEMORY = "memoryMB";

    @Parameter( label = "Number of CPUs per job" )
    public int numWorkers;
    public static final String WORKERS = "threads";

    @Parameter()
    public File outputDirectory;

    @Parameter()
    public File classifierFile;

    @Parameter()
    public String inputModality;

    @Parameter()
    public String inputImagePath;

    public String imageJ = ImageJCommandsSubmitter.IMAGEJ_EXECTUABLE_ALMF_CLUSTER_XVFB;

    @Parameter()
    public FinalInterval interval;
    public static final String INTERVAL = "interval";

    public boolean quitAfterRun = true;

    FinalInterval inputImageIntervalXYZT;

    public void run()
    {

        Path jobDirectory = Paths.get( "/g/cba/cluster/" + username );

        List< Path > dataSets = new ArrayList<>();
        dataSets.add( Paths.get( inputImagePath ) );

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

        // TODO: tile loop

        for ( Path dataSet : dataSets )
        {
            commandsSubmitter.clearCommands();
            setCommandAndParameterStrings( commandsSubmitter, dataSet, classifierPath );
            jobFutures.add( commandsSubmitter.submitCommands( memoryMB, numWorkers, queue ) );
        }

        return jobFutures;
    }


    private void setCommandAndParameterStrings( ImageJCommandsSubmitter commandsSubmitter, Path inputImagePath, Path classifierPath )
    {

        String outputDirectory = inputImagePath.getParent() + "--classification" + "/"; //+ "DataSet--" + simpleDataSetName;
        String dataSetID = getSimpleString( inputImagePath.getFileName().toString() );

        // TODO: put to fiji-slurm
        commandsSubmitter.addLinuxCommand( "hostname" );
        commandsSubmitter.addLinuxCommand( "lscpu" );
        commandsSubmitter.addLinuxCommand( "free -m" );
        commandsSubmitter.addLinuxCommand( "START_TIME=$SECONDS" );

        Map< String, Object > parameters = new HashMap<>();

        parameters.clear();
        parameters.put( ApplyClassifierCommand.DATASET_ID, dataSetID );
        parameters.put( IOUtils.INPUT_IMAGE_PATH, inputImagePath );
        parameters.put( ApplyClassifierCommand.CLASSIFIER_FILE, classifierPath );
        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( IOUtils.OUTPUT_MODALITY, IOUtils.SAVE_AS_TIFF_SLICES );
        parameters.put( ApplyClassifierCommand.MEMORY, memoryMB );
        parameters.put( ApplyClassifierCommand.THREADS, numWorkers );
        parameters.put( ApplyClassifierCommand.INPUT_IMAGE_INTERVAL, inputImageIntervalXYZT );
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
