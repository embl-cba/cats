package de.embl.cba.trainableDeepSegmentation.commands;

import de.embl.cba.cluster.ImageJCommandsSubmitter;
import de.embl.cba.cluster.JobFuture;
import de.embl.cba.cluster.SlurmQueue;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils;
import de.embl.cba.trainableDeepSegmentation.utils.SlurmUtils;
import de.embl.cba.utils.fileutils.PathMapper;
import net.imagej.ImageJ;
import net.imglib2.FinalInterval;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.TextWidget;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static de.embl.cba.trainableDeepSegmentation.DeepSegmentation.logger;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.T;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.XYZT;
import static de.embl.cba.trainableDeepSegmentation.utils.Utils.getSimpleString;


@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Development>Apply Classifier On Slurm" )
public class ApplyClassifierOnSlurmCommand implements Command
{

    @Parameter( label = "Username" )
    private String userName = "tischer";
    public static String USER_NAME = "userName";

    @Parameter( label = "Password", style = TextWidget.PASSWORD_STYLE, persist = false )
    private String password;
    public static String PASSWORD = "password";

    @Parameter( label = "Queue", choices = { SlurmQueue.DEFAULT_QUEUE, SlurmQueue.ONE_DAY_QUEUE, SlurmQueue.ONE_WEEK_QUEUE, SlurmQueue.BIGMEM_QUEUE, SlurmQueue.GPU_QUEUE } )
    public String queue = SlurmQueue.DEFAULT_QUEUE;
    public static String QUEUE = "queue";

    @Parameter( label = "Number of jobs" )
    public int numJobs = 20;
    public static String NUM_JOBS = "numJobs";

    @Parameter( label = "Number of CPUs per job" )
    public int numWorkers;
    public static final String NUM_WORKERS = "numWorkers";

    @Parameter( label = "Classifier file (must be cluster accessible)" )
    public File classifierFile;
    public static final String CLASSIFIER_FILE = "classifierFile";

    @Parameter()
    public String inputModality;

    @Parameter()
    public File inputImageFile;

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

        String jobDirectory = "/g/cba/cluster/" + userName;

        List< Path > dataSets = new ArrayList<>();
        dataSets.add( inputImageFile.toPath() );

        ArrayList< JobFuture > jobFutures = submitJobsOnSlurm( imageJ, jobDirectory, classifierFile.toPath(), dataSets );

        SlurmUtils.monitorJobProgress( jobFutures, logger );

    }


    private ArrayList< JobFuture > submitJobsOnSlurm( String imageJ, String jobDirectory, Path classifierPath, List< Path > dataSets )
    {

        ImageJCommandsSubmitter commandsSubmitter = new ImageJCommandsSubmitter(
                ImageJCommandsSubmitter.EXECUTION_SYSTEM_EMBL_SLURM,
                jobDirectory ,
                imageJ,
                userName, password );

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

        String dataSetID = getDataSetID( inputImagePath );


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
        parameters.put( IOUtils.INPUT_IMAGE_FILE, PathMapper.asEMBLClusterMounted( inputImagePath ) );

        parameters.put( IOUtils.INPUT_IMAGE_VSS_DIRECTORY, PathMapper.asEMBLClusterMounted( inputImageVSSDirectory ));
        parameters.put( IOUtils.INPUT_IMAGE_VSS_PATTERN, inputImageVSSPattern );
        parameters.put( IOUtils.INPUT_IMAGE_VSS_SCHEME, inputImageVSSScheme );
        parameters.put( IOUtils.INPUT_IMAGE_VSS_HDF5_DATA_SET_NAME, inputImageVSSHdf5DataSetName );

        parameters.put( ApplyClassifierCommand.CLASSIFICATION_INTERVAL, intervalXYZT );

        parameters.put( ApplyClassifierCommand.CLASSIFIER_FILE, PathMapper.asEMBLClusterMounted( classifierPath ) );

        parameters.put( IOUtils.OUTPUT_MODALITY, IOUtils.SAVE_AS_TIFF_SLICES );
        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, PathMapper.asEMBLClusterMounted( outputDirectory ) );

        parameters.put( ApplyClassifierCommand.NUM_WORKERS, numWorkers );
        parameters.put( ApplyClassifierCommand.MEMORY_MB, getApproximatelyNeededMemoryMB( tile ) );

        parameters.put( ApplyClassifierCommand.SAVE_RESULTS_TABLE, false );
        parameters.put( ApplyClassifierCommand.QUIT_AFTER_RUN, true );

        commandsSubmitter.addIJCommandWithParameters( ApplyClassifierCommand.PLUGIN_NAME , parameters );


    }

    private String getDataSetID( Path inputImagePath )
    {
        String dataSetID = getSimpleString( inputImagePath.getFileName().toString()  );

        if ( dataSetID.equals( "" ) )
        {
            dataSetID = "dataSet";
        }
        return dataSetID;
    }


    public static void main(final String... args) throws Exception
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ij.command().run( ApplyClassifierOnSlurmCommand.class, true );

    }
}
