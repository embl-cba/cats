package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.cluster.SlurmCommandSubmitter;
import de.embl.cba.cluster.SlurmJobFuture;
import de.embl.cba.cluster.SlurmJobSubmitter;
import de.embl.cba.cluster.job.ImageJCommandSlurmJob;
import de.embl.cba.cluster.logger.Logger;
import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.TextWidget;
import de.embl.cba.trainableDeepSegmentation.commands.AnalyzeObjectsCommand;
import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierCommand;
import de.embl.cba.trainableDeepSegmentation.commands.Commands;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


@Plugin(type = Command.class, menuPath = "Plugins>Development>EMBL-CBA>Slurm Classification" )
public class RunSylwiasWorkflowOnSlurm implements Command
{

    @Parameter(label = "Username" )
    private String username;

    @Parameter(label = "Password", style = TextWidget.PASSWORD_STYLE, persist = false )
    private String password;

    @Parameter (label = "Directory", style = "directory", required = false )
    public File inputDirectory;

    @Parameter (label = "Classifier", required = false  )
    public File classifierFile;

    @Parameter (label = "ImageJ", required = false  )
    public String imageJ;

    public void run()
    {

        String imageJ = ImageJCommandSlurmJob.ALMF_CLUSTER_IMAGEJ_XVFB;

        Path classifierPath = Paths.get( "/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/01.classifier" );

        ArrayList< Path > dataSetPatterns = getDataSetPatterns( inputDirectory );

        ArrayList< SlurmJobFuture > jobFutures = submitJobs( imageJ, classifierPath, dataSetPatterns );

        for ( SlurmJobFuture jobFuture : jobFutures )
        {
            Logger.log( jobFuture.getStatus() );
            if ( jobFuture.isDone() )
            {
                Logger.log( jobFuture.getOutput() );
                Logger.log( jobFuture.getError() );
                jobFutures.remove( jobFuture );
            }
        }

        Logger.log( "All jobs finished." );

    }

    private ArrayList< SlurmJobFuture > submitJobs( String imageJ, Path classifierPath, ArrayList< Path > dataSetPatterns )
    {
        ArrayList< SlurmJobFuture > jobFutures = new ArrayList<>( );
        for ( Path dataSetPattern : dataSetPatterns )
        {
            ArrayList< String > commands = createCommands( imageJ, dataSetPattern, classifierPath );
            jobFutures.add( SlurmJobSubmitter.submit( commands, username, password ) );
        }
        return jobFutures;
    }

    private ArrayList< Path > getDataSetPatterns( File inputDirectory )
    {
        ArrayList< Path > patterns =  new ArrayList<>( );
        String directory = "/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/4x_2p7mm_100umsteps_trans_001/data";

        patterns.add( Paths.get( directory + "/" + ".*--W00016--P00003--.*" ) );
        patterns.add( Paths.get( directory + "/" + ".*--W00016--P00004--.*" ) );

        return patterns;
    }

    private static ArrayList< String > createCommands( String imageJ, Path inputImagePath, Path classifierPath )
    {

        String outputDirectory = inputImagePath.getParent() + "--analysis" + "/" + inputImagePath.getFileName();

        Map< String, Object > parameters = new HashMap<>();

        ArrayList< String > commands = new ArrayList<>( );

        parameters.clear();
        parameters.put( ApplyClassifierCommand.INPUT_IMAGE_PATH, inputImagePath );
        parameters.put( ApplyClassifierCommand.CLASSIFIER_PATH, classifierPath );
        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( ApplyClassifierCommand.OUTPUT_MODALITY, ApplyClassifierCommand.SAVE_AS_TIFF_FILES );
        parameters.put( AnalyzeObjectsCommand.QUIT_AFTER_RUN, true );
        commands.add( Commands.createImageJPluginCommandLineCall( imageJ,"ApplyClassifierCommand" , parameters ) );

        parameters.clear();
        parameters.put( AnalyzeObjectsCommand.INPUT_IMAGE_PATH, new File( outputDirectory + "/foreground.tif" ) );
        parameters.put( AnalyzeObjectsCommand.LOWER_THRESHOLD, 1 );
        parameters.put( AnalyzeObjectsCommand.UPPER_THRESHOLD, 255 );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_MODALITY, AnalyzeObjectsCommand.SHOW );
        parameters.put( AnalyzeObjectsCommand.QUIT_AFTER_RUN, true );
        commands.add( Commands.createImageJPluginCommandLineCall( imageJ,"AnalyzeObjectsCommand" , parameters ) );

        return commands;
    }


    public static void main(final String... args) throws Exception
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ij.command().run( RunSylwiasWorkflowOnSlurm.class, true );

    }
}
