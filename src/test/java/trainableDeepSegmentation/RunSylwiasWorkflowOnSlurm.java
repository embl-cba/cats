package trainableDeepSegmentation;

import de.embl.cba.cluster.plugins.SlurmCommandRunnerPlugin;
import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.TextWidget;
import trainableDeepSegmentation.commands.AnalyzeObjectsCommand;
import trainableDeepSegmentation.commands.ApplyClassifierCommand;
import trainableDeepSegmentation.commands.Commands;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


@Plugin(type = Command.class, menuPath = "Plugins>Institutes>EMBL>3D-Transmission-Slurm" )
public class RunSylwiasWorkflowOnSlurm implements Command
{

    @Parameter(label = "Username" )
    private String username;

    @Parameter(label = "Password", style = TextWidget.PASSWORD_STYLE, persist = false )
    private String password;

    @Parameter (label = "Directory", style = "directory" )
    public File inputDirectory;

    @Parameter (label = "Classifier" )
    public File classifierFile;

    public void run()
    {

        Path classifierPath = Paths.get( classifierFile.getAbsolutePath() );

        // ensure remote existence of imagej with all dependencies

        ArrayList< Path > dataSetPatterns = getDataSetPatterns( inputDirectory );

        for ( Path dataSetPattern : dataSetPatterns )
        {
            ArrayList< String > commands = createCommands( dataSetPattern, classifierPath );
            Map< String, Object > parameters = configureSlurmCommandRunnerParameters( commands, username, password );
            runCommandOnCluster( parameters );
        }


    }

    private ArrayList< Path > getDataSetPatterns( File inputDirectory )
    {
        ArrayList< Path > patterns =  new ArrayList<>( );
        Path path = Paths.get( TestingUtils.TEST_RESOURCES + "/image-sequence/" + ".*--W00016--P00004--.*" );
        patterns.add( path );
        return patterns;
    }

    private static void runCommandOnCluster( Map< String, Object > parameters )
    {
        ImageJ ij = new ImageJ();
        ij.command().run( SlurmCommandRunnerPlugin.class, false, parameters );
    }

    private static Map< String, Object > configureSlurmCommandRunnerParameters( ArrayList< String > commands,
                                                                                String username,
                                                                                String password )
    {
        Map< String, Object > parameters = new HashMap<>(  );
        parameters.put( "username", username );
        parameters.put( "password", password );
        parameters.put( "commands", commands );
        return parameters;
    }

    private static ArrayList< String > createCommands( Path inputImagePath, Path classifierPath )
    {

        String outputDirectory = inputImagePath.getParent() + "--analysis" + "/" + inputImagePath.getFileName();

        Map< String, Object > parameters = new HashMap<>();

        ArrayList< String > commands = new ArrayList<>( );

        parameters.clear();
        parameters.put( ApplyClassifierCommand.INPUT_IMAGE_PATH, inputImagePath );
        parameters.put( ApplyClassifierCommand.CLASSIFIER_PATH, classifierPath );
        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( ApplyClassifierCommand.OUTPUT_MODALITY, ApplyClassifierCommand.SAVE_AS_TIFF_FILES );
        commands.add( Commands.createCommand( "ApplyClassifierCommand" , parameters ) );

        parameters.clear();
        parameters.put( AnalyzeObjectsCommand.INPUT_IMAGE_PATH, new File( outputDirectory + "/foreground.tif" ) );
        parameters.put( AnalyzeObjectsCommand.LOWER_THRESHOLD, 1 );
        parameters.put( AnalyzeObjectsCommand.UPPER_THRESHOLD, 255 );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_MODALITY, AnalyzeObjectsCommand.SHOW );
        parameters.put( AnalyzeObjectsCommand.QUIT_AFTER_RUN, true );
        commands.add( Commands.createCommand( "AnalyzeObjectsCommand" , parameters ) );

        return commands;
    }


    public static void main(final String... args) throws Exception
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ij.command().run( RunSylwiasWorkflowOnSlurm.class, true );

    }
}
