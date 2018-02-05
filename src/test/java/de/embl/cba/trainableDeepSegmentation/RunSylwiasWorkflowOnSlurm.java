package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.cluster.ImageJCommandsSubmitter;
import de.embl.cba.cluster.JobFuture;
import de.embl.cba.cluster.logger.Logger;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import ij.ImagePlus;
import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.TextWidget;
import de.embl.cba.trainableDeepSegmentation.commands.AnalyzeObjectsCommand;
import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierCommand;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import embl.cba.fileutils.FileRegMatcher;

import static de.embl.cba.trainableDeepSegmentation.utils.IOUtils.asEMBLClusterMounted;


// /g/almf/software/Fiji.app/ImageJ-linux64 --run "Apply Classifier" "quitAfterRun='true',inputImagePath='/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/small-test-image-sequences/.*--W00016--P00004--.*',classifierPath='/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/01.classifier',outputDirectory='/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/small-test-image-sequences--analysis/.*--W00016--P00004--.*',outputModality='Save class probabilities as tiff files'"

@Plugin(type = Command.class, menuPath = "Plugins>Development>EMBL-CBA>Sylwia Slurm" )
public class RunSylwiasWorkflowOnSlurm implements Command
{

    @Parameter(label = "Username" )
    private String username;

    @Parameter(label = "Password", style = TextWidget.PASSWORD_STYLE, persist = false )
    private String password;

    @Parameter (label = "Data directory", style = "directory", required = false )
    public File inputDirectory;

    @Parameter (label = "Job directory", style = "directory", required = false )
    public File jobDirectory;

    @Parameter (label = "Classifier", required = false  )
    public File classifierFile;

    static String masterRegExp="(?<treatment>.+)--W(?<well>\\d+)--P(?<position>\\d+)--Z(?<slice>\\d+)--T(?<timePoint>\\d+)--(?<channel>.+)\\.tif";

    static String[] datasetGroups={"treatment","well","position"};

    public void run()
    {
        if ( inputDirectory == null )
        {
            //inputDirectory = new File( "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/small-image-sequences" );
            inputDirectory = new File( "/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/small-test-image-sequences/" );
        }

        if ( jobDirectory == null )
        {
            jobDirectory = new File( "/g/cba/cluster/sylwia" );
        }


        Path classifierPath = Paths.get( "/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/transmission-cells-3d.classifier" );

        List< Path > dataSetPatterns = getDataSetPatterns( inputDirectory.toString() );

        List< Path > clusterMountedDataSetPatterns = asEMBLClusterMounted( dataSetPatterns );

        String imageJ = ImageJCommandsSubmitter.IMAGEJ_EXECTUABLE_ALMF_CLUSTER_XVFB;

        ArrayList< JobFuture > jobFutures = submitJobsOnSlurm(
                imageJ,
                asEMBLClusterMounted( jobDirectory.toPath() ) ,
                asEMBLClusterMounted( classifierPath ),
                asEMBLClusterMounted( clusterMountedDataSetPatterns ) );

        monitorJobProgress( jobFutures );

    }

    public static List< Path > getDataSetPatterns( String directory )
    {
        System.out.println("Start test");

        FileRegMatcher regMatcher = new FileRegMatcher();

        regMatcher.setParameters( masterRegExp, datasetGroups );

        regMatcher.matchFiles( directory );

        List< File > filePatterns = regMatcher.getMatchedFilesList();

        List< Path > filePatternPaths = new ArrayList<>();

        for ( File f : filePatterns )
        {
            String pattern = f.getAbsolutePath();
            //pattern = pattern.replaceAll( "\\(.*?\\)" , ".*" );
            //System.out.println( f.getAbsolutePath() );
            //System.out.println( pattern );
            filePatternPaths.add( Paths.get( pattern) );
        }

        return filePatternPaths;
    }

    private void monitorJobProgress( ArrayList< JobFuture > jobFutures )
    {

        for ( JobFuture jobFuture : jobFutures )
        {
            try
            {
                HashMap< String, Object > output = jobFuture.get();
                Logger.log( (String) output.get( JobFuture.STD_OUT ) );
            }
            catch ( InterruptedException e )
            {
                e.printStackTrace();
            }
            catch ( ExecutionException e )
            {
                e.printStackTrace();
            }

        }

        Logger.log( "All jobs finished." );
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
            jobFutures.add( commandsSubmitter.submitCommands() );
        }

        return jobFutures;
    }


    /*
    private ArrayList< Path > getDataSetPatterns( File inputDirectory )
    {
        ArrayList< Path > patterns =  new ArrayList<>( );

        patterns.add( Paths.get( inputDirectory.getAbsolutePath() + "/" + ".*--W00016--P00003--.*" ) );
        patterns.add( Paths.get( inputDirectory.getAbsolutePath() + "/" + ".*--W00016--P00004--.*" ) );

        return patterns;
    }
    */

    private static void setCommandAndParameterStrings( ImageJCommandsSubmitter commandsSubmitter, Path inputImagePath, Path classifierPath )
    {

        String dataSetName = inputImagePath.getFileName().toString().replace( ".*", "" ).trim();
        String trimmedDataSetName = dataSetName.replaceAll("\\(.*?\\)" , "" );
        trimmedDataSetName = trimmedDataSetName.replace("\\" , "" );
        trimmedDataSetName = trimmedDataSetName.replace(".tif" , "" );
        String outputDirectory = inputImagePath.getParent() + "--analysis" + "/" + "DataSet--" + trimmedDataSetName;

        Map< String, Object > parameters = new HashMap<>();

        parameters.clear();
        parameters.put( ApplyClassifierCommand.INPUT_IMAGE_PATH, inputImagePath );
        parameters.put( ApplyClassifierCommand.CLASSIFIER_PATH, classifierPath );
        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( ApplyClassifierCommand.OUTPUT_MODALITY, ApplyClassifierCommand.SAVE_AS_TIFF_FILES );
        parameters.put( AnalyzeObjectsCommand.QUIT_AFTER_RUN, true );
        commandsSubmitter.addCommand( ApplyClassifierCommand.PLUGIN_NAME , parameters );

        parameters.clear();
        parameters.put( AnalyzeObjectsCommand.INPUT_IMAGE_PATH, new File( outputDirectory + "/foreground.tif" ) );
        parameters.put( AnalyzeObjectsCommand.LOWER_THRESHOLD, 1 );
        parameters.put( AnalyzeObjectsCommand.UPPER_THRESHOLD, 255 );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_DIRECTORY, new File( outputDirectory ) );
        parameters.put( AnalyzeObjectsCommand.OUTPUT_MODALITY, AnalyzeObjectsCommand.SAVE );
        parameters.put( AnalyzeObjectsCommand.QUIT_AFTER_RUN, true );
        commandsSubmitter.addCommand( AnalyzeObjectsCommand.PLUGIN_NAME , parameters );
    }


    public static void main(final String... args) throws Exception
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ij.command().run( RunSylwiasWorkflowOnSlurm.class, true );

    }
}
