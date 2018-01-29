package de.embl.cba.trainableDeepSegmentation.commands;

import ij.IJ;
import ij.ImagePlus;
import ij.measure.ResultsTable;
import inra.ijpb.measure.GeometricMeasures3D;
import net.imagej.DatasetService;
import net.imagej.ops.OpService;
import org.scijava.ItemVisibility;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.command.DynamicCommand;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import de.embl.cba.trainableDeepSegmentation.objectanalysis.ObjectAnalysis;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static de.embl.cba.trainableDeepSegmentation.commands.AnalyzeObjectsCommand.PLUGIN_NAME;


@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>EMBL-CBA>" + PLUGIN_NAME )
public class AnalyzeObjectsCommand extends DynamicCommand
{

    public static final String PLUGIN_NAME = "Analyze Objects";

    @Parameter
    public UIService uiService;

    @Parameter
    public DatasetService datasetService;

    @Parameter
    public LogService logService;

    @Parameter
    public ThreadService threadService;

    @Parameter
    public OpService opService;

    @Parameter
    public StatusService statusService;

    @Parameter( visibility = ItemVisibility.MESSAGE )
    private String message
            = "<html>"  +
            "<br>Analyse Objects<br>" +
            "...<br>";

    @Parameter (label = "Input image", required = true )
    public File inputImagePath;
    public static final String INPUT_IMAGE_PATH = "inputImagePath";

    @Parameter( label = "Lower Threshold", required = true )
    public int lowerThreshold = 1;
    public static final String LOWER_THRESHOLD = "lowerThreshold";

    @Parameter( label = "Upper Threshold", required = true )
    public int upperThreshold = 255;
    public static final String UPPER_THRESHOLD = "upperThreshold";

    @Parameter( label = "Output modality", choices = { SHOW, SAVE } )
    public String outputModality;
    public static final String OUTPUT_MODALITY = "outputModality";
    public static final String SAVE = "Save results table";
    public static final String SHOW = "Show results table";

    @Parameter( label = "Output folder", style = "directory" )
    public File outputDirectory;
    public static final String OUTPUT_DIRECTORY = "outputDirectory";

    @Parameter( label = "Quit ImageJ after running", required = false )
    public boolean quitAfterRun = false;
    public static final String QUIT_AFTER_RUN = "quitAfterRun";

    ImagePlus inputImage;

    public void run()
    {

        Map<String, Object> inputs = getInputs();

        logCommandLineCall();

        inputImage = IOUtils.loadImage( inputImagePath );

        ImagePlus labelMask = ObjectAnalysis.createLabelMaskForChannelAndFrame( inputImage, 1, 1, 1, lowerThreshold, upperThreshold );

        ResultsTable volumes = GeometricMeasures3D.volume( labelMask.getStack(), new double[]{ 1, 1, 1 } );

        if ( outputModality.equals( SAVE ) )
        {
            IOUtils.createDirectoryIfNotExists( outputDirectory.getPath() );

            volumes.save( outputDirectory + File.separator + inputImage.getTitle() + "--volumes.csv" );
        }

        if ( outputModality.equals( SHOW ) )
        {
            volumes.show( inputImage.getTitle() + "--volumes" );
        }

        if ( quitAfterRun )  IJ.run( "Quit" );

    }

    private void logCommandLineCall()
    {
        Map<String, Object> parameters = new HashMap<>( );
        parameters.put( INPUT_IMAGE_PATH, inputImagePath );
        parameters.put( LOWER_THRESHOLD, lowerThreshold );
        parameters.put( UPPER_THRESHOLD, upperThreshold );
        parameters.put( OUTPUT_MODALITY, outputModality );
        parameters.put( OUTPUT_DIRECTORY, outputDirectory );
        parameters.put( QUIT_AFTER_RUN, quitAfterRun );
        IJ.log( Commands.createImageJPluginCommandLineCall( "ImageJ", PLUGIN_NAME, parameters ) );
    }


}
