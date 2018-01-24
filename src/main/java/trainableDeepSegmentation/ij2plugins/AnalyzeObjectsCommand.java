package trainableDeepSegmentation.ij2plugins;

import edu.mines.jtk.util.ArrayMath;
import ij.IJ;
import ij.ImagePlus;
import ij.measure.ResultsTable;
import ij.plugin.Duplicator;
import inra.ijpb.binary.BinaryImages;
import inra.ijpb.measure.GeometricMeasures3D;
import inra.ijpb.morphology.AttributeFiltering;
import inra.ijpb.segment.Threshold;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import org.scijava.ItemVisibility;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import trainableDeepSegmentation.WekaSegmentation;

import java.io.File;


@Plugin(type = Command.class,
        menuPath = "Plugins>Segmentation>EMBL-CBA>Analyze Objects" )
public class AnalyzeObjectsCommand implements Command
{

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
    public File inputImageFile;
    public static final String INPUT_IMAGE_FILE = "inputImagePath";

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
    public File outputFolder;
    public static final String OUTPUT_FOLDER = "outputDirectory";

    ImagePlus inputImage;


    public void run()
    {

        IJ.log( "Command:" );

        String command = "";
        command += "AnalyzeObjectsCommand";
        command += " \"";
        command += INPUT_IMAGE_FILE + "=" + inputImageFile;
        command += "," + LOWER_THRESHOLD + "=" + lowerThreshold;
        command += "\" ";

        IJ.log( command );


        inputImage = IOUtils.loadImage( inputImageFile );

        ImagePlus labelMask = ObjectSegmentationUtils.createLabelMaskForChannelAndFrame( inputImage, 1, 1, 1, lowerThreshold, upperThreshold );

        ResultsTable volumes = GeometricMeasures3D.volume( labelMask.getStack(), new double[]{ 1, 1, 1 } );

        if ( outputModality.equals( SAVE ) )
        {
            IOUtils.createDirectoryIfNotExists( outputFolder.getPath() );

            volumes.save( outputFolder + File.separator + inputImage.getTitle() + "--volumes.csv" );
        }

        if ( outputModality.equals( SHOW ) )
        {
            volumes.show( inputImage.getTitle() + "--volumes" );
        }

    }


}
