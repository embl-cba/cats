package de.embl.cba.cats.ui;

import de.embl.cba.cluster.commands.Commands;
import de.embl.cba.cats.utils.IOUtils;
import ij.IJ;
import ij.ImagePlus;
import ij.measure.ResultsTable;
import inra.ijpb.measure.GeometricMeasures3D;


import net.imagej.DatasetService;
import net.imagej.ops.OpService;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import de.embl.cba.cats.objects.ObjectSegmentation;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static de.embl.cba.cats.ui.AnalyzeObjectsCommand.PLUGIN_NAME;


@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Development>" + PLUGIN_NAME )
public class AnalyzeObjectsCommand implements Command
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

    @Parameter ( label = "Input image" )
    public File inputImageFile;
    public static final String INPUT_IMAGE_FILE = "inputImageFile";

    @Parameter( label = "Lower threshold", required = true )
    public int lowerThreshold = 1;
    public static final String LOWER_THRESHOLD = "lowerThreshold";

    @Parameter( label = "Upper threshold", required = true )
    public int upperThreshold = 255;
    public static final String UPPER_THRESHOLD = "upperThreshold";

    @Parameter( label = "Minimum number of voxels per object" )
    public int minNumVoxels = 10;
    public static final String MIN_NUM_VOXELS = "minNumVoxels";

    @Parameter( label = "Output modality", choices = { IOUtils.SHOW, IOUtils.SAVE } )
    public String outputModality;
    public static final String OUTPUT_MODALITY = "outputModality";

    @Parameter( label = "Output folder", style = "directory" )
    public File outputDirectory;
    public static final String OUTPUT_DIRECTORY = "outputDirectory";

    @Parameter( label = "Quit ImageJ after running", required = false )
    public boolean quitAfterRun = false;
    public static final String QUIT_AFTER_RUN = "quitAfterRun";

    @Parameter( label = "Dataset ID" )
    public String dataSetID = "dataSet";
    public static final String DATASET_ID = "dataSetID";


    ImagePlus inputImage;

    public void run()
    {

        logService.info( "# " + PLUGIN_NAME );
        logCommandLineCall();

        logService.info( "Loading image " + inputImageFile );
        inputImage = IOUtils.openImageWithIJOpenImage( inputImageFile );

        logService.info( "Creating label mask");
        ImagePlus labelMask = ObjectSegmentation.createLabelMaskForChannelAndFrame( inputImage, 1, 1, minNumVoxels, lowerThreshold, upperThreshold );

        ArrayList< ResultsTable > resultsTables = new ArrayList<>(  );
        ArrayList< String > resultsTableNames = new ArrayList<>(  );

        logService.info( "Measuring object geometries");

        resultsTables.add( GeometricMeasures3D.volume( labelMask.getStack(), new double[]{ 1, 1, 1 } ) );
        resultsTableNames.add( "Volume" );
        addInputImageFileAndPathName( resultsTables.get( 0 ) );

        resultsTables.add( GeometricMeasures3D.boundingBox( labelMask.getStack() ) );
        resultsTableNames.add( "BoundingBox" );

        resultsTables.add( GeometricMeasures3D.inertiaEllipsoid( labelMask.getStack(), new double[]{ 1, 1, 1 } ) );
        resultsTableNames.add( "Ellipsoid" );

        if ( outputModality.equals( IOUtils.SAVE ) )
        {
            resultsTables.add( saveLabelMask( labelMask ) );
            resultsTableNames.add( "LabelMask" );

            for ( int i = 0; i < resultsTables.size(); ++i )
            {
                saveTable( resultsTables, resultsTableNames, i );
            }
        }

        if ( outputModality.equals( IOUtils.SHOW ) )
        {
            for ( int i = 0; i < resultsTables.size(); ++i )
            {
                resultsTables.get( i ).show( resultsTableNames.get( i ) );
            }

            labelMask.show();
        }

        if ( quitAfterRun ) Commands.quitImageJ( logService );

    }

    private ResultsTable saveLabelMask( ImagePlus labelMask )
    {
        String labelMaskDirectory = outputDirectory.getAbsolutePath();
        String labelMaskFileName = inputImage.getTitle() + "--LabelMask--AnalyzeObjects.tif";;
        String labelMaskPath = labelMaskDirectory + File.separator + labelMaskFileName;

        logService.info( "Saving label mask: " + labelMaskPath );
        IJ.saveAsTiff( labelMask, labelMaskPath );

        ij.measure.ResultsTable resultsTable = new ij.measure.ResultsTable();
        resultsTable.incrementCounter();

        resultsTable.addValue( "FileName_AnalyzeObjects_LabelMask_IMG", labelMaskFileName  );
        resultsTable.addValue( "PathName_AnalyzeObjects_LabelMask_IMG",  labelMaskDirectory );

        return ( resultsTable );
    }

    private void saveTable( ArrayList< ResultsTable > resultsTables, ArrayList< String > resultsTableNames, int i )
    {
        String tablePath = outputDirectory + File.separator + inputImage.getTitle() + "--" + resultsTableNames.get( i ) + "--AnalyzeObjects.csv";
        logService.info( "Saving results table: " + tablePath );
        IOUtils.createDirectoryIfNotExists( outputDirectory.getPath() );
        addDataSetId( resultsTables.get( i ) );
        resultsTables.get( i ).save( tablePath );
    }

    private void addInputImageFileAndPathName( ResultsTable resultsTable )
    {
        if ( resultsTable.getCounter() == 0 )
        {
            resultsTable.incrementCounter();
        }

        resultsTable.addValue( "FileName_AnalyzeObjects_InputImage_IMG", inputImageFile.getName() );
        resultsTable.addValue( "PathName_AnalyzeObjects_InputImage_IMG", inputImageFile.getParent() );

        for ( int i = 0; i < resultsTable.size(); ++i )
        {
            resultsTable.setValue("FileName_AnalyzeObjects_InputImage_IMG", i, inputImageFile.getName()  );
            resultsTable.setValue("PathName_AnalyzeObjects_InputImage_IMG", i, inputImageFile.getParent()  );
        }
    }

    private void addDataSetId( ResultsTable resultsTable )
    {
        if ( resultsTable.getCounter() == 0 )
        {
            resultsTable.incrementCounter();
        }

        resultsTable.addValue( "DataSetID_FACT", dataSetID );

        for ( int i = 0; i < resultsTable.size(); ++i )
        {
            resultsTable.setValue( "DataSetID_FACT", i, dataSetID );
        }
    }


    private void logCommandLineCall()
    {
        Map<String, Object> parameters = new HashMap<>( );
        parameters.put( INPUT_IMAGE_FILE, inputImageFile );
        parameters.put( LOWER_THRESHOLD, lowerThreshold );
        parameters.put( UPPER_THRESHOLD, upperThreshold );
        parameters.put( OUTPUT_MODALITY, outputModality );
        parameters.put( OUTPUT_DIRECTORY, outputDirectory );
        parameters.put( QUIT_AFTER_RUN, quitAfterRun );
        IJ.log( Commands.createImageJPluginCommandLineCall( "ImageJ", PLUGIN_NAME, parameters ) );
    }


}
