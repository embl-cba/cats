package de.embl.cba.cats.ui;

/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */


import de.embl.cba.cats.CATS;
import de.embl.cba.cats.results.ResultExportSettings;
import de.embl.cba.cats.utils.IOUtils;
import de.embl.cba.cats.utils.StringUtils;
import de.embl.cba.cluster.commands.Commands;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.DatasetService;
import net.imagej.legacy.IJ1Helper;
import net.imagej.legacy.LegacyService;
import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.type.numeric.RealType;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static de.embl.cba.cats.ui.ApplyClassifierAdvancedCommand.PLUGIN_NAME;
import static de.embl.cba.cats.utils.IntervalUtils.XYZT;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>CATS>"+PLUGIN_NAME )
public class ApplyClassifierAdvancedCommand<T extends RealType<T>> implements Command
{
    public static final String PLUGIN_NAME = "Apply Classifier (Advanced)";

    @Parameter
    public LegacyService legacyService;

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

    @Parameter( label = "Input modality", choices = {
            IOUtils.OPEN_USING_IMAGEJ1,
            IOUtils.OPEN_USING_IMAGEJ1_IMAGE_SEQUENCE,
            IOUtils.OPEN_USING_IMAGE_J1_VIRTUAL,
            IOUtils.OPEN_USING_LAZY_LOADING_TOOLS } , required = true )

    public String inputModality;

    @Parameter ( label = "Input image path" )
    public File inputImageFile;

    @Parameter (label = "Classification interval (zero-based, leave empty to classify whole data set) [x0,x1,y0,y1,z0,z1,t0,t1]", required = false )
    public String classificationIntervalXYZT = WHOLE_IMAGE;
    public static final String CLASSIFICATION_INTERVAL = "classificationIntervalXYZT";
    public static final String WHOLE_IMAGE = "Whole image";

    @Parameter (label = "Classifier" )
    public File classifierFile;
    public static final String CLASSIFIER_FILE = "classifierFile";

    @Parameter( label = "Output modality", choices = {
            IOUtils.SHOW_AS_ONE_IMAGE,
            IOUtils.SAVE_AS_TIFF_STACKS,
            IOUtils.SAVE_AS_IMARIS } , required = true )

    public String outputModality;
    public static final String OUTPUT_MODALITY = "outputModality";

    @Parameter( label = "Output folder", style = "directory" )
    public File outputDirectory;
    public static final String OUTPUT_DIRECTORY = "outputDirectory";

    @Parameter( label = "Quit ImageJ after running", required = false )
    public boolean quitAfterRun = false;
    public static final String QUIT_AFTER_RUN = "quitAfterRun";

    @Parameter( label = "Memory [MB]" )
    public long memoryMB;
    public static final String MEMORY_MB = "memoryMB";

    @Parameter( label = "Threads" )
    public int numWorkers;
    public static final String NUM_WORKERS = "numWorkers";

    @Parameter( label = "Dataset ID" )
    public String dataSetID = "dataSet";
    public static final String DATASET_ID = "dataSetID";

    @Parameter( required = false )
    public String inputImageVSSDirectory;

    @Parameter( required = false )
    public String inputImageVSSScheme;

    @Parameter( required = false )
    public String inputImageVSSPattern;

    @Parameter( required = false )
    public String inputImageVSSHdf5DataSetName;

    @Parameter( required = false )
    public boolean saveResultsTable = false;
    public static final String SAVE_RESULTS_TABLE = "saveResultsTable";

    ImagePlus inputImage;
    CATS cats;
    String outputFileType;

    // /Applications/Fiji.app/Contents/MacOS/ImageJ-macosx --run "Apply Classifier" "quitAfterRun='true',inputImageFile='/Users/tischer/Documents/fiji-plugin-cats/src/test/resources/image-sequence/.*--W00016--P00003--.*',classifierFile='/Users/tischer/Documents/fiji-plugin-cats/src/test/resources/transmission-cells-3d.classifier',outputModality='Save de.embl.cba.bigdataprocessor.Point3Dobabilities as tiff files',outputDirectory='/Users/tischer/Documents/fiji-plugin-cats/src/test/resources/image-sequence--classified'"

    // xvfb-run -a /g/almf/software/Fiji.app/ImageJ-linux64 --run "Apply Classifier" "quitAfterRun='true',inputImageFile='/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/small-test-image-sequences/.*--W00016--P00004--.*',classifierFile='/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/transmission-cells-3d.classifier',outputDirectory='/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/small-test-image-sequences--classified/DataSet--W00016--P00004--',outputModality='Save class probabilities as tiff files'"

    public void run()
    {
        logService.info( "# " + PLUGIN_NAME );

        logCommandLineCall();

        logMultipleInstanceListenerState();

        setInputImage();

        applyClassifier();

        manageProbabilitiesOutput();

        saveResultsTable();

        if ( quitAfterRun )  if ( quitAfterRun ) Commands.quitImageJ( logService );

    }

    private void logMultipleInstanceListenerState()
    {
        final IJ1Helper helper = legacyService.getIJ1Helper();
        logService.info( "isRMIEnabled: " + helper.isRMIEnabled() );
    }

    private void setInputImage()
    {
        logService.info( "Loading: " + inputImageFile );

        if ( inputModality.equals( IOUtils.OPEN_USING_IMAGEJ1 ) )
        {
            inputImage = IOUtils.openImageWithIJOpenImage( inputImageFile );
        }
        else if ( inputModality.equals( IOUtils.OPEN_USING_IMAGE_J1_VIRTUAL ) )
        {
            inputImage = IOUtils.openImageWithIJOpenVirtualImage( inputImageFile );
        }
        else if ( inputModality.equals( IOUtils.OPEN_USING_IMAGEJ1_IMAGE_SEQUENCE ) )
        {
            inputImage = IOUtils.openImageWithIJImportImageSequence( inputImageFile );
        }
        else if ( inputModality.equals( IOUtils.OPEN_USING_LAZY_LOADING_TOOLS ) )
        {
            logService.info( "Opening with Lazy Loading Tools" );
            logService.info( "Directory: " + inputImageVSSDirectory );
            logService.info( "Naming scheme: " + inputImageVSSScheme );
            logService.info( "Exclusion: " + inputImageVSSPattern );
            logService.info( "Hdf5 data set name: " + inputImageVSSHdf5DataSetName );
            inputImage = IOUtils.openImageLazy( inputImageVSSDirectory, inputImageVSSScheme, inputImageVSSPattern, inputImageVSSHdf5DataSetName );
        }

        logService.info( "Loaded image: " + inputImage.getTitle() );
        logService.info( "Image dimensions: X = " + inputImage.getWidth() + "; Y = " + inputImage.getHeight() );

        inputImage.setTitle( dataSetID );
    }

    private void manageProbabilitiesOutput()
    {
        if ( outputModality.equals( IOUtils.STREAM_TO_RESULT_IMAGE_DISK ) )
            logService.info( "Results have been saved into the specified disk resident image folder." );
        else
            saveOutputImages();
    }

    private void saveOutputImages()
    {
        logService.info( "Saving classification output images..." );

        if ( outputModality.equals( IOUtils.SHOW_AS_ONE_IMAGE ) )
            cats.getResultImage().getWholeImageCopy().show();
        else
            saveAsFiles();

        logService.info( "Saving classification output images: done!" );

    }

    private void saveResultsTable()
    {
        if ( saveResultsTable )
        {
            logService.info( "Saving results table..." );

            ij.measure.ResultsTable resultsTable = new ij.measure.ResultsTable();
            resultsTable.incrementCounter();

            resultsTable.addValue( "DataSetID_FACT", dataSetID );

            resultsTable.addValue( "FileName_ApplyClassifier_InputImage_IMGR", inputImageFile.getName() );
            resultsTable.addValue( "PathName_ApplyClassifier_InputImage_IMGR", inputImageFile.getParent() );

            for ( String className : cats.getClassNames() )
            {
                resultsTable.addValue( "FileName_ApplyClassifier_" + className + "_IMG",
                        dataSetID + "--" + className + outputFileType );
                resultsTable.addValue( "PathName_ApplyClassifier_" + className + "_IMG",
                        outputDirectory.getPath() );
            }

            logService.info( "Saving results table: " + outputDirectory.getPath() + "/" + dataSetID + "--ApplyClassifier.csv" );
            resultsTable.save( outputDirectory.getPath() + "/" + dataSetID + "--ApplyClassifier.csv" );
            logService.info( "Saving results table: done!" );

        }
    }


    private void saveAsFiles()
    {

        ResultExportSettings resultExportSettings = new ResultExportSettings();
        resultExportSettings.directory = outputDirectory.getAbsolutePath();
        resultExportSettings.exportNamesPrefix = dataSetID + "--";
        resultExportSettings.classNames = cats.getClassNames();
        resultExportSettings.saveRawData = false;
        resultExportSettings.inputImagePlus = inputImage;

        if( outputModality.equals(  IOUtils.SAVE_AS_IMARIS ) )
        {
            logService.info( "Saving as Imaris..." );
            outputFileType = ".ims";
            resultExportSettings.exportType = ResultExportSettings.IMARIS_STACKS;
            cats.getResultImage().exportResults( resultExportSettings );
        }

        if( outputModality.equals(  IOUtils.SAVE_AS_TIFF_STACKS ) )
        {
            logService.info( "Saving as Tiff stacks..." );
            outputFileType = ".tif";
            resultExportSettings.exportType = ResultExportSettings.SAVE_AS_CLASS_PROBABILITY_TIFF_STACKS;
            cats.getResultImage().exportResults( resultExportSettings );
        }

        if( outputModality.equals( IOUtils.SAVE_AS_MULTI_CLASS_TIFF_SLICES ) )
        {
            logService.info( "Saving as Tiff slices..." );
            outputFileType = ".tif";
            resultExportSettings.exportType = ResultExportSettings.CLASS_PROBABILITIES_TIFF_SLICES;
            cats.getResultImage().exportResults( resultExportSettings );
        }


    }

    private void applyClassifier( )
    {
        logService.info( "Applying classifier: " + classifierFile );

        cats = new CATS();
        cats.setNumThreads( numWorkers );
        cats.setMaxMemoryBytes( memoryMB * 1000000L ); // MB -> Byte
        cats.setInputImage( inputImage );

        if ( outputModality.equals( IOUtils.STREAM_TO_RESULT_IMAGE_DISK ) )
        {
            cats.setResultImageDisk( outputDirectory.getAbsolutePath() );
        }
        else
        {

            if ( classificationIntervalXYZT.equals( WHOLE_IMAGE ) || classificationIntervalXYZT.equals("") )
                cats.setResultImageRAM();
            else
                cats.setResultImageRAM( getInterval() );
        }

        cats.loadClassifier( classifierFile.getAbsolutePath() );

        if ( classificationIntervalXYZT.equals( WHOLE_IMAGE ) || classificationIntervalXYZT.equals("") )
            cats.applyClassifierWithTiling();
        else
        {
            FinalInterval interval = getInterval();
            cats.applyClassifierWithTiling( interval );
        }

        logService.info( "Applying classifier: done." );

    }

    private FinalInterval getInterval()
    {
        int[] minMaxXYZT = StringUtils.delimitedStringToIntegerArray( classificationIntervalXYZT, "," );
        long[] min = new long[ 5 ];
        long[] max = new long[ 5 ];

        int i = 0;
        for ( int d : XYZT )
        {
            min[ d ] = minMaxXYZT[ i++ ];
            max[ d ] = minMaxXYZT[ i++ ];
        }

        return new FinalInterval( min, max );
    }

    private void logCommandLineCall()
    {
        Map<String, Object> parameters = new HashMap<>( );
        parameters.put( IOUtils.INPUT_IMAGE_FILE, inputImageFile );
        parameters.put( IOUtils.OUTPUT_MODALITY, outputModality );
        parameters.put( OUTPUT_DIRECTORY, outputDirectory );
        parameters.put( QUIT_AFTER_RUN, quitAfterRun );
        parameters.put( CLASSIFIER_FILE, classifierFile );
        IJ.log( Commands.createImageJPluginCommandLineCall( "ImageJ", PLUGIN_NAME, parameters ) );
    }


}