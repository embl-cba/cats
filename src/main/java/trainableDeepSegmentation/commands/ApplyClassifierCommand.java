package trainableDeepSegmentation.commands;

/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */


import ij.IJ;
import ij.ImagePlus;
import ij.io.FileSaver;
import net.imagej.*;
import net.imagej.ops.OpService;
import net.imglib2.type.numeric.RealType;
import org.scijava.ItemVisibility;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.command.DynamicCommand;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import trainableDeepSegmentation.WekaSegmentation;
import trainableDeepSegmentation.results.Utils;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static trainableDeepSegmentation.commands.ApplyClassifierCommand.PLUGIN_NAME;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>EMBL-CBA>"+PLUGIN_NAME )
public class ApplyClassifierCommand<T extends RealType<T>> extends DynamicCommand
{
    public static final String PLUGIN_NAME = "Apply Classifier";

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
            "<br>Apply Classifier<br>" +
            "...<br>";

    @Parameter (label = "Input image", required = true )
    public File inputImagePath;
    public static final String INPUT_IMAGE_PATH = "inputImagePath";

    @Parameter (label = "Classifier",  required = true )
    public File classifierPath;
    public static final String CLASSIFIER_PATH = "classifierPath";

    @Parameter( label = "Output modality", choices = { ApplyClassifierCommand.SHOW_AS_ONE_IMAGE, ApplyClassifierCommand.SAVE_AS_TIFF_FILES, ApplyClassifierCommand.SAVE_AS_IMARIS } , required = true )
    public String outputModality;
    public static final String OUTPUT_MODALITY = "outputModality";
    public static final String SAVE_AS_IMARIS = "Save class probabilities as imaris files";
    public static final String SAVE_AS_TIFF_FILES = "Save class probabilities as tiff files";
    public static final String SHOW_AS_ONE_IMAGE = "Show all probabilities in one image";

    @Parameter( label = "Output folder", style = "directory" )
    public File outputDirectory;
    public static final String OUTPUT_DIRECTORY = "outputDirectory";

    @Parameter( label = "Quit ImageJ after running", required = false )
    public boolean quitAfterRun = false;
    public static final String QUIT_AFTER_RUN = "quitAfterRun";

    ImagePlus inputImage;

    WekaSegmentation wekaSegmentation;

    public void run()
    {

        logCommandLineCall();

        inputImage = IOUtils.loadImage( inputImagePath );

        applyClassifier();

        if ( outputModality.equals( SHOW_AS_ONE_IMAGE ) )
        {
            wekaSegmentation.getResultImage().getWholeImageCopy().show();
        }

        if ( outputModality.equals( SAVE_AS_TIFF_FILES ) )
        {
            saveProbabilitiesAsSeparateTiffFiles();
        }

        if( outputModality.equals(  SAVE_AS_IMARIS ) )
        {
            saveProbabilitiesAsImarisFiles();
        }

        if ( quitAfterRun )  IJ.run( "Quit" );

    }

    private void saveProbabilitiesAsOneTiff()
    {
        ImagePlus result = wekaSegmentation.getResultImage().getWholeImageCopy();
        String savingPath = "" + outputDirectory + File.separator + inputImage.getTitle() + "--classified.tif";
        WekaSegmentation.logger.info("\n# Saving " + savingPath + "...");
        FileSaver fileSaver = new FileSaver( result );
        fileSaver.saveAsTiff( savingPath );
        WekaSegmentation.logger.info("...done.");
    }

    private void saveProbabilitiesAsSeparateTiffFiles()
    {
        wekaSegmentation.getResultImage().saveClassesAsFiles( outputDirectory.getPath(), null, null, Utils.SEPARATE_TIFF_FILES );
    }

    private void saveProbabilitiesAsImarisFiles()
    {
        wekaSegmentation.getResultImage().saveClassesAsFiles( outputDirectory.getPath(), null, null, Utils.SEPARATE_IMARIS );
    }

    private void applyClassifier( )
    {
        wekaSegmentation = new WekaSegmentation( );
        wekaSegmentation.setInputImage( inputImage );
        wekaSegmentation.setResultImageRAM( );
        wekaSegmentation.loadClassifier( classifierPath.getAbsolutePath() );
        wekaSegmentation.applyClassifierWithTiling();
    }

    private void logCommandLineCall()
    {

        Map<String,Object> inputs = getInputs();

        Map<String, Object> parameters = new HashMap<>( );
        parameters.put( INPUT_IMAGE_PATH, inputImagePath );
        parameters.put( OUTPUT_MODALITY, outputModality );
        parameters.put( OUTPUT_DIRECTORY, outputDirectory );
        parameters.put( QUIT_AFTER_RUN, quitAfterRun );
        parameters.put( CLASSIFIER_PATH, classifierPath );
        IJ.log( Commands.createCommand( PLUGIN_NAME, parameters ) );
    }


}