package trainableDeepSegmentation.ij2plugins;

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
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import trainableDeepSegmentation.WekaSegmentation;
import trainableDeepSegmentation.results.Utils;

import java.io.File;

@Plugin(type = Command.class,
        menuPath = "Plugins>Segmentation>EMBL-CBA>Apply Trainable Weka Deep Classifier" )

public class ApplyClassifierCommand<T extends RealType<T>> implements Command
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


    ImagePlus inputImage;

    WekaSegmentation wekaSegmentation;

    public void run()
    {

        logCommand();

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

    }

    private void logCommand()
    {
        IJ.log( "Command:" );

        String command = "";
        command += "ApplyClassifierCommand";
        command += " \"";
        command += INPUT_IMAGE_PATH + "=" + inputImagePath;
        command += "," + CLASSIFIER_PATH + "=" + classifierPath;
        command += "," + OUTPUT_DIRECTORY + "=" + outputDirectory;
        command += "," + OUTPUT_MODALITY + "=" + outputModality;
        command += "\" ";

        IJ.log( command );
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


}