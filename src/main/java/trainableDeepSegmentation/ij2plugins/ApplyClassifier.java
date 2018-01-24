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
import org.scijava.widget.WidgetService;
import trainableDeepSegmentation.WekaSegmentation;
import trainableDeepSegmentation.results.Utils;

import java.io.File;

@Plugin(type = Command.class,
        menuPath = "Plugins>Segmentation>Apply Trainable Weka Deep Classifier (Test)" )

public class ApplyClassifier<T extends RealType<T>> implements Command
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
    public File inputImageFile;

    @Parameter (label = "Classifier",  required = true )
    public File classifierFile;

    public static final String SHOW_AS_ONE_IMAGE = "Show all probabilities in one image";
    public static final String SAVE_AS_TIFF_FILES = "Save class probabilities as tiff files";
    public static final String SAVE_AS_IMARIS = "Save class probabilities as imaris files";

    @Parameter( label = "Output modality", choices = { SHOW_AS_ONE_IMAGE, SAVE_AS_TIFF_FILES, SAVE_AS_IMARIS } , required = true )
    public String outputModality;

    @Parameter( label = "Output folder", style = "directory" )
    public File outputFolder;


    ImagePlus input;

    WekaSegmentation wekaSegmentation;

    public void run()
    {
        loadImage();
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

    private void saveProbabilitiesAsOneTiff()
    {
        ImagePlus result = wekaSegmentation.getResultImage().getWholeImageCopy();
        String savingPath = "" + outputFolder + File.separator + input.getTitle() + "--classified.tif";
        WekaSegmentation.logger.info("\n# Saving " + savingPath + "...");
        FileSaver fileSaver = new FileSaver( result );
        fileSaver.saveAsTiff( savingPath );
        WekaSegmentation.logger.info("...done.");
    }

    private void saveProbabilitiesAsSeparateTiffFiles()
    {
        String folder = outputFolder.getPath();
        wekaSegmentation.getResultImage().saveClassesAsFiles( outputFolder.getPath(), null, null, Utils.SEPARATE_TIFF_FILES );
    }

    private void saveProbabilitiesAsImarisFiles()
    {
        wekaSegmentation.getResultImage().saveClassesAsFiles( outputFolder.getPath(), null, null, Utils.SEPARATE_IMARIS );
    }

    private void loadImage()
    {
        if ( inputImageFile.getName().contains( ".*" ) )
        {
            loadImageWithImportImageSequence();
        }
        else
        {
            loadImageWithIJOpenImage();
        }
    }

    private void loadImageWithIJOpenImage()
    {
        input = IJ.openImage( inputImageFile.getAbsolutePath() );
    }

    private void loadImageWithImportImageSequence()
    {
        String directory = inputImageFile.getParent();
        String regExp = inputImageFile.getName();
        IJ.run("Image Sequence...", "open=["+ directory +"]" +"file=(" + regExp + ") sort");
        input = IJ.getImage();
        input.setTitle( regExp );
    }

    private void applyClassifier( )
    {
        wekaSegmentation = new WekaSegmentation( );
        wekaSegmentation.setInputImage( input );
        wekaSegmentation.setResultImageRAM( );
        wekaSegmentation.loadClassifier( classifierFile.getAbsolutePath() );
        wekaSegmentation.applyClassifierWithTiling();
    }


}