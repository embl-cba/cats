package de.embl.cba.trainableDeepSegmentation.commands;

/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */


import de.embl.cba.cluster.commands.Commands;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import ij.IJ;
import ij.ImagePlus;
import ij.io.FileSaver;
import net.imagej.*;
import net.imagej.ops.OpService;
import net.imglib2.type.numeric.RealType;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import de.embl.cba.trainableDeepSegmentation.*;
import de.embl.cba.trainableDeepSegmentation.results.Utils;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierCommand.PLUGIN_NAME;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>EMBL-CBA>"+PLUGIN_NAME )
public class ApplyClassifierCommand<T extends RealType<T>> implements Command
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

    @Parameter (label = "Input image" )
    public File inputImagePath;
    public static final String INPUT_IMAGE_PATH = "inputImagePath";

    @Parameter (label = "Classifier" )
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

    DeepSegmentation deepSegmentation;

    // /Applications/Fiji.app/Contents/MacOS/ImageJ-macosx --run "Apply Classifier" "quitAfterRun='true',inputImagePath='/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/image-sequence/.*--W00016--P00003--.*',classifierPath='/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/transmission-cells-3d.classifier',outputModality='Save class probabilities as tiff files',outputDirectory='/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/image-sequence--classified'"

    // xvfb-run -a /g/almf/software/Fiji.app/ImageJ-linux64 --run "Apply Classifier" "quitAfterRun='true',inputImagePath='/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/small-test-image-sequences/.*--W00016--P00004--.*',classifierPath='/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/transmission-cells-3d.classifier',outputDirectory='/g/cba/tischer/projects/transmission-3D-stitching-organoid-size-measurement--data/small-test-image-sequences--classified/DataSet--W00016--P00004--',outputModality='Save class probabilities as tiff files'"

    public void run()
    {

        logService.info( "# " + PLUGIN_NAME );
        logCommandLineCall();

        logService.info( "Loading: " + inputImagePath );
        inputImage = IOUtils.loadImage( inputImagePath );

        logService.info( "Applying classifier: " + classifierPath );
        applyClassifier();

        if ( outputModality.equals( SHOW_AS_ONE_IMAGE ) )
        {
            deepSegmentation.getResultImage().getWholeImageCopy().show();
        }

        if ( outputModality.equals( SAVE_AS_TIFF_FILES ) )
        {
            saveProbabilitiesAsSeparateTiffFiles();
        }

        if( outputModality.equals(  SAVE_AS_IMARIS ) )
        {
            saveProbabilitiesAsImarisFiles();
        }

        if ( quitAfterRun )  if ( quitAfterRun ) Commands.quitImageJ( logService );

    }

    private void saveProbabilitiesAsOneTiff()
    {
        ImagePlus result = deepSegmentation.getResultImage().getWholeImageCopy();
        String savingPath = "" + outputDirectory + File.separator + inputImage.getTitle() + "--classified.tif";
        logService.info( "Save results: " + savingPath);
        DeepSegmentation.logger.info("\n# Saving " + savingPath + "...");
        FileSaver fileSaver = new FileSaver( result );
        fileSaver.saveAsTiff( savingPath );
        DeepSegmentation.logger.info("...done.");
    }

    private void saveProbabilitiesAsSeparateTiffFiles()
    {
        deepSegmentation.getResultImage().saveClassesAsFiles( outputDirectory.getPath(), null, null, Utils.SEPARATE_TIFF_FILES );
    }

    private void saveProbabilitiesAsImarisFiles()
    {
        deepSegmentation.getResultImage().saveClassesAsFiles( outputDirectory.getPath(), null, null, Utils.SEPARATE_IMARIS );
    }

    private void applyClassifier( )
    {
        deepSegmentation = new DeepSegmentation( );
        deepSegmentation.setInputImage( inputImage );
        deepSegmentation.setResultImageRAM( );
        deepSegmentation.loadClassifier( classifierPath.getAbsolutePath() );
        deepSegmentation.applyClassifierWithTiling();
    }

    private void logCommandLineCall()
    {
        Map<String, Object> parameters = new HashMap<>( );
        parameters.put( INPUT_IMAGE_PATH, inputImagePath );
        parameters.put( OUTPUT_MODALITY, outputModality );
        parameters.put( OUTPUT_DIRECTORY, outputDirectory );
        parameters.put( QUIT_AFTER_RUN, quitAfterRun );
        parameters.put( CLASSIFIER_PATH, classifierPath );
        IJ.log( Commands.createImageJPluginCommandLineCall( "ImageJ", PLUGIN_NAME, parameters ) );
    }


}