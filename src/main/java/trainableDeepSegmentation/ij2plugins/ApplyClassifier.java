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

    @Parameter (label = "Input image", required = true )
    public File inputImageFile;

    @Parameter (label = "Classifier",  required = true )
    public File classifierFile;

    @Parameter( label = "Output folder", style = "directory", required = true )
    public File outputFolder;

    @Parameter( visibility = ItemVisibility.MESSAGE )
    private String message02
            = "<html>"  +
            "<br>Test<br>" +
            "Test.<br>";

    ImagePlus input;

    WekaSegmentation wekaSegmentation;

    public void run()
    {
        loadImage();
        applyClassifier();
        saveProbabilitiesAsTiff();

    }

    private void saveProbabilitiesAsTiff()
    {
        ImagePlus result = wekaSegmentation.getResultImage().getWholeImageCopy();
        String savingPath = "" + outputFolder + File.separator + input.getTitle() + "--classified.tif";
        WekaSegmentation.logger.info("\n# Saving " + savingPath + "...");
        FileSaver fileSaver = new FileSaver( result );
        fileSaver.saveAsTiff( savingPath );
        WekaSegmentation.logger.info("...done.");
    }

    private void loadImage()
    {
        input = IJ.openImage( inputImageFile.getAbsolutePath() );
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