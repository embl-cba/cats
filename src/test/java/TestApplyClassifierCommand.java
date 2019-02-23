import de.embl.cba.cats.utils.IOUtils;
import de.embl.cba.cats.utils.IntervalUtils;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.ImageJ;
import de.embl.cba.cats.ui.ApplyClassifierCommand;
import net.imglib2.FinalInterval;

import java.io.File;
import java.util.HashMap;
import java.util.Map;


import static de.embl.cba.cats.utils.IntervalUtils.Z;
import static de.embl.cba.cats.utils.IntervalUtils.getIntervalAsCsvString;

public class TestApplyClassifierCommand
{

    // Main
    public static void main(final String... args) throws Exception {

        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ImagePlus imp = IJ.openImage("fib-sem--cell--8x8x8nm.zip" );

        FinalInterval fullImageInterval = IntervalUtils.getInterval( imp );
        long[] min = new long[ 5 ];
        long[] max = new long[ 5 ];
        fullImageInterval.min( min );
        fullImageInterval.max( max );
        min[ Z ] = 0; max[ Z ] = 5;
        FinalInterval classificationInterval = new FinalInterval( min, max );

        Map< String, Object > parameters = new HashMap<>(  );

        parameters.clear();

        parameters.put( ApplyClassifierCommand.DATASET_ID, "007" );
        parameters.put( IOUtils.INPUT_MODALITY, IOUtils.OPEN_USING_IMAGEJ1 );
        parameters.put( IOUtils.INPUT_IMAGE_FILE, new File( "/Users/tischer/Documents/fiji-plugin-CATS/src/test/resources/fib-sem--cell--8x8x8nm.zip" )  );

        File classifierFile = new File( "/Volumes/cba/tischer/projects/em-automated-segmentation--data/fib-sem--cell.classifier" );
        String outputDirectory = "/Volumes/cba/tischer/projects/em-automated-segmentation--data/fib-sem--cell--classification/";

        parameters.put( ApplyClassifierCommand.CLASSIFICATION_INTERVAL, getIntervalAsCsvString( classificationInterval ) );

        parameters.put( ApplyClassifierCommand.CLASSIFIER_FILE, classifierFile );

        parameters.put( IOUtils.OUTPUT_MODALITY, IOUtils.SAVE_AS_MULTI_CLASS_TIFF_SLICES );

        parameters.put( ApplyClassifierCommand.OUTPUT_DIRECTORY, outputDirectory  );

        parameters.put( ApplyClassifierCommand.NUM_WORKERS, 4 );
        parameters.put( ApplyClassifierCommand.MEMORY_MB, 8000 );

        parameters.put( ApplyClassifierCommand.SAVE_RESULTS_TABLE, false );
        parameters.put( ApplyClassifierCommand.QUIT_AFTER_RUN, true );

        ij.command().run( ApplyClassifierCommand.class, false, parameters );

    }

}
