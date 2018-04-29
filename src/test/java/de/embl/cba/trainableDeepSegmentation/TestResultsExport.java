package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.trainableDeepSegmentation.results.ResultExportSettings;
import de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.imglib2.FinalInterval;

import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.*;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.C;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.T;

public class TestResultsExport
{
    public static void main( final String[] args )
    {

        new ImageJ();

        // Open Image
        //
        ImagePlus imp = IJ.openImage( "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/fib-sem--cell--8x8x8nm.zip" );

        FinalInterval fullImageInterval = IntervalUtils.getInterval( imp );
        long[] min = new long[ 5 ];
        long[] max = new long[ 5 ];
        fullImageInterval.min( min );
        fullImageInterval.max( max );
        max[ X ] = 50;
        max[ Y ] = 50;
        min[ Z ] = 2; max[ Z ] = 5;
        FinalInterval interval = new FinalInterval( min, max );

        DeepSegmentation deepSegmentation = new DeepSegmentation( );
        deepSegmentation.setInputImage( imp );
        deepSegmentation.setResultImageRAM( interval );
        deepSegmentation.loadInstancesAndMetadata( "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/fib-sem--cell--8x8x8nm.ARFF" );

        deepSegmentation.classifierNumTrees = 10;
        deepSegmentation.trainClassifier( );
        deepSegmentation.applyClassifierWithTiling( interval );

        deepSegmentation.getInputImage().show();
        deepSegmentation.getResultImage().getWholeImageCopy().show();

        ResultExportSettings resultExportSettings = new ResultExportSettings();
        resultExportSettings.directory = "/tmp/tmp";
        resultExportSettings.exportType = ResultExportSettings.SEPARATE_MULTI_CLASS_TIFF_SLICES;
        resultExportSettings.classNames = deepSegmentation.getClassNames();

        deepSegmentation.getResultImage().exportResults( resultExportSettings );
    }

}
