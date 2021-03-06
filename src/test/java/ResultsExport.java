import bdv.img.imaris.Imaris;
import bdv.spimdata.SpimDataMinimal;
import bdv.util.BdvFunctions;
import de.embl.cba.cats.CATS;
import de.embl.cba.cats.results.ResultExportSettings;
import de.embl.cba.cats.utils.IntervalUtils;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.imglib2.FinalInterval;

import java.io.File;
import java.io.IOException;


public class ResultsExport
{
    public static void main( final String[] args )
    {
        new ImageJ();

//        fibSemCell();

        objects3d2Channels();
    }

    private static void objects3d2Channels()
    {
        // Open Image
        //
        ImagePlus inputImagePlus = IJ.openImage( "/Users/tischer/Documents/fiji-plugin-cats/src/test/resources/3d-objects-test/3d-objects-input/3d-objects-2ch.zip" );

        inputImagePlus.show();

        CATS cats = new CATS();
        cats.setInputImage( inputImagePlus );
        cats.setResultImageDisk( "/Users/tischer/Documents/fiji-plugin-cats/src/test/resources/3d-objects-test/3d-objects-probabilities" );
        cats.loadInstancesAndMetadata( "/Users/tischer/Documents/fiji-plugin-cats/src/test/resources/3d-objects-test/3d-objects-instances/3d-objects.ARFF"  );

        ResultExportSettings resultExportSettings = new ResultExportSettings();
        resultExportSettings.directory = "/Users/tischer/Documents/fiji-plugin-cats/src/test/resources/3d-objects-test/3d-objects-export";
        resultExportSettings.exportType = ResultExportSettings.SAVE_AS_IMARIS_STACKS;
        resultExportSettings.classNames = cats.getClassNames();
        resultExportSettings.timePointsFirstLast = new int[]{ 0, 0 };
        resultExportSettings.saveRawData = true;
        resultExportSettings.inputImagePlus = inputImagePlus;

        deleteFolder( new File( resultExportSettings.directory ) );

        cats.getResultImage().exportResults( resultExportSettings );

        try
        {
            SpimDataMinimal spimData = Imaris.openIms(
                    ResultsExport.class.getResource(
                            "3d-objects-export/meta.ims" ).getFile() );
            BdvFunctions.show( spimData );
        }
        catch ( IOException e )
        {
            e.printStackTrace();
        }
    }

    public static void deleteFolder(File folder) {
        File[] files = folder.listFiles();
        if(files!=null) { //some JVMs return null for empty dirs
            for(File f: files) {
                if(f.isDirectory()) {
                    deleteFolder(f);
                } else {
                    f.delete();
                }
            }
        }
        folder.delete();
    }

    private static void fibSemCell()
    {
        // Open Image
        //
        ImagePlus imp = IJ.openImage( "/Users/tischer/Documents/fiji-plugin-CATS/src/test/resources/fib-sem--cell--8x8x8nm--nt2.zip" );

        FinalInterval fullImageInterval = IntervalUtils.getInterval( imp );
        long[] min = new long[ 5 ];
        long[] max = new long[ 5 ];
        fullImageInterval.min( min );
        fullImageInterval.max( max );
        //max[ X ] = 50;
        //max[ Y ] = 50;
        //min[ Z ] = 2; max[ Z ] = 5;
        FinalInterval interval = new FinalInterval( min, max );

        CATS cats = new CATS();
        cats.setInputImage( imp );
        cats.setResultImageRAM( interval );
        cats.loadInstancesAndMetadata( "/Users/tischer/Documents/fiji-plugin-CATS/src/test/resources/fib-sem--cell--8x8x8nm.ARFF" );

        cats.classifierNumTrees = 10;
        cats.trainClassifier( "fib-sem--cell--8x8x8nm.tif" );
        cats.applyClassifierWithTiling( interval );

        cats.getInputImage().show();
        cats.getResultImage().getWholeImageCopy().show();

        ResultExportSettings resultExportSettings = new ResultExportSettings();
        resultExportSettings.directory = "/Users/tischer/Desktop/tmp4";
        resultExportSettings.exportType = ResultExportSettings.SAVE_AS_IMARIS_STACKS;
        resultExportSettings.classNames = cats.getClassNames();
        resultExportSettings.timePointsFirstLast = new int[]{ 0, 1 };
        resultExportSettings.saveRawData = true;
        resultExportSettings.inputImagePlus = imp;

        cats.getResultImage().exportResults( resultExportSettings );

        try
		{
			SpimDataMinimal spimData = Imaris.openIms( "/Users/tischer/Desktop/tmp4/meta.ims" );
			BdvFunctions.show( spimData );
		} catch ( IOException e )
		{
			e.printStackTrace();
		}
    }

}
