package trainableDeepSegmentation;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.io.FileSaver;
import ij.measure.ResultsTable;
import inra.ijpb.measure.GeometricMeasures3D;

import java.io.File;

/**
 * Created by tischi on 04/10/17.
 */
public class TestSylwia {

    final static String ROOT = "/Users/tischi/Documents/transmission-3D-stitching-organoid-size-measurement";

    final static Integer WELL_NUM = 50;
    final static Integer POS_NUM = 1;
    final static String INPUT_DIR = ROOT+"/data/scanR-small-crop";
    final static String OUTPUT_DIR = ROOT+"/data/scanR-small-crop--out";
    final static String CLASSIFIER_PATH = ROOT+"/code/cluster-test/01.classifier";


    public static void main( final String[] args )
    {
        final Integer MIN_VOXELS = 10;

        new ImageJ();

        // Load data
        //
        String wellString = String.format( "%05d", WELL_NUM );
        String posString = String.format( "%05d", POS_NUM );
        String regExp = ".*--W" + wellString + "--P" + posString + "--Z.*--T00000--Trans.tif";

        IJ.run("Image Sequence...",
                "open=["+ INPUT_DIR +"]"
                +"file=("+regExp+") sort");

        ImagePlus inputImagePlus = IJ.getImage();

        // Pixel classification
        //
        WekaSegmentation wekaSegmentation = new WekaSegmentation( );
        wekaSegmentation.setInputImage( inputImagePlus );
        wekaSegmentation.setResultImageRAM( );
        wekaSegmentation.loadClassifier( CLASSIFIER_PATH );
        wekaSegmentation.applyClassifierWithTiling();

        // Connected components
        //
        ImagePlus labelMask = null;
        //ImagePlus labelMask = wekaSegmentation.computeClassLabelMask( 1, MIN_VOXELS, 12, 20 );
        //labelMask.show();

        //  Object bounding boxes
        //
        ResultsTable rt_bb = GeometricMeasures3D.boundingBox( labelMask.getStack() );
        rt_bb.show( "Bounding boxes" );

        //  Object volumes
        //
        double[] resolution = new double[3];
        resolution[0] = 1.0;
        resolution[1] = 1.0;
        resolution[2] = 1.0;
        ResultsTable rt_v = GeometricMeasures3D.volume( labelMask.getStack(), resolution );
        rt_v.show( "Volumes" );

        WekaSegmentation.logger.info( "\nNumber of objects: " + rt_v.size() );

        // Saving results
        //
        String savingPath;

        savingPath = "" + OUTPUT_DIR + File.separator + "R"+ regExp + "--labelMask.tif";
        WekaSegmentation.logger.info("\n# Saving " + savingPath + "...");
        FileSaver fileSaver = new FileSaver( labelMask );
        fileSaver.saveAsTiff( savingPath );
        WekaSegmentation.logger.info("...done.");

        savingPath = "" + OUTPUT_DIR + File.separator + "R"+ regExp + "--bounding-boxes.csv";
        WekaSegmentation.logger.info("\n# Saving " + savingPath + "...");
        rt_bb.save(  savingPath );
        WekaSegmentation.logger.info("...done.");

        savingPath = "" + OUTPUT_DIR + File.separator + "R"+ regExp + "--volumes.csv";
        WekaSegmentation.logger.info("\n# Saving " + savingPath + "...");
        rt_v.save(  savingPath );
        WekaSegmentation.logger.info("...done.");

        System.exit(0);
        IJ.run("Quit");

    }


}
