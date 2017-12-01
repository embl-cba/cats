package trainableDeepSegmentation;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.io.FileSaver;

import java.io.File;

/**
 * Created by tischi on 04/10/17.
 */
public class TestSylwia {

    final static String ARG_CLASSIFIER_PATH = "/Volumes/almf/group/ALMFstuff/ALMF_ImageAnalysisProjects/transmission" +
            "-3D-stitching-organoid-size-measurement--data/01.classifier";
    final static Integer ARG_WELL_NUM = 50;
    final static Integer ARG_POS_NUM = 1;
    final static String ARG_INPUT_DIRECTORY = "/Volumes/almf/group/ALMFstuff/ALMF_ImageAnalysisProjects/transmission-3D-stitching-organoid-size-measurement--data/data/scanR-test-data-02-crop";
    final static String ARG_OUTPUT_DIRECTORY = "/Volumes/almf/group/ALMFstuff/ALMF_ImageAnalysisProjects/transmission-3D-stitching-organoid-size-measurement--data/data/scanR-test-data-02-crop--out";
    final static Integer MIN_NUM_VOXELS = 10;

    public static void main( final String[] args )
    {
        new ImageJ();


        // Load data
        //
        String wellString = String.format( "%05d", ARG_WELL_NUM );
        String posString = String.format( "%05d", ARG_POS_NUM );
        String regExp = ".*--W" + wellString + "--P" + posString + "--Z.*--T00000--Trans.tif";

        IJ.run("Image Sequence...",
                "open=["+ ARG_INPUT_DIRECTORY +"]"
                +"file=("+regExp+") sort");

        ImagePlus inputImagePlus = IJ.getImage();

        // Segment objects
        //
        WekaSegmentation wekaSegmentation = new WekaSegmentation( );
        wekaSegmentation.setInputImage( inputImagePlus );
        wekaSegmentation.setResultImageRAM( );
        wekaSegmentation.loadClassifier( ARG_CLASSIFIER_PATH );
        wekaSegmentation.applyClassifierWithTiling();


        // Analyse objects
        //
        ImagePlus labelMask = wekaSegmentation.analyzeObjects( MIN_NUM_VOXELS );
        labelMask.show();

        // Save image
        String outFilePath = ARG_OUTPUT_DIRECTORY + File.separator + "R"+ regExp + "--labelMask.tif";
        WekaSegmentation.logger.info("\n# Saving file " + outFilePath + "...");
        FileSaver fileSaver = new FileSaver( labelMask );
        fileSaver.saveAsTiff( outFilePath );
        WekaSegmentation.logger.info("...done.");


    }


}
