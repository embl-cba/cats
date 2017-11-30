package trainableDeepSegmentation;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

/**
 * Created by tischi on 04/10/17.
 */
public class TestHeadless {

    final static String INPUT_IJ_HYPERSTACK = "/Users/tischi/Desktop/segmentation-challenges/brainiac2-mit-edu-SNEMI3D/combined-clahe.tif";
    final static String CLASSIFIER_DIR = "/Users/tischi/Desktop/segmentation-challenges/brainiac2-mit-edu-SNEMI3D/classifiers/";
    final static String CLASSIFIER_FILE = "01.classifier";

    public static void main( final String[] args )
    {
        new ImageJ();
        ImagePlus inputImagePlus = IJ.openImage( INPUT_IJ_HYPERSTACK );

        WekaSegmentation wekaSegmentation = new WekaSegmentation( );
        wekaSegmentation.setInputImage( inputImagePlus );
        wekaSegmentation.setResultImageRAM( );
        wekaSegmentation.loadClassifier( CLASSIFIER_DIR, CLASSIFIER_FILE );
        wekaSegmentation.applyClassifierWithTiling();

        ImagePlus result = wekaSegmentation.getResultImage().getImagePlus();
    }


}
