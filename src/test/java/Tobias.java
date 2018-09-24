import de.embl.cba.cats.CATS;
import de.embl.cba.cats.instances.InstancesAndMetadata;
import de.embl.cba.cats.results.ResultExportSettings;
import de.embl.cba.cats.results.ResultImage;
import ij.IJ;

import java.io.File;
import java.util.ArrayList;

public class Tobias
{

	final static String IMAGES_DIR = "/Volumes/almfspim/tischi/test_tobi";
    final static String INSTANCES_DIR = "/Volumes/almfspim/tischi/label_tobi";
	final static String RESULTS_DIR = "/Volumes/almfspim/tischi/result_tobi";
	final static String[] TRAIN = new String[]{ "data0", "data1", "data2" };
	final static String[] TEST = new String[]{ "data3" };

	final static String IMAGE_FILE_EXTENSION = ".tif";
	final static String INSTANCES_FILE_EXTENSION = ".ARFF";
	public static final String CLASSIFIER = "classifier.classifier";


	public static void main( final String[] args )
	{

		recomputeInstances();

		buildClassifier();

		applyClassifier();

	}

	private static void applyClassifier()
	{
		for ( String test : TEST )
		{
			final CATS cats = new CATS();
			// CATS needs one image even though it is not used at all here
			cats.setInputImage( IJ.openImage( IMAGES_DIR + File.separator + test + IMAGE_FILE_EXTENSION ) );
			cats.setResultImageRAM();
			cats.loadClassifier( INSTANCES_DIR, CLASSIFIER );
			cats.applyClassifierWithTiling();

			final ResultImage resultImage = cats.getResultImage();
			final ResultExportSettings resultExportSettings = new ResultExportSettings();
			resultExportSettings.inputImagePlus = cats.getInputImage();
			resultExportSettings.exportType = ResultExportSettings.TIFF_STACKS;
			resultExportSettings.directory = RESULTS_DIR;
			resultExportSettings.exportNamesPrefix = test + "-";
			resultExportSettings.classNames = cats.getClassNames();
			resultImage.exportResults( resultExportSettings );
		}
	}

	private static void buildClassifier()
	{
		final CATS cats = new CATS();
		// CATS needs one image even though it is not used at all here
		cats.setInputImage( IJ.openImage( IMAGES_DIR + File.separator + TRAIN[ 0 ] + IMAGE_FILE_EXTENSION ) );

		final ArrayList< String > instancesKeys = new ArrayList<>();

		for ( String train : TRAIN )
		{
			instancesKeys.add( cats.loadInstancesAndMetadata( INSTANCES_DIR, train + INSTANCES_FILE_EXTENSION ) );
		}

		final InstancesAndMetadata combinedInstancesAndMetadata = cats.getInstancesManager().getCombinedInstancesAndMetadata( instancesKeys );
		cats.trainClassifierWithFeatureSelection( combinedInstancesAndMetadata );
		cats.saveClassifier( INSTANCES_DIR, CLASSIFIER );
	}

	private static void recomputeInstances()
	{
		for ( String train : TRAIN )
		{
			CATS cats = new CATS();
			cats.setInputImage( IJ.openImage( IMAGES_DIR + File.separator + train + IMAGE_FILE_EXTENSION ) );
			cats.setResultImageRAM();
			cats.loadInstancesAndMetadata( INSTANCES_DIR, train + INSTANCES_FILE_EXTENSION );
			cats.recomputeLabelInstances();
			cats.saveInstances( INSTANCES_DIR, train + INSTANCES_FILE_EXTENSION  );
		}
	}


	public static void createIfNotExists( String dir )
    {
        File file = new File( dir );
        if ( !file.exists() )
        {
            file.mkdir();
        }

    }

}
