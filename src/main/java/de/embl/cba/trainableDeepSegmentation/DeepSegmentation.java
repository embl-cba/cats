package de.embl.cba.trainableDeepSegmentation;

import java.awt.*;
import java.awt.image.ColorModel;
import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.GZIPOutputStream;

import de.embl.cba.bigDataTools.VirtualStackOfStacks.VirtualStackOfStacks;
import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierOnSlurmCommand;
import de.embl.cba.trainableDeepSegmentation.features.DownSampler;
import de.embl.cba.trainableDeepSegmentation.labelimagetraining.AccuracyEvaluation;
import de.embl.cba.trainableDeepSegmentation.postprocessing.ObjectReview;
import de.embl.cba.trainableDeepSegmentation.postprocessing.ObjectSegmentation;
import de.embl.cba.trainableDeepSegmentation.postprocessing.SegmentedObjects;
import de.embl.cba.trainableDeepSegmentation.settings.FeatureSettings;
import de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin;
import de.embl.cba.trainableDeepSegmentation.utils.CommandUtils;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import de.embl.cba.utils.logging.IJLazySwingLogger;
import de.embl.cba.utils.logging.Logger;

import de.embl.cba.trainableDeepSegmentation.weka.fastRandomForest.FastRandomForest;
import fiji.util.gui.GenericDialogPlus;
import ij.*;
import ij.gui.*;
import ij.io.FileInfo;
import ij.io.FileSaver;
import ij.measure.Calibration;
import ij.measure.ResultsTable;
import ij.plugin.Duplicator;
import ij.plugin.MacroInstaller;
import ij.process.ImageProcessor;
import inra.ijpb.binary.BinaryImages;
import inra.ijpb.measure.GeometricMeasures3D;
import inra.ijpb.morphology.AttributeFiltering;
import inra.ijpb.segment.Threshold;
import javafx.geometry.Point3D;

import net.imglib2.FinalInterval;
import de.embl.cba.trainableDeepSegmentation.classification.AttributeSelector;
import de.embl.cba.trainableDeepSegmentation.classification.ClassifierInstancesMetadata;
import de.embl.cba.trainableDeepSegmentation.classification.ClassifierManager;
import de.embl.cba.trainableDeepSegmentation.classification.ClassifierUtils;
import de.embl.cba.trainableDeepSegmentation.labels.examples.Example;
import de.embl.cba.trainableDeepSegmentation.labels.examples.ExamplesUtils;
import de.embl.cba.trainableDeepSegmentation.features.FeatureProvider;
import de.embl.cba.trainableDeepSegmentation.instances.InstancesAndMetadata;
import de.embl.cba.trainableDeepSegmentation.instances.ReusableDenseInstance;
import de.embl.cba.trainableDeepSegmentation.results.ResultImage;
import de.embl.cba.trainableDeepSegmentation.results.ResultImageDisk;
import de.embl.cba.trainableDeepSegmentation.results.ResultImageFrameSetter;
import de.embl.cba.trainableDeepSegmentation.instances.InstancesUtils;
import de.embl.cba.trainableDeepSegmentation.instances.InstancesManager;
import de.embl.cba.trainableDeepSegmentation.results.ResultImageRAM;
import de.embl.cba.trainableDeepSegmentation.settings.SettingsUtils;

//import inra.ijpb.segment.Threshold;

import de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils;
import de.embl.cba.trainableDeepSegmentation.utils.ThreadUtils;
import net.imglib2.util.Intervals;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.*;


/**
 * This class contains all the library methods to perform image segmentation
 * based on the Weka classifiersComboBox.
 */
public class DeepSegmentation
{

	/**
	 * maximum number of classes (labels) allowed
	 */
	public static final int MAX_NUM_CLASSES = 20;
	public static final String RESULT_IMAGE_DISK_SINGLE_TIFF = "Disk";
	public static final String RESULT_IMAGE_RAM = "RAM";


    /** available classColors for available classes */
    public Color[] classColors = new Color[]{
            Color.gray,
            Color.green,
            Color.red,
            Color.blue,
            Color.cyan,
            Color.pink,
            Color.white,
            Color.magenta,
            Color.orange,
            Color.black,
            Color.yellow,
            Color.gray,
            Color.green,
            Color.red,
            Color.blue,
            Color.cyan,
            Color.pink,
            Color.white,
            Color.magenta,
            Color.orange,
            Color.black
    };

    public DeepSegmentationIJ1Plugin deepSegmentationIJ1Plugin;

	public String resultImageType = RESULT_IMAGE_RAM;

	/**
	 * array of lists of Rois for each slice (vector index)
	 * and each class (arraylist index) of the instances image
	 */
	private ArrayList<Example > examples;
	/**
	 * image to be used in the instances
	 */
	private ImagePlus inputImage;

	private ResultImage resultImage = null;

    public static void reserveKeyboardShortcuts()
    {
        // reserve shortcuts
        String macros = "macro 'shortcut 1 [1]' {};\n"
                + "macro 'shortcut 2 [2]' {};"
                + "macro 'shortcut 3 [3]' {};"
                + "macro 'shortcut 4 [4]' {};"
                + "macro 'shortcut 5 [5]' {};"
                + "macro 'shortcut 6 [6]' {};"
                + "macro 'shortcut 7 [7]' {};"
                + "macro 'shortcut 8 [8]' {};"
                + "macro 'shortcut 9 [9]' {};"
                + "macro 'shortcut r [r]' {};"
                + "macro 'shortcut p [p]' {};"
                + "macro 'shortcut u [u]' {};"
                + "macro 'shortcut g [g]' {};"
                + "macro 'shortcut n [n]' {};"
                + "macro 'shortcut b [b]' {};"
                + "macro 'shortcut d [d]' {};"
                + "macro 'shortcut s [s]' {};"
                ;
        new MacroInstaller().install(macros);
    }

    public void setMaxMemory( long maxMemory )
	{
		this.maxMemory = maxMemory;
	}

	public long getMaxMemory( )
	{
		return maxMemory;
	}

	private long maxMemory;

	public ResultImage getResultImageBgFg()
	{
		return resultImageBgFg;
	}

	public void setResultImageBgFg( ResultImage resultImageBgFg )
	{
		this.resultImageBgFg = resultImageBgFg;
	}

	public void setResultImageBgFgRAM( )
	{
		resultImageBgFg = new ResultImageRAM( this, getInputImageDimensions() );

		logger.info("Allocated memoryMB for resultImagePlus image." );

	}

	public void setResultImageBgFgDisk( String directory )
	{

		resultImageBgFg = new ResultImageDisk( this, directory, getInputImageDimensions() );

		logger.info("Created disk-resident resultImagePlus image: " + directory);


	}

	public Color[] getColors()
    {
        return classColors;
    }

	public Map< String, SegmentedObjects > getSegmentedObjectsMap()
	{
		return segmentedObjectsMap;
	}

	public void addObjects( SegmentedObjects segmentedObjects )
	{
		if ( segmentedObjectsMap == null  )
		{
			segmentedObjectsMap = new HashMap<>(  );
		}

		segmentedObjectsMap.put( segmentedObjects.name, segmentedObjects );
	}

	public void segmentObjects()
	{
		ObjectSegmentation objectSegmentation = new ObjectSegmentation( this );

		SegmentedObjects objects = objectSegmentation.runFromUI( );

		if ( objects != null )
		{
			addObjects( objects );
		}

	}


	private static void sleep()
	{
		try
		{
			Thread.sleep( 300 );
		}
		catch ( InterruptedException e )
		{
			e.printStackTrace();
		}
	}

	private Map< String, SegmentedObjects > segmentedObjectsMap;

	private ResultImage resultImageBgFg = null;

	/** features to be used in the instances */
	//private FeatureImagesMultiResolution  featureImages = null;
	/**
	 * set of instancesComboBox from the user's traces
	 */
	private Instances trainingData = null;
	/**
	 * current classifier
	 */
	public FastRandomForest classifier = null;
	/**
	 * train header
	 */
	private Instances trainHeader = null;
	/**
	 * default classifier (Fast Random Forest)
	 */
	private FastRandomForest rf;

	/**
	 * current number of trees in the fast random forest classifier
	 */
	public int classifierNumTrees = 200;
	/**
	 * number of random features per node in the fast random forest classifier
	 */
	private int classifierNumRandomFeatures = 50;

	public String classifierBatchSizePercent = "66";

	/**
	 * fraction of random features per node in the fast random forest classifier
	 */
	public double classifierFractionFeaturesPerNode = 0.1;

	/**
	 * maximum depth per tree in the fast random forest classifier
	 */
	public int classifierMaxDepth = 0;


	public boolean debugLogLabelPixelValues = false;
    public boolean computeExampleFeatureValuesAtMultipleRegionOffest = false;
    public String featureImageToBeShown = "";
    public boolean debugUseWholeImageForFeatureComputation = false;

    /**
	 * list of class names on the loaded data
	 */
	private ArrayList<String> loadedClassNames = null;

	private boolean updateFeatures = false;

	public int[] minTileSizes = new int[]{162, 162, 81};

	public String tileSizeSetting = "auto";

	public FeatureSettings featureSettings = new FeatureSettings();

	public String featureSelectionMethod = FEATURE_SELECTION_RELATIVE_USAGE;

	public double featureSelectionValue = 1.0;

	private boolean computeFeatureImportance = false;

	public void setNumThreads( int numThreads )
	{
		this.numThreads = numThreads;
		threadsRegion = (int) ( Math.sqrt( numThreads ) + 0.5 );
		threadsPerRegion = (int) ( Math.sqrt( numThreads ) + 0.5 );
		threadsClassifierTraining = numThreads;
	}

	public int numThreads;
	public int threadsRegion;
	public int threadsPerRegion;
	public int threadsClassifierTraining;


	public int tilingDelay = 2000; // milli-seconds

	public double uncertaintyLutDecay = 0.5;

	public double accuracy = 4.0;

	public double memoryFactor = 10.0;

	public static final IJLazySwingLogger logger = new IJLazySwingLogger();

	private ArrayList<UncertaintyRegion> uncertaintyRegions = new ArrayList<>();

	public boolean isTrainingCompleted = true;

	public int maximumMultithreadedLevel = 10;

	public int getLabelImageTrainingIteration()
	{
		return labelImageTrainingIteration;
	}

	public void setLabelImageTrainingIteration( int labelImageTrainingIteration )
	{
		this.labelImageTrainingIteration = labelImageTrainingIteration;
	}

	private int labelImageTrainingIteration = 2;

	public Logger getLogger()
	{
		return logger;
	}

	public ImagePlus getInputImage()
	{
		return inputImage;
	}


	public File getInputImageFile()
    {
        FileInfo fileInfo = inputImage.getOriginalFileInfo();

        File inputImageFile =  new File(fileInfo.directory + File.separator + fileInfo.fileName );

        return inputImageFile;
    }

	/**
	 * @param logFileName
	 * @param logFileDirectory
	 * @return returns true if the log file could be sucessfully created
	 */

	public boolean setAndCreateLogDirRelative( String directory )
	{

		String logFileDirectory = directory.replaceAll( "/$", "" ) + "--log";
		String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new Date());
		String logFileName = "log-" + timeStamp + ".txt";

		logger.setLogFileNameAndDirectory( logFileName, logFileDirectory );

		logger.isFileLogging = true;

		return (true);
	}

	public boolean setAndCreateLogDirAbsolute( String directory )
	{
		String logFileDirectory = directory;
		String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").
				format(new Date());
		String logFileName = "log-" + timeStamp + ".txt";

		logger.setLogFileNameAndDirectory( logFileName, logFileDirectory );

		logger.isFileLogging = true;

		return (true);
	}


	public boolean getComputeFeatureImportance()
	{
		return computeFeatureImportance;
	}

	double getAvgExampleLength()
	{
		double totalLength = 0;

		for ( Example example : examples )
		{
			totalLength += example.points.length;
		}

		return totalLength / getNumExamples();
	}

	public void setBatchSizePercent( String batchSizePercent )
	{
		if ( batchSizePercent.equals( "Auto" ) || batchSizePercent.equals( "auto" )  )
		{
			classifierBatchSizePercent = "auto";
		}
		else
		{
			try
			{
				classifierBatchSizePercent = "" + Integer.parseInt( batchSizePercent );
			}
			catch ( NumberFormatException e )
			{
				logger.error( "Batch size must be a number (1-100) or 'auto'" +
						"\nSetting to 'auto' now." );
				classifierBatchSizePercent = "auto";
			}
		}

	}

	public int getBatchSizePercent( )
	{
		int avgNumPointsFromEachExamplePerTree = 5;

		if ( classifierBatchSizePercent.equals("auto") && getNumExamples() > 0 )
		{
			logger.info( "\n# Auto setting batch size..." );

			double avgExampleLength = getAvgExampleLength();
			logger.info( "Average example length: " + avgExampleLength );
			double batchSizePercent =
					avgNumPointsFromEachExamplePerTree
							* ( 100.0 / avgExampleLength ) ;
			logger.info( "Batch size [%]: " + batchSizePercent );

			return Math.max( 1, (int) batchSizePercent );

		}
		else
		{
			return Integer.parseInt( classifierBatchSizePercent );
		}
	}


	public boolean recomputeLabelInstances = false;

	public void setImageBackground( int background )
	{
		if ( background != featureSettings.imageBackground && getNumExamples() > 0 )
		{
			logger.warning( "Image background value has changed. " +
					"Feature values for labels will thus be recomputed during " +
					"next update." );
			recomputeLabelInstances = true;
		}
		featureSettings.imageBackground = background;
	}

	public int getImageBackground()
	{
		return featureSettings.imageBackground;
	}


	private String imagingModality = de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin.FLUORESCENCE_IMAGING;

	public AtomicInteger totalThreadsExecuted = new AtomicInteger(0);

	public AtomicLong pixelsClassified = new AtomicLong(0);

	public AtomicLong rfStatsTreesEvaluated = new AtomicLong(0);

	private AtomicInteger classifierStatsMaximumTreesUsed = new AtomicInteger(0);

	public boolean hasTrainingData()
	{
		if ( trainingData != null )
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	public final static String START_NEW_INSTANCES = "Start new instances";
	public final static String APPEND_TO_PREVIOUS_INSTANCES = "Append to previous instances";

	public void trainFromLabelImage(
			String instancesKey,
			String modality,
			int numIterations,
			int zChunkSize,
			int nxyTiles,
			int localRadius,
			int numInstanceSetsPerTilePlaneClass,
			long maxNumInstances,
			int minNumInstancesBeforeNewTraining,
			int numTrainingTrees,
			int numClassificationTrees,
			int minNumVoxels,
			ArrayList< Double > classWeights,
			String directory,
			FinalInterval trainInterval,
			FinalInterval applyInterval )
	{

	    int labelChannel = 1; // TODO: make chooseable

		if ( trainInterval == null ) return;

		featureSelectionMethod = FEATURE_SELECTION_NONE;
		classifierBatchSizePercent = "66";
		threadsClassifierTraining = zChunkSize;

		// TODO: obtain from label image
		featureSettings.classNames = new ArrayList<>();
		featureSettings.classNames.add( "label_im_class_0" );
		featureSettings.classNames.add( "label_im_class_1" );

		FeatureProvider featureProvider = null;
		ArrayList< long[][] > labelImageClassificationAccuraciesHistory = null;
		ArrayList< Integer > numInstancesHistory = null;

		ImageProcessor ipLabelImageInstancesDistribution = null;

        boolean isFirstTime = true;
		int numTrueObjects = -1;
		String classifierKey = null;

		ArrayList< FinalInterval > tiles = getXYTiles( trainInterval, nxyTiles, getInputImageDimensions() );

		long numInstances = 0;
		long numInstancesAtLastTraining = 0;


		iterationLoop:
		for ( int i = 0; i < numIterations; ++i )
		{
			for ( int iTile = 0; iTile < tiles.size(); ++iTile )
			{

				FinalInterval tile = tiles.get( iTile );

				if ( featureProvider == null || tiles.size() > 1 )
				{
					logger.info( "\n# Computing image features for tile " + ( iTile + 1 ) );
					logInterval( tile );
					featureProvider = new FeatureProvider( this );
					featureProvider.isLogging( true );
					featureProvider.setInterval( tile );
					featureProvider.setCacheSize( zChunkSize );
					featureProvider.computeFeatures( numThreads );
				}

				int numSliceChunks = ( int ) tile.dimension( Z ) / zChunkSize;
				int numSlicesInCurrentChunk;
				int numThreadsPerSlice;
				ExecutorService exe;
				long[] zChunk;

				ResultImageFrameSetter resultImageFrameSetter = resultImage.getFrameSetter( tile );

				ArrayList< long[] > zChunks = getZChunks( numSliceChunks, tile );

				for ( int iChunk = 0; iChunk < zChunks.size(); ++iChunk )
				{

					zChunk = zChunks.get( iChunk );

					logger.info( "\n# Instances from label image"
							+ ", tile " + (iTile+1) + "/" + tiles.size()
							+ ", iteration " + (i+1) + "/" + numIterations
							+ ", zMin " + zChunk[0] + ", zMax " + zChunk[1] );

					if ( stopCurrentTasks ) return;

					numSlicesInCurrentChunk = ( int ) ( zChunk[ 1 ] - zChunk[ 0 ] + 1 );
					numThreadsPerSlice = 1;

					// create new instances
					exe = Executors.newFixedThreadPool( numThreads );
					ArrayList< Future<InstancesAndMetadata> > futures = new ArrayList<>();

					for ( int z = ( int ) zChunk[ 0 ]; z <= zChunk[ 1 ]; ++z )
					{

						FinalInterval newInstancesInterval = fixDimension( tile, Z, z );

						futures.add(
								exe.submit(
										InstancesUtils.getUsefulInstancesFromLabelImage(
												this,
												inputImage,
												labelChannel,
												getResultImage(),
												ipLabelImageInstancesDistribution,
												featureProvider,
												instancesKey,
												newInstancesInterval,
												numInstanceSetsPerTilePlaneClass,
												numThreadsPerSlice,
												localRadius,
												classWeights )
								)
						);

					}

					for ( Future<InstancesAndMetadata> future : futures )
					{

						InstancesAndMetadata instancesAndMetadata = null;

						try
						{
							instancesAndMetadata = future.get();
						}
						catch ( InterruptedException e )
						{
							e.printStackTrace();
						}
						catch ( ExecutionException e )
						{
							e.printStackTrace();
						}

						if ( isFirstTime && modality.equals( START_NEW_INSTANCES ) )
						{
							instancesKey = instancesManager.putInstancesAndMetadata( instancesAndMetadata );
							isFirstTime = false;
						}
						else
						{
							instancesManager.getInstancesAndMetadata( instancesKey ).append( instancesAndMetadata );
						}


					}

					futures = null;
					exe.shutdown();


					// Train classifier
					//

					InstancesAndMetadata instancesAndMetadata = getInstancesManager().getInstancesAndMetadata(instancesKey);

					numInstances = instancesAndMetadata.getInstances().numInstances();

					logger.info( "Number of instances: " + numInstances );

					boolean enoughNewInstancesForNewTraining =
							( numInstances - numInstancesAtLastTraining ) >= minNumInstancesBeforeNewTraining;

					if ( classifierKey == null || enoughNewInstancesForNewTraining )
					{
						numInstancesAtLastTraining = numInstances;
						classifierNumTrees = numTrainingTrees;
						classifierKey = trainClassifier( instancesAndMetadata );
					}

					// Apply classifier to next chunk
					//

					applyLabelImageTrainingClassifierToNextChunk( featureProvider, classifierKey, resultImageFrameSetter, zChunks, iChunk );

					numInstances = instancesManager.getInstancesAndMetadata( instancesKey ).getInstances().numInstances();

					if ( numInstances >= maxNumInstances ) break;


				} // chunks

				numInstances = saveLabelImageTrainingCurrentInstances( instancesKey, directory );

				if ( numInstances >= maxNumInstances ) break;


			} // tiles

			classifierKey = evaluateCurrentAccuracies( instancesKey, numClassificationTrees, applyInterval, i );

			if ( numInstances >= maxNumInstances ) break;

		} // iterations

		logger.info( "\n\n# Training from label image: DONE.");

	}

	private void applyLabelImageTrainingClassifierToNextChunk( FeatureProvider featureProvider, String classifierKey, ResultImageFrameSetter resultImageFrameSetter, ArrayList< long[] > zChunks, int iChunk )
	{

		int iClassificationChunk = iChunk < ( zChunks.size() - 1 ) ? iChunk + 1 : 0;
        long[] zChunk = zChunks.get( iClassificationChunk );

        int numSlicesInCurrentChunk = ( int ) ( zChunk[ 1 ] - zChunk[ 0 ] + 1 );
        int numThreadsPerSlice = 1;

        ExecutorService exe = Executors.newFixedThreadPool( numThreads );
		ArrayList< Future > classificationFutures = new ArrayList<>();

		logger.info( "\n# Applying classifier to: zMin " + zChunk[ 0 ]
                + "; zMax " + zChunk[ 1 ]
                + "; using " + numThreads + " workers." );

		for ( int z = ( int ) zChunk[ 0 ]; z <= zChunk[ 1 ]; ++z )
        {

            classificationFutures.add(
                    exe.submit(
                            classifyZChunk(
                                    featureProvider,
                                    resultImageFrameSetter,
                                    z, z,
                                    getClassifierManager().getInstancesHeader( classifierKey ),
                                    getClassifierManager().getClassifier( classifierKey ),
                                    accuracy,
                                    numThreadsPerSlice,
                                    false
                            )
                    )
            );
        }

		ThreadUtils.joinThreads( classificationFutures, logger, exe );

	}

	private long saveLabelImageTrainingCurrentInstances( String instancesKey, String directory )
	{
		long numInstances;
		numInstances = instancesManager
                .getInstancesAndMetadata( instancesKey )
                .getInstances().numInstances();

		if( directory != null )
        {
            logger.info( "\n# Saving instances..." );
            InstancesUtils.saveInstancesAndMetadataAsARFF( instancesManager.getInstancesAndMetadata( instancesKey ), directory, "Instances-" + numInstances + ".ARFF" );
            logger.info( "...done" );
        }
		return numInstances;
	}

	private String evaluateCurrentAccuracies( String instancesKey, int numClassificationTrees, FinalInterval applyInterval, int i )
	{

		logger.info( "\n#" +
                "\n# --------------------------------------------------------------------------------------------------------" +
                "\n# ----------------------------------- Evaluate current accuracy ------------------------------------------" +
                "\n# --------------------------------------------------------------------------------------------------------" +
                "\n#" );


		// Classify everything, properly
		//
		InstancesAndMetadata instancesAndMetadata = getInstancesManager().getInstancesAndMetadata( instancesKey );

		classifierNumTrees = numClassificationTrees;
		String classifierKey = trainClassifier( instancesAndMetadata );

		applyClassifierWithTiling( classifierKey, applyInterval );

		// Report results from training region
		//
			/*
			if ( numTrueObjects == -1 )
			{
				int t = 0;
				ImagePlus classLabelMask = computeClassLabelMask(
						labelImage, t,
						10, 1, 1 );
				ResultsTable rtTruth = GeometricMeasures3D.volume( classLabelMask.getStack(), new double[]{ 1, 1, 1 } );
				numTrueObjects = rtTruth.size();
			}
			*/

		//logger.info( "\n# True number of segmentedObjectsMap in training image: " + numTrueObjects);

		//segmentObjects( minNumVoxels, 0 , "" + i + "-train" , directory );

		// Report results from test image
		//
		// segmentObjects( minNumVoxels, 1, ""+i+"-test" , directory );


		computeLabelImageBasedAccuracies( "" + i + "-accuracies-train" , 1, getInterval( getInputImage() ) );


		return classifierKey;
	}


	private void analyzeObjects( int minNumVoxels, int t, String title, String directory )
	{
		ImagePlus twoClassImage = computeTwoClassImage( resultImage, t );
		twoClassImage.setTitle( title + "-bgfg" );
		//twoClassImage.show();
		saveImage( twoClassImage, directory );

		ImagePlus classLabelMask = computeClassLabelMask( twoClassImage, t, minNumVoxels, 11, 20 );
		classLabelMask.setTitle( title + "-labels" );
		//classLabelMask.show();

		ResultsTable rt = GeometricMeasures3D.volume( classLabelMask.getStack() , new double[]{1,1,1});
		logger.info( "\n# Classified number of segmentedObjectsMap in " + title + ": " + rt.size() );
	}

	private void saveImage( ImagePlus imp, String directory )
	{
		if( directory != null )
		{
			FileSaver fileSaver = new FileSaver( imp );
			fileSaver.saveAsTiff( directory + File.separator + imp.getTitle() + ".tif" );
		}

	}

	public void computeLabelImageBasedAccuracies( String title,
                                                  int labelChannel,
                                                  FinalInterval interval )
	{


		for ( long t = interval.min( T ); t <= interval.max( T ); ++t )
        {
            ImagePlus accuraciesImage = IntervalUtils.createImagePlus( interval );

            long[][][] accuracies = AccuracyEvaluation.computeLabelImageBasedAccuracies(
                    inputImage,
                    getResultImage(),
                    labelChannel,
                    accuraciesImage,
                    interval,
                    (int) t,
                    this
            );

            AccuracyEvaluation.reportClassificationAccuracies( accuracies, (int) t, logger );

            //accuraciesImage.setTitle( title + "_t" + t );
            //accuraciesImage.show();
        }

	}



	private ArrayList<Integer> featuresToShow = null;

	public boolean stopCurrentTasks = false;

	public boolean isBusy = false;

	private int currentUncertaintyRegion = 0;

	private ImagePlus labelImage = null;

	private InstancesManager instancesManager = new InstancesManager();

	public ClassifierManager getClassifierManager()
	{
		return classifierManager;
	}

	private ClassifierManager classifierManager = new ClassifierManager();

	public String getInputImageTitle()
	{
		return ( inputImage.getTitle() );
	}

	/**
	 * Default constructor.
	 *
	 * @param trainingImage The image to be segmented/trained
	 */
	public DeepSegmentation( ImagePlus trainingImage )
	{
		initialize();
		setInputImage(trainingImage);
	}

	public long[] getInputImageDimensions()
	{
		long[] dimensions = new long[5];
		IntervalUtils.getInterval(  inputImage ).dimensions( dimensions );
		return dimensions;
	}

	/**
	 * No-image constructor. If you use this constructor, the image has to be
	 * set using setInputImage().
	 */
	public DeepSegmentation()
	{
		initialize();
	}

	private void initialize()
	{

		// set class label names
		char[] alphabet = "abcdefghijklmnopqrstuvwxyz".toCharArray();

		featureSettings.classNames = new ArrayList<>();
		featureSettings.classNames.add(new String("background"));
		featureSettings.classNames.add(new String("foreground"));

		// Initialization of Fast Random Forest classifier
		rf = new FastRandomForest();
		rf.setNumTrees( classifierNumTrees );
		rf.setNumFeatures( classifierNumRandomFeatures );
		rf.setSeed((new Random()).nextInt());
		rf.setNumThreads( threadsClassifierTraining );
		rf.setMaxDepth( classifierMaxDepth );
		//rf.setBatchSize("50");
		rf.setComputeImportances(true);
		classifier = rf;

		// initialize the examples
		examples = new ArrayList<Example>();

		maxMemory = IJ.maxMemory();

		setNumThreads( Prefs.getThreads() );

	}




	/**
	 * Set the instances image (single image or stack)
	 *
	 * @param imp instances image
	 */
	public void setInputImage( ImagePlus imp )
	{
		if ( imp == null )
		{
			logger.error( "Input image is NULL." );
			return;
		}

		this.inputImage = imp;


		Calibration calibration = inputImage.getCalibration();

        featureSettings.anisotropy = 1.0 * calibration.pixelDepth / calibration.pixelWidth;

        if( calibration.pixelWidth != calibration.pixelHeight )
        {
            logger.error("Image calibration in x and y is not the same; currently cannot take this into " +
                    "account; but you can still use this plugin, may work anyway...");
        }

        Set< Integer > channelsToConsider = new TreeSet<>();
        for ( int c = 0; c < inputImage.getNChannels(); c++ )
        {
            channelsToConsider.add(c); // zero-based
        }

        featureSettings.activeChannels = channelsToConsider;

    }

	/**
	 * Adds a ROI to the list of examples for a certain class
	 * and slice.
	 *
	 * @param classNum the number of the class
	 * @param roi      the ROI containing the new example
	 * @param n        number of the current slice
	 */
	public void addExample(Example example)
	{
		examples.add(example);
	}

	public Example createExample(int classNum, Point[] points, int strokeWidth, int z, int t)
	{
		Example example = new Example(classNum, points, strokeWidth, z, t);
		return (example);
	}

	/**
	 * Remove an example list from a class and specific slice
	 *
	 * @param classNum the number of the examples' class
	 * @param nSlice   the slice number
	 * @param index    the index of the example list to remove
	 */
	public void removeExample( int classNum, int z, int t, int index )
	{
		int i = 0;
		for (int iExample = 0; iExample < examples.size(); iExample++)
		{
			Example example = examples.get(iExample);
			if ((example.z == z)
					&& (example.t == t)
					&& (example.classNum == classNum))
			{
				if ( index == i++ ) // i'th example for this z,t,class
				{
					examples.remove( iExample );
					return;
				}
			}

		}

	}

	public Rectangle getExampleRectangleBounds(Example example)
	{
		int xMin = example.points[0].x;
		int xMax = example.points[0].x;
		int yMin = example.points[0].y;
		int yMax = example.points[0].y;

		for (Point point : example.points)
		{
			xMin = point.x < xMin ? point.x : xMin;
			xMax = point.x > xMax ? point.x : xMax;
			yMin = point.y < yMin ? point.y : yMin;
			yMax = point.y > yMax ? point.y : yMax;
		}

		xMin -= example.strokeWidth + 2; // +2 just to be on the save side
		xMax += example.strokeWidth + 2;
		yMin -= example.strokeWidth + 2;
		yMax += example.strokeWidth + 2;

		return (new Rectangle(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1));
	}

	/**
	 * Check whether the example is valid, i.e.
	 * - not too close to the image boundary
	 *
	 * @param example
	 * @return
	 */
	/*
	public boolean isValidExample( Example example )
	{
		int[][] bounds = getExampleBoundingInterval( example );
		int[][] borders = getClassifiableImageBorders();

		for ( int i = 0; i < 3; ++i )
		{
			if ( bounds[i][0] < borders[i][0] ) return false;
		}

		for ( int i = 0; i < 3; ++i )
		{
			if ( bounds[i][1] > borders[i][1] ) return false;
		}

		return true;

	}*/

	/**
	 * Return the list of examples for a certain class.
	 *
	 * @param classNum the number of the examples' class
	 * @param n        the slice number
	 */
	public ArrayList<Roi> getLabelRois( int classNum, int z, int t)
	{
		ArrayList<Roi> rois = new ArrayList<>();

		for (Example example : examples)
		{
			if ((example.z == z)
					&& (example.t == t)
					&& (example.classNum == classNum))
			{
				float[] x = new float[example.points.length];
				float[] y = new float[example.points.length];
				for (int iPoint = 0; iPoint < example.points.length; iPoint++)
				{
					x[iPoint] = (float) example.points[iPoint].getX();
					y[iPoint] = (float) example.points[iPoint].getY();
				}
				Roi roi = new PolygonRoi(x, y, PolygonRoi.FREELINE);
				roi.setStrokeWidth((double) example.strokeWidth);
				rois.add(roi);
			}

		}
		return rois;
	}

	public ArrayList<Example> getExamples()
	{
	    return examples;
	}

	public int getNumExamples()
	{
		if ( examples == null ) return 0;
		else return examples.size();
	}

	public void setExamples( ArrayList<Example> examples )
	{
		this.examples = examples;
	}

	public void setLabelROIs(ArrayList<Example> examples)
	{
		this.examples = examples;
	}

	/**
	 * Get the current number of classes.
	 *
	 * @return the current number of classes
	 */
	public int getNumClasses()
	{
		return featureSettings.classNames.size();
	}

	/**
	 * Set the name of a class.
	 *
	 * @param classNum class index
	 * @param label    new name for the class
	 */
	public void setClassLabel(int classNum, String label)
	{
		getClassNames().set(classNum, label);
	}

	/**
	 * Get the label name of a class.
	 *
	 * @param classNum class index
	 */
	public String getClassName( int classNum )
	{
		return getClassNames().get( classNum );
	}


	/**
	 * Get the label name of a class.
	 *
	 * @param classNum class index
	 */
	public void addClass(String className)
	{
        if( getNumClasses() == MAX_NUM_CLASSES )
        {
            IJ.showMessage("Sorry...", "Maximum number of classes has been reached. Class could not be added");
            return;
        }

		featureSettings.classNames.add(className);
	}


    public boolean changeClassNamesDialog()
    {
        GenericDialogPlus gd = new GenericDialogPlus("Class names");

        for( int i = 0; i < getNumClasses(); i++)
            gd.addStringField("Class "+(i+1), getClassName(i), 15);

        gd.showDialog();

        if ( gd.wasCanceled() )
            return false;

        for( int i = 0; i < getNumClasses(); i++)
        {
            String s = gd.getNextString();
            if (null == s || 0 == s.length()) {
                IJ.log("Invalid name for class " + (i+1));
                continue;
            }
            s = s.trim();
            if( ! s.equals( getClassName(i) ) )
            {
                if (0 == s.toLowerCase().indexOf("add to "))
                    s = s.substring(7);

                setClassLabel(i, s);
            }
        }

        return true;
    }


	public void setResultImage( ResultImage resultImage )
	{
		this.resultImage = resultImage;
	}

	public void setResultImageDisk( String directory )
	{
		ResultImage resultImage = new ResultImageDisk( this, directory, getInputImageDimensions() );
		setResultImage( resultImage );

		// setAndCreateLogDirRelative( directory ); // this is slow...

		logger.info("Created disk-resident classification resultImagePlus image: " + directory);
	}

	public void setResultImageRAM( )
	{
		ResultImage resultImage = new ResultImageRAM( this, getInputImageDimensions() );
		setResultImage( resultImage );
	}

    public void setResultImageRAM( FinalInterval interval )
    {
        ResultImage resultImage = new ResultImageRAM( this, interval );
        setResultImage( resultImage );
    }


    public ResultImage getResultImage()
	{
		return ( resultImage );
	}

    public void assignResultImage( String resultImageType )
    {
        this.resultImageType = resultImageType;

        if ( resultImageType.equals( RESULT_IMAGE_DISK_SINGLE_TIFF ) )
        {
            String directory = IJ.getDirectory("Select directory with pixel probabilities");

            if( directory == null )
            {
                logger.error( "No resultImagePlus image was assigned now.\n " +
                        "You can later click [FeatureSettings] and assign one." );
                return;
            }

            setResultImageDisk( directory  );


        }
        else if ( resultImageType.equals( RESULT_IMAGE_RAM ))
        {

            setResultImageRAM();

            logger.info("Allocated memory for result image." );

        }

    }

    public boolean featureSettingsDialog( boolean showAdvancedSettings )
    {
        GenericDialogPlus gd = new GenericDialogPlus("Image Feature Settings");

        for ( int i = 0; i < 5; ++i )
        {
            gd.addNumericField( "Binning " + ( i + 1 ), featureSettings.binFactors.get( i ), 0 );
        }

        gd.addNumericField("Maximal convolution depth", featureSettings.maxDeepConvLevel, 0);
        gd.addNumericField("z/xy anisotropy", featureSettings.anisotropy, 10);
        gd.addStringField("Feature computation: Channels to consider (one-based) [ID,ID,..]",
                FeatureSettings.getAsCSVString( featureSettings.activeChannels, 1 ) );

        if ( showAdvancedSettings )
        {
            gd.addStringField( "Bounding box offsets ",
                    FeatureSettings.getAsCSVString( featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels, 0 ) );

            gd.addChoice( "Downsampling method", new String[]{
                            DownSampler.BIN_AVERAGE,
                            DownSampler.BIN_MAXIMUM,
                            DownSampler.TRANSFORMJ_SCALE_LINEAR,
                            DownSampler.TRANSFORMJ_SCALE_CUBIC },
                    DownSampler.getString( featureSettings.downSamplingMethod ) );

            gd.addStringField( "Smoothing sigmas [pixels x/y] ",
                    FeatureSettings.getAsCSVString( featureSettings.smoothingScales, 0 ) );

            gd.addCheckbox( "Compute Gaussian filter", featureSettings.computeGaussian );

            gd.addCheckbox( "Use log2 transformation", featureSettings.log2 );

            gd.addCheckbox( "Consider multiple bounding box offsets during loading", considerMultipleBoundingBoxOffsetsDuringInstancesLoading );

        }

        gd.showDialog();

        if ( gd.wasCanceled() )
            return false;

        FeatureSettings newFeatureSettings = getFeatureSettingsFromGenericDialog( gd, showAdvancedSettings );
        boolean settingsChanged = ! featureSettings.equals( newFeatureSettings );
        featureSettings = newFeatureSettings;

        if ( settingsChanged )
        {
            if ( getInstancesManager().getNumInstancesSets() > 0 )
            {
                recomputeLabelFeaturesAndRetrainClassifier();
            }
        }

        return true;
    }


    public void recomputeLabelFeaturesAndRetrainClassifier()
    {
        recomputeLabelInstances = true;

        updateLabelInstancesAndMetadata();

        trainClassifierWithFeatureSelection( getCurrentLabelInstancesAndMetadata() );
    }

    public void updateLabelInstancesAndTrainClassifier()
    {
        recomputeLabelInstances = false;

        updateLabelInstancesAndMetadata();

		trainClassifierFromCurrentLabelInstances();

    }

    public void trainClassifierFromCurrentLabelInstances()
	{
		trainClassifierWithFeatureSelection( getCurrentLabelInstancesAndMetadata() );
	}

	public void updateLabelInstances()
	{
		recomputeLabelInstances = false;

		updateLabelInstancesAndMetadata();
	}


    public FeatureSettings getFeatureSettingsFromGenericDialog( GenericDialogPlus gd, boolean showAdvancedSettings )
    {
        FeatureSettings newFeatureSettings = featureSettings.copy();

        for ( int i = 0; i < 5; ++i )
        {
            newFeatureSettings.binFactors.set( i, (int) gd.getNextNumber() );
        }

        newFeatureSettings.maxDeepConvLevel = (int) gd.getNextNumber();
        newFeatureSettings.anisotropy = gd.getNextNumber();
        newFeatureSettings.setActiveChannels( gd.getNextString() );

        if ( showAdvancedSettings )
        {
            newFeatureSettings.setBoundingBoxExpansionsForGeneratingInstancesFromLabels( gd.getNextString() );
            newFeatureSettings.downSamplingMethod = DownSampler.getID( gd.getNextChoice() );
            newFeatureSettings.setSmoothingScales( gd.getNextString() );
            newFeatureSettings.computeGaussian = gd.getNextBoolean();
            newFeatureSettings.log2 = gd.getNextBoolean();
            considerMultipleBoundingBoxOffsetsDuringInstancesLoading = gd.getNextBoolean();

        }

        return newFeatureSettings;
    }

    public boolean initialisationDialog(  )
    {

        IJ.run( inputImage, "Properties...", "");

        GenericDialog gd = new NonBlockingGenericDialog("Set up");

        gd.addMessage( "DATA SET NAME\n \n" +
                "Please enter/confirm the name of this data set.\n" +
                "This is important for keeping track of which instances have been trained with which image." );
        gd.addStringField( "Name", inputImage.getTitle(), 50 );

        gd.addMessage( "RESULT IMAGE\n \n" +
                "For large data sets it can be necessary to store the results " +
                "on disk rather than in RAM.\n" +
                "The speed of this plugin does hardly depend on this choice.\n" +
                "If you choose 'Disk' a dialog window will appear to select the storage directory.\n" +
                "You can point to a directory containing previous segmentation results and they will be loaded (not overwritten)." );

        gd.addChoice( "Location" ,
                new String[]{
                        RESULT_IMAGE_DISK_SINGLE_TIFF,
                        RESULT_IMAGE_RAM },
                        RESULT_IMAGE_RAM );


        gd.showDialog();

        if ( gd.wasCanceled() ) return false;

        inputImage.setTitle( gd.getNextString()  );
        assignResultImage( gd.getNextChoice() );

        return true;
    }



    public void reviewObjects( )
    {
        ObjectReview objectReview = new ObjectReview( this );
        objectReview.runUI( );
    }

    public boolean showClassifierSettingsDialog()
    {
        GenericDialogPlus gd = new GenericDialogPlus("Classifier featureSettings");

        gd.addNumericField("Number of trees",
                classifierNumTrees, 0);

        gd.addStringField("Batch size per tree in percent", classifierBatchSizePercent) ;

        gd.addNumericField("Fraction of random features per node", classifierFractionFeaturesPerNode, 2);

        gd.addChoice("Feature selection method", new String[]
                        {
                                FEATURE_SELECTION_RELATIVE_USAGE,
                                FEATURE_SELECTION_ABSOLUTE_USAGE,
                                FEATURE_SELECTION_TOTAL_NUMBER,
                                FEATURE_SELECTION_NONE,
                        },
                FEATURE_SELECTION_RELATIVE_USAGE );

        gd.addNumericField("Feature selection value", featureSelectionValue, 1);

        gd.showDialog();

        if ( gd.wasCanceled() )
            return false;

        // Set classifier and options
        classifierNumTrees = (int) gd.getNextNumber();
        setBatchSizePercent( gd.getNextString() );
        classifierFractionFeaturesPerNode = (double) gd.getNextNumber();
        featureSelectionMethod = gd.getNextChoice();
        featureSelectionValue = gd.getNextNumber();

        return true;
    }


    /**
	 * bag class for getting the resultImagePlus of the loaded classifier
	 */
	private static class LoadedProject {
		private AbstractClassifier newClassifier = null;
		private FeatureSettings newFeatureSettings = null;
		private ArrayList<Example> newExamples = null;
	}


	/**
	 * Returns the current classifier.
	 */
	public AbstractClassifier getClassifier()
	{
		return classifier;
	}

	/**
	 * Write current project into a file
	 *
	 * @param filename name (with complete path) of the destination file
	 * @return false if error
	 */
	public boolean saveProject(String filename)
	{
		File sFile = null;
		boolean saveOK = true;

		logger.info("Saving project to disk...");

		try
		{
			sFile = new File(filename);
			OutputStream os = new FileOutputStream(sFile);
			if (sFile.getName().endsWith(".gz"))
			{
				os = new GZIPOutputStream(os);
			}
			ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
			objectOutputStream.writeObject(classifier);
			objectOutputStream.writeObject( featureSettings );
			objectOutputStream.writeObject(getExamples());
			objectOutputStream.flush();
			objectOutputStream.close();
		}
		catch (Exception e)
		{
			IJ.error("Save Failed", "Error when saving project to disk");
			logger.info(e.toString());
			saveOK = false;
		}
		if (saveOK)
		{
			IJ.log("Saved project to " + filename);
		}

		return saveOK;
	}

	/**
	 * Set current classifier
	 *
	 * @param cls new classifier
	 */
	public void setClassifier( FastRandomForest cls )
	{
		this.classifier = cls;
	}

	/**
	 * Set current featureSettings
	 *
	 * @param featureSettings
	 */
	public void setFeatureSettings( FeatureSettings featureSettings )
	{
		this.featureSettings = featureSettings;
	}

	public void loadClassifier( File file )
	{
		loadClassifier( file.getParent().toString(),  file.getName().toString() );
	}

	public void loadClassifier( String filePath )
	{
		Path p = Paths.get(filePath);
		loadClassifier( p.getParent().toString(),  p.getFileName().toString());
	}

	public void loadClassifier( String directory, String filename )
	{
		ClassifierInstancesMetadata classifierInstancesMetadata = ClassifierUtils.loadClassifierInstancesMetadata( directory, filename );
		SettingsUtils.setSettingsFromInstancesMetadata( featureSettings, classifierInstancesMetadata.instancesAndMetadata );
		classifierManager.setClassifier( classifierInstancesMetadata );
	}


	public void saveClassifier( String directory, String filename )
	{
		String key = classifierManager.getMostRecentClassifierKey();
		classifierManager.saveClassifier( key, directory, filename  );
	}


	/**
	 * Balance number of instancesComboBox per class
	 *
	 * @param instances input set of instancesComboBox
	 * @return resampled set of instancesComboBox
	 */
	public static Instances balanceTrainingData( Instances instances )
	{

		long[] classDistribution = InstancesUtils.getClassDistribution( instances );
		Arrays.sort( classDistribution );

		long numMissingInstances = 0;
		long mostOccuringClass = classDistribution[ classDistribution.length - 1 ];

		for ( int i = 0; i < classDistribution.length - 1; ++i )
		{
			numMissingInstances += mostOccuringClass - classDistribution[ i ];
		}

		int sampleSizePercent = (int) ( 100.0 * ( instances.size() + numMissingInstances ) / instances.size() );

		final Resample filter = new Resample();
		Instances filteredIns = null;
		filter.setBiasToUniformClass(1.0);
		try
		{
			filter.setInputFormat(instances);
			filter.setNoReplacement(false); // with replacement
			filter.setSampleSizePercent(sampleSizePercent);
			filteredIns = Filter.useFilter(instances, filter);
		}
		catch (Exception e)
		{
			IJ.log("Error when resampling input data!");
			e.printStackTrace();
		}

		return filteredIns;

	}

	/**
	 * Homogenize number of instancesComboBox per class (in the loaded instances data)
	 *
	 * @deprecated use balanceTrainingData
	 */
	public void homogenizeTrainingData()
	{
		balanceTrainingData();
	}

	private boolean isUpdatedFeatureList = false;

	/**
	 * Balance number of instancesComboBox per class (in the loaded instances data)
	 */
	public void balanceTrainingData()
	{
		final Resample filter = new Resample();
		Instances filteredIns = null;
		filter.setBiasToUniformClass(1.0);
		try
		{
			filter.setInputFormat(trainingData);
			filter.setNoReplacement(false);
			filter.setSampleSizePercent(100);
			filteredIns = Filter.useFilter(trainingData, filter);
		}
		catch (Exception e)
		{
			IJ.log("Error when resampling input data!");
			e.printStackTrace();
		}
		trainingData = filteredIns;
	}


	private ArrayList< Integer > setInstancesClassesToBgFg( Instances instances )
	{
		ArrayList< Integer > originalClasses = new ArrayList< >( );

		int classIndex = instances.classIndex();

		for ( Instance instance : instances )
		{
			originalClasses.add( (int) instance.value( classIndex ) );

			if ( instance.value( classIndex ) > 0 )
			{
				instance.setValue( classIndex, 1 );
			}
		}

		return originalClasses;
	}


	private void resetInstancesClasses( Instances instances,
										ArrayList< Integer > classes )
	{
		// TODO
	}


	public void applyBgFgClassification( FinalInterval interval,
										 String key )
	{
		InstancesAndMetadata instancesAndMetadata = getInstancesManager().getInstancesAndMetadata( key );
		applyBgFgClassification( interval, instancesAndMetadata);
	}

	public void replaceByDistanceMap( ResultImageRAM resultImageMemory )
	{
		// We need to loop through the frames because the
		// 3-D distance transform code has no hyperstack logic
		for ( int frame = 1; frame <= inputImage.getNFrames(); ++frame )
		{
			ImagePlus imp = resultImageMemory.getFrame( frame );

			// IJ.run(imp, "Exact Euclidean Distance Transform (3D)", "");
			//

			// Replace values by distance map values
			//
			ImageStack distanceMap = null;
			for( int slice = 1; slice <= imp.getNSlices(); ++slice )
			{
				resultImageMemory.setProcessor(
						distanceMap.getProcessor( slice ),
						slice, frame );
			}
		}
	}

	public void applyBgFgClassification( FinalInterval interval,
										 InstancesAndMetadata instancesAndMetadata )
	{
		setResultImageBgFgRAM( );

		ArrayList< Integer > originalClasses =
				setInstancesClassesToBgFg( instancesAndMetadata.getInstances() );

		String classifierBgFgKey = trainClassifierWithFeatureSelection( instancesAndMetadata );

		resetInstancesClasses( instancesAndMetadata.getInstances(), originalClasses );

		// reroute classification to BgFg resultImagePlus image
		ResultImage originalResultImage = resultImage;
		resultImage = resultImageBgFg;

		applyClassifierWithTiling(
				classifierBgFgKey,
				interval );

		// reset resultImagePlus image
		resultImage = originalResultImage;

		// compute distance transform on whole data set
		// - it is important to do it on the whole data set, because
		// we need to recompute the examples' instance values, which
		// can be anywhere

		replaceByDistanceMap( (ResultImageRAM )resultImageBgFg );

		// Recompute instance values, now including the distance map
		//
		//featureSettings.activeChannels.remove( 0 );
		featureSettings.activeChannels.add( FeatureProvider.FG_DIST_BG_IMAGE );

		logger.info( "\n# Recomputing instances including distance image" );
		String bgFgInstancesKey = "BgFgTraining";
		recomputeLabelInstances = true;
		updateLabelInstancesAndMetadata( bgFgInstancesKey );
		recomputeLabelInstances = false;

		// Retrain classifier
		//
		String classifierKey  = trainClassifierWithFeatureSelection( bgFgInstancesKey );

		applyClassifierWithTiling( classifierKey,
				interval,
				-1,
				null,
				false );

		resultImage.getDataCubeCopy( interval ).show();






	}

	public String loadInstancesAndMetadata( String filePath )
	{
		Path p = Paths.get(filePath);

		String key = loadInstancesAndMetadata( p.getParent().toString(),  p.getFileName().toString());

		return key;
	}

	public boolean considerMultipleBoundingBoxOffsetsDuringInstancesLoading = false;

	public String loadInstancesAndMetadata( String directory, String fileName )
	{
		InstancesAndMetadata instancesAndMetadata = InstancesUtils.loadInstancesAndMetadataFromARFF( directory, fileName );

		if ( instancesAndMetadata == null )
		{
			logger.error( "Loading failed..." );
			return null;
		}

		String key = getInstancesManager().putInstancesAndMetadata( instancesAndMetadata );

		if ( InstancesUtils.getNumLabelIds( instancesAndMetadata ) > 1 && instancesAndMetadata.getRelationName().equals( inputImage.getTitle() ) )
		{
            logger.info( "# Loaded instances relation name matches image name => Populating labels..." );
            logger.info( "Creating examples from instances..." );
			setExamples( ExamplesUtils.getExamplesFromInstancesAndMetadata( instancesAndMetadata, considerMultipleBoundingBoxOffsetsDuringInstancesLoading ) );
			examplesFeatureNames = instancesAndMetadata.getAttributeNames();
			logger.info( "..done." );
			currentLabelsInstancesKey = key;
		}
		else
        {
            logger.info( "\nLoaded instances relation name: " + instancesAndMetadata.getRelationName()
                            + "\ndoes not match image name: " + inputImage.getTitle()
                            + "\nInstances can be used for training a classifier but not for adding more annotations"
                            + " based on current image." );
        }


		SettingsUtils.setSettingsFromInstancesMetadata( featureSettings, instancesAndMetadata );

		setImageBackground( featureSettings.imageBackground );

		return key;

	}

	public boolean saveInstances( String key, String directory, String filename )
	{
		logger.info("\n# Saving instances " + key + " to " + directory + File.separator + filename );

		InstancesAndMetadata instancesAndMetadata = getInstancesManager().getInstancesAndMetadata( key );

		if ( instancesAndMetadata == null )
        {
            logger.error( "Saving instances failed." );
            return false;
        }

		boolean success = InstancesUtils.saveInstancesAndMetadataAsARFF( instancesAndMetadata, directory, filename );

		if ( success )
		{
			logger.info( "...done." );
		}
		else
		{
			logger.error( "Saving instances failed." );
		}

		return success;
	}

	public void updateLabelInstancesAndMetadata()
	{
		currentLabelsInstancesKey = getKeyFromImageTitle();
		updateLabelInstancesAndMetadata( currentLabelsInstancesKey );
	}

	private String currentLabelsInstancesKey;

	private String getCurrentLabelsInstancesKey()
	{
		return currentLabelsInstancesKey;
	}

	public InstancesAndMetadata getCurrentLabelInstancesAndMetadata()
	{
		return getInstancesManager().getInstancesAndMetadata( currentLabelsInstancesKey );
	}

	public void updateLabelInstancesAndMetadata( String instancesAndMetadataKey )
	{
		computeUpdatedExamplesInstances();
		putExamplesIntoInstancesAndMetadata( instancesAndMetadataKey );
	}

	public synchronized String putExamplesIntoInstancesAndMetadata( String instancesAndMetadataKey )
	{
		if ( getNumExamples() > 0 )
		{
			InstancesAndMetadata instancesAndMetadata =
					InstancesUtils.createInstancesAndMetadataFromExamples(
							getExamples(),
							instancesAndMetadataKey,
                            featureSettings,
							examplesFeatureNames,
							getClassNames() );

			getInstancesManager().putInstancesAndMetadata( instancesAndMetadata );

			return instancesAndMetadataKey;
		}
		else
		{
			return null;
		}

	}

	private String getKeyFromImageTitle( )
	{
		String instancesName = getInputImageTitle(); //.split( "--" )[ 0 ];

		return instancesName;
	}

	public void computeUpdatedExamplesInstances()
	{
		ArrayList< Example > examplesNeedingInstanceUpdate = getExamplesNeedingInstancesUpdate();

		ArrayList< ArrayList< Example > > groupedExamples = groupExamplesBySpatialProximity( examplesNeedingInstanceUpdate );

		updateInstancesValuesOfGroupedExamples( groupedExamples );
	}

	private void updateInstancesValuesOfGroupedExamples( ArrayList< ArrayList< Example > > groupedExamples )
	{
		ExecutorService exe = Executors.newFixedThreadPool( threadsRegion ); // because internally there are threadsPerRegion used for feature computation!
		ArrayList<Future > futures = new ArrayList<>();

		isUpdatedFeatureList = false;

		logger.info("Computing features values for " + groupedExamples.size() + " label groups using " + numThreads + " threads." );

		for (int i = 0; i < groupedExamples.size(); i++)
		{
			ArrayList<Example> neighboringExamples = groupedExamples.get(i);
			futures.add (
					exe.submit(
							setExamplesInstanceValues( neighboringExamples, i, groupedExamples.size() - 1)
					)
			);
		}

		ThreadUtils.joinThreads( futures, logger, exe );


		if ( groupedExamples.size() > 0 )
		{
			logger.info("Computed feature values for " + groupedExamples.size() + " labels; " + "total number of labels is " + getNumExamples() );
		}
	}

	private ArrayList< ArrayList< Example > > groupExamplesBySpatialProximity( ArrayList< Example > examplesNeedingInstanceUpdate )
	{
		ArrayList<ArrayList<Example>> exampleList = new ArrayList<>();

        if( debugUseWholeImageForFeatureComputation )
        {
            // put all examples into one group
            exampleList.add( examplesNeedingInstanceUpdate );
            return exampleList;
        }

		for (int iExampleWithoutFeatures = 0; iExampleWithoutFeatures < examplesNeedingInstanceUpdate.size(); iExampleWithoutFeatures++)
		{
			// figure out which examples are spatially close,
			// put them together and compute the feature images
			// for them in one go; this saves time.
			ArrayList<Example> neighboringExamples = new ArrayList<>();

			Rectangle exampleBounds = getExampleRectangleBounds( examplesNeedingInstanceUpdate.get( iExampleWithoutFeatures ) );

			Point3D exampleLocation = new Point3D(
					exampleBounds.getX(),
					exampleBounds.getY(),
					examplesNeedingInstanceUpdate.get(iExampleWithoutFeatures).z
			);

			neighboringExamples.add( examplesNeedingInstanceUpdate.get( iExampleWithoutFeatures ) );

			Boolean includeNextExample = true;

			iExampleWithoutFeatures++;

			while ( includeNextExample && ( iExampleWithoutFeatures < examplesNeedingInstanceUpdate.size() ) )
			{
				Rectangle nextExampleBounds = getExampleRectangleBounds(examplesNeedingInstanceUpdate.get(iExampleWithoutFeatures));

				Point3D nextExampleLocation = new Point3D(
						nextExampleBounds.getX(),
						nextExampleBounds.getY(),
						examplesNeedingInstanceUpdate.get(iExampleWithoutFeatures).z
				);

				if ( exampleLocation.distance( nextExampleLocation ) < getFeatureVoxelSizeAtMaximumScale() )
				{
					neighboringExamples.add( examplesNeedingInstanceUpdate.get(iExampleWithoutFeatures) );
					iExampleWithoutFeatures++;
				}
				else
				{
					includeNextExample = false;
					iExampleWithoutFeatures--;
				}

			}

			exampleList.add( neighboringExamples );

		}

		return exampleList;
	}

	private ArrayList< Example > getExamplesNeedingInstancesUpdate()
	{
		ArrayList< Example > examplesWithoutFeatures = new ArrayList<>();

		for ( Example example : examples )
		{
			if ( recomputeLabelInstances )
			{
				// add all examples to the list
				examplesWithoutFeatures.add( example );
			}
			else
			{
				// add examples that need feature re-computation
				if ( example.instanceValuesArrays == null && ! example.instanceValuesAreCurrentlyBeingComputed )
				{
					example.instanceValuesAreCurrentlyBeingComputed = true;
					examplesWithoutFeatures.add( example );
				}
			}
		}

		recomputeLabelInstances = false;

		return examplesWithoutFeatures;
	}

	public ArrayList< String > examplesFeatureNames = null;


	public ImagePlus computeTwoClassImage( ResultImage resultImage, int t )
	{
		ImageStack newStack = null;

		int probabilityRange = resultImage.getProbabilityRange();

		for ( int z = 0; z < inputImage.getNSlices(); ++z )
		{
			ImageProcessor ipResult = resultImage.getSlice( z + 1, t + 1 );
			ImageProcessor ip = ipResult.duplicate();

			for ( int x = 0; x < inputImage.getWidth(); ++x )
			{
				for ( int y = 0; y < inputImage.getHeight(); ++y )
				{

					byte value = ( byte ) ipResult.get( x, y );
					if ( value <= probabilityRange && value > 0 )
					{
						ip.set( x, y, probabilityRange - value );
					}

				}

			}
			if ( newStack == null )
			{
				newStack = new ImageStack( ip.getWidth(), ip.getHeight(), ( ColorModel ) null );
			}
			newStack.addSlice( "", ip );
		}

		return new ImagePlus("two-classes", newStack);
	}

	public boolean isExampleInstanceValuesAreCurrentlyBeingComputed()
	{
	    for ( Example example : examples )
        {
            if ( example.instanceValuesAreCurrentlyBeingComputed ) return true;
        }

        return false;
	}

	private Runnable setExamplesInstanceValues( ArrayList<Example> examples, int counter, int counterMax )
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return () -> {

            setExampleInstanceValuesAreCurrentlyBeingComputed( examples, true );

            logger.info("Label set " + (counter + 1) + "/" + (counterMax + 1) + ": " + "Computing features for " + examples.size() + " label(s)...");

            clearInstancesValues( examples );

			ArrayList< FinalInterval > exampleListBoundingIntervals = getBoundingIntervals( examples );

			for ( int iBoundingInterval = 0; iBoundingInterval < exampleListBoundingIntervals.size(); ++iBoundingInterval )
			{

			    FeatureProvider featureProvider = new FeatureProvider( this );
				featureProvider.setInterval( exampleListBoundingIntervals.get( iBoundingInterval ) );
				featureProvider.computeFeatures( threadsPerRegion );

				int nf = featureProvider.getNumAllFeatures();

				this.examplesFeatureNames = featureProvider.getAllFeatureNames();

				double[][][] featureSlice = featureProvider.getReusableFeatureSlice();

				// extract the feature values at
				// the respective z-position of each example
				for ( Example example : examples )
				{
					example.instanceValuesAreCurrentlyBeingComputed = true;

					ArrayList< double[] > instanceValuesArray = new ArrayList<>();

					int z = example.z;

					featureProvider.setFeatureSlicesValues( z, featureSlice, 1 );

					for ( Point point : example.points )
					{
						int xGlobal = ( int ) point.getX();
						int yGlobal = ( int ) point.getY();

						double[] values = new double[ nf + 1 ];
						featureProvider.setFeatureValuesAndClassIndex( values, xGlobal, yGlobal, featureSlice, example.classNum );
						instanceValuesArray.add( values );

						if ( debugLogLabelPixelValues )
						{
							logger.info( "# " + counter
									+ " x " + xGlobal
									+ " y " + yGlobal
									+ " z " + example.z
									+ " value[0] " + values[ 0 ] );
						}
					}

					example.instanceValuesArrays.add( instanceValuesArray );

				}

				// logger.info( "Bounding interval " + iBoundingInterval );

			}

            setExampleInstanceValuesAreCurrentlyBeingComputed( examples, false );

            logger.info( "Label set " + ( counter + 1 ) + "/" + ( counterMax + 1 ) + ": " + "...done" );

		};
	}

    private void setExampleInstanceValuesAreCurrentlyBeingComputed( ArrayList< Example > examples, boolean b )
    {
        for ( Example example : examples )
        {
            example.instanceValuesAreCurrentlyBeingComputed = b;
        }
    }

    private void clearInstancesValues( ArrayList< Example > examples )
	{
		for ( Example example : examples )
        {
            example.instanceValuesArrays = new ArrayList<>();
        }
	}

	private ArrayList< FinalInterval > getBoundingIntervals( ArrayList< Example > examples )
	{
		ArrayList< FinalInterval > exampleListBoundingIntervals = new ArrayList<>();
		FinalInterval exampleListBoundingInterval = getExampleListBoundingInterval( examples );

		for ( int offset : featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels )
        {
            long[] offsets = new long[5];
            offsets[ X ] = offset;
            offsets[ Y ] = offset;

            if ( inputImage.getNSlices() > 1 )
            {
                offsets[ Z ] = offset;
            }

            exampleListBoundingIntervals.add( Intervals.expand( exampleListBoundingInterval, offsets ) );
        }

        return exampleListBoundingIntervals;
	}

	private void debugInstanceValuesComputation()
	{
		// Below is some code for debugging
		//

					/*
					IJ.log(" x,y: " + x + ", " + y + "; max = " + featureSlice.length + ", " + featureSlice[0].length );
					IJ.log("");
					IJ.log(randomNum+" x,y,z global: " + point.getX() + "," + point.getY() + "," + example.z );
					IJ.log(randomNum+" x,y local: " + x + "," + y );

					String[] valueString = new String[4];
					valueString[0] = "";
					valueString[1] = "";
					valueString[2] = "";
					valueString[3] = "";
					*/


					/*if ( f < 7 )
						valueString[0] = valueString[0] + (int)Math.round(instanceValues[f]) + ", ";
					else if ( f < 7+24 )
						valueString[1] = valueString[1] + (int)Math.round(instanceValues[f]) + ", ";
					else if ( f < 7+24+27 )
						valueString[2] = valueString[2] + (int)Math.round(instanceValues[f]) + ", ";
					else
						valueString[3] = valueString[3] + (int)Math.round(instanceValues[f]) + ", ";
						*/

					/*
					for ( String s : valueString )
						IJ.log(randomNum + " x,y,z global: " + point.getX() + "," + point.getY()+ "," + example.z + " values "+s);
						*/
	}

	public long getMaximalNumberOfVoxelsPerRegion()
	{
		long currentMemory = IJ.currentMemory();
		long freeMemory = maxMemory - currentMemory;

		long maxNumVoxelsPerRegion = (long) 1.0 * freeMemory / ( getApproximateNeededBytesPerVoxel( memoryFactor ) * threadsRegion * threadsPerRegion);

		long maxNumRegionWidth = (long) Math.pow( maxNumVoxelsPerRegion, 1.0 / 3 );

		//log.setShowDebug(true);
		//log.debug("memoryMB factor " + memoryFactor);
		//log.debug("maxNumVoxelsPerRegion " + maxNumVoxelsPerRegion);
		//log.debug("memoryPerRegionMemoryEstimate [MB] " +
		//		(maxNumVoxelsPerRegion * getApproximateNeededBytesPerVoxel() / 1000000));

		return maxNumVoxelsPerRegion;
	}


	public int getMaximalRegionSize()
	{
		// TODO: this is wrong if the regions are not cubic...
		int maxNumRegionWidth = (int) Math.pow( getMaximalNumberOfVoxelsPerRegion(), 1.0 / 3 );
		// to keep it kind of interactive limit the maximal size
		// to something (500 is arbitrary)
		maxNumRegionWidth = Math.min( maxNumRegionWidth, 500 );

		// remove borders, which go into the memoryMB
		// considerations, but should not be explicitely
		// asked for
		maxNumRegionWidth -= 2 * getFeatureBorderSizes()[ X];
		return maxNumRegionWidth;
	}

	private int[] point3DToInt(Point3D point3D)
	{
		int[] xyz = new int[3];
		xyz[0] = (int) point3D.getX();
		xyz[1] = (int) point3D.getY();
		xyz[2] = (int) point3D.getZ();
		return (xyz);
	}

	private Point[] getPointsFromExample( Example example )
	{
		final int width = Math.round( example.strokeWidth );
		Point[] p = example.points;
		int n = example.points.length;

		double x1, y1;
		double x2 = p[0].getX() - (p[1].getX() - p[0].getX());
		double y2 = p[0].getY() - (p[1].getY() - p[0].getY());

		//double x2=p.xpoints[0]-(p.xpoints[1]-p.xpoints[0]);
		//double y2=p.ypoints[0]-(p.ypoints[1]-p.ypoints[0]);
		ArrayList<Point> points = new ArrayList<>();

		for (int i = 0; i < n; i++)
		{
			x1 = x2;
			y1 = y2;
			x2 = p[i].getX();
			y2 = p[i].getY();

			if ((x1 != x2) || (y1 != y2))
			{
				double dx = x2 - x1;
				double dy = y1 - y2;
				double length = (float) Math.sqrt(dx * dx + dy * dy);
				dx /= length;
				dy /= length;
				double x = x2 - dy * (width - 1) / 2.0;
				double y = y2 - dx * (width - 1) / 2.0;

				int n2 = width;
				do
				{
					int ix = (int) ( x + 0.5 );
					int iy = (int) ( y + 0.5 );
					points.add( new Point(ix, iy) );
					x += dy;
					y += dx;
				} while (--n2 > 0);
			}
		}
		Point[] pointArray = points.toArray(new Point[points.size()]);
		return pointArray;
	}

	public FinalInterval getExampleBoundingInterval(Example example)
	{
		ArrayList<Example> examples = new ArrayList<>();
		examples.add(example);
		return (getExampleListBoundingInterval(examples));
	}

	public FinalInterval getExampleListBoundingInterval(ArrayList<Example> examples)
	{
		Rectangle rectangle = getExampleRectangleBounds( examples.get(0) );

		long[] min = new long[5];
		long[] max = new long[5];

		min[ X] = (int) rectangle.getMinX();
		max[ X] = (int) rectangle.getMaxX();

		min[ Y] = (int) rectangle.getMinY();
		max[ Y] = (int) rectangle.getMaxY();

		min[ Z] = examples.get(0).z;
		max[ Z] = examples.get(0).z;

		min[ T] = examples.get(0).t;
		max[ T] = examples.get(0).t;

		for ( Example example : examples )
		{
			rectangle = getExampleRectangleBounds( example );

			min[ X] = (int) rectangle.getMinX() < min[ X] ? (int) rectangle.getMinX() : min[ X];
			max[ X] = (int) rectangle.getMaxX() > max[ X] ? (int) rectangle.getMaxX() : max[ X];


			min[ Y] = (int) rectangle.getMinY() < min[ Y] ? (int) rectangle.getMinY() : min[
					Y];
			max[ Y] = (int) rectangle.getMaxY() > max[ Y] ? (int) rectangle.getMaxY() : max[ Y];

			min[ Z] = example.z < min[ Z] ? example.z : min[ Z];
			max[ Z] = example.z > max[ Z] ? example.z : max[ Z];
		}



		FinalInterval interval = new FinalInterval( min, max );

		return (interval);

	}


	public int getFeatureVoxelSizeAtMaximumScale()
	{

		int maxFeatureVoxelSize = 1;

		for ( int b : featureSettings.binFactors )
		{
			if ( b > 0 )
			{
				maxFeatureVoxelSize *= b;
			}
		}

		return maxFeatureVoxelSize;
	}


	public ImagePlus computeClassLabelMask(
			ImagePlus impIn,
			int t,
			int minNumVoxels,
			int lower,
			int upper)
	{

		int conn = 6;

		long start = System.currentTimeMillis();

		logger.info( "\n# Computing label mask..." );

		Duplicator duplicator = new Duplicator();
		ImagePlus imp = duplicator.run( impIn, 1, 1, 1, impIn.getNSlices(), t+1, t+1);

		logger.info( "Threshold: " + lower + ", " + upper );
		ImagePlus th = Threshold.threshold( imp, lower, upper );

		logger.info( "MinNumVoxels: " + minNumVoxels );
		ImagePlus th_sf = new ImagePlus( "",
				AttributeFiltering.volumeOpening( th.getStack(), minNumVoxels) );

		logger.info( "Connectivity: " + conn );
		ImagePlus cc = BinaryImages.componentsLabeling( th_sf, conn, 16);
		//logger.info( "...done." );

		logger.info( "...done! It took [min]:" + (System.currentTimeMillis() - start ) / ( 1000 * 60) );

		return cc;
	}

	public int[] getFeatureBorderSizes()
	{
		int[] borderSize = new int[ 5 ];

		borderSize[ X ] = borderSize[ Y ] = getFeatureVoxelSizeAtMaximumScale();

		// Z: deal with 2-D case and anisotropy
		if ( getInputImageDimensions()[ Z ] == 1 )
		{
			borderSize[ Z ] = 0;
		}
		else
		{
			borderSize[ Z ] = (int) Math.ceil(( 1.0 * getFeatureVoxelSizeAtMaximumScale() / featureSettings.anisotropy ) );
		}

		return (borderSize);
	}

	public ArrayList<String> getClassNamesAsArrayList()
	{
		ArrayList<String> classes = new ArrayList<>();
		for (int i = 0; i < getNumClasses(); i++)
		{
			classes.add("class" + i);
		}
		return classes;
	}

	/*
	 This is needed for the 2nd instances, because
	 the feature values are not recomputed, but the
	 classifier is trained using only the selected features
	 of the first instances; this means that
	 it is critical that the sequence in which the features
	 are computed during the actual classification
	 is the same is it was during the instances.
	 This should be fine, but is not entirely trivial, because
	 the feature computation itself is altered, because only
	 the necessary features are computed!
	*/


	/**
	 * Add instancesComboBox from a labeled image in a random and balanced way.
	 * For convention, the label zero is used to define pixels with no class
	 * assigned. The rest of integer values correspond to the order of the
	 * classes (1 for the first class, 2 for the second class, etc.).
	 *
	 * @param labelImage labeled image (labels are positive integer or 0)
	 * @param featureStack corresponding feature stack
	 * @param numSamples number of samples to add of each class
	 * @return false if error
	 */


	public String getMinutes( long now, long begin )
	{
		double minutes = 1.0 * ( now - begin ) / ( 1000.0 * 60 );
		String minutesS = String.format( "%.1f", minutes );
		return ( minutesS );
	}

	public void setLabelImage( ImagePlus labelImage )
	{
		this.labelImage = labelImage;
	}

	public ImagePlus getLabelImage( )
	{
		return ( labelImage );
	}

	public boolean hasLabelImage()
	{
		return ( labelImage != null );
	}

	public void setTrainingData( Instances instances )
	{
		this.trainingData = instances;
	}


	public InstancesManager getInstancesManager()
	{
		return instancesManager;
	}


	private HashMap< String, String > getCurrentMetaData()
	{
		HashMap< String, String > metaData = new HashMap<>();

		return metaData;
	}


	public final static String FEATURE_SELECTION_ABSOLUTE_USAGE = "Absolute usage";
	public final static String FEATURE_SELECTION_RELATIVE_USAGE = "Relative usage";
	public final static String FEATURE_SELECTION_TOTAL_NUMBER = "Total number";
	public final static String FEATURE_SELECTION_NONE = "None";


	public void setFeatureSelection( String method, double value )
	{
		featureSelectionMethod = method;
		featureSelectionValue = value;

	}

	public String trainClassifierWithFeatureSelection ( String key )
	{
		return trainClassifierWithFeatureSelection( instancesManager.getInstancesAndMetadata( key ) );
	}


	public String trainClassifierWithFeatureSelection ( InstancesAndMetadata instancesAndMetadata )
	{

		String classifierKey = trainClassifier( instancesAndMetadata, null );

		if ( ! featureSelectionMethod.equals( FEATURE_SELECTION_NONE ) )
		{
			FastRandomForest classifier = classifierManager.getClassifier( classifierKey );
			InstancesAndMetadata instancesWithFeatureSelection = null;
			int[] attIndicesWindow;

			switch ( featureSelectionMethod )
			{
				case FEATURE_SELECTION_RELATIVE_USAGE:
					ArrayList< Integer > goners =
							AttributeSelector.getGonersBasedOnUsage(
									classifier,
									instancesAndMetadata.getInstances(),
									featureSelectionValue,
									-1,
									logger );

					//instancesWithFeatureSelection = InstancesUtils.removeAttributes( instancesAndMetadata, goners );
                    attIndicesWindow = InstancesUtils.getAttIndicesWindowByRemovingAttributes( instancesAndMetadata, goners );
					break;

				case FEATURE_SELECTION_TOTAL_NUMBER:
					ArrayList< Integer > keepers =
							AttributeSelector.getMostUsedAttributes(
									classifier,
									instancesAndMetadata.getInstances(),
									( int ) featureSelectionValue,
									logger );

					//instancesWithFeatureSelection = InstancesUtils.onlyKeepAttributes( instancesAndMetadata, keepers );
					attIndicesWindow = InstancesUtils.getAttIndicesWindowByKeepingAttributes( keepers );
					break;

				case FEATURE_SELECTION_ABSOLUTE_USAGE:
                    ArrayList< Integer > goners2 =
                            AttributeSelector.getGonersBasedOnUsage(
                                    classifier,
                                    instancesAndMetadata.getInstances(),
                                    -1,
                                    (int) featureSelectionValue,
                                    logger );

                    //instancesWithFeatureSelection = InstancesUtils.removeAttributes( instancesAndMetadata, goners2 );
                    attIndicesWindow = InstancesUtils.getAttIndicesWindowByRemovingAttributes( instancesAndMetadata, goners2 );
                    break;
                default:
                    attIndicesWindow = null;
                    break;
			}

			logger.info( "\n# Second training with feature subset" );

			classifierKey = trainClassifier( instancesAndMetadata, attIndicesWindow );

		}

		return classifierKey;

	}

	public String trainClassifier( )
	{
		return trainClassifier( getCurrentLabelInstancesAndMetadata(), null );
	}

	public long[] getNumLabelInstancesPerClass()
	{
		return InstancesUtils.getNumInstancesPerClass( getCurrentLabelInstancesAndMetadata().getInstances() );
	}

	public String trainClassifier( String key )
	{
		return trainClassifier( instancesManager.getInstancesAndMetadata( key ), null );
	}

    public String trainClassifier( InstancesAndMetadata instancesAndMetadata )
    {
        return trainClassifier( instancesAndMetadata, null );
    }

    /**
	 * Train classifier with the current instances
	 * and current classifier featureSettings
	 * and current active features
	 */
	public String trainClassifier( InstancesAndMetadata instancesAndMetadata, int[] attIndicesWindow  )
	{
		isTrainingCompleted = false;

		// Train the classifier on the current data
		logger.info( "\n# Train classifier" );

		InstancesUtils.logInstancesInformation( instancesAndMetadata.getInstances() );

		final long start = System.currentTimeMillis();

		if (Thread.currentThread().isInterrupted())
		{
			logger.warning("Classifier instances was interrupted.");
			return null;
		}

		// Set up the classifier
		classifierNumRandomFeatures = (int) Math.ceil(1.0 * instancesAndMetadata.getInstances().numAttributes() * classifierFractionFeaturesPerNode );

		FastRandomForest classifier = new FastRandomForest();

		classifier.setSeed( (new Random()).nextInt() );
		classifier.setMaxDepth( classifierMaxDepth );
		classifier.setNumTrees( classifierNumTrees );
		classifier.setNumFeatures( classifierNumRandomFeatures );
		classifier.setNumThreads( threadsClassifierTraining );
		classifier.setBatchSize( "" + getBatchSizePercent() );
		classifier.setComputeImportances( false ); // using own method currently
        classifier.setAttIndicesWindow( attIndicesWindow  );

		// balance traces training data
		// Instances balancedInstances = instancesAndMetadata.getInstances();


		//logger.info( "\nUsing label balancing strategy..." );
		classifier.setLabelIds( instancesAndMetadata.getLabelList() );

		try
		{
			logger.info( "\n" );
			classifier.buildClassifier( instancesAndMetadata.getInstances() );
		}
		catch (InterruptedException ie)
		{
			logger.info( "Classifier construction was interrupted." );
			return null;
		}
		catch (Exception e)
		{
			IJ.showMessage( e.getMessage() );
			e.printStackTrace();
			return null;
		}

		final long end = System.currentTimeMillis();

		ClassifierUtils.reportClassifierCharacteristics( classifier, instancesAndMetadata.getInstances() );

		logger.info("Trained classifier in " + (end - start) + " ms.");

		isTrainingCompleted = true;

        String classifierKey;

		if ( attIndicesWindow != null )
        {
            // remove unused attributes from instances...
            ArrayList< Integer > attIndicesWindowList = asList( attIndicesWindow );
            InstancesAndMetadata instancesAndMetadataAttributeSubset = InstancesUtils.onlyKeepAttributes( instancesAndMetadata, attIndicesWindowList, 1 );

            // ...and keep the classifier, only with used instances (important for feature computation)
            // also note that the "buildClassifier" command above takes care of reassigning the attribute
            // ids on the tree nodes to only account for the used attributes.
            classifierKey = getClassifierManager().setClassifier( classifier, instancesAndMetadataAttributeSubset );
        }
        else
        {
            classifierKey = getClassifierManager().setClassifier( classifier, instancesAndMetadata );
        }

		return classifierKey;
	}

    private ArrayList<Integer> asList( int[] array )
    {
        ArrayList<Integer> list = new ArrayList<>();

        for ( int i = 0; i < array.length; ++i )
        {
            list.add( array[ i ] );
        }

        return list;
    }


    public void applyClassifierWithTiling()
	{
		String mostRecentClassifierKey = getClassifierManager().getMostRecentClassifierKey();
		FinalInterval interval = IntervalUtils.getIntervalWithChannelsDimensionAsSingleton( getInputImage() );
		applyClassifierWithTiling( mostRecentClassifierKey, interval, -1, null , false );
	}

	public void applyClassifierWithTiling( FinalInterval interval )
	{
		String mostRecentClassifierKey = getClassifierManager().getMostRecentClassifierKey();
		applyClassifierWithTiling( mostRecentClassifierKey, interval, -1, null , false );
	}

	public boolean hasClassifier()
	{
		return getClassifierManager().getMostRecentClassifierKey() != null;
	}

	public void applyClassifierWithTiling( String classifierKey, FinalInterval interval )
	{
		applyClassifierWithTiling(  classifierKey, interval, -1, null , false );
	}


	public void applyClassifierOnSlurm(  Map< String, Object > parameters )
	{
		FinalInterval interval = IntervalUtils.getIntervalWithChannelsDimensionAsSingleton( inputImage );
		applyClassifierOnSlurm( parameters, interval );
	}

    public void applyClassifierOnSlurm( FinalInterval interval )
    {
        applyClassifierOnSlurm( new HashMap<>(  ), interval );
    }

	public void applyClassifierOnSlurm(  Map< String, Object > parameters, FinalInterval interval )
    {
        configureInputImageLoading( parameters );

        parameters.put( IOUtils.OUTPUT_DIRECTORY, ((ResultImageDisk)resultImage).getDirectory() );

        parameters.put( ApplyClassifierOnSlurmCommand.INTERVAL, interval );

        parameters.put( ApplyClassifierOnSlurmCommand.NUM_WORKERS, 16 );

        CommandUtils.runSlurmCommand( parameters );
	}

    public void addLabelFromImageRoi( int classNum )
    {

        if ( classNum >= getNumClasses() )
        {
            logger.error( "Class number " + classNum + " does not exist; cannot add label.");
            return;
        }

        if ( isBusy )
        {
            logger.error( "Sorry, but I am busy and cannot add a new label right now...");
            return;
        }

        final Roi roi = inputImage.getRoi();
        if (null == roi) return;
        inputImage.killRoi();

        Point[] points = roi.getContainedPoints();

        final int z = inputImage.getZ() - 1;
        final int t = inputImage.getT() - 1;

        Example newExample = createExample( classNum, points, (int)roi.getStrokeWidth(), z, t );

        addExample( newExample );

        if ( false ) // TODO: instant label update
        {
            Thread thread = new Thread()
            {
                public void run()
                {
                    updateLabelInstancesAndMetadata();
                }
            }; thread.start();
        }

    }



	private void configureInputImageLoading( Map< String, Object > parameters )
    {

        if ( inputImage.getStack() instanceof VirtualStackOfStacks )
		{
			parameters.put( IOUtils.INPUT_MODALITY, IOUtils.OPEN_USING_LAZY_LOADING_TOOLS );
			VirtualStackOfStacks vss = ( VirtualStackOfStacks ) inputImage.getStack();
			parameters.put( IOUtils.INPUT_IMAGE_VSS_DIRECTORY, vss.getDirectory() );
            parameters.put( IOUtils.INPUT_IMAGE_VSS_PATTERN, vss.getFilterPattern() );
            parameters.put( IOUtils.INPUT_IMAGE_VSS_SCHEME, vss.getNamingScheme() );
            parameters.put( IOUtils.INPUT_IMAGE_VSS_HDF5_DATA_SET_NAME, vss.getH5DataSet() );
			parameters.put( IOUtils.INPUT_IMAGE_FILE, new File("") ) ;
			System.out.println( "IOUtils.INPUT_IMAGE_VSS_DIRECTORY: " + vss.getDirectory() );
        }
		else if ( inputImage.getStack() instanceof VirtualStack )
		{
			parameters.put( IOUtils.INPUT_MODALITY, IOUtils.OPEN_USING_IMAGE_J1_VIRTUAL );
			parameters.put( IOUtils.INPUT_IMAGE_FILE, getInputImageFile() );
		}
		else
        {
            parameters.put( IOUtils.INPUT_MODALITY, IOUtils.OPEN_USING_IMAGEJ1 );
			parameters.put( IOUtils.INPUT_IMAGE_FILE, getInputImageFile() );
		}
    }


    public void applyClassifierWithTiling( String classifierKey, FinalInterval interval, Integer numTiles, FeatureProvider externalFeatureProvider, boolean doNotLog )
	{

		logger.info("\n# Apply classifier");

		// set up tiling
        if ( debugUseWholeImageForFeatureComputation )
        {
            numTiles = 1;
        }

		ArrayList<FinalInterval> tiles = createTiles( interval, IntervalUtils.getInterval( inputImage ), numTiles,false, this );

		// set up multi-threading

		int adaptedThreadsPerRegion = threadsPerRegion;
		int adaptedRegionThreads = threadsRegion;

		if ( tiles.size() == 1 )
		{
			adaptedThreadsPerRegion = numThreads;
			adaptedRegionThreads = 1;
		}

		logger.info( "Tile threads: " + adaptedRegionThreads );
		logger.info( "Threads per tile: " + adaptedThreadsPerRegion );

		// submit tiles to executor service

		ExecutorService exe = Executors.newFixedThreadPool( adaptedRegionThreads );
		ArrayList<Future> futures = new ArrayList<>();

		pixelsClassified.set(0L);
		rfStatsTreesEvaluated.set(0L);

		long startTime = System.currentTimeMillis();

		int tileCounter = 0;

		for ( FinalInterval tile : tiles )
		{
			futures.add(
					exe.submit(
							applyClassifier(
									classifierKey,
									tile,
									adaptedThreadsPerRegion,
									++tileCounter,
									tiles.size(),
									externalFeatureProvider,
									doNotLog )
					)
			);
		}


		// collect results

		int regionCounter = 0;
		long maximumMemoryUsage = 0L;

		for (Future future : futures)
		{
			try
			{
				if ( !stopCurrentTasks )
				{
					future.get();

					regionCounter++;

					reportClassificationProgress(startTime,
							maximumMemoryUsage,
							regionCounter,
							tiles);

					System.gc();
				}
				else
				{
					break;
				}
			}
			catch ( Exception e ) // TODO: is it ok to catch all exceptions like this?
			{
				errorHandler( "Apply classifier", e );
			}

		}

		// we're done
		exe.shutdown();

	}


	private void errorHandler( String moduleName, Exception e )
	{
		logger.error("There was an error (see log); trying to stop all ongoing computations..." );
		logger.info("Error in " + moduleName + "\n" + e.toString());
		stopCurrentTasks = true;
		e.printStackTrace();
	}

	private void reportClassificationProgress(
			long startTime,
			long maximumMemoryUsage,
			long tileCounter,
			ArrayList<FinalInterval> tiles)
	{
		if ( tileCounter == 0 || pixelsClassified.get() == 0 )  return;

		long timeUsed = (System.currentTimeMillis() - startTime);
		double timeUsedPerTile = 1.0 * timeUsed / tileCounter;
		long regionsLeft = tiles.size() - tileCounter;
		double minutesLeft = 1.0 * regionsLeft * timeUsedPerTile / (1000 * 60);
		double minutesCurr = 1.0 * timeUsed / (1000 * 60);
		double rate = 1.0 * pixelsClassified.get() / timeUsed;

		String timeInfo = String.format("Time (spent, left) [min]: " + "%.1f, %.1f", minutesCurr, minutesLeft);
		timeInfo += " (" + (int) (rate) + " kv/s)";


		long avgTreesUsed = (long) 1.0 * rfStatsTreesEvaluated.get() / pixelsClassified.get();

		long currentMemoryUsage = IJ.currentMemory();

		if (currentMemoryUsage > maximumMemoryUsage)
			maximumMemoryUsage = currentMemoryUsage;

		String tileInfo = "Tile: "
				+ tileCounter + "/" + tiles.size();

		String memoryInfo = "Memory (current, max, avail) [MB]: "
				+ currentMemoryUsage / 1000000L
				+ ", " + maximumMemoryUsage / 1000000L
				+ ", " + IJ.maxMemory() / 1000000L;

		String treeInfo = "Trees (avg, max, avail): "
				+ avgTreesUsed
				+ ", " + classifierStatsMaximumTreesUsed
				+ ", " + classifierNumTrees;

		logger.progress("Classification", tileInfo
				+ "; " + timeInfo
				+ "; " + treeInfo
				+ "; " + memoryInfo);

	}

	private void storeUncertainties()
	{

		/*
		UncertaintyRegion unitedUncertaintyRegion = new UncertaintyRegion();
		unitedUncertaintyRegion.xyzt[3] = r5D.t;

		for ( UncertaintyRegion uncertaintyRegion : uncertaintyRegions )
		{
			unitedUncertaintyRegion.sumUncertainty += uncertaintyRegion.sumUncertainty;

			unitedUncertaintyRegion.maxUncertainty =
					uncertaintyRegion.maxUncertainty > unitedUncertaintyRegion.maxUncertainty ?
							uncertaintyRegion.maxUncertainty : unitedUncertaintyRegion.maxUncertainty;

			for ( int i = 0; i < 3; i++ )
			{
				unitedUncertaintyRegion.xyzt[i] += uncertaintyRegion.xyzt[i];
			}
		}

		for ( int i = 0; i < 3; i++ )
		{
			unitedUncertaintyRegion.xyzt[i] /= unitedUncertaintyRegion.sumUncertainty;
		}

		unitedUncertaintyRegion.avgUncertainty = unitedUncertaintyRegion.sumUncertainty / ( nx * ny * nz );

		this.uncertaintyRegions.add( unitedUncertaintyRegion );
		*/

	}

	private void waitMilliseconds( long milliSeconds)
	{
		try
		{
			System.gc();
			Thread.sleep(milliSeconds);
		} catch (InterruptedException e)
		{
			e.printStackTrace();
		}
	}

	private FastRandomForest getClassifierCopy()
	{
		FastRandomForest classifierCopy = null;

		classifierCopy = classifier;

		return classifierCopy;
	}

	/**
	 * Apply current classifier to a set of feature vectors (given in a feature
	 * stack array). The classification if performed in a multi-threaded way
	 * using as many numWorkers as defined by the user.
	 *
	 * @param featureImages   feature stack array
	 * @param numThreads      The number of numWorkers to use. Set to zero for auto-detection.
	 * @param probabilityMaps probability flag. Tue: probability maps are calculated, false: binary classification
	 * @return resultImagePlus image containing the probability maps or the binary classification
	 */
	public Runnable applyClassifier(
			String classifierKey,
			final FinalInterval tileInterval,
			final int numThreads,
			final int tileCounter,
			final int tileCounterMax,
			final FeatureProvider externalFeatureProvider,
			boolean doNotLog )
	{

		return () ->
		{

			if ( ThreadUtils.stopThreads( logger, stopCurrentTasks, tileCounter, tileCounterMax ) ) return;

			boolean isLogging = configureLogging( tileCounter, doNotLog );

			checkForBackgroundRegion();

			FeatureProvider featureProvider = configureFeatureProvider( classifierKey, tileInterval, numThreads, externalFeatureProvider, isLogging );

			ArrayList< long[] > zChunks = getZChunks( numThreads, tileInterval );

			final ResultImageFrameSetter resultSetter = resultImage.getFrameSetter( tileInterval );

			ExecutorService exe = Executors.newFixedThreadPool( numThreads );
			ArrayList< Future > futures = new ArrayList<>();

			if ( isLogging ) logger.info( "Classifying pixels..." );

			int numThreadsPerZChunk = 1;

			if ( zChunks.size() == 1 )
			{
				numThreadsPerZChunk = numThreads;
			}

			for ( long[] zChunk : zChunks )
			{
				if ( ThreadUtils.stopThreads( logger, exe, stopCurrentTasks, tileCounter, tileCounterMax ) ) return;

				UncertaintyRegion uncertaintyRegion = new UncertaintyRegion();
				uncertaintyRegions.add(uncertaintyRegion);

				futures.add(
						exe.submit(
								classifyZChunk(
										featureProvider,
										resultSetter,
										zChunk[0], zChunk[1],
										classifierManager.getInstancesHeader( classifierKey ),
										classifierManager.getClassifier( classifierKey ),
										accuracy,
										numThreadsPerZChunk,
										isLogging
								)
						)
				);

			}

			// wait until done
			ThreadUtils.joinThreads( futures, logger, exe );

			// store classification results
			resultSetter.close();

			// store uncertainty information
			//storeUncertainties();

		};
	}

	private boolean configureLogging( int tileCounter, boolean doNotLog )
	{
		boolean isLogging = (tileCounter <= threadsRegion );
		if( doNotLog ) isLogging = false;
		return isLogging;
	}

	private void checkForBackgroundRegion()
	{
		// TODO: check whether this is a background region
			/*
			if ( featureSettings.imageBackground > 0 )
			{
				// check whether the region is background
				if ( isBackgroundRegion( imageToClassify, featureSettings.imageBackground) )
				{
					// don't classify, but leave all classification pixels as is, hopefully 0...
					pixelsClassified.addAndGet( nx * ny * nz  );
					return;
				}
			}*/
	}

	private FeatureProvider configureFeatureProvider( String classifierKey, FinalInterval tileInterval, int numThreads, FeatureProvider externalFeatureProvider, boolean isLogging )
	{
		FeatureProvider featureProvider;

		if ( externalFeatureProvider == null )
		{
			featureProvider = new FeatureProvider( this );
			featureProvider.setFeatureListSubset( classifierManager.getClassifierAttributeNames( classifierKey ) );
			featureProvider.setInterval( tileInterval );
			featureProvider.isLogging( isLogging );
			featureProvider.setShowFeatureImageTitle( featureImageToBeShown );
			featureProvider.computeFeatures( numThreads );
		}
		else
		{
			featureProvider = externalFeatureProvider;
            featureProvider.setShowFeatureImageTitle( featureImageToBeShown );
            featureProvider.setFeatureListSubset( classifierManager.getClassifierAttributeNames( classifierKey ) );
		}
		return featureProvider;
	}


	// Configure z-chunking

	private ArrayList< long[] > getZChunks( int numChunks, FinalInterval tileInterval )
	{

		ArrayList< long[] > zChunks = new ArrayList<>();

		int sliceChunk;

		if ( tileInterval.dimension( Z ) < numChunks )
		{
			sliceChunk = 1;
		}
		else
		{
			sliceChunk = (int) Math.ceil ( 1.0 * tileInterval.dimension( Z ) / numChunks );
		}

		new ArrayList<>();

		for ( long zSlice = tileInterval.min( Z );
			  zSlice <= tileInterval.max( Z );
			  zSlice += sliceChunk )
		{

			long[] zChunk = new long[2];

			if ( zSlice + sliceChunk >= tileInterval.max( Z ) )
			{
				// last chunk can be smaller
				zChunk[0] = zSlice;
				zChunk[1] = tileInterval.max( Z );
				zChunks.add( zChunk );
				break;
			}
			else
			{
				zChunk[0] = zSlice;
				zChunk[1] = zSlice + sliceChunk - 1;
				zChunks.add( zChunk );
			}


		}

		return ( zChunks );
	}


	/**
	 * Checks whether the image only contains background pixels
	 * @param imp
	 * @return
	 */
	private boolean isBackgroundRegion( ImagePlus imp , int threshold )
	{
		ImageStack stack = imp.getStack();

		if ( stack.getBitDepth() == 8 )
		{
			for ( int z = 1; z <= stack.getSize(); z++ )
			{
				byte[] pixels = (byte[]) stack.getProcessor(z).getPixels();
				for ( int i = 0; i < pixels.length; i++ )
				{
					if ( pixels[i] > threshold )
					{
						return(false);
					}
				}
			}
		}
		else if ( stack.getBitDepth() == 16 )
		{
			for ( int z = 1; z <= stack.getSize(); z++ )
			{
				short[] pixels = (short[]) stack.getProcessor(z).getPixels();
				for ( int i = 0; i < pixels.length; i++ )
				{
					if ( pixels[i] > threshold )
					{
						return(false);
					}
				}
			}

		}
		else if ( stack.getBitDepth() == 32  )
		{
			for ( int z = 1; z <= stack.getSize(); z++ )
			{
				float[] pixels = (float[]) stack.getProcessor(z).getPixels();
				for ( int i = 0; i < pixels.length; i++ )
				{
					if ( pixels[i] > threshold )
					{
						return(false);
					}
				}
			}

		}

		return(true);

	}


	/**
	 * Classify instances concurrently
	 *
	 * @param featureImages feature stack array with the feature vectors
	 * @param instancesHeader empty set of instances containing the data structure (attributes and classes)
	 * @param first index of the first instance to classify (considering the feature stack array as a 1D array)
	 * @param numInstances number of instances to classify in this thread
	 * @param classifier current classifier
	 * @param counter auxiliary counter to be able to update the progress bar
	 * @param probabilityMaps if true return a probability map for each class instead of a classified image
	 * @return classification resultImagePlus
	 */
	private Runnable classifyZChunk(
			FeatureProvider featureProvider,
			ResultImageFrameSetter resultSetter,
			long zMin, long zMax,
			//UncertaintyRegion uncertaintyRegion,
			final Instances instancesHeader,
			final FastRandomForest classifier,
			double accuracy,
			int numThreads,
			boolean isLogging )
	{

		return new Runnable() {

			public void run()
			{

				if ( Thread.currentThread().isInterrupted() )
					return;

				final int FIRST_CLASS_ID = 0, SECOND_CLASS_ID = 1, FIRST_CLASS_PROB = 2,
						SECOND_CLASS_PROB = 3, NUM_TREES_EVALUATED = 4;


				FinalInterval interval = featureProvider.getInterval();

				// create reusable instance
				double[] featureValues = new double[ featureProvider.getNumActiveFeatures() + 1 ];
				final ReusableDenseInstance ins = new ReusableDenseInstance( 1.0, featureValues );
				ins.setDataset( instancesHeader );

				// create empty reusable feature slice
				double[][][] featureSlice = null;

				// create empty, reusable results array
				double[] result = null;

				try
				{

					for ( long z = zMin; z <= zMax; ++z )
					{
						if ( isLogging )
						{
							// logger.info( "Classifying slice " + z + "...; chunk of this thread contains: min = " + zMin + ", max = " + zMax  );
						}

						featureSlice = featureProvider.getCachedFeatureSlice( ( int ) z );
						if ( featureSlice == null )
						{
							featureSlice = featureProvider.getReusableFeatureSlice();
							featureProvider.setFeatureSlicesValues( ( int ) z, featureSlice, numThreads );
						}

						for ( long y = interval.min( Y ); y <= interval.max( Y ); ++y )
						{
							for ( long x = interval.min( X ); x <= interval.max( X ); ++x )
							{

								// set instance values
								featureValues = featureProvider.getValuesFromFeatureSlice( ( int ) x, ( int ) y, featureSlice );
								ins.setValues( 1.0, featureValues );

								result = classifier.distributionForInstance( ins, accuracy );

								double certainty = ( result[ FIRST_CLASS_PROB ] - result[ SECOND_CLASS_PROB ] );

								resultSetter.set( x, y, z, ( int ) result[ FIRST_CLASS_ID ], certainty );

								pixelsClassified.incrementAndGet();

								// record tree usage stats
								rfStatsTreesEvaluated.addAndGet( ( int ) result[ NUM_TREES_EVALUATED ] );
								if ( result[ NUM_TREES_EVALUATED ] > classifierStatsMaximumTreesUsed.get() )
								{
									classifierStatsMaximumTreesUsed.set( ( int ) result[ NUM_TREES_EVALUATED ] );
								}
							}
						}
					}

				} catch ( Exception e )
				{
					IJ.showMessage( "Could not apply Classifier!" );
					e.printStackTrace();
					return;
				}
			}
		};
	}

	public UncertaintyRegion getUncertaintyRegion( int i )
	{
		if ( i >= uncertaintyRegions.size())
		{
			logger.error("Selected uncertainty region does not exist.");
			return (null);
		}

		Collections.sort( uncertaintyRegions, Collections.reverseOrder() );
		return(uncertaintyRegions.get(i) );
	}

	public void resetUncertaintyRegions()
	{
		uncertaintyRegions = new ArrayList<>();
	}

	public void deleteUncertaintyRegion( int i )
	{
		uncertaintyRegions.remove( i );
	}

	public int getNumUncertaintyRegions()
	{
		return ( uncertaintyRegions.size() );
	}

	public static int[] maxIndicies(double[] doubles) {
		double maximum = 0.0D;
		int[] maxIndicies = new int[2];

		// 1st maximum
		for(int i = 0; i < doubles.length; ++i) {
			if(i == 0 || doubles[i] >= maximum) {
				maxIndicies[0] = i;  // 1st max
				maximum = doubles[i];
			}
		}

		// 2nd maximum
		maximum = 0.0D;
		for(int i = 0; i < doubles.length; ++i) {
			if(i == 0 || doubles[i] >= maximum) {
				if ( i != maxIndicies[0])
				{
					maxIndicies[1] = i;  // 1st max
					maximum = doubles[i];
				}
			}
		}

		return maxIndicies;
	}


	/**
	 * Set the minimum sigma (radius) to use in the features
	 * @param sigma minimum sigma (radius) to use in the features filters
	 */
	public void setMinimumSigma(float sigma)
	{
		// not used
	}

	/**
	 * Get the minimum sigma (radius) to use in the features
	 * @return minimum sigma (radius) to use in the features
	 */
	public float getMinimumSigma()
	{
		return 0;
	}


	public ArrayList < Integer > getFeaturesToShow()
	{
		return featuresToShow;
	}

	public String getFeaturesToShowAsString()
	{
		String ss = "";
		for ( Integer id : featuresToShow )
		{
			ss += !ss.equals("") ? ("," + id) : id;
		}
		return ss;
	}

	public void setFeaturesToShowFromString( String featuresToShow )
	{
		String[] ss = featuresToShow.split(",");
		this.featuresToShow = new ArrayList<>();
		for ( String s : ss)
		{
			try
			{
				this.featuresToShow.add( Integer.parseInt(s.trim()) );
			}
			catch ( Exception e )
			{
				// probably not an integer
			}
		}
	}

	public String getTileSizeSetting()
	{
		String s = tileSizeSetting;
		return ( s );
	}

	public void setTileSizeSetting( String s )
	{
		tileSizeSetting = s;

		if ( tileSizeSetting.equals("auto") )
		{

		}
	}

	// TODO:
    // move below two functions to featureSettings and make one function for the comma separated list


    /**
	 * Get maximum depth of the random forest
	 * @return maximum depth of the random forest
	 */
	public int getClassifierMaxDepth()
	{
		return classifierMaxDepth;
	}


	public boolean hasResultImage()
	{
		return ( resultImage != null );
	}

	/**
	 * Get the current class labels
	 * @return array containing all the class labels
	 */
	public ArrayList<String> getClassNames()
	{
		return featureSettings.classNames;
	}

	public Set<String> getSegmentedObjectsNames()
	{
		if ( segmentedObjectsMap != null )
		{
			return segmentedObjectsMap.keySet();
		}
		else
		{
			return null;
		}
	}

}

/**
 *
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * Authors: Ignacio Arganda-Carreras (iarganda@mit.edu), Verena Kaynig (verena.kaynig@inf.ethz.ch),
 *          Albert Cardona (acardona@ini.phys.ethz.ch)
 */

