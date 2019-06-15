package de.embl.cba.cats;

import de.embl.cba.bigdataprocessor.virtualstack2.VirtualStack2;
import de.embl.cba.cats.classification.AttributeSelector;
import de.embl.cba.cats.classification.ClassifierInstancesMetadata;
import de.embl.cba.cats.classification.ClassifierManager;
import de.embl.cba.cats.classification.ClassifierUtils;
import de.embl.cba.cats.features.DownSampler;
import de.embl.cba.cats.features.FeatureProvider;
import de.embl.cba.cats.features.settings.FeatureSettings;
import de.embl.cba.cats.features.settings.FeatureSettingsUtils;
import de.embl.cba.cats.instances.InstancesAndMetadata;
import de.embl.cba.cats.instances.InstancesManager;
import de.embl.cba.cats.instances.InstancesUtils;
import de.embl.cba.cats.instances.ReusableDenseInstance;
import de.embl.cba.cats.labelimagetraining.AccuracyEvaluation;
import de.embl.cba.cats.labels.Label;
import de.embl.cba.cats.labels.LabelManager;
import de.embl.cba.cats.labels.LabelUtils;
import de.embl.cba.cats.objects.ObjectReview;
import de.embl.cba.cats.objects.ObjectSegmentation;
import de.embl.cba.cats.objects.SegmentedObjects;
import de.embl.cba.cats.results.ResultImage;
import de.embl.cba.cats.results.ResultImageDisk;
import de.embl.cba.cats.results.ResultImageFrameSetter;
import de.embl.cba.cats.results.ResultImageRAM;
import de.embl.cba.cats.ui.ApplyClassifierOnSlurmCommand;
import de.embl.cba.cats.utils.*;
import de.embl.cba.classifiers.weka.FastRandomForest;
import de.embl.cba.utils.logging.IJLazySwingLogger;
import de.embl.cba.utils.logging.Logger;
import fiji.util.gui.GenericDialogPlus;
import ij.*;
import ij.gui.GenericDialog;
import ij.gui.NonBlockingGenericDialog;
import ij.gui.Roi;
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
import de.embl.cba.bigdataprocessor.utils.Point3D;
import net.imglib2.FinalInterval;
import net.imglib2.util.Intervals;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.awt.*;
import java.awt.image.ColorModel;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static de.embl.cba.cats.utils.IntervalUtils.*;


public class CATS
{
	public static final int MAX_NUM_CLASSES = 20;
	public static final String RESULT_IMAGE_DISK_SINGLE_TIFF = "Disk";
	public static final String RESULT_IMAGE_RAM = "RAM";

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

	public String resultImageType = RESULT_IMAGE_RAM;
	private ImagePlus inputImage;
	private ResultImage resultImage = null;
	private ObjectReview objectReview;

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

    public void setMaxMemoryBytes( long maxMemory )
	{
		this.maxMemoryBytes = maxMemory;
	}

	public long getMaxMemoryBytes( )
	{
		return maxMemoryBytes;
	}

	private long maxMemoryBytes;

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
    public boolean computeLabelFeatureValuesAtMultipleRegionOffest = false;
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

	public int getNumThreads()
	{
		return numThreads;
	}

	public int numThreads;
	public int threadsRegion;
	public int threadsPerRegion;
	public int threadsClassifierTraining;


	public int tilingDelay = 2000; // milli-seconds

	public double uncertaintyLutDecay = 0.5;

	public double accuracy = 4.0;

	public double memoryFactor = 5;

	public static final IJLazySwingLogger logger = new IJLazySwingLogger();

	private ArrayList<UncertaintyRegion > uncertaintyRegions = new ArrayList<>();

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
		int avgNumPointsFromEachLabelPerTree = 5;

		if ( classifierBatchSizePercent.equals("auto") && labelManager.getNumLabels() > 0 )
		{
			logger.info( "\n# Auto setting batch size..." );

			double avgLabelLength = labelManager.getAverageNumberOfPointsPerLabel();
			logger.info( "Average label length: " + avgLabelLength );
			double batchSizePercent =
					avgNumPointsFromEachLabelPerTree
							* ( 100.0 / avgLabelLength ) ;
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
		if ( background != featureSettings.imageBackground && labelManager.getNumLabels() > 0 )
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
            InstancesUtils.saveInstancesAndMetadataAsARFF(
            		instancesManager.getInstancesAndMetadata( instancesKey ),
					directory,
					"Instances-" + numInstances + ".ARFF" );
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

	private LabelManager labelManager = new LabelManager();

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
	public CATS( ImagePlus trainingImage )
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
	public CATS()
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

		// initialize the labels

		labelManager = new LabelManager();

		setMaxMemoryBytes( IJ.maxMemory() );

		setNumThreads( Prefs.getThreads() );

	}


	public LabelManager getLabelManager()
	{
		return labelManager;
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

        if(  Math.round( calibration.pixelWidth * 100.0) != Math.round( calibration.pixelWidth * 100.0)  )
            logger.error("Image calibration in x and y is not the same; currently cannot take this into " +
                    "account; but you can still use this plugin, may work anyway...");

        Set< Integer > channelsToConsider = new TreeSet<>();
        for ( int c = 0; c < inputImage.getNChannels(); c++ )
            channelsToConsider.add(c); // zero-based

        featureSettings.activeChannels = channelsToConsider;

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
	public void setClassName( int classNum, String label )
	{
		getClassNames().set( classNum, label );
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

                setClassName(i, s);
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

        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Image Feature Settings");

        for ( int i = 0; i < 5; ++i )
        {
            gd.addNumericField( "Binning " + ( i + 1 ), featureSettings.binFactors.get( i ), 0 );
        }

        gd.addNumericField("Maximal convolution depth", featureSettings.maxDeepConvLevel, 0);
        gd.addNumericField("z/xy anisotropy", featureSettings.anisotropy, 10);
        gd.addStringField("Feature computation: Channels to consider (one-based) [ID,ID,..]",
                FeatureSettings.getAsCSVString( featureSettings.activeChannels, 1 ) );
        gd.addCheckbox( "Only use difference features", featureSettings.onlyUseDifferenceFeatures );
		gd.addCheckbox( "Normalize intensities block-wise (not recommended...)", featureSettings.normalize );

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

        return settingsChanged;
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


	public void recomputeLabelInstances()
	{
		this.recomputeLabelInstances = true;

		updateLabelInstancesAndMetadata();
	}

    public FeatureSettings getFeatureSettingsFromGenericDialog( NonBlockingGenericDialog gd, boolean showAdvancedSettings )
    {
        FeatureSettings newFeatureSettings = featureSettings.copy();

        for ( int i = 0; i < 5; ++i )
        {
            newFeatureSettings.binFactors.set( i, (int) gd.getNextNumber() );
        }

        newFeatureSettings.maxDeepConvLevel = (int) gd.getNextNumber();
        newFeatureSettings.anisotropy = gd.getNextNumber();
        newFeatureSettings.setActiveChannels( gd.getNextString() );
		newFeatureSettings.onlyUseDifferenceFeatures = gd.getNextBoolean();
		newFeatureSettings.normalize = gd.getNextBoolean();


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


    public void imageCalibrationDialog()
	{
		IJ.run( inputImage, "Properties...", "");

		Calibration calibration = inputImage.getCalibration();

		featureSettings.anisotropy = 1.0 * calibration.pixelDepth / calibration.pixelWidth;

	}

    public boolean initialisationDialog(  )
    {

        GenericDialog gd = new NonBlockingGenericDialog("Set up");

        gd.addMessage( "DATA SET NAME\n \n" +
                "Please enter/confirm the name of this data set.\n" +
                "This is important for keeping track of which instances have been trained with which image." );

        gd.addStringField( "  ", inputImage.getTitle(), 50 );

        gd.addMessage( "RESULT IMAGE LOCATION\n \n" +
                "For large data sets it can be necessary to store the results " +
                "on disk rather than in RAM.\n" +
                "The speed of this plugin does hardly depend on this choice.\n" +
                "If you choose 'Disk' a dialog window will appear to select the storage directory.\n" +
                "You can point to a directory containing previous segmentation results and they will be loaded (not overwritten)." );

        gd.addChoice( "  " ,
                new String[]{
                        RESULT_IMAGE_DISK_SINGLE_TIFF,
                        RESULT_IMAGE_RAM },
						RESULT_IMAGE_DISK_SINGLE_TIFF );


        gd.showDialog();

        if ( gd.wasCanceled() ) return false;

        inputImage.setTitle( gd.getNextString()  );
        assignResultImage( gd.getNextChoice() );

        return true;
    }

    public void reviewObjects( )
    {
		objectReview = new ObjectReview( this );
        objectReview.runUI( );
    }

	public ObjectReview getObjectReview( )
	{
		return objectReview;
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
		private ArrayList<Label > newLabels = null;
	}


	/**
	 * Returns the current classifier.
	 */
	public AbstractClassifier getClassifier()
	{
		return classifier;
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
		loadClassifier( file.getParent(), file.getName() );
	}

	public void loadClassifier( String filePath )
	{
		Path p = Paths.get(filePath);
		loadClassifier( p.getParent().toString(),  p.getFileName().toString());
	}

	public void loadClassifier( String directory, String filename )
	{
		ClassifierInstancesMetadata classifierInstancesMetadata =
				ClassifierUtils.loadClassifierInstancesMetadata( directory, filename );
		final FeatureSettings loadedFeatureSettings =
				FeatureSettingsUtils.getFeatureSettingsFromInstancesMetadata(
						classifierInstancesMetadata.instancesAndMetadata );

		if ( ! featureSettings.equals( loadedFeatureSettings ) )
			logger.info( "Feature settings have been changed!" );

		featureSettings = loadedFeatureSettings;

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
		InstancesAndMetadata instancesAndMetadata =
				getInstancesManager().getInstancesAndMetadata( key );
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
		// we need to recompute the labels' instance values, which
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

		String key = loadInstancesAndMetadata(
				p.getParent().toString(),
				p.getFileName().toString());

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
            logger.info( "# Loaded instances relation name matches image name => Populating image labels..." );
			labelManager.setLabels( LabelUtils.getLabelsFromInstancesAndMetadata( instancesAndMetadata, considerMultipleBoundingBoxOffsetsDuringInstancesLoading ) );
			labelsFeatureNames = instancesAndMetadata.getAttributeNames();
			currentImageLabelInstancesKey = key;
		}
		else
        {
            logger.info( "\nLoaded instances relation name: " + instancesAndMetadata.getRelationName()
                            + "\ndoes not match image name: " + inputImage.getTitle()
                            + "\nLabels will thus not be populated." );
        }


        logger.info( "Setting feature computation settings from loaded instances..." );

		final FeatureSettings loadedFeatureSettings = FeatureSettingsUtils.getFeatureSettingsFromInstancesMetadata( instancesAndMetadata );

		if ( ! featureSettings.equals( loadedFeatureSettings ) )
		{
			logger.info( "Feature settings have been changed!" );
		}

		featureSettings = loadedFeatureSettings;

		setImageBackground( featureSettings.imageBackground );

		return key;

	}

	public boolean saveInstances(  String directory, String filename )
	{
		String key = getCurrentImageLabelInstancesKey();

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
		currentImageLabelInstancesKey = getKeyFromImageTitle();
		updateLabelInstancesAndMetadata( currentImageLabelInstancesKey );
	}

	private String currentImageLabelInstancesKey;

	private String getCurrentImageLabelInstancesKey()
	{
		return currentImageLabelInstancesKey;
	}

	public InstancesAndMetadata getCurrentLabelInstancesAndMetadata()
	{
		return getInstancesManager().getInstancesAndMetadata( currentImageLabelInstancesKey );
	}

	public void updateLabelInstancesAndMetadata( String instancesAndMetadataKey )
	{
		computeUpdatedLabelsInstances();
		putLabelsIntoInstancesAndMetadata( instancesAndMetadataKey );
	}

	public synchronized String putLabelsIntoInstancesAndMetadata( String instancesAndMetadataKey )
	{
		if ( labelManager.getNumLabels() > 0 )
		{
			InstancesAndMetadata instancesAndMetadata =
					InstancesUtils.createInstancesAndMetadataFromLabels(
							labelManager.getLabels(),
							instancesAndMetadataKey,
                            featureSettings,
							labelsFeatureNames,
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

	public void computeUpdatedLabelsInstances()
	{
		ArrayList< Label > labelsNeedingInstanceUpdate = getLabelsNeedingInstancesUpdate();

		ArrayList< ArrayList< Label > > groupedLabels = groupLabelsBySpatialProximity( labelsNeedingInstanceUpdate );

		updateInstancesValuesOfGroupedLabels( groupedLabels );
	}

	private void updateInstancesValuesOfGroupedLabels( ArrayList< ArrayList< Label > > groupedLabels )
	{
		ExecutorService exe = Executors.newFixedThreadPool( threadsRegion ); // because internally there are threadsPerRegion used for feature computation!
		ArrayList<Future > futures = new ArrayList<>();

		isUpdatedFeatureList = false;

		logger.info("Computing features values for " + groupedLabels.size() + " label groups using " + numThreads + " threads." );

		for (int i = 0; i < groupedLabels.size(); i++)
		{
			ArrayList<Label > neighboringLabels = groupedLabels.get(i);
			futures.add (
					exe.submit(
							computeLabelsInstanceValues( neighboringLabels, i, groupedLabels.size() - 1)
					)
			);
		}

		ThreadUtils.joinThreads( futures, logger, exe );


		if ( groupedLabels.size() > 0 )
		{
			logger.info("Computed feature values for " + groupedLabels.size() + " labels; " + "total number of labels is " + labelManager.getNumLabels() );
		}
	}

	private ArrayList< ArrayList< Label > > groupLabelsBySpatialProximity( ArrayList< Label > labelsNeedingInstanceUpdate )
	{
		ArrayList<ArrayList<Label >> labelList = new ArrayList<>();

        if( debugUseWholeImageForFeatureComputation )
        {
            // put all labels into one group
            labelList.add( labelsNeedingInstanceUpdate );
            return labelList;
        }

		for (int iLabelWithoutFeatures = 0; iLabelWithoutFeatures < labelsNeedingInstanceUpdate.size(); iLabelWithoutFeatures++)
		{
			// figure out which labels are spatially close,
			// put them together and compute the feature images
			// for them in one go; this saves time.
			ArrayList<Label > neighboringLabels = new ArrayList<>();

			Rectangle labelBounds = LabelUtils.getLabelRectangleBounds( labelsNeedingInstanceUpdate.get( iLabelWithoutFeatures ) );

			Point3D labelLocation = new Point3D(
					labelBounds.getX(),
					labelBounds.getY(),
					labelsNeedingInstanceUpdate.get(iLabelWithoutFeatures).z
			);

			neighboringLabels.add( labelsNeedingInstanceUpdate.get( iLabelWithoutFeatures ) );

			Boolean includeNextLabel = true;

			iLabelWithoutFeatures++;

			while ( includeNextLabel && ( iLabelWithoutFeatures < labelsNeedingInstanceUpdate.size() ) )
			{
				Rectangle nextLabelBounds = LabelUtils.getLabelRectangleBounds(labelsNeedingInstanceUpdate.get(iLabelWithoutFeatures));

				Point3D nextLabelLocation = new Point3D(
						nextLabelBounds.getX(),
						nextLabelBounds.getY(),
						labelsNeedingInstanceUpdate.get(iLabelWithoutFeatures).z
				);

				if ( labelLocation.distance( nextLabelLocation ) < getFeatureVoxelSizeAtMaximumScale() )
				{
					neighboringLabels.add( labelsNeedingInstanceUpdate.get(iLabelWithoutFeatures) );
					iLabelWithoutFeatures++;
				}
				else
				{
					includeNextLabel = false;
					iLabelWithoutFeatures--;
				}

			}

			labelList.add( neighboringLabels );

		}

		return labelList;
	}

	private ArrayList< Label > getLabelsNeedingInstancesUpdate( )
	{
		ArrayList< Label > labelsWithoutFeatures = new ArrayList<>();

		for ( Label label : labelManager.getLabels() )
		{
			if ( recomputeLabelInstances )
			{
				// add all labels to the list
				labelsWithoutFeatures.add( label );
			}
			else
			{
				// add labels that need feature re-computation
				if ( label.instanceValuesArrays == null && ! label.instanceValuesAreCurrentlyBeingComputed )
				{
					label.instanceValuesAreCurrentlyBeingComputed = true;
					labelsWithoutFeatures.add( label );
				}
			}
		}

		recomputeLabelInstances = false;

		return labelsWithoutFeatures;
	}

	public ArrayList< String > labelsFeatureNames = null;


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

	private Runnable computeLabelsInstanceValues( ArrayList< Label > labels, int counter, int counterMax )
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return () -> {

            LabelUtils.setLabelInstanceValuesAreCurrentlyBeingComputed( labels, true );

            logger.info("Label set " + (counter + 1) + "/" + (counterMax + 1) + ": " + "Computing features for " + labels.size() + " label(s)...");

            LabelUtils.clearInstancesValues( labels );

			ArrayList< FinalInterval > labelListBoundingIntervals = getBoundingIntervals( labels );

			for ( int iBoundingInterval = 0; iBoundingInterval < labelListBoundingIntervals.size(); ++iBoundingInterval )
			{

			    FeatureProvider featureProvider = new FeatureProvider( this );
				featureProvider.setInterval( labelListBoundingIntervals.get( iBoundingInterval ) );
				featureProvider.computeFeatures( threadsPerRegion );

				int nf = featureProvider.getNumAllFeatures();

				this.labelsFeatureNames = featureProvider.getAllFeatureNames();

				double[][][] featureSlice = featureProvider.getReusableFeatureSlice();

				// extract the feature values at
				// the respective z-position of each label
				for ( Label label : labels )
				{
					label.instanceValuesAreCurrentlyBeingComputed = true;

					ArrayList< double[] > instanceValuesArray = new ArrayList<>();

					int z = label.z;

					featureProvider.setFeatureSlicesValues( z, featureSlice, 1 );

					for ( Point point : label.points )
					{
						int xGlobal = ( int ) point.getX();
						int yGlobal = ( int ) point.getY();

						double[] values = new double[ nf + 1 ];
						featureProvider.setFeatureValuesAndClassIndex( values, xGlobal, yGlobal, featureSlice, label.classNum );
						instanceValuesArray.add( values );

						if ( debugLogLabelPixelValues )
						{
							logger.info( "# " + counter
									+ " x " + xGlobal
									+ " y " + yGlobal
									+ " z " + label.z
									+ " value[0] " + values[ 0 ] );
						}
					}

					label.instanceValuesArrays.add( instanceValuesArray );

				}

				// logger.info( "Bounding interval " + iBoundingInterval );

			}

            LabelUtils.setLabelInstanceValuesAreCurrentlyBeingComputed( labels, false );

            logger.info( "Label set " + ( counter + 1 ) + "/" + ( counterMax + 1 ) + ": " + "...done" );

		};
	}

	private ArrayList< FinalInterval > getBoundingIntervals( ArrayList< Label > labels )
	{
		ArrayList< FinalInterval > labelListBoundingIntervals = new ArrayList<>();
		FinalInterval labelListBoundingInterval = getLabelListBoundingInterval( labels );

		for ( int offset : featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels )
        {
            long[] offsets = new long[5];
            offsets[ X ] = offset;
            offsets[ Y ] = offset;

            if ( inputImage.getNSlices() > 1 )
            {
                offsets[ Z ] = offset;
            }

            labelListBoundingIntervals.add( Intervals.expand( labelListBoundingInterval, offsets ) );
        }

        return labelListBoundingIntervals;
	}

	private void debugInstanceValuesComputation()
	{
		// Below is some code for debugging
		//

					/*
					IJ.log(" x,y: " + x + ", " + y + "; max = " + featureSlice.length + ", " + featureSlice[0].length );
					IJ.log("");
					IJ.log(randomNum+" x,y,z global: " + point.getX() + "," + point.getY() + "," + label.z );
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
						IJ.log(randomNum + " x,y,z global: " + point.getX() + "," + point.getY()+ "," + label.z + " values "+s);
						*/
	}

	public long getMaximalNumberOfVoxelsPerRegion( int numFeatures )
	{
		long currentMemory = IJ.currentMemory();
		long freeMemory = maxMemoryBytes - currentMemory;
		double maxNumVoxelsPerRegion =  1.0 * freeMemory;
		maxNumVoxelsPerRegion /= 1.0 * getApproximatelyNeededBytesPerVoxel( numFeatures );
		maxNumVoxelsPerRegion /= 1.0 * threadsRegion * threadsPerRegion;
		return (long) maxNumVoxelsPerRegion;
	}


	private boolean isInputImage2D()
	{
		return ( inputImage.getNSlices() == 1 );
	}


	public int getMaximalRegionWidth( int numFeatures )
	{
		int maxRegionWidth;

		if ( isInputImage2D() || this.featureSettings.anisotropy > 10 )
			maxRegionWidth = ( int ) Math.pow( getMaximalNumberOfVoxelsPerRegion( numFeatures ), 1.0 / 2.0 );
		else
			maxRegionWidth = ( int ) Math.pow( getMaximalNumberOfVoxelsPerRegion( numFeatures ), 1.0 / 3.0 );

		// remove borders, which go into the memory
		// considerations, but should not be explicitly
		// asked for
		maxRegionWidth -= 2 * getFeatureBorderSizes()[ X];

//		if ( this.featureSettings.normalize )
//		{
//			// make it consistent with what has been used during training, otherwise
//			// the normalisations might be very different.
//			// this will make it slower of course
//			maxRegionWidth = getFeatureVoxelSizeAtMaximumScale();
//		}

		return maxRegionWidth;
	}

	private int[] point3DToInt(Point3D point3D)
	{
		int[] xyz = new int[3];
		xyz[0] = (int) point3D.getX();
		xyz[1] = (int) point3D.getY();
		xyz[2] = (int) point3D.getZ();
		return (xyz);
	}

	private Point[] getPointsFromLabel( Label label )
	{
		final int width = Math.round( label.strokeWidth );
		Point[] p = label.points;
		int n = label.points.length;

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

	public FinalInterval getLabelBoundingInterval(Label label )
	{
		ArrayList<Label > labels = new ArrayList<>();
		labels.add( label );
		return (getLabelListBoundingInterval( labels ));
	}

	public FinalInterval getLabelListBoundingInterval(ArrayList<Label > labels )
	{
		Rectangle rectangle = LabelUtils.getLabelRectangleBounds( labels.get(0) );

		long[] min = new long[5];
		long[] max = new long[5];

		min[ X ] = (int) rectangle.getMinX();
		max[ X ] = (int) rectangle.getMaxX();

		min[ Y ] = (int) rectangle.getMinY();
		max[ Y ] = (int) rectangle.getMaxY();

		min[ Z ] = labels.get(0).z;
		max[ Z ] = labels.get(0).z;

		min[ T ] = labels.get(0).t;
		max[ T ] = labels.get(0).t;

		for ( Label label : labels )
		{
			rectangle = LabelUtils.getLabelRectangleBounds( label );

			min[ X] = (int) rectangle.getMinX() < min[ X] ? (int) rectangle.getMinX() : min[ X ];
			max[ X] = (int) rectangle.getMaxX() > max[ X] ? (int) rectangle.getMaxX() : max[ X ];


			min[ Y] = (int) rectangle.getMinY() < min[ Y] ? (int) rectangle.getMinY() : min[ Y ];
			max[ Y] = (int) rectangle.getMaxY() > max[ Y] ? (int) rectangle.getMaxY() : max[ Y ];

			min[ Z] = label.z < min[ Z] ? label.z : min[ Z];
			max[ Z] = label.z > max[ Z] ? label.z : max[ Z];
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

		// TODO: deal with 2-D case and anisotropy

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

	public long[] getNumInstancesPerClass()
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
	public String trainClassifier(
			InstancesAndMetadata instancesAndMetadata,
			int[] attIndicesWindow  )
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
            classifierKey = getClassifierManager().
					setClassifier( classifier, instancesAndMetadataAttributeSubset );
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

		FinalInterval interval =
				IntervalUtils.getIntervalWithChannelsDimensionAsSingleton( getInputImage() );

		applyClassifierWithTiling(
				mostRecentClassifierKey,
				interval,
				-1,
				null ,
				false );
	}

	public void applyClassifierWithTiling( FinalInterval classificationInterval )
	{
		String mostRecentClassifierKey =
				getClassifierManager().getMostRecentClassifierKey();

		applyClassifierWithTiling(
				mostRecentClassifierKey,
				classificationInterval,
				-1,
				null ,
				false );
	}

	public boolean hasClassifier()
	{
		return getClassifierManager().getMostRecentClassifierKey() != null;
	}

	public void applyClassifierWithTiling( String classifierKey, FinalInterval interval )
	{
		applyClassifierWithTiling(
				classifierKey,
				interval,
				-1,
				null ,
				false );
	}

	public void applyClassifierOnSlurm(  Map< String, Object > parameters )
	{
		FinalInterval interval =
				IntervalUtils.getIntervalWithChannelsDimensionAsSingleton( inputImage );
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

        Label newLabel = LabelUtils.createLabel( classNum, points, (int)roi.getStrokeWidth(), z, t );

        labelManager.addLabel( newLabel );

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

        if ( inputImage.getStack() instanceof VirtualStack2 )
		{
			parameters.put( IOUtils.INPUT_MODALITY, IOUtils.OPEN_USING_LAZY_LOADING_TOOLS );
			VirtualStack2 vss = ( VirtualStack2 ) inputImage.getStack();
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


    public void applyClassifierWithTiling( String classifierKey,
										   FinalInterval classificationInterval,
										   Integer numTiles,
										   FeatureProvider externalFeatureProvider,
										   boolean doNotLog )
	{
		logger.info("\n# Apply classifier");

		if ( classificationInterval == null )
		{
			CATS.logger.error( "The classification interval was NULL." );
			return;
		}

		isBusy = true;

		// set up tiling
        if ( debugUseWholeImageForFeatureComputation )
            numTiles = 1;

		ArrayList<FinalInterval> tiles = createTiles(
				classificationInterval,
				IntervalUtils.getInterval( inputImage ),
				numTiles,
				classifierManager.getClassifierAttributeNames( classifierKey ).size(),
				featureSettings.normalize,
				false,
				this );

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

		for ( Future future : futures )
		{
			try
			{
				if ( ! stopCurrentTasks )
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

		isBusy = false;

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

		return () -> {

			if ( Thread.currentThread().isInterrupted() )
				return;

			final int FIRST_CLASS_ID = 0, SECOND_CLASS_ID = 1, FIRST_CLASS_PROB = 2,
					SECOND_CLASS_PROB = 3, NUM_TREES_EVALUATED = 4;


			FinalInterval interval = featureProvider.getInterval();

			// create reusable instance
			double[] featureValues = new double[ featureProvider.getNumActiveFeatures() + 1 ];
			final ReusableDenseInstance ins = new ReusableDenseInstance(
					1.0,
					featureValues );
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
						featureProvider.setFeatureSlicesValues( ( int ) z,
								featureSlice, numThreads );
					}

					for ( long y = interval.min( Y ); y <= interval.max( Y ); ++y )
					{
						for ( long x = interval.min( X ); x <= interval.max( X ); ++x )
						{

							// set instance values
							featureValues =
									featureProvider.getValuesFromFeatureSlice(
											( int ) x, ( int ) y, featureSlice );
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

