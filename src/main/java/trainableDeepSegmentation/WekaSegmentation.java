package trainableDeepSegmentation;

import java.awt.Point;
import java.awt.Rectangle;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import bigDataTools.Region5D;
import bigDataTools.VirtualStackOfStacks.VirtualStackOfStacks;
import bigDataTools.utils.Utils;
import bigDataTools.logging.Logger;
import bigDataTools.logging.IJLazySwingLogger;
import ij.gui.PolygonRoi;
import javafx.geometry.Point3D;

import hr.irb.fastRandomForest.FastRandomForest;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Prefs;
import ij.gui.Roi;
import ij.process.ImageProcessor;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;


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

/**
 * This class contains all the library methods to perform image segmentation
 * based on the Weka classifiers.
 */
public class WekaSegmentation {

	/** maximum number of classes (labels) allowed */
	public static final int MAX_NUM_CLASSES = 10;

	public static final int CLASS_LUT_WIDTH = 20;

	/** array of lists of Rois for each slice (vector index)
	 * and each class (arraylist index) of the training image */
	private ArrayList<Example> examples;
	/** image to be used in the training */
	private ImagePlus inputImage;
	/** result image after classification */
	private ImagePlus classifiedImage;
	/** features to be used in the training */
	//private FeatureImagesMultiResolution  featureImages = null;
	/** set of instances from loaded data (previously saved segmentation) */
	private Instances labelImageTrainingData = null;
	/** set of instances from the user's traces */
	private Instances trainingData = null;
	/** current classifier */
	public AbstractClassifier classifier = null;
	/** train header */
	private Instances trainHeader = null;
	/** default classifier (Fast Random Forest) */
	private FastRandomForest rf;

	/** names of the current classes */

	private int[] imgDims = new int[5];

	// Random Forest parameters
	/** current number of trees in the fast random forest classifier */
	private int numOfTrees = 200;
	/** number of random features per node in the fast random forest classifier */
	private int numRandomFeatures = 50;
	/** fraction of random features per node in the fast random forest classifier */
	public double fractionRandomFeatures = 0.1;
	/** maximum depth per tree in the fast random forest classifier */
	public int maxDepth = 0;
	/** maximum depth per tree in the fast random forest classifier */
	private int batchSizePercent = 100;
	/** list of class names on the loaded data */
	private ArrayList<String> loadedClassNames = null;

	private boolean updateFeatures = false;

	public int[] minTileSizes = new int[]{162,162,81};

	public String tileSizeSetting = "auto";

	public Settings settings = new Settings();

	public int minFeatureUsage = 0;

	public double minFeatureUsageFactor = 2.0;

	private boolean computeFeatureImportance = false;

	public int numRegionThreads = (int) Math.sqrt( Prefs.getThreads() ) + 1;

	public int numThreadsPerRegion = (int) Math.sqrt( Prefs.getThreads() ) + 1;

	public int numRfTrainingThreads = Prefs.getThreads();

	public int tilingDelay = 500; // milli-seconds

	public double uncertaintyLUTdecay = 0.5;

	private double avgRfTreeSize = 0.0;

	public double accuracy = 3.0;

	public double memoryFactor = 1.0;

	public int labelImageNumInstancesPerClass = 1000;

	private static Logger logger = new IJLazySwingLogger();

	private ArrayList< UncertaintyRegion > uncertaintyRegions = new ArrayList<>();

	public boolean isTrainingCompleted = true;

	public Logger getLogger()
	{
		return logger;
	}

	/**
	 *
	 * @param logFileName
	 * @param logFileDirectory
	 * @return returns true if the log file could be sucessfully created
	 */

	public boolean setLogFileNameAndDirectory( String logFileName, String logFileDirectory )
	{

		((IJLazySwingLogger)logger).setLogFileNameAndDirectory( logFileName ,
				logFileDirectory );
		((IJLazySwingLogger)logger).isFileLogging = true;

		return ( true );
	}

	public boolean getComputeFeatureImportance()
	{
		return computeFeatureImportance;
	}

	/** use neighborhood flag */
	private boolean useNeighbors = false;

	public AtomicInteger totalThreadsExecuted = new AtomicInteger(0);

	public AtomicLong pixelsClassified = new AtomicLong(0);

	public AtomicLong rfStatsTreesEvaluated = new AtomicLong(0);

	private AtomicInteger rfStatsMaximumTreesUsed = new AtomicInteger( 0 );

	/**
	 * flag to set the resampling of the training data in order to guarantee
	 * the same number of instances per class (class balance)
	 * */
	private boolean balanceClasses = false;

	/** Project folder name. It is used to stored temporary data if different from null */
	private String projectFolder = null;

	private ArrayList < Integer > featuresToShow = null;

	/** executor service to launch threads for the library operations */
	private ExecutorService exe = Executors.newFixedThreadPool( numThreadsPerRegion );

	public boolean stopCurrentThreads = false;

	private int currentUncertaintyRegion = 0;

	private ImagePlus labelImage = null;

	/**
	 * Default constructor.
	 *
	 * @param trainingImage The image to be segmented/trained
	 */
	public WekaSegmentation( ImagePlus trainingImage )
	{
		initialize();
		setInputImage( trainingImage );
	}

	private void setImgDims( )
	{
		imgDims[0] = inputImage.getWidth();
		imgDims[1] = inputImage.getHeight();
		imgDims[2] = inputImage.getNSlices();
		imgDims[3] = inputImage.getNChannels();
		imgDims[4] = inputImage.getNFrames();
	}

	public int[] getImgDims()
	{
		return( imgDims );
	}

	/**
	 * No-image constructor. If you use this constructor, the image has to be
	 * set using setInputImage().
	 */
	public WekaSegmentation()
	{
		initialize();
	}

	private void initialize()
	{
		// set class label names
		char[] alphabet = "abcdefghijklmnopqrstuvwxyz".toCharArray();

		settings.classNames = new ArrayList<>();
		settings.classNames.add( new String("background") );
		settings.classNames.add( new String("foreground") );

		// Initialization of Fast Random Forest classifier
		rf = new FastRandomForest();
		rf.setNumTrees(numOfTrees);
		rf.setNumFeatures(numRandomFeatures);
		rf.setSeed((new Random()).nextInt());
		rf.setNumThreads(numRfTrainingThreads);
		rf.setMaxDepth(maxDepth);
		//rf.setBatchSize("50");
		rf.setComputeImportances(true);
		classifier = rf;
		

		// initialize the examples
		examples = new ArrayList<Example>();

	}

	/**
	 * Set the training image (single image or stack)
	 *
	 * @param imp training image
	 */
	public void setInputImage(ImagePlus imp)
	{
		this.inputImage = imp;
		setImgDims();
	}

	/**
	 * Adds a ROI to the list of examples for a certain class
	 * and slice.
	 *
	 * @param classNum the number of the class
	 * @param roi the ROI containing the new example
	 * @param n number of the current slice
	 */
	public void addExample( Example example )
	{
		examples.add(example);
	}

	public Example createExample(int classNum, Point[] points, int strokeWidth, int z, int t)
	{
		Example example = new Example(classNum, points, strokeWidth, z, t);
		return( example );
	}

	/**
	 * Remove an example list from a class and specific slice
	 *
	 * @param classNum the number of the examples' class
	 * @param nSlice the slice number
	 * @param index the index of the example list to remove
	 */
	public void deleteExample(int classNum, int z, int t, int index)
	{
		int i = 0;
		for ( int iExample = 0; iExample < examples.size(); iExample++ )
		{
			Example example = examples.get(iExample);
			if (  (example.z == z)
					&& (example.t == t)
					&& (example.classNum == classNum) )
			{
				if ( index == i++ )
				{
					examples.remove( iExample );
					return;
				}
			}

		}

	}


	public Rectangle getExampleBounds( Example example )
	{
		int xMin = example.points[0].x;
		int xMax = example.points[0].x;
		int yMin = example.points[0].y;
		int yMax = example.points[0].y;

		for ( Point point : example.points  )
		{
			xMin = point.x < xMin ? point.x : xMin;
			xMax = point.x > xMax ? point.x : xMax;
			yMin = point.y < yMin ? point.y : yMin;
			yMax = point.y > yMax ? point.y : yMax;
		}

		xMin -= example.strokeWidth;
		xMax += example.strokeWidth;
		yMin -= example.strokeWidth;
		yMax += example.strokeWidth;

		return ( new Rectangle( xMin, yMin, xMax-xMin+1, yMax-yMin+1 ) );
	}

	/**
	 * Check whether the example is valid, i.e.
	 * - not too close to the image boundary
	 *
	 * @param example
	 * @return
	 */
	public boolean isValidExample( Example example )
	{
		int[][] bounds = getExample3DBounds( example );
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

	}

	/**
	 * Return the list of examples for a certain class.
	 *
	 * @param classNum the number of the examples' class
	 * @param n the slice number
	 */
	public ArrayList<Roi> getExampleRois(int classNum, int z, int t)
	{
		ArrayList<Roi> rois = new ArrayList<>();

		for ( Example example : examples )
		{
			if ( (example.z == z)
					&& (example.t == t)
					&& (example.classNum == classNum))
			{
				float[] x = new float[ example.points.length ];
				float[] y = new float[ example.points.length ];
				for ( int iPoint = 0; iPoint < example.points.length; iPoint++ )
				{
					x[iPoint] = (float) example.points[iPoint].getX();
					y[iPoint] = (float) example.points[iPoint].getY();
				}
				Roi roi = new PolygonRoi( x, y, PolygonRoi.FREELINE );
				roi.setStrokeWidth((double)example.strokeWidth);
				rois.add(roi);
			}

		}
		return rois;
	}

	public ArrayList< Example > getExamples()
	{
		return examples;
	}

	public void setLabelROIs(ArrayList< Example > examples )
	{
		this.examples = examples;
	}

	public int getNumExamples(int classNum)
	{
		int n =0 ;
		for ( Example example : examples )
		{
			if ( example.classNum == classNum )
			{
				n++;
			}
		}
		return n;
	}

	public void setAllFeaturesActive()
	{
		if ( settings.featureList == null )
			return;

		for ( Feature feature : settings.featureList )
		{
			feature.isActive = true;
		}
	}

	public void deactivateRarelyUsedFeatures()
	{
		settings.activeFeatureNames = new ArrayList<>();

		for ( Feature feature : settings.featureList )
		{
			if ( feature.usageInRF < minFeatureUsage )
			{
				feature.isActive = false;
			}
			else
			{
				feature.isActive = true;
				settings.activeFeatureNames.add( feature.name);
			}
		}
	}

	public int getNumClassesInExamples()
	{
		Set<Integer> classNums = new HashSet<>();

		for ( Example example : examples )
		{
			classNums.add( example.classNum );
		}
		return classNums.size();
	}

	public int[] getNumExamplesPerClass()
	{
		int[] numExamplesPerClass = new int[ getNumClasses() ];

		for ( Example example : examples )
		{
			numExamplesPerClass[ example.classNum ] ++;
		}

		return ( numExamplesPerClass );
	}
	

	/**
	 * Get the current number of classes.
	 *
	 * @return the current number of classes
	 */
	public int getNumClasses()
	{
		return settings.classNames.size();
	}

	/**
	 * Set the name of a class.
	 *
	 * @param classNum class index
	 * @param label new name for the class
	 */
	public void setClassLabel(int classNum, String label)
	{
		getClassNames().set( classNum, label );
	}

	/**
	 * Get the label name of a class.
	 *
	 * @param classNum class index
	 */
	public String getClassName(int classNum)
	{
		return getClassNames().get( classNum );
	}


	/**
	 * Get the label name of a class.
	 *
	 * @param classNum class index
	 */
	public void addClass( String className )
	{
		settings.classNames.add( className );
	}


	/**
	 * Get current classification result
	 * @return classified image
	 */
	public ImagePlus getClassifiedImage()
	{
		return classifiedImage;
	}

	public void setOutputImage(ImagePlus imp )
	{
		classifiedImage = imp;
	}

	private double[] getFeatureValues(
			ArrayList < double[][][] > featureSlices,
			int x,
			int y,
			int classNum )
	{

		int numFeaturesAllChannels = 0;
		for ( int c = 0; c < featureSlices.size(); c++ )
		{
			// Different channels might have different numbers of
			// features, because during actual classification
			// only active features have been computed.
			numFeaturesAllChannels += featureSlices.get(c)[0][0].length;
		}

		if ( classNum > -1 ) numFeaturesAllChannels++;

		double[] values = new double[ numFeaturesAllChannels ];

		int iFeature = 0;

		for ( int c = 0; c < featureSlices.size(); c++ )
		{
			int numFeaturesThisChannel = featureSlices.get(c)[0][0].length;
			for (int i = 0; i < numFeaturesThisChannel; i++)
			{
				values[ iFeature++ ] = featureSlices.get(c)[x][y][i];
			}
		}

		if( classNum > -1 ) values[ iFeature ] = classNum;

		return ( values );
	}

	/**
	 * bag class for getting the result of the loaded classifier
	 */
	private static class LoadedProject {
		private AbstractClassifier newClassifier = null;
		private Settings newSettings = null;
		private ArrayList < Example > newExamples = null;
	}



	/**
	 * Returns the current classifier.
	 */
	public AbstractClassifier getClassifier() {
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


		try {
			sFile = new File(filename);
			OutputStream os = new FileOutputStream(sFile);
			if (sFile.getName().endsWith(".gz"))
			{
				os = new GZIPOutputStream(os);
			}
			ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
			objectOutputStream.writeObject( classifier );
			objectOutputStream.writeObject( settings );
			objectOutputStream.writeObject( getExamples() );
			objectOutputStream.flush();
			objectOutputStream.close();
		}
		catch ( Exception e )
		{
			IJ.error("Save Failed", "Error when saving project to disk");
			logger.info( e.toString() );
			saveOK = false;
		}
		if (saveOK)
		{
			IJ.log("Saved project to " + filename);
		}

		return saveOK;
	}


	/**
	 * Read header classifier from a .model file
	 * @param pathName complete path and file name
	 * @return false if error
	 */
	public boolean loadProject( String pathName ) throws IOException
	{
		AbstractClassifier newClassifier = null;
		Settings newSettings = null;
		ArrayList < Example > newExamples = null;

		logger.info("Loading project from disk...");

		File selected = new File( pathName );
		try {
			InputStream is = new FileInputStream( selected );
			if (selected.getName().endsWith(".gz"))
			{
				is = new GZIPInputStream(is);
			}
			try {
				ObjectInputStream objectInputStream = new ObjectInputStream( is );
				newClassifier = (AbstractClassifier) objectInputStream.readObject();
				newSettings = (Settings) objectInputStream.readObject();
				newExamples = (ArrayList<Example>)  objectInputStream.readObject();
				objectInputStream.close();
			} catch (Exception e) {
				logger.error("Error while loading project");
				logger.info( e.toString() );
				return false;
			}
		}
		catch (Exception e)
		{
			logger.error( "Error while loading project" );
			e.printStackTrace();
			return false;
		}

		/*
		// TODO: check stuff here...
		try{
			// Check if the loaded information corresponds to current state of the segmentator
			// (the attributes can be adjusted, but the classes must match)
			if(!adjustSegmentationStateToData(newHeader))
			{
				IJ.log("Error: current segmentator state could not be updated to loaded data requirements (attributes and classes)");
				return false;
			}
		}catch(Exception e)
		{
			IJ.log("Error while adjusting data!");
			e.printStackTrace();
			return false;
		}
		*/

		setClassifier( newClassifier );
		setSettings( newSettings );
		setLabelROIs( newExamples );

		logger.info("Loaded project: " + pathName );

		return true;
	}

	/**
	 * Set current classifier
	 * @param cls new classifier
	 */
	public void setClassifier(AbstractClassifier cls)
	{
		this.classifier = cls;
	}

	/**
	 * Set current settings
	 * @param settings
	 */
	public void setSettings( Settings settings)
	{
		this.settings = settings;
	}


	/**
	 * Homogenize number of instances per class
	 *
	 * @param data input set of instances
	 * @return resampled set of instances
	 * @deprecated use balanceTrainingData
	 */
	public static Instances homogenizeTrainingData(Instances data)
	{
		return WekaSegmentation.balanceTrainingData( data );
	}

	/**
	 * Balance number of instances per class
	 *
	 * @param data input set of instances
	 * @return resampled set of instances
	 */
	public static Instances balanceTrainingData( Instances data )
	{
		final Resample filter = new Resample();
		Instances filteredIns = null;
		filter.setBiasToUniformClass(1.0);
		try {
			filter.setInputFormat(data);
			filter.setNoReplacement(false);
			filter.setSampleSizePercent(100);
			filteredIns = Filter.useFilter(data, filter);
		} catch (Exception e) {
			IJ.log("Error when resampling input data!");
			e.printStackTrace();
		}
		return filteredIns;

	}

	/**
	 * Homogenize number of instances per class (in the loaded training data)
	 * @deprecated use balanceTrainingData
	 */
	public void homogenizeTrainingData()
	{
		balanceTrainingData();
	}

	/**
	 * Balance number of instances per class (in the loaded training data)
	 */
	public void balanceTrainingData()
	{
		final Resample filter = new Resample();
		Instances filteredIns = null;
		filter.setBiasToUniformClass(1.0);
		try {
			filter.setInputFormat(this.labelImageTrainingData);
			filter.setNoReplacement(false);
			filter.setSampleSizePercent(100);
			filteredIns = Filter.useFilter(this.labelImageTrainingData, filter);
		} catch (Exception e) {
			IJ.log("Error when resampling input data!");
			e.printStackTrace();
		}
		this.labelImageTrainingData = filteredIns;
	}

	/**
	 * Create training instances out of the user markings
	 * @return set of instances (feature vectors in Weka format)
	 */
	public Instances createTrainingInstancesFromROIs(boolean recomputeFeatures )
	{

		int nonEmpty = 0;

		for(int i = 0; i < getNumClasses(); i++)
		{
			for (int j = 0; j < inputImage.getImageStackSize(); j++)
			{
				if ( getNumExamples(i) > 0 )
				{
					nonEmpty++;
					break;
				}
			}
		}

		if ( nonEmpty < 2 )
		{
			logger.error( "Cannot train without at least 2 sets of examples!" );
			return null;
		}


		ArrayList< Example > examplesWithoutFeatures = new ArrayList<>();

		for( Example example : examples )
		{
			if ( recomputeFeatures )
			{
				// add all examples to the list
				examplesWithoutFeatures.add( example );
			}
			else
			{
				// add examples that need feature recomputation
				if ( example.instanceValuesArray == null )
				{
					examplesWithoutFeatures.add( example );
				}
			}
		}


		// compute feature values for examples
		//
		ArrayList< ArrayList < Example > > exampleList = new ArrayList<>();

		for (int iExampleWithoutFeatures = 0;
			 iExampleWithoutFeatures < examplesWithoutFeatures.size();
			 iExampleWithoutFeatures ++ )
		{
			// figure out which examples are spatially close,
			// put them together and compute the feature images
			// for them in one go; this saves time.
			ArrayList < Example > neighboringExamples = new ArrayList<>();

			Rectangle exampleBounds = getExampleBounds( examplesWithoutFeatures.get(iExampleWithoutFeatures) );

			Point3D exampleLocation = new Point3D(
					exampleBounds.getX(),
					exampleBounds.getY(),
					examplesWithoutFeatures.get(iExampleWithoutFeatures).z
					);

			neighboringExamples.add( examplesWithoutFeatures.get(iExampleWithoutFeatures) );

			Boolean includeNextExample = true;

			iExampleWithoutFeatures++;
			while ( includeNextExample && (iExampleWithoutFeatures < examplesWithoutFeatures.size() ) )
			{
				Rectangle nextExampleBounds = getExampleBounds( examplesWithoutFeatures.get(iExampleWithoutFeatures) );

				Point3D nextExampleLocation = new Point3D(
						nextExampleBounds.getX(),
						nextExampleBounds.getY(),
						examplesWithoutFeatures.get(iExampleWithoutFeatures).z
				);

				if ( exampleLocation.distance( nextExampleLocation ) < getFeatureVoxelSizeAtMaximumScale() )
				{
					neighboringExamples.add( examplesWithoutFeatures.get( iExampleWithoutFeatures ) );
					iExampleWithoutFeatures++;
				}
				else
				{
					includeNextExample = false;
					iExampleWithoutFeatures--;
				}

			}

			exampleList.add(neighboringExamples) ;
		}

		// Compute feature values for examples
		//
		ExecutorService exe = Executors.newFixedThreadPool( numRegionThreads );
		ArrayList<Future> futures = new ArrayList<>();
		boolean updateFeatureList = true;
		for ( int i = 0; i < exampleList.size(); i++ )
		{
			ArrayList< Example > neighboringExamples = exampleList.get( i );
			futures.add(
					exe.submit(
							setExamplesInstanceValues(
									neighboringExamples,
									i, exampleList.size() - 1,
									updateFeatureList) ) );
			updateFeatureList = false; // only needed once
			this.totalThreadsExecuted.addAndGet(1);
		}

		trainableDeepSegmentation.utils.Utils.joinThreads( futures, logger );
		exe.shutdown();

		// TODO:
		// - there is a bug, as it quite often never reaches below line
		// or at least does not log anything
		if ( exampleList.size() > 0 )
		{
			logger.info( "Computed feature values for all new annotations." );
		}

		// prepare training data
		int numActiveFeatures = getNumActiveFeatures();
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int f = 0; f < numActiveFeatures; f++)
		{
			attributes.add( new Attribute("feat_"+f) );
		}
		attributes.add( new Attribute("class", getClassNamesAsArrayList() ) );

		// initialize set of instances
		Instances trainingData = new Instances("segment", attributes, 1);
		// Set the index of the class attribute
		trainingData.setClassIndex( numActiveFeatures );

		// add and report training values
		int[] numExamplesPerClass = new int[ getNumClassesInExamples() ];
		int[] numExamplePixelsPerClass = new int[ getNumClassesInExamples() ];

		for ( Example example : examples )
		{
			// loop over all pixels of the example
			for ( double[] values : example.instanceValuesArray )
			{
				// loop over all features of the pixel
				double[] activeValues = new double[ numActiveFeatures + 1 ]; // +1, for class value
				int iActiveValue = 0;
				for ( int i = 0; i < values.length - 1; i++ ) // -1, because last value is the class value
				{
					// during previous training, features that were
					// not used in the RF often enough
					// were set inactive.
					if ( settings.featureList.get(i).isActive )
					{
						activeValues[ iActiveValue++ ] = values[ i ];
					}
				}
				activeValues[ iActiveValue ] = example.classNum;
				trainingData.add(new DenseInstance(1.0, activeValues));
			}
			numExamplesPerClass[ example.classNum ] += 1;
			numExamplePixelsPerClass[ example.classNum ] += example.instanceValuesArray.size();
		}

		logger.info("## Annotation information: ");
		for ( int iClass = 0; iClass < getNumClassesInExamples(); iClass++ )
		{
			logger.info(getClassNames().get(iClass) + ": "
					+ numExamplesPerClass[iClass] + " labels; "
					+ numExamplePixelsPerClass[iClass] + " pixels");
		}

		if (trainingData.numInstances() == 0)
			return null;

		logger.info("Memory usage [MB]: " + IJ.currentMemory()/1000000L + "/" + IJ.maxMemory()/1000000L);

		return trainingData;
	}

	private Runnable setExamplesInstanceValues( ArrayList< Example > examples,
												int counter,
												int counterMax,
												boolean updateFeatureList)
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return () -> {

			logger.info("" + (counter+1) + "/" + (counterMax+1) + ": " +
					"Computing features for " + examples.size() + " labels..." );

			// determine which dataCube of the image that we need
			int[][] bounds = getExamples3DBounds( examples );
			int[] borders = getFeatureBorderSizes();
			int[] sizes = new int[3];

			for ( int i = 0; i < 3; ++i )
			{
				// add one to width and height, as, e.g., a horizontal line has zero height.
				int exampleWidth = ( bounds[i][1] - bounds[i][0] + 1 );
				sizes[i] = borders[i] * ( 2 + (int) Math.ceil( 1.0 * exampleWidth / borders[i] ) );
				sizes[i]++; // This is necessary for the 2D case, where otherwise the z-size would be zero

				if ( sizes[i] > imgDims[i] )
				{
					logger.error("Cannot compute the image features, " +
							"because size of features if too large " +
							"compared with the size of the image. " +
							"Please go to [Settings] and reduce the downsampling " +
							"factor or the maximum downsampling level.");
				}
				//sizes[i] = sizes[i] > imgDims[i] ? imgDims[i] : sizes[i];
			}

			Point3D center = new Point3D(
					( bounds[0][0] + bounds[0][1] ) / 2,
					( bounds[1][0] + bounds[1][1] ) / 2,
					( bounds[2][0] + bounds[2][1] ) / 2
			);

			final Region5D region5D = new Region5D();
			region5D.t = examples.get(0).t;
			region5D.c = 0;
			region5D.size = new Point3D( sizes[0], sizes[1], sizes[2] );
			region5D.offset = Utils.computeOffsetFromCenterSize( center, region5D.size );
			region5D.offset = shiftOffsetToStayInBounds( region5D.offset, region5D.size );
			region5D.subSampling = new Point3D(1, 1, 1);

			// compute feature values for pixels at examples
			// compute all features, because we do not know,
			// which ones we might need in the future for the trainings
			ArrayList < FeatureImagesMultiResolution > featureImagesChannels= new ArrayList<>();
			for ( int c : settings.activeChannels)
			{
				region5D.c = c;
				FeatureImagesMultiResolution featureImages = new FeatureImagesMultiResolution();
				featureImages.setOriginalImage( Utils.getDataCube(inputImage, region5D, new int[]{-1,-1}, 1 ) );
				featureImages.wekaSegmentation = this;
				featureImages.updateFeaturesMT( "ch" + c, false,
						featuresToShow, numThreadsPerRegion,
						true); // computeAll = true
				featureImagesChannels.add( featureImages );
			}

			/* update feature list, which might have been changed during
			this training, because the user might have altered the
			feature computation settings.
			this is only need for one of the new examples, thus the
			if statement
			*/
			if ( updateFeatureList )
			{
				updateFeatureList(featureImagesChannels);
			}

			final int[] borderSizes = getFeatureBorderSizes();

			// prepare reusable featureSlice arrays
			ArrayList < double[][][] > featureSlices = new ArrayList<>();
			for (int c = 0; c < settings.activeChannels.size(); c++ )
			{
				featureSlices.add( new double
						[featureImagesChannels.get(c).getWidth() - 2 * borderSizes[0] ]
						[featureImagesChannels.get(c).getHeight() - 2 * borderSizes[1] ]
						[featureImagesChannels.get(c).getNumFeatures() ]
				);
			}

			// extract the feature values at
			// the respective z-position of each example
			for ( Example example : examples )
			{
				// obtain feature values at example's z-position
				long start = System.currentTimeMillis();

				int z = (int) (example.z - region5D.offset.getZ());
				int xs = borderSizes[0];
				int xe = featureImagesChannels.get(0).getWidth() - borderSizes[0] - 1;
				int ys = borderSizes[1];
				int ye = featureImagesChannels.get(0).getHeight() - borderSizes[1] - 1;

				for (int c = 0; c < settings.activeChannels.size(); c++ )
				{
					featureImagesChannels.get( c ).setFeatureSliceRegion(
							z, xs, xe, ys, ye, featureSlices.get( c ));

				}
				long duration = System.currentTimeMillis() - start;

				// code for debugging
				//
				// get and set feature values at the x,y positions of the example
				// Point[] points = example.points;
				// int randomNum = ThreadLocalRandom.current().nextInt(1, 100 + 1);
				// IJ.log("NEW EXAMPLE " + randomNum);

				example.instanceValuesArray = new ArrayList<>();
				for ( Point point : getPointsFromExample( example ) )
				{
					// set instance values for this pixel
					//
					int x = (int) (point.getX() - region5D.offset.getX() - xs);
					int y = (int) (point.getY() - region5D.offset.getY() - ys);

					example.instanceValuesArray.add(
							getFeatureValues( featureSlices, x, y, example.classNum ) );

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
			}

			logger.info("" + (counter+1) + "/" + (counterMax+1) + ": " + "...done" );

		};
	}

	private synchronized void updateFeatureList( ArrayList < FeatureImagesMultiResolution > featureImagesChannels )
	{
		settings.featureList = new ArrayList<>();

		for (int c = 0; c < settings.activeChannels.size(); ++c)
		{
			for (String featureName : featureImagesChannels.get(c).getFeatureNames())
			{
				settings.featureList.add( new Feature(featureName, 0, true) );
			}
		}
	}

	public long getNeededBytesPerVoxel()
	{
		long oneByte = 8;
		long floatingPointImp = 32;
		long mem = (long) memoryFactor * floatingPointImp / oneByte;
		return ( mem );
	}

	public long getMaximalNumberOfVoxelsPerRegion()
	{
		long maxMemory = IJ.maxMemory();
		long currentMemory = IJ.currentMemory();
		long freeMemory = maxMemory - currentMemory;

		long maxNumVoxelsPerRegion = (long) 1.0 * freeMemory /
				( getNeededBytesPerVoxel() * numRegionThreads * numThreadsPerRegion );

		long maxNumRegionWidth = (long) Math.pow( maxNumVoxelsPerRegion, 1.0/3 );

		logger.setShowDebug( true );
		logger.debug( "memory factor " + memoryFactor );
		logger.debug( "maxNumVoxelsPerRegion " + maxNumVoxelsPerRegion );
		logger.debug( "memoryPerRegionMemoryEstimate [MB] " +
				(maxNumVoxelsPerRegion * getNeededBytesPerVoxel() / 1000000) );

		return maxNumVoxelsPerRegion;
	}

	public int getMaximalRegionSize()
	{
		// TODO: this is wrong if the regions are not cubic...
		int maxNumRegionWidth = (int) Math.pow( getMaximalNumberOfVoxelsPerRegion(), 1.0/3 );
		// to keep it kind of interactive limit the maximal size
		// to something (500 is arbitrary)
		maxNumRegionWidth = Math.min( maxNumRegionWidth, 500 );

		// remove borders, which go into the memory
		// considerations, but should not be explicitely
		// asked for
		maxNumRegionWidth -= 2 * getFeatureBorderSizes()[0];
		return maxNumRegionWidth;
	}


	private int[] point3DToInt( Point3D point3D )
	{
		int[] xyz = new int[3];
		xyz[0] = (int) point3D.getX();
		xyz[1] = (int) point3D.getY();
		xyz[2] = (int) point3D.getZ();
		return( xyz );
	}

	public Point3D shiftOffsetToStayInBounds( Point3D pOffset, Point3D pSizes )
	{
		// ensure to stay within image bounds
		int[] offset = point3DToInt( pOffset );
		int[] sizes = point3DToInt( pSizes );

		for (int i = 0; i < 3; ++i)
		{
			offset[i] = offset[i] + sizes[i] > imgDims[i] ? imgDims[i] - sizes[i] : offset[i];
			offset[i] = offset[i] < 0 ? 0 : offset[i];
		}

		Point3D pOffsetShifted = new Point3D(offset[0], offset[1], offset[2]);
		return ( pOffsetShifted );
	}

	private Point[] getPointsFromExample(Example example)
	{
		final int width = Math.round(example.strokeWidth);
		Point[] p = example.points;
		int n = example.points.length;

		double x1, y1;
		double x2=p[0].getX()-(p[1].getX()-p[0].getX());
		double y2=p[0].getY()-(p[1].getY()-p[0].getY());

		//double x2=p.xpoints[0]-(p.xpoints[1]-p.xpoints[0]);
		//double y2=p.ypoints[0]-(p.ypoints[1]-p.ypoints[0]);
		ArrayList < Point > points = new ArrayList<>();

		for (int i=0; i<n; i++)
		{
			x1 = x2;
			y1 = y2;
			x2 = p[i].getX();
			y2 = p[i].getY();

			if ( (x1!=x2) || (y1!=y2) )
			{
				double dx = x2 - x1;
				double dy = y1 - y2;
				double length = (float) Math.sqrt(dx * dx + dy * dy);
				dx /= length;
				dy /= length;
				double x = x2 - dy * (width-1) / 2.0;
				double y = y2 - dx * (width-1) / 2.0;

				int n2 = width;
				do
				{
					int ix = (int) x;
					int iy = (int) y;
					points.add(new Point(ix, iy));
					x += dy;
					y += dx;
				} while (--n2 > 0);
			}
		}
		Point[] pointArray = points.toArray(new Point[points.size()]);
		return pointArray;
	}

	public int[][] getExample3DBounds( Example example )
	{
		ArrayList<Example> examples = new ArrayList<>();
		examples.add( example );
		int[][] bounds = getExamples3DBounds(examples);
		return( bounds );
	}

	public int[][] getExamples3DBounds( ArrayList < Example > examples )
	{

		Rectangle bounds = getExampleBounds(examples.get(0));

		int xMin = (int) bounds.getMinX();
		int xMax = (int) bounds.getMaxX();
		int yMin = (int) bounds.getMinY();
		int yMax = (int) bounds.getMaxY();
		int zMin = examples.get(0).z;
		int zMax = examples.get(0).z;

		for (Example example : examples)
		{
			bounds = getExampleBounds( example );

			xMin = (int) bounds.getMinX() < xMin ? (int) bounds.getMinX() : xMin;
			xMax = (int) bounds.getMaxX() > xMax ? (int) bounds.getMaxX() : xMax;

			yMin = (int) bounds.getMinY() < yMin ? (int) bounds.getMinY() : yMin;
			yMax = (int) bounds.getMaxY() > yMax ? (int) bounds.getMaxY() : yMax;

			zMin = example.z < zMin ? example.z : zMin;
			zMax = example.z > zMax ? example.z : zMax;
		}

		int[][] bounds3D = new int[3][2];

		bounds3D[0][0] = xMin;
		bounds3D[1][0] = yMin;
		bounds3D[2][0] = zMin;
		bounds3D[0][1] = xMax;
		bounds3D[1][1] = yMax;
		bounds3D[2][1] = zMax;

		return( bounds3D );

	}

	/**
	 * Add training samples from a FreeRoi with thickness of 1 pixel
	 *
	 * @param trainingData set of instances to add to
	 * @param classIndex class index value
	 * @param sliceNum number of 2d slice being processed
	 * @param r thin free line roi
	 * @return number of instances added
	 */
	public int getFeatureVoxelSizeAtMaximumScale()
	{
		int maxFeatureVoxelSize = (int) Math.pow( settings.downSamplingFactor,
				settings.maxResolutionLevel );
		return maxFeatureVoxelSize;
	}

	public int[] getFeatureBorderSizes()
	{
		// TODO:
		// - check whether this is too conservative
		int[] borderSize = new int[3];
		borderSize[0] = borderSize[1] = getFeatureVoxelSizeAtMaximumScale();
		// 2D case:
		borderSize[2] = ( imgDims[2] == 1 ) ? 0 : (int) (1.0 * getFeatureVoxelSizeAtMaximumScale() / settings.anisotropy);
		//borderSize[2] = borderSize[2] == 0 ? 1 : borderSize[2];
		return( borderSize );
	}

	public int[][] getClassifiableImageBorders()
	{
		int[] borderSizes = getFeatureBorderSizes();

		int[][] borders = new int[3][2];

		borders[0][0] = borderSizes[0];
		borders[1][0] = borderSizes[1];
		borders[2][0] = borderSizes[2];
		borders[0][1] = inputImage.getWidth() - borderSizes[0] - 1;
		borders[1][1] = inputImage.getHeight() - borderSizes[1] - 1;
		borders[2][1] = inputImage.getNSlices() - borderSizes[2] - 1;

		return( borders );
	}

	public ArrayList<String> getClassNamesAsArrayList()
	{
		ArrayList<String> classes = new ArrayList<>();
		for(int i = 0; i < getNumClasses(); i++)
		{
			classes.add("class"+i);
		}
		return classes;
	}

	/*
	 This is needed for the 2nd training, because
	 the feature values are not recomputed, but the
	 classifier is trained using only the selected features
	 of the first training; this means that
	 it is critical that the sequence in which the features
	 are computed during the actual classification
	 is the same is it was during the training.
	 This should be fine, but is not entirely trivial, because
	 the feature computation itself is altered, because only
	 the necessary features are computed!
	*/
	public int getNumActiveFeatures()
	{
		int numActiveFeatures = 0;

		if ( settings.featureList != null )
		{
			for ( Feature feature : settings.featureList )
			{
				if ( feature.isActive )
				{
					numActiveFeatures++;
				}
			}
			return ( numActiveFeatures );
		}
		else
		{
			return ( 0 );
		}
	}

	public int getNumAllFeatures()
	{
		if ( settings.featureList == null )
		{
			return ( 0 );
		}
		else
		{
			return ( settings.featureList.size() );
		}
	}


	/**
	 * Add instances from a labeled image in a random and balanced way.
	 * For convention, the label zero is used to define pixels with no class
	 * assigned. The rest of integer values correspond to the order of the
	 * classes (1 for the first class, 2 for the second class, etc.).
	 *
	 * @param labelImage labeled image (labels are positive integer or 0)
	 * @param featureStack corresponding feature stack
	 * @param numSamples number of samples to add of each class
	 * @return false if error
	 */

	/**
	 * Questions:
	 * - how many of the label image pixels did you use and why?
	 * - can you give an example of why balancing is important?
	 *
	 * 	 */
	public Instances createTrainingInstancesFromLabelImageRegion(
			Region5D region5D,
			int numInstancesPerClass,
			boolean balanceInstances,
			boolean recomputeFeatures)
	{

		int xs = (int) region5D.offset.getX();
		int ys = (int) region5D.offset.getY();
		int zs = (int) region5D.offset.getZ();
		int nx = (int) region5D.size.getX();
		int ny = (int) region5D.size.getY();
		int nz = (int) region5D.size.getZ();

		// Load inputImage region into RAM
		ImagePlus trainingImageRegion = Utils.getDataCube(inputImage,
				region5D, new int[]{-1,-1}, 1 );

		// Compute features
		FeatureImagesMultiResolution featureImages = new FeatureImagesMultiResolution();
		featureImages.setOriginalImage( trainingImageRegion );
		featureImages.wekaSegmentation = this;
		featureImages.updateFeaturesMT(
				"ch" + region5D.c,
				true,
				featuresToShow,
				numThreadsPerRegion,
				true );


		// Create lists of coordinates of pixels of each class
		//
		ArrayList<Point3D>[] classCoordinates = new ArrayList[ getNumClasses() ];
		for(int i = 0; i < getNumClasses() ; i ++)
			classCoordinates[ i ] = new ArrayList<>();

		for(int z = zs; z < zs + nz; ++z )
		{
			ImageProcessor ip = labelImage.getStack().getProcessor( z + 1 );

			for (int y = ys; y < ys + ny; y++)
			{
				for (int x = xs; x < xs + nx; x++)
				{
					int classIndex = ip.get( x, y );
					classCoordinates[classIndex].add( new Point3D(x, y, z) );
				}
			}
		}

		// Get image features
		//
		ArrayList < double[][][] > featureSlices = new ArrayList<>();

		for (int z = 0; z < nz; ++z )
		{
			ArrayList < double[][][] > featureSlicesThisPlane = new ArrayList<>();

			for ( int c = 0; c < 1; c++ )
			{

			}
			double[][][] slice = new double [nx][ny][ featureImages.getNumFeatures() ];

			featureImages.setFeatureSliceRegion(
					z,
					0, nx - 1,
					0, ny - 1,
					slice );

			featureSlices.add( slice );
		}

		// Select random samples from each class
		Random rand = new Random();

		for( int i = 0; i < numInstancesPerClass; i++ )
		{
			for( int cl = 0; cl < getNumClasses() ; cl ++ )
			{
				if( !classCoordinates[ cl ].isEmpty() )
				{
					int randomSample = rand.nextInt( classCoordinates[ cl ].size() );

					int z = (int) classCoordinates[ cl ].get( randomSample ).getZ();

					// We have to put the featureSlice for this z-plane into
					// an ArrayList, because there could be multiple channels,
					// and this is what 'getFeatureValues' expects as input
					ArrayList< double[][][] > featureSliceChannels = new ArrayList<>();
					featureSliceChannels.add( featureSlices.get( z ) );

					double[] featureValues = getFeatureValues( featureSliceChannels,
							(int) classCoordinates[ cl ].get( randomSample ).getX(),
							(int) classCoordinates[ cl ].get( randomSample ).getY()
							, cl );

					DenseInstance denseInstance = new DenseInstance(
							1.0,
							featureValues);

					addInstanceToLabelImageTrainingData( denseInstance );

				}
			}
		}

		//for( int j = 0; j < numOfClasses ; j ++ )
		//	IJ.log("Added " + numSamples + " instances of '" + loadedClassNames.get( j ) +"'.");

		logger.progress("Label image training dataset updated ",
				"(" + labelImageTrainingData.numInstances() +
				" instances, " + labelImageTrainingData.numAttributes() +
				" attributes, " + labelImageTrainingData.numClasses() + " classes).");

		return null;

	}

	private synchronized void addInstanceToLabelImageTrainingData( Instance instance )
	{
		labelImageTrainingData.add( instance );
	}

	public void setLabelImage( ImagePlus labelImage )
	{
		this.labelImage = labelImage;
	}

	public void setTrainingInstancesFromLabelImage( ImagePlus labelImageTrainingData )
	{

		/*
		final long start = System.currentTimeMillis();

		// set classes
		settings.classNames = new ArrayList<>();
		int maxLabel = 1; // TODO: determine from Label image
		for (int iClass = 0; iClass <= maxLabel; ++iClass)
		{
			settings.classNames.add("class_" + iClass);
		}
		logger.info("Found " + (maxLabel + 1) + " classes in label image.");


		// Create loaded training data if it does not exist yet
		if (null == labelImageTrainingData)
		{
			IJ.log("Initializing label image training data...");

			// Create instances
			ArrayList<Attribute> attributes = getAttributes();

			labelImageTrainingData = new Instances("segment", attributes, 1);
			labelImageTrainingData.setClassIndex(labelImageTrainingData.numAttributes() - 1);
		}

		int numInstancesPerClass = 100; // TODO: how to determine this?
		boolean balanceInstances = true; // TODO: ...

		// TODO:
		// here a loop over subsets could be implemented
		// in case not everything is fitting into RAM at once
		Region5D region5D = new Region5D();
		region5D.size = new Point3D(
				labelImage.getWidth(),
				labelImage.getHeight(),
				labelImage.getNSlices());
		region5D.offset = new Point3D(0, 0, 0);
		region5D.t = 0;
		region5D.c = 0;
		region5D.subSampling = new Point3D(1, 1, 1);

		labelImageTrainingData = createTrainingInstancesFromLabelImageRegion(
				region5D,
				numInstancesPerClass,
				balanceInstances,
				numThreadsPerRegion
		);

		final long end = System.currentTimeMillis();
		logger.info("Created training data from label image in " + (end - start) + " ms");
		*/
	}

	public final int TRAIN_FROM_ROIS = 0, TRAIN_FROM_LABEL_IMAGE = 1;

	/**
	 * Train classifier with the current instances
	 * and current classifier settings
	 * and current active features
	 */
	public boolean trainClassifier( int trainingDataSource ,
									boolean recomputeFeatures,
									Region5D labelImageRegion,
									int labelImageNumInstancesPerClass)
	{
		if ( Thread.currentThread().isInterrupted() )
		{
			IJ.log("Classifier training was interrupted.");
			return false;
		}

		isTrainingCompleted = false;

		logger.info("Creating training data... ");

		if ( trainingDataSource == TRAIN_FROM_ROIS )
		{

			final long start = System.currentTimeMillis();

			trainingData = createTrainingInstancesFromROIs( recomputeFeatures );

			final long end = System.currentTimeMillis();

			logger.info("...created training data from ROIs in " + (end - start) + " ms");
		}
		else if ( trainingDataSource == TRAIN_FROM_LABEL_IMAGE)
		{
			final long start = System.currentTimeMillis();

			trainingData = createTrainingInstancesFromLabelImageRegion(
					labelImageRegion,
					labelImageNumInstancesPerClass,
					true,
					recomputeFeatures );

			final long end = System.currentTimeMillis();

			logger.info("...created training data from label image in " + (end - start) + " ms");
		}

		// Resample data if necessary
		//
		// TODO: check code and ask ignacio
		if( balanceClasses )
		{
			final long start = System.currentTimeMillis();
			IJ.showStatus("Balancing classes distribution...");
			IJ.log("Balancing classes distribution...");
			trainingData = balanceTrainingData( trainingData );
			final long end = System.currentTimeMillis();
			IJ.log("Done. Balancing classes distribution took: " + (end-start) + "ms");
		}


		// Train the classifier on the current data
		logger.info("Training classifier...");

		final long start = System.currentTimeMillis();

		if ( Thread.currentThread().isInterrupted() )
		{
			logger.warning("Classifier training was interrupted.");
			return false;
		}

		// Set up the classifier
		numRandomFeatures = (int) Math.ceil( 1.0 * getNumActiveFeatures()
				* fractionRandomFeatures );
		rf.setSeed( (new Random()).nextInt() );
		rf.setMaxDepth( maxDepth);
		rf.setNumTrees( getNumTrees() );
		rf.setNumThreads( numRfTrainingThreads);
		rf.setNumFeatures( numRandomFeatures );
		rf.setBatchSize("" + getBatchSizePercent());
		rf.setComputeImportances( false ); // using own method currently


		try
		{
			classifier.buildClassifier( trainingData );
			this.totalThreadsExecuted.addAndGet( numRfTrainingThreads );
		}
		catch (InterruptedException ie)
		{
			logger.info( "Classifier construction was interrupted.");
			return false;
		}
		catch(Exception e)
		{
			IJ.showMessage(e.getMessage());
			e.printStackTrace();
			return false;
		}

		final long end = System.currentTimeMillis();

		logger.info("Trained classifier in " + (end - start) + " ms.");

		reportClassifierCharacteristics();

		isTrainingCompleted = true;

		return true;
	}


	public void reportClassifierCharacteristics()
	{
		// Print classifier information
		logger.info( classifier.toString() );

		// Compute characteristics about the RF

		int numDecisionNodes = ((FastRandomForest) classifier).getDecisionNodes();

		int[] attributeUsages = ((FastRandomForest) classifier).getAttributeUsages();
		int iUsedFeature = 0;
		int totalFeatureUsage = 0;

		for ( int usage : attributeUsages )
		{
			totalFeatureUsage += usage;
		}

		for ( Feature feature : settings.featureList )
		{
			if ( feature.isActive )
			{
				feature.usageInRF = attributeUsages[ iUsedFeature++ ];
			}
		}

		avgRfTreeSize = numDecisionNodes / getNumTrees();
		double avgTreeDepth = 1.0 + Math.log(avgRfTreeSize) / Math.log( 2.0 ) ;
		double randomFeatureUsage = 1.0 * numDecisionNodes / getNumActiveFeatures();
		minFeatureUsage = (int) Math.ceil( minFeatureUsageFactor * randomFeatureUsage );

		ArrayList<Feature> sortedFeatureList = new ArrayList<>( settings.featureList );
		sortedFeatureList.sort(Comparator.comparing(Feature::getUsageInRF).reversed());

		logger.info("# 20 most used features: ");
		for ( int f = 19; f >= 0; f--)
		{
			Feature feature = sortedFeatureList.get( f );
			int featureID = settings.featureList.indexOf( feature );
			logger.info("Usage: " + feature.usageInRF + "; ID: " + featureID +
					"; Name: " + feature.name);
		}

		logger.info("Features considered for training: "
				+ getNumActiveFeatures() +
				"/" + getNumAllFeatures() +
				";     debug info: attributeUsages.length: " + attributeUsages.length );
		logger.info("Number of decision nodes in RF: "
				+ numDecisionNodes +
				";     debug info: total feature usage in RF: " + totalFeatureUsage);
		logger.info(String.format("Random feature usage: numDecisionNodes " +
				"/ numUsedFeatures = %.2f", randomFeatureUsage));
		logger.info("Average number of decision nodes per tree: " +
				avgRfTreeSize );
		logger.info("Average tree depth: log2(numDecisionNodes) + 1 = " +
				avgTreeDepth );
		logger.info("Minimum feature usage factor: " +
				minFeatureUsageFactor);
		logger.info("Minimum feature usage: " +
				"ceil ( minFeatureUsageFactor * " +
				"randomFeatureUsage ) = " + minFeatureUsage);


	}

	/**
	 *
	 * @param region5D
	 * @param sizesMinMax
	 * @return
	 */
	public Runnable postProcess(final Region5D region5D, int[] sizesMinMax)
	{
		return () ->
		{
			if ( stopCurrentThreads || Thread.currentThread().isInterrupted())
			{
				//logger.info("Thread "+counter+"/"+counterMax+" stopped.");
				return;
			}

			/*
			logger.info("Classifying region "+counter+"/"+counterMax+" at "
							+ region5DToClassify.offset.getX() + ","
							+ region5DToClassify.offset.getY() + ","
							+ region5DToClassify.offset.getZ() + "..."
			);*/
			//logger.info("Memory usage [MB]: " + IJ.currentMemory() / 1000000L + "/" + IJ.maxMemory() / 1000000L);

			long start = System.currentTimeMillis();

			ImagePlus imp = null;

			if (region5D != null)
			{
				imp = Utils.getDataCube(inputImage, region5D, new int[]{-1,-1}, 1);
			}
			else
			{
				imp = classifiedImage; // whole image
			}

			// remove small objects
			trainableDeepSegmentation.utils.Utils.filterSmallObjects3D(imp, sizesMinMax);

			// save changed classification image
			ExecutorService exe = Executors.newFixedThreadPool(numThreadsPerRegion);
			ArrayList < Future > futures = new ArrayList<>();
			for ( int z = 0; z < imp.getNSlices(); ++z )
			{
				Region5D region5DThisSlice = new Region5D();
				region5DThisSlice.size = new Point3D(imp.getWidth(), imp.getHeight(), 1);
				region5DThisSlice.offset = new Point3D(
						region5D.offset.getX(),
						region5D.offset.getY(),
						region5D.offset.getZ() + z);
				region5DThisSlice.t = region5D.t;
				region5DThisSlice.c = 0;
				region5DThisSlice.subSampling = new Point3D(1, 1, 1);
				byte[] thisSlice = (byte[]) imp.getStack().getProcessor(z+1).getPixels();
				futures.add(
						exe.submit(
								setClassificationResult(
										classifiedImage,
										region5DThisSlice,
										thisSlice
								)
						)
				);
			}

			trainableDeepSegmentation.utils.Utils.joinThreads( futures, logger );
			exe.shutdown();
		};

	}

	public void applyClassifier( int[] xyztStart, int[] xyztEnd )
	{

		int[] imgDims = getImgDims();
		int[] tileSizes = new int[3];
		int[] xyztNumTiles = new int[4];

		// TODO:
		// - function that test whether current feature settings
		// are compatible with image dimensions!

		// tile sizes
		// TODO: put into extra function
		for ( int i = 0; i < 3; ++i )
		{
			int size = ( xyztEnd[i] - xyztStart[i] + 1 );

			if ( tileSizeSetting.equals("auto") )
			{
				if ( xyztEnd[i] - xyztStart[i] <= getMaximalRegionSize() )
				{
					// everything can be computed at once
					tileSizes[i] = size;
				}
				else
				{
					// we need to tile
					int n = (int) Math.ceil( (1.0 * size) / getMaximalRegionSize() );
					tileSizes[i] = size / n ;
				}

				// make sure sizes fit into image
				tileSizes[i] = Math.min ( tileSizes[i], imgDims[i] );


			}
		}

		// Compute some stuff
		// TODO: check the logic here again...

		// xyz
		int[] distances = new int[3];
		int iTotal = 0, numRegions = 1;
		String xyztSizes = "";

		for ( int i = 0; i < 3; ++i )
		{
			distances[i] = tileSizes[i];
			distances[i] = distances[i] < 1 ? 1 : distances[i];
			xyztNumTiles[i] = (int) Math.ceil( ( xyztEnd[i] - xyztStart[i] + 1 ) / distances[i] );
			xyztSizes += "" + (xyztEnd[i] - xyztStart[i] + 1) +" ";
			numRegions *= xyztNumTiles[i];
		}

		// time
		xyztNumTiles[3] = (xyztEnd[3] - xyztStart[3]) + 1;
		xyztSizes += "" + xyztNumTiles[3] +" ";
		numRegions *= xyztNumTiles[3];

		logger.info("Selected region size [x,y,z,t]: "  + xyztSizes );

		int[] borders = getFeatureBorderSizes();
		logger.info("Tile size (excl. borders) [x,y,z]: "
				+ ( tileSizes[0] + 2 * borders[0] )
				+ ", " + ( tileSizes[1] + 2 * borders[1] )
				+ ", " + ( tileSizes[2] + 2 * borders[2] )
				+ " (" + tileSizes[0]
				+ ", " + tileSizes[1]
				+ ", " + tileSizes[2] + ")"

		);

		// if there is only one region use all threads for it
		int numAdaptedThreadsPerRegion = ( numRegions == 1 ) ?
				Prefs.getThreads() : numThreadsPerRegion;
		int numAdaptedRegionThreads = ( numRegions == 1 ) ?
				1 : numRegionThreads;

		logger.info("Number of tiles: " + numRegions );
		logger.info("Tile threads: " + numAdaptedRegionThreads );
		logger.info("Threads per tile: " + numAdaptedThreadsPerRegion );

		// classify all tiles

		pixelsClassified.set( 0L );
		rfStatsTreesEvaluated.set( 0L );

		ExecutorService exe = Executors.newFixedThreadPool(
				numAdaptedRegionThreads );
		ArrayList< Future > futures = new ArrayList<>();

		ArrayList<int[]> positions = new ArrayList<>();

		long startTime = System.currentTimeMillis();

		for (int t = xyztStart[3]; t <= xyztEnd[3]; t += 1)
		{
			for (int z = xyztStart[2]; z <= xyztEnd[2]; z += distances[2])
			{
				for (int y = xyztStart[1]; y <= xyztEnd[1]; y += distances[1])
				{
					for (int x = xyztStart[0]; x <= xyztEnd[0]; x += distances[0])
					{
						// create region to be classified
						Region5D region5D = new Region5D();
						region5D.t = t;
						region5D.c = 0; // TODO: does not matter, right?
						region5D.offset = new Point3D(x, y, z);
						region5D.size = new Point3D( tileSizes[0], tileSizes[1], tileSizes[2] );
						region5D.subSampling = new Point3D(1, 1, 1);

						// remember, just for reporting to user
						positions.add(new int[]{
								(int)region5D.offset.getX(),
								(int)region5D.offset.getY(),
								(int)region5D.offset.getZ() + 1,
								region5D.t + 1});

						try
						{
							if (! stopCurrentThreads )
							{
								totalThreadsExecuted.addAndGet( 1 );

								futures.add(
										exe.submit(
												applyClassifierToTile(
														region5D,
														numAdaptedThreadsPerRegion,
														++iTotal,
														numRegions)
										)
								);
							}
						}
						catch ( OutOfMemoryError e )
						{
							logger.error("Out of memory: " + e.toString());
							stopCurrentThreads = true;
						}
					}
				}
			}
		}

		int regionsClassified = 0;
		long nThreadsLast = totalThreadsExecuted.get();
		long maximumMemoryUsage = 0L;
		long totalMemory = IJ.maxMemory();

		for ( Future future : futures )
		{
			try
			{
				if ( ! stopCurrentThreads )
				{
					future.get();

					regionsClassified++;
					long nThreadsNew = totalThreadsExecuted.get() - nThreadsLast;

					long milliSeconds = (System.currentTimeMillis() - startTime);
					double milliSecondsPerRegion = 1.0 * milliSeconds / regionsClassified;
					int regionsToGo = numRegions - regionsClassified;
					double minutesLeft = 1.0 * regionsToGo * milliSecondsPerRegion / (1000 * 60);
					double minutesCurr = 1.0 * milliSeconds / (1000 * 60);

					String timeInfo = String.format("Time (spent, left) [min]: " +
							"%.1f, %.1f", minutesCurr, minutesLeft );

					double rate = 1.0 * pixelsClassified.get() /
							milliSeconds;
					long avgTreesUsed = (long) 1.0 * rfStatsTreesEvaluated.get() /
							pixelsClassified.get();

					long currentMemoryUsage = IJ.currentMemory();

					if ( currentMemoryUsage > maximumMemoryUsage )
						maximumMemoryUsage = currentMemoryUsage;

					// TODO:
					// - max memory usage must be monitored during computation
					String memoryInfo = "Memory (current, max, avail) [MB]: "
							+ currentMemoryUsage / 1000000L
							+ ", " + maximumMemoryUsage / 1000000L
							+ ", " + totalMemory / 1000000L;

					String treeInfo = "Trees (avg, max, avail): "
							+ avgTreesUsed
							+ ", " + rfStatsMaximumTreesUsed
							+ ", " + getNumTrees();

					logger.progress("Region",
							Arrays.toString(positions.get(regionsClassified - 1))
									+ "; " + (regionsClassified) + "/" + numRegions
									+ "; " + timeInfo
									+ " (" + (int) (rate) + " kv/s)"
									+ "; " + treeInfo
									+ "; " + memoryInfo

					);

					nThreadsLast = totalThreadsExecuted.get();

					System.gc();

				}
				else
				{
					break;
				}
			}
			catch (InterruptedException e)
			{
				e.printStackTrace();
			}
			catch (ExecutionException e)
			{
				e.printStackTrace();
			}

		}

		// we're done
		exe.shutdown();

	}


	/**
	 * Apply current classifier to a set of feature vectors (given in a feature
	 * stack array). The classification if performed in a multi-threaded way
	 * using as many threads as defined by the user.
	 *
	 * @param featureImages feature stack array
	 * @param numThreads The number of threads to use. Set to zero for auto-detection.
	 * @param probabilityMaps probability flag. Tue: probability maps are calculated, false: binary classification
	 * @return result image containing the probability maps or the binary classification
	 */
	public Runnable applyClassifierToTile(
				final Region5D r5D,
				final int numThreads,
				final int counter,
				final int counterMax)
	{

		return () ->
		{

			// append for non-classifiable borders
			int[] borders = getFeatureBorderSizes();
			r5D.size = new Point3D(
					r5D.size.getX() + 2 * borders[0],
					r5D.size.getY() + 2 * borders[1],
					r5D.size.getZ() + 2 * borders[2]);

			r5D.offset = new Point3D(
					r5D.offset.getX() - borders[0],
					r5D.offset.getY() - borders[1],
					r5D.offset.getZ() - borders[2]);

			// TODO:
			// remove shiftOffset, once out-of-bounds are implemented!
			r5D.offset = shiftOffsetToStayInBounds(
					r5D.offset,
					r5D.size );

			if ( stopCurrentThreads || Thread.currentThread().isInterrupted())
			{
				logger.progress("Thread stopped:", "" + counter + "/" + counterMax);
				return;
			}

			// wait a bit to get threads out-of-sync
			if ( counter < numRegionThreads )
			{
				try
				{
					System.gc();
					Thread.sleep( counter * tilingDelay);
				} catch (InterruptedException e)
				{
					e.printStackTrace();
				}
			}


			//logger.info("Classifying region "+counter+"/"+counterMax+" at "
			//		+ region5DToClassify.offset.getX() + ","
			//		+ region5DToClassify.offset.getY() + ","
			//		+ region5DToClassify.offset.getZ() + "..."
			//);
			//logger.info("Memory usage [MB]: " + IJ.currentMemory() / 1000000L + "/" + IJ.maxMemory() / 1000000L);

			long start = System.currentTimeMillis();

			ImagePlus imageToClassify = null;

			final int[] borderSizes = getFeatureBorderSizes();
			int nx = 0;
			int ny = 0;
			int nz = 0;
			int zs = 0;

			ArrayList < FeatureImagesMultiResolution > featureImagesChannels = new ArrayList<>();
			for ( int c : settings.activeChannels)
			{
				r5D.c = c;
				imageToClassify = Utils.getDataCube(inputImage, r5D, new int[]{-1,-1}, 1 );

				// border pixels cannot be classified,
				// because the interpolated features
				// cannot not be computed properly
				// thus leave them out
				nx = imageToClassify.getWidth() - 2 * borderSizes[0];
				ny = imageToClassify.getHeight() - 2 * borderSizes[1];
				nz = imageToClassify.getNSlices() > 1 ? imageToClassify.getNSlices() - 2 * borderSizes[2] : 1;
				zs = imageToClassify.getNSlices() > 1 ? borderSizes[2] : 0;

				// TODO:
				// - implement better multi-channel treatment for background threshold
				// - explicitly set classification to zero
				if ( settings.backgroundThreshold > 0 )
				{
					// check whether the region is background
					if ( isBackgroundRegion( imageToClassify, settings.backgroundThreshold) )
					{
						// don't classify, but leave all classification pixels as is, hopefully 0...
						pixelsClassified.addAndGet( nx * ny * nz  );
						return;
					}
				}

				FeatureImagesMultiResolution featureImages = new FeatureImagesMultiResolution();
				featureImages.setOriginalImage( imageToClassify );
				featureImages.wekaSegmentation = this;
				featureImages.updateFeaturesMT( "ch" + c,
						true,
						featuresToShow,
						numThreads,
						false ); // computeAll = false
				featureImagesChannels.add( featureImages );
			}

			if ( counterMax == 1 )
			{
				logger.info("Features computed in [ms]: " +
						(System.currentTimeMillis() - start) +
				", using " + numThreads + " threads");
			}


			start = System.currentTimeMillis();

			// create instances information (each instance needs a pointer to this)
			Instances dataInfo = new Instances("segment", getAttributes(), 1);
			dataInfo.setClassIndex( dataInfo.numAttributes() - 1 );

			// distribute classification across different threads
			int slicesPerChunk;
			if ( nz < numThreads )
			{
				slicesPerChunk = 1;
			}
			else
			{
				slicesPerChunk = (int) Math.ceil ( 1.0 * nz / numThreads );
			}

			ExecutorService exe = Executors.newFixedThreadPool( numThreads );
			ArrayList<Future<byte[][]>> classificationFutures = new ArrayList<>();
			ArrayList<UncertaintyRegion> uncertaintyRegions = new ArrayList<>();

			for ( int slicesClassified = 0; slicesClassified < nz; slicesClassified += slicesPerChunk )
			{

				if ( stopCurrentThreads || Thread.currentThread().isInterrupted() )
				{
					logger.progress("Thread stopped:", "" + counter + "/" + counterMax);
					exe.shutdownNow();
					return;
				}

				if ( slicesClassified + slicesPerChunk >= nz )
				{
					// adapt the last chunk
					slicesPerChunk = nz - slicesClassified;
				}

				AbstractClassifier classifierCopy = null;
				try
				{
					// The Weka random forest classifiers do not need to be duplicated on each thread
					// (that saves much memory)
					if (classifier instanceof FastRandomForest || classifier instanceof RandomForest)
						classifierCopy = classifier;
					else
						classifierCopy = (AbstractClassifier) (AbstractClassifier.makeCopy(classifier));
				}
				catch (Exception e)
				{
					IJ.log("Error: classifier could not be copied to classify in a multi-thread way.");
					e.printStackTrace();
				}

				Region5D region5DThread = new Region5D();
				region5DThread.size = new Point3D(nx, ny, slicesPerChunk);
				region5DThread.offset = new Point3D(borderSizes[0], borderSizes[1], slicesClassified + zs);
				region5DThread.t = 0;
				region5DThread.c = 0;
				region5DThread.subSampling = new Point3D(1, 1, 1);

				UncertaintyRegion uncertaintyRegion = new UncertaintyRegion();
				uncertaintyRegions.add( uncertaintyRegion );

				classificationFutures.add(
						exe.submit(
								classifyRegion(
										featureImagesChannels,
										region5DThread,
										r5D,
										uncertaintyRegion,
										dataInfo,
										classifierCopy
								)
						)
				); this.totalThreadsExecuted.addAndGet(1);

			}

			ArrayList<byte[]> classificationResults = new ArrayList<>();

			// Join classifications
			try
			{
				for (Future<byte[][]> future : classificationFutures)
				{
					byte[][] data = future.get();

					if (Thread.currentThread().isInterrupted())
						return;

					for (byte[] classifiedSlice : data )
					{
						classificationResults.add( classifiedSlice );
					}
				}
			} catch (InterruptedException e)
			{
				IJ.log("INTERRPUT: " + e.toString());
				return;
			} catch (ExecutionException e)
			{
				IJ.log("ERROR: " + e.toString());
				return;
			} catch (OutOfMemoryError err)
			{
				IJ.log("ERROR: applyClassifier run out of memory. Please, "
						+ "use a smaller input image or fewer features.");
				err.printStackTrace();
				return;
			}
			// clean up
			classificationFutures = null;
			exe.shutdown();

			// unite uncertainty measurements from different threads
			//
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

			if ( counterMax == 1 )
			{
				logger.info("Classification computed in [ms]: " + (System.currentTimeMillis() - start)
						+ ", using " + numThreads + " threads");
			}

			start = System.currentTimeMillis();

			//
			// put (local) classification results into (big) results image
			//
			exe = Executors.newFixedThreadPool( numThreads );
			ArrayList<Future> savingFutures = new ArrayList<>();
			int iSlice = 0;
			for (byte[] classifiedSlice : classificationResults)
			{
				Region5D region5DThisSlice = new Region5D();
				region5DThisSlice.size = new Point3D(nx, ny, 1);
				int offsetZ;
				if ( imageToClassify.getNSlices() > 1 )
				{
					int regionOffset = (int) r5D.offset.getZ();
					offsetZ = regionOffset + borderSizes[2] + iSlice;
				}
				else
				{
					offsetZ = iSlice;
				}
				region5DThisSlice.offset = new Point3D(
						(int) r5D.offset.getX() + borderSizes[0],
						(int) r5D.offset.getY() + borderSizes[1],
						offsetZ);
				region5DThisSlice.t = r5D.t;
				region5DThisSlice.c = 0;
				region5DThisSlice.subSampling = new Point3D(1, 1, 1);

				savingFutures.add(
						exe.submit(
								setClassificationResult(
										classifiedImage,
										region5DThisSlice,
										classifiedSlice
								)
						)
				); this.totalThreadsExecuted.addAndGet( 1 );
				iSlice++;

			}

			trainableDeepSegmentation.utils.Utils.joinThreads( savingFutures, logger );
			exe.shutdown();

			if ( counterMax == 1 )
			{
				logger.info("Saved classification results in " + (System.currentTimeMillis() - start) + " ms");
			}
		};

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

	private static Runnable setClassificationResult(
			ImagePlus classifiedImage,
			Region5D region5D,
			byte[] classifiedSlice
	)
	{

		if (Thread.currentThread().isInterrupted())
			return null;

		return () -> {

			if ( classifiedImage.getStack() instanceof VirtualStackOfStacks )
			{
				VirtualStackOfStacks classifiedImageStack = (VirtualStackOfStacks) classifiedImage.getStack();
				try
				{
					classifiedImageStack.setAndSaveBytePixels(classifiedSlice, region5D);
				} catch (IOException e)
				{
					logger.warning("WekaSegmentation.setClassificationResult: " + e.toString());
				}
			}
			else
			{
				int z = (int)region5D.offset.getZ();
				int t = (int)region5D.t;
				int xs = (int)region5D.offset.getX();
				int ys = (int)region5D.offset.getY();
				int nx = (int)region5D.size.getX();
				int ny = (int)region5D.size.getY();
				int ye = ys + ny - 1;
				int xe = xs + nx - 1;

				int n = classifiedImage.getStackIndex( 1, z+1, t+1 );
				ImageProcessor ip = classifiedImage.getStack().getProcessor( n );

				int i = 0;
				for ( int y = ys ; y <= ye; y++ )
					for ( int x = xs ; x <= xe; x++ )
						ip.set(x, y, classifiedSlice[i++]);
			}

		};

	}

	public ArrayList<String> getSettingActiveFeatureNames()
	{
		ArrayList<String> names = new ArrayList<>();

		for (Feature feature : settings.featureList)
		{
			if ( feature.isActive )
				names.add( feature.name);
		}

		return names;
	}

	public ArrayList<Attribute> getAttributes()
	{
		ArrayList<Attribute> attributes = new ArrayList<>();

		for ( Feature feature : settings.featureList )
		{
			attributes.add( new Attribute( feature.name) );
		}

		attributes.add( new Attribute("class", getClassNamesAsArrayList()) );

		return attributes;
	}

	/**
	 * Classify instances concurrently
	 *
	 * @param featureImages feature stack array with the feature vectors
	 * @param dataInfo empty set of instances containing the data structure (attributes and classes)
	 * @param first index of the first instance to classify (considering the feature stack array as a 1D array)
	 * @param numInstances number of instances to classify in this thread
	 * @param classifier current classifier
	 * @param counter auxiliary counter to be able to update the progress bar
	 * @param probabilityMaps if true return a probability map for each class instead of a classified image
	 * @return classification result
	 */
	private Callable<byte[][]> classifyRegion(
			final ArrayList < FeatureImagesMultiResolution >  featureImagesChannels,
			final Region5D region5D, // region within the featureImages
			final Region5D region5Dglobal, // region within the whole image
			UncertaintyRegion uncertaintyRegion,
			final Instances dataInfo,
			final AbstractClassifier classifier)
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return new Callable<byte[][]>(){

			@Override
			public byte[][] call(){

				int nx = (int) region5D.size.getX();
				int ny = (int) region5D.size.getY();
				int nz = (int) region5D.size.getZ();

				int xs = (int) region5D.offset.getX();
				int ys = (int) region5D.offset.getY();
				int zs = (int) region5D.offset.getZ();

				int t = region5D.t;

				int xe = xs + nx - 1;
				int ye = ys + ny - 1;
				int ze = zs + nz - 1;

				int ogx = xs + (int) region5Dglobal.offset.getX();
				int ogy = ys + (int) region5Dglobal.offset.getY();
				int ogz = zs + (int) region5Dglobal.offset.getZ();

				final byte[][] classificationResult = new byte[nz][nx*ny];

				// reusable array to be filled for each instance
				final double[] values = new double[ getNumActiveFeatures() + 1 ]; // +1 for (unused) class value

				// create empty reusable instance
				final ReusableDenseInstance ins =
						new ReusableDenseInstance( 1.0, values );
				ins.setDataset( dataInfo );

				// create reusable feature slices
				ArrayList < double[][][] > featureSlices = new ArrayList<>();
				for (int c = 0; c < settings.activeChannels.size(); c++ )
				{
					featureSlices.add( new double
									[ nx ]
									[ ny ]
									[ featureImagesChannels.get(c).getNumFeatures() ]);
				}

				try
				{
					double[] result;

					int iInstanceThisSlice = 0;
					int zPrevious = -1;

					for ( int z = 0; z < nz; z++ )
					{
						if (z != zPrevious)
						{
							if (Thread.currentThread().isInterrupted())
								return null;

							zPrevious = z;

							for (int c = 0; c < settings.activeChannels.size(); c++ )
							{
								featureImagesChannels.
										get(c).
										setFeatureSliceRegion(
												zs + z, xs, xe, ys, ye,
												featureSlices.get(c));
							}

							iInstanceThisSlice = 0;
						}

						/*
						String[] valueString = new String[4];
						valueString[0] = "";
						valueString[1] = "";
						valueString[2] = "";
						valueString[3] = "";*/

						// here counting starts from 0, because
						// the featureSlice already contains
						// the proper pixels
						for (int y = 0; y < ny; y++)
						{
							for (int x = 0; x < nx; x++)
							{
								// set reusable instance
								ins.setValues( 1.0, getFeatureValues( featureSlices, x, y, -1 ) );

								boolean evalUntilSignificant = true;

								/*
								Compute classification:
								result[0] = most likley class ID
								result[1] = 2nd most likely class ID
								result[2] = most likely class probability
								result[3] = 2nd most likely class probability
								result[4] = number of trees needed to reach significance
								*/

								result = ((FastRandomForest)classifier).
										distributionForInstance( ins ,
												evalUntilSignificant,
												accuracy );

								pixelsClassified.incrementAndGet();

								// record tree usage stats

								rfStatsTreesEvaluated.addAndGet( (int) result[4] );

								if ( result[4] > rfStatsMaximumTreesUsed.get() )
								{
									rfStatsMaximumTreesUsed.set( (int) result[4] );
								}


								// double uncertainty = result[ 4 ] / numOfTrees;
								double uncertainty = result[ 2 ] - result [ 3 ];

								int certainty = (int) ( (1.0 - uncertainty) *
										(double)( CLASS_LUT_WIDTH-1 ) );
								int classOffset = (int) result[0] * CLASS_LUT_WIDTH;
								classificationResult[z][iInstanceThisSlice++] = (byte) (classOffset + certainty);

								uncertaintyRegion.sumUncertainty += uncertainty;
								uncertaintyRegion.xyzt[0] += ( x + ogx ) * uncertainty;
								uncertaintyRegion.xyzt[1] += ( y + ogy ) * uncertainty;
								uncertaintyRegion.xyzt[2] += ( z + ogz ) * uncertainty;

								if ( uncertainty > uncertaintyRegion.maxUncertainty )
								{
									uncertaintyRegion.maxUncertainty = uncertainty;
								}

								/*
								CODE FOR DEBUGGING

								int xGlobal = x + (int)region5D.offset.getX() + (int)region5Dglobal.offset.getX();
								int yGlobal = y + (int)region5D.offset.getY() + (int)region5Dglobal.offset.getY();
								int zGlobal = z + (int)region5D.offset.getZ() + (int)region5Dglobal.offset.getZ();

								if ( (xGlobal == 1838) & (yGlobal == 716) & (zGlobal == 588) )
								{
									IJ.log("global offset " + region5Dglobal.offset);
									IJ.log("local offset " + region5D.offset);
									IJ.log("x,y,z global: " + xGlobal + "," + yGlobal + "," + zGlobal);

									//featureImages.getMultiResolutionFeatureImageArray()[0][0].show();

									for (int f = 0; f < nf; f++)
									{
										if (f < 7)
											valueString[0] = valueString[0] + f + ":" + (featureSlice[x][y][f]) + ", ";
										else if (f < 7 + 24)
											valueString[1] = valueString[1] + f + ":" +(int) Math.round(featureSlice[x][y][f]) + ", ";
										else if (f < 7 + 24 + 27)
											valueString[2] = valueString[2] + f + ":" +(int) Math.round(featureSlice[x][y][f]) + ", ";
										else
											valueString[3] = valueString[3] + f + ":" +(int) Math.round(featureSlice[x][y][f]) + ", ";
									}

									for (String s : valueString)
										IJ.log(" x,y global: " + xGlobal + "," + yGlobal + "," + zGlobal + " values " + s);

									IJ.log("Classification "+ classificationResult[z][iInstanceThisSlice-1] );

								}*/


							}
						}

					}


				}
				catch(Exception e)
				{
						IJ.showMessage("Could not apply Classifier!");
						e.printStackTrace();
						return null;
				}

				return classificationResult;
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

	/**
	 * Set current number of trees (for random forest training)
	 */
	public void setNumTrees(int n)
	{
		numOfTrees = n;
	}

	/**
	 * Get current number of trees (for random forest training)
	 * @return number of trees
	 */
	public int getNumTrees()
	{
		return numOfTrees;
	}

	public void setBatchSizePercent( int percent )
	{
		batchSizePercent = percent;
	}

	public int getBatchSizePercent()
	{
		return batchSizePercent;
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

	public boolean isFeatureNeeded(String featureImageTitle)
	{
		if ( ! isTrainingCompleted )
		{
			// during training always all features are needed,
			// because we do not know yet which are going to be
			// useful enough
			return true;
		}

		for ( String featureName : settings.activeFeatureNames)
		{
			if ( featureName.equals(featureImageTitle) )
			{
				return ( true );
			}
		}
		return ( false );
	}

	public boolean isFeatureOrChildrenNeeded(String featureImageTitle)
	{
		if ( ! isTrainingCompleted )
		{
			// during training always all features are needed,
			// because we do not know yet which are going to be
			// useful enough
			return true;
		}

		for ( String featureName : settings.activeFeatureNames)
		{
			if ( featureName.contains( featureImageTitle ) )
			{
				return ( true );
			}
		}
		return ( false );
	}

	public String getActiveChannelsAsString()
	{
		String ss = "";
		for ( int s : settings.activeChannels)
		{
			if ( !ss.equals("") )
				ss += ("," + (s+1)); // one-based
			else
				ss += ""+(s+1);
		}
		return ss;
	}

	public void setActiveChannelsFromString(String activeChannels)
	{
		String[] ss = activeChannels.split(",");
		this.settings.activeChannels = new ArrayList<>();
		for ( String s : ss)
		{
			this.settings.activeChannels.add(Integer.parseInt(s.trim()) - 1); // zero-based
		}
	}

	/**
	 * Get maximum depth of the random forest
	 * @return maximum depth of the random forest
	 */
	public int getMaxDepth()
	{
		return maxDepth;
	}

	/**
	 * Set the flag to balance the class distributions
	 * @param homogenizeClasses boolean flag to enable/disable the class balance
	 * @deprecated use setDoClassBalance
	 */
	public void setDoHomogenizeClasses( boolean homogenizeClasses )
	{
		this.balanceClasses = homogenizeClasses;
	}

	/**
	 * Get the boolean flag to enable/disable the class balance
	 * @return flag to enable/disable the class balance
	 * @deprecated use doClassBalance
	 */
	public boolean doHomogenizeClasses()
	{
		return balanceClasses;
	}

	/**
	 * Set the flag to balance the class distributions
	 * @param balanceClasses boolean flag to enable/disable the class balance
	 */
	public void setDoClassBalance( boolean balanceClasses )
	{
		this.balanceClasses = balanceClasses;
	}

	/**
	 * Get the boolean flag to enable/disable the class balance
	 * @return flag to enable/disable the class balance
	 */
	public boolean doClassBalance()
	{
		return balanceClasses;
	}

	/**
	 * Set feature update flag
	 * @param updateFeatures new feature update flag
	 */
	public void setUpdateFeatures(boolean updateFeatures)
	{
		this.updateFeatures = updateFeatures;
	}

	/**
	 * Forces the feature stack to be updated whenever it is needed next.
	 */
	public void setFeaturesDirty()
	{
		updateFeatures = true;
		return;
	}

	/**
	 * Update fast random forest classifier with new values
	 *
	 * @param newNumTrees new number of trees
	 * @param newRandomFeatures new number of random features per tree
	 * @param newMaxDepth new maximum depth per tree
	 * @return false if error
	 */
	public boolean updateClassifier(
			int newNumTrees,
			int newRandomFeatures,
			int newMaxDepth)
	{
		if(newNumTrees < 1 || newRandomFeatures < 0)
			return false;
		numOfTrees = newNumTrees;
		numRandomFeatures = newRandomFeatures;
		maxDepth = newMaxDepth;

		rf.setNumTrees(numOfTrees);
		rf.setNumFeatures(numRandomFeatures);
		rf.setMaxDepth(maxDepth);

		return true;
	}

	/**
	 * Merge two datasets of Weka instances in place
	 * @param first first (and destination) dataset
	 * @param second second dataset
	 */
	public void mergeDataInPlace(Instances first, Instances second)
	{
		for(int i=0; i<second.numInstances(); i++)
			first.add(second.get(i));
	}

	/**
	 * Get the current class labels
	 * @return array containing all the class labels
	 */
	public ArrayList<String> getClassNames()
	{
		return settings.classNames;
	}

}

