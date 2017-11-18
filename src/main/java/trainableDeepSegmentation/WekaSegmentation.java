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
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import bigDataTools.logging.Logger;
import bigDataTools.logging.IJLazySwingLogger;
import ij.gui.PolygonRoi;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import javafx.geometry.Point3D;

import hr.irb.fastRandomForest.FastRandomForest;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Prefs;
import ij.gui.Roi;
import net.imglib2.FinalInterval;
import trainableDeepSegmentation.classification.ClassifierManager;
import trainableDeepSegmentation.examples.Example;
import trainableDeepSegmentation.results.ResultImage;
import trainableDeepSegmentation.results.ResultImageFrameSetter;
import trainableDeepSegmentation.instances.InstancesCreator;
import trainableDeepSegmentation.instances.InstancesManager;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
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
 * based on the Weka classifiersComboBox.
 */
public class WekaSegmentation {

	/**
	 * maximum number of classes (labels) allowed
	 */
	public static final int MAX_NUM_CLASSES = 10;

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

	public ResultImage getResultImageBgFg()
	{
		return resultImageBgFg;
	}

	public void setResultImageBgFg( ResultImage resultImageBgFg )
	{
		this.resultImageBgFg = resultImageBgFg;
	}

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
	 * names of the current classes
	 */

	private long[] imgDims = new long[5];

	// Random Forest parameters
	/**
	 * current number of trees in the fast random forest classifier
	 */
	private int numOfTrees = 200;
	/**
	 * number of random features per node in the fast random forest classifier
	 */
	private int numRandomFeatures = 50;
	/**
	 * fraction of random features per node in the fast random forest classifier
	 */
	public double fractionRandomFeatures = 0.1;
	/**
	 * maximum depth per tree in the fast random forest classifier
	 */
	public int maxDepth = 0;
	/**
	 * maximum depth per tree in the fast random forest classifier
	 */
	private int batchSizePercent = 10;
	/**
	 * list of class names on the loaded data
	 */
	private ArrayList<String> loadedClassNames = null;

	private boolean updateFeatures = false;

	public int[] minTileSizes = new int[]{162, 162, 81};

	public String tileSizeSetting = "auto";

	public Settings settings = new Settings();

	public double minFeatureUsageFactor = 0.0; // TODO: does not seem

	private boolean computeFeatureImportance = false;

	public int regionThreads = (int) ( Math.sqrt( Prefs.getThreads()) + 0.5 );

	public int threadsPerRegion = (int) ( Math.sqrt(Prefs.getThreads()) + 0.5 );

	public int numRfTrainingThreads = Prefs.getThreads();

	public int tilingDelay = 2000; // milli-seconds

	public double uncertaintyLUTdecay = 0.5;

	public double accuracy = 4.0;

	public double memoryFactor = 1.0;

	public int labelImageNumInstancesPerClass = 1000;

	public static IJLazySwingLogger logger = new IJLazySwingLogger();

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

	/**
	 * @param logFileName
	 * @param logFileDirectory
	 * @return returns true if the log file could be sucessfully created
	 */

	public boolean setLogFile( String directory )
	{
		String logFileDirectory = directory.substring(0, directory.length() - 1)
				+ "--log";
		String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").
				format(new Date());
		String logFileName = "log-" + timeStamp + ".txt";

		logger.setLogFileNameAndDirectory( logFileName, logFileDirectory );
		return (true);
	}

	public boolean getComputeFeatureImportance()
	{
		return computeFeatureImportance;
	}

	/**
	 * use neighborhood flag
	 */
	private boolean useNeighbors = false;

	public static final String SEM_IMAGING = "Scanning EM";
	public static final String FLUORESCENCE_IMAGING = "Fluorescence";

	public String getImagingModality()
	{
		return imagingModality;
	}

	public void setImagingModality( String imagingModality )
	{
		this.imagingModality = imagingModality;
	}

	private String imagingModality = FLUORESCENCE_IMAGING;

	public AtomicInteger totalThreadsExecuted = new AtomicInteger(0);

	public AtomicLong pixelsClassified = new AtomicLong(0);

	public AtomicLong rfStatsTreesEvaluated = new AtomicLong(0);

	private AtomicInteger rfStatsMaximumTreesUsed = new AtomicInteger(0);

	public int getNumLabelImageInstancesPerPlaneAndClass()
	{
		return labelImageInstancesPerPlaneAndClass;
	}

	public void setLabelImageInstancesPerPlaneAndClass( int labelImageInstancesPerPlaneAndClass )
	{
		this.labelImageInstancesPerPlaneAndClass = labelImageInstancesPerPlaneAndClass;
	}

	private int labelImageInstancesPerPlaneAndClass = 50;

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


	private FinalInterval labelImageInterval = null;
	private FeatureProvider labelImageFeatureProvider = null;
	private ArrayList< int[][] > labelImageClassificationAccuraciesHistory = null;
	private ArrayList< Integer > numInstancesHistory = null;
	private ImageProcessor ipLabelImageInstancesDistribution = null;
	// show dialog
	final String NEW = "Start new instances";
	final String APPEND = "Append to previous instances";

	public void trainIterativeFromLabelImage(
			String instancesKey,
			String modality,
			int numIterations,
			FinalInterval interval )
	{

		boolean isFirstTime = false;

		if ( modality.equals( NEW ) )
		{
			IntervalUtils.logInterval( interval );

			logger.info( "\n# Computing features for label image region...");
			labelImageFeatureProvider = new FeatureProvider( this );
			labelImageFeatureProvider.isLogging( true );
			labelImageFeatureProvider.setInterval( interval );
			labelImageFeatureProvider.setActiveChannels( settings.activeChannels );
			labelImageFeatureProvider.computeFeatures( Prefs.getThreads() );

			labelImageInterval = interval;
			if ( labelImageInterval == null ) return;
			labelImageClassificationAccuraciesHistory = new ArrayList<>();
			numInstancesHistory = new ArrayList<>();
			ipLabelImageInstancesDistribution = new ShortProcessor(
					( int ) labelImageInterval.dimension( IntervalUtils.X ),
					( int ) labelImageInterval.dimension( IntervalUtils.Y ) );

			isFirstTime = true;

			batchSizePercent = 100; // since we chose uncorrelated features anyway

		}

		for ( int i = 0; i < numIterations; ++i )
		{

			for ( int z = (int) labelImageInterval.min( IntervalUtils.Z ); z <= labelImageInterval.max( IntervalUtils.Z ); ++z )
			{

				logger.info("\n# Iterative instances from label image, iteration " + i + ", slice " + z);

				if ( stopCurrentTasks )
				{
					return;
				}

				// create new instances

				FinalInterval newInstancesInterval =
						IntervalUtils.fixDimension( labelImageInterval, IntervalUtils.Z, z );

				Instances instances = InstancesCreator.createUsefulInstancesFromLabelImage(
						this,
						getLabelImage(),
						getResultImage(),
						ipLabelImageInstancesDistribution,
						labelImageFeatureProvider,
						instancesKey,
						newInstancesInterval,
						getNumLabelImageInstancesPerPlaneAndClass(),
						Prefs.getThreads(),
						logger,
						isFirstTime);


				if ( isFirstTime && modality.equals( NEW ) )
				{
					instancesKey = getInstancesManager().putInstances( instances );
				}
				else
				{
					instancesKey = getInstancesManager().appendInstances( instances );
				}

				isFirstTime = false;

				instances = getInstancesManager().getInstances( instancesKey );

				numInstancesHistory.add( instances.size() );

				InstancesManager.logInstancesInformation( instances, logger );

				if ( instances == null ) return;

				FastRandomForest classifier = trainClassifier( instances );

				if ( classifier == null ) return;

				String key = getClassifierManager().setClassifier( classifier, instances );

				if ( minFeatureUsageFactor > 0 )
				{
					// TODO: does not work
					performFeatureSelection();
				}

				// apply classifier to next z-plane
				logger.info( "\n# Apply classifier" );

				int classificationZ = z + 1;
				if( classificationZ > labelImageInterval.max( IntervalUtils.Z ) )
				{
					classificationZ = (int) labelImageInterval.min( IntervalUtils.Z );
				}

				FinalInterval classificationInterval =
						IntervalUtils.fixDimension( labelImageInterval, IntervalUtils.Z, classificationZ );

				applyClassifierWithTiling(
						key,
						classificationInterval,
						Prefs.getThreads(),
						labelImageFeatureProvider );


				// record progress
				ImageProcessor ipAccuracy = null;

				if ( classificationZ == labelImageInterval.max( IntervalUtils.Z ) )
				{
					// record progress
					ipAccuracy = new ShortProcessor(
							( int ) labelImageInterval.dimension( IntervalUtils.X ),
							( int ) labelImageInterval.dimension( IntervalUtils.Y ) );
				}

				labelImageClassificationAccuraciesHistory.add(
						InstancesCreator.getAccuracies(
								getLabelImage(),
								getResultImage(),
								ipAccuracy,
								labelImageInterval
						) );


				if( classificationZ == labelImageInterval.max( IntervalUtils.Z ) )
				{
					ImagePlus impCorrectness = new ImagePlus( "correctness " + i, ipAccuracy );
					impCorrectness.show();
				}

			}


		}

		logger.info( "..done." );

		// report what happened
		long intervalVolume = labelImageInterval.dimension( IntervalUtils.X )
				* labelImageInterval.dimension( IntervalUtils.Y )
				* labelImageInterval.dimension( IntervalUtils.Z );

		for ( int i = 0; i < numInstancesHistory.size(); ++i )
		{
			InstancesCreator.reportClassificationAccuracies( labelImageClassificationAccuraciesHistory.get( i ), logger );
			logger.info( "# Instances: Total: " + numInstancesHistory.get( i )
					+ String.format("; Percent: %.2f" , ( 100.0 * numInstancesHistory.get( i ) / intervalVolume ) ));
		}

		ImagePlus impInstancesDistribution = new ImagePlus( "instance distribution" , ipLabelImageInstancesDistribution );
		impInstancesDistribution.show();


	}

	public void performFeatureSelection()
	{

		/*
		ArrayList< Integer > goners = AttributeSelector.getGoners(
				classifier,
				instances,
				minFeatureUsageFactor,
				getLogger() );

		Instances instances2 = InstancesCreator.removeAttributes(
				instances, goners );

		logger.info( "\n# Second Training" );

		FastRandomForest classifier2 = trainClassifier( instances2 );

		key = getClassifierManager().
				setClassifier( classifier2, instances2 );
				*/
	}


	/**
	 * flag to set the resampling of the instances data in order to guarantee
	 * the same number of instancesComboBox per class (class balance)
	 */
	private boolean balanceClasses = false;

	/**
	 * Project folder name. It is used to stored temporary data if different from null
	 */
	private String projectFolder = null;

	private ArrayList<Integer> featuresToShow = null;

	/**
	 * executor service to launch threads for the library operations
	 */
	private ExecutorService exe = Executors.newFixedThreadPool(threadsPerRegion);

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
	public WekaSegmentation(ImagePlus trainingImage)
	{
		initialize();
		setInputImage(trainingImage);
	}

	private void setInputImageDimensions()
	{
		imgDims[ IntervalUtils.X ] = inputImage.getWidth();
		imgDims[ IntervalUtils.Y ] = inputImage.getHeight();
		imgDims[ IntervalUtils.C ] = inputImage.getNChannels();
		imgDims[ IntervalUtils.Z ] = inputImage.getNSlices();
		imgDims[ IntervalUtils.T ] = inputImage.getNFrames();
	}

	public long[] getInputImageDimensions()
	{
		return (imgDims);
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
		settings.classNames.add(new String("background"));
		settings.classNames.add(new String("foreground"));

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
	 * Set the instances image (single image or stack)
	 *
	 * @param imp instances image
	 */
	public void setInputImage(ImagePlus imp)
	{
		this.inputImage = imp;
		setInputImageDimensions();
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
	public ArrayList<Roi> getExampleRois(int classNum, int z, int t)
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
		return settings.classNames.size();
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
	public String getClassName(int classNum)
	{
		return getClassNames().get(classNum);
	}


	/**
	 * Get the label name of a class.
	 *
	 * @param classNum class index
	 */
	public void addClass(String className)
	{
		settings.classNames.add(className);
	}

	public void setResultImage( ResultImage resultImage )
	{
		this.resultImage = resultImage;
	}

	public ResultImage getResultImage()
	{
		return ( resultImage );
	}


	/**
	 * bag class for getting the result of the loaded classifier
	 */
	private static class LoadedProject {
		private AbstractClassifier newClassifier = null;
		private Settings newSettings = null;
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
			objectOutputStream.writeObject(settings);
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
	 * Read header classifier from a .model file
	 *
	 * @param pathName complete path and file name
	 * @return false if error
	 */
	public boolean loadProject(String pathName) throws IOException
	{
		FastRandomForest newClassifier = null;
		Settings newSettings = null;
		ArrayList<Example> newExamples = null;

		logger.info("Loading project from disk...");

		File selected = new File(pathName);
		try
		{
			InputStream is = new FileInputStream(selected);
			if (selected.getName().endsWith(".gz"))
			{
				is = new GZIPInputStream(is);
			}
			try
			{
				ObjectInputStream objectInputStream = new ObjectInputStream(is);
				newClassifier = (FastRandomForest) objectInputStream.readObject();
				newSettings = (Settings) objectInputStream.readObject();
				newExamples = (ArrayList<Example>) objectInputStream.readObject();
				objectInputStream.close();
			} catch (Exception e)
			{
				logger.error("Error while loading project");
				logger.info(e.toString());
				return false;
			}
		} catch (Exception e)
		{
			logger.error("Error while loading project");
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

		setClassifier(newClassifier);
		setSettings(newSettings);
		setLabelROIs(newExamples);

		logger.info("...loaded project: " + pathName);

		return true;
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
	 * Set current settings
	 *
	 * @param settings
	 */
	public void setSettings(Settings settings)
	{
		this.settings = settings;
	}


	/**
	 * Balance number of instancesComboBox per class
	 *
	 * @param data input set of instancesComboBox
	 * @return resampled set of instancesComboBox
	 */
	public static Instances balanceTrainingData(Instances data)
	{
		final Resample filter = new Resample();
		Instances filteredIns = null;
		filter.setBiasToUniformClass(1.0);
		try
		{
			filter.setInputFormat(data);
			filter.setNoReplacement(false); // with replacement => adds more of underrepresented class
			filter.setSampleSizePercent(100);
			filteredIns = Filter.useFilter(data, filter);
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
										 Instances instances )
	{

		ArrayList< Integer > originalClasses = setInstancesClassesToBgFg( instances );
		FastRandomForest classifierBgFg = trainClassifier( instances );
		String classifierBgFgKey = classifierManager.setClassifier( classifierBgFg, instances );
		resetInstancesClasses( instances, originalClasses );

		// reroute classification to BgFg result image
		ResultImage originalResultImage = resultImage;
		resultImage = resultImageBgFg;

		applyClassifierWithTiling( classifierBgFgKey,
				interval, null, null );

		// reset
		resultImage = originalResultImage;

		ImagePlus imp1 = resultImageBgFg.getDataCube( interval );
		imp1.setTitle( "BgFg" );
		imp1.show();

		resultImage.getDataCube( interval ).show();


		// apply bgfg classifier

		// apply distance transfrom to bgfg image

		// add bgfg result image to active channels

		// recompute instances, now including the bgfg result image

		/*
		updateExamples( true );
		Instances instances2 = InstancesCreator.createInstancesAndMetadataFromLabels(
				getExamples(),
				getInputImageTitle(),
				settings,
				latestFeatureNames,
				getClassNames() );
		*/


		//


		// train classifier with new instances and all classes

		// apply classifier




	}


	public void updateExamples( boolean recomputeFeatures )
	{
		ArrayList<Example> examplesWithoutFeatures = new ArrayList<>();

		for ( Example example : examples )
		{
			if ( recomputeFeatures )
			{
				// add all examples to the list
				examplesWithoutFeatures.add( example );
			}
			else
			{
				// add examples that need feature re-computation
				if ( example.instanceValuesArray == null )
				{
					examplesWithoutFeatures.add( example );
				}
			}
		}

		// compute feature values for examples
		//
		ArrayList<ArrayList<Example>> exampleList = new ArrayList<>();

		for (int iExampleWithoutFeatures = 0;
			 iExampleWithoutFeatures < examplesWithoutFeatures.size();
			 iExampleWithoutFeatures++)
		{
			// figure out which examples are spatially close,
			// put them together and compute the feature images
			// for them in one go; this saves time.
			ArrayList<Example> neighboringExamples = new ArrayList<>();

			Rectangle exampleBounds = getExampleRectangleBounds(examplesWithoutFeatures.get(iExampleWithoutFeatures));

			Point3D exampleLocation = new Point3D(
					exampleBounds.getX(),
					exampleBounds.getY(),
					examplesWithoutFeatures.get(iExampleWithoutFeatures).z
			);

			neighboringExamples.add( examplesWithoutFeatures.get(iExampleWithoutFeatures) );

			Boolean includeNextExample = true;

			iExampleWithoutFeatures++;
			while (includeNextExample && (iExampleWithoutFeatures < examplesWithoutFeatures.size()))
			{
				Rectangle nextExampleBounds = getExampleRectangleBounds(examplesWithoutFeatures.get(iExampleWithoutFeatures));

				Point3D nextExampleLocation = new Point3D(
						nextExampleBounds.getX(),
						nextExampleBounds.getY(),
						examplesWithoutFeatures.get(iExampleWithoutFeatures).z
				);

				if ( exampleLocation.distance(nextExampleLocation) < getFeatureVoxelSizeAtMaximumScale())
				{
					neighboringExamples.add(examplesWithoutFeatures.get(iExampleWithoutFeatures));
					iExampleWithoutFeatures++;
				}
				else
				{
					includeNextExample = false;
					iExampleWithoutFeatures--;
				}

			}

			exampleList.add(neighboringExamples);
		}

		// Compute feature values for examples
		//
		ExecutorService exe = Executors.newFixedThreadPool(regionThreads);
		ArrayList<Future> futures = new ArrayList<>();

		isUpdatedFeatureList = false;

		for (int i = 0; i < exampleList.size(); i++)
		{
			ArrayList<Example> neighboringExamples = exampleList.get(i);
			futures.add(
					exe.submit(
							this.setExamplesInstanceValues(
									neighboringExamples,
									i, exampleList.size() - 1)));
			this.totalThreadsExecuted.addAndGet(1);
		}

		ThreadUtils.joinThreads(futures, logger);
		exe.shutdown();

		// TODO:
		// - there is a bug, as it quite often never reaches below line
		// or at least does not log anything
		if (exampleList.size() > 0)
		{
			logger.info("Computed feature values for all new annotations.");
		}

		return;

	}

	public ArrayList< String > latestFeatureNames = null;


	private Runnable setExamplesInstanceValues(ArrayList<Example> examples,
											   int counter,
											   int counterMax )
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return () -> {

			logger.info("" + (counter + 1) + "/" + (counterMax + 1) + ": " +
					"Computing features for " + examples.size() + " labels...");

			FinalInterval exampleListBoundingInterval = getExampleListBoundingInterval( examples );

			FeatureProvider featureProvider = new FeatureProvider( this );
			featureProvider.setActiveChannels(settings.activeChannels);
			featureProvider.setInterval(exampleListBoundingInterval);
			featureProvider.computeFeatures(threadsPerRegion);

			int nf = featureProvider.getNumAllFeatures();

			this.latestFeatureNames = featureProvider.getAllFeatureNames();

			double[][][] featureSlice  = featureProvider.getReusableFeatureSlice();;

			// extract the feature values at
			// the respective z-position of each example
			for (Example example : examples)
			{
				example.instanceValuesArray = new ArrayList<>();

				int z = example.z;

				featureProvider.setFeatureSlicesValues( z, featureSlice, 1 );

				for ( Point point : getPointsFromExample(example) )
				{
					// global coordinates of this example point
					int x = (int) point.getX();
					int y = (int) point.getY();

					// Note: x and y are global coordinates
					// setFeatureValuesAndClassIndex will use the exampleListBoundingInterval
					// to compute the correct coordinates in the featureSlice
					// TODO: check that this extracts  the right values!
					double[] values = new double[nf + 1];

					featureProvider.setFeatureValuesAndClassIndex(
							values, x, y, featureSlice, example.classNum);

					example.instanceValuesArray.add( values );

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

			logger.info("" + (counter + 1) + "/" + (counterMax + 1) + ": " + "...done");

		};
	}

	public long getNeededBytesPerVoxel()
	{
		long oneByte = 8;
		long floatingPointImp = 32;
		long mem = (long) memoryFactor * floatingPointImp / oneByte;
		return (mem);
	}

	public long getMaximalNumberOfVoxelsPerRegion()
	{
		long maxMemory = IJ.maxMemory();
		long currentMemory = IJ.currentMemory();
		long freeMemory = maxMemory - currentMemory;

		long maxNumVoxelsPerRegion = (long) 1.0 * freeMemory /
				(getNeededBytesPerVoxel() * regionThreads * threadsPerRegion);

		long maxNumRegionWidth = (long) Math.pow(maxNumVoxelsPerRegion, 1.0 / 3);

		//log.setShowDebug(true);
		//log.debug("memory factor " + memoryFactor);
		//log.debug("maxNumVoxelsPerRegion " + maxNumVoxelsPerRegion);
		//log.debug("memoryPerRegionMemoryEstimate [MB] " +
		//		(maxNumVoxelsPerRegion * getNeededBytesPerVoxel() / 1000000));

		return maxNumVoxelsPerRegion;
	}

	public int getMaximalRegionSize()
	{
		// TODO: this is wrong if the regions are not cubic...
		int maxNumRegionWidth = (int) Math.pow(getMaximalNumberOfVoxelsPerRegion(), 1.0 / 3);
		// to keep it kind of interactive limit the maximal size
		// to something (500 is arbitrary)
		maxNumRegionWidth = Math.min(maxNumRegionWidth, 500);

		// remove borders, which go into the memory
		// considerations, but should not be explicitely
		// asked for
		maxNumRegionWidth -= 2 * getFeatureBorderSizes()[ IntervalUtils.X];
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

		min[ IntervalUtils.X] = (int) rectangle.getMinX();
		max[ IntervalUtils.X] = (int) rectangle.getMaxX();

		min[ IntervalUtils.Y] = (int) rectangle.getMinY();
		max[ IntervalUtils.Y] = (int) rectangle.getMaxY();

		min[ IntervalUtils.Z] = examples.get(0).z;
		max[ IntervalUtils.Z] = examples.get(0).z;

		min[ IntervalUtils.T] = examples.get(0).t;
		max[ IntervalUtils.T] = examples.get(0).t;

		for ( Example example : examples )
		{
			rectangle = getExampleRectangleBounds( example );

			min[ IntervalUtils.X] = (int) rectangle.getMinX() < min[ IntervalUtils.X] ? (int) rectangle.getMinX() : min[ IntervalUtils.X];
			max[ IntervalUtils.X] = (int) rectangle.getMaxX() > max[ IntervalUtils.X] ? (int) rectangle.getMaxX() : max[ IntervalUtils.X];


			min[ IntervalUtils.Y] = (int) rectangle.getMinY() < min[ IntervalUtils.Y] ? (int) rectangle.getMinY() : min[
					IntervalUtils.Y];
			max[ IntervalUtils.Y] = (int) rectangle.getMaxY() > max[ IntervalUtils.Y] ? (int) rectangle.getMaxY() : max[ IntervalUtils.Y];

			min[ IntervalUtils.Z] = example.z < min[ IntervalUtils.Z] ? example.z : min[ IntervalUtils.Z];
			max[ IntervalUtils.Z] = example.z > max[ IntervalUtils.Z] ? example.z : max[ IntervalUtils.Z];
		}



		FinalInterval interval = new FinalInterval( min, max );

		return (interval);

	}

	/**
	 * Add instances samples from a FreeRoi with thickness of 1 pixel
	 *
	 * @param trainingData set of instances to add to
	 * @param classIndex   class index value
	 * @param sliceNum     number of 2d slice being processed
	 * @param r            thin free line roi
	 * @return number of instances added
	 */
	public int getFeatureVoxelSizeAtMaximumScale()
	{
		int maxFeatureVoxelSize = (int) Math.pow(settings.binFactor,
				settings.maxBinLevel );
		return maxFeatureVoxelSize;
	}

	public int[] getFeatureBorderSizes()
	{
		// TODO:
		// - check whether this is too conservative
		int[] borderSize = new int[5];

		borderSize[ IntervalUtils.X] = borderSize[ IntervalUtils.Y] = getFeatureVoxelSizeAtMaximumScale();

		// Z: deal with 2-D case and anisotropy
		if (imgDims[ IntervalUtils.Z] == 1)
		{
			borderSize[ IntervalUtils.Z] = 0;
		}
		else
		{
			borderSize[ IntervalUtils.Z] = (int) (1.0 * getFeatureVoxelSizeAtMaximumScale() / settings.anisotropy);
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

	public void setLabelImage(ImagePlus labelImage)
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


	/**
	 * Train classifier with the current instances
	 * and current classifier settings
	 * and current active features
	 */
	public FastRandomForest trainClassifier( Instances instances )
	{
		isTrainingCompleted = false;

		// Train the classifier on the current data
		logger.info("\n# Train classifier");

		final long start = System.currentTimeMillis();

		if (Thread.currentThread().isInterrupted())
		{
			logger.warning("Classifier instances was interrupted.");
			return null;
		}

		// Set up the classifier
		numRandomFeatures = (int) Math.ceil(1.0 * instances.numAttributes() * fractionRandomFeatures);

		FastRandomForest classifier = new FastRandomForest();
		classifier.setSeed((new Random()).nextInt());
		classifier.setMaxDepth(maxDepth);
		classifier.setNumTrees(getNumTrees());
		classifier.setNumThreads(numRfTrainingThreads);
		classifier.setNumFeatures(numRandomFeatures);
		classifier.setBatchSize("" + getBatchSizePercent());
		classifier.setComputeImportances(false); // using own method currently

		// balance traces training data

		Instances balancedInstances = instances;

		if ( labelImage == null )
		{
			logger.info( "Balancing training data..." );
			balancedInstances = balanceTrainingData( instances );
		}

		try
		{
			classifier.buildClassifier( balancedInstances );
		}
		catch (InterruptedException ie)
		{
			logger.info("Classifier construction was interrupted.");
			return null;
		} catch (Exception e)
		{
			IJ.showMessage(e.getMessage());
			e.printStackTrace();
			return null;
		}

		final long end = System.currentTimeMillis();

		reportClassifierCharacteristics( classifier, instances );

		logger.info("Trained classifier in " + (end - start) + " ms.");

		isTrainingCompleted = true;

		return classifier;
	}

	public void applyClassifierWithTiling( String classifierKey,
										   FinalInterval interval,
										   Integer numTiles,
										   FeatureProvider featureProvider )
	{

		logger.info("\n# Apply classifier");

		// set up tiling
		ArrayList<FinalInterval> tiles = createTiles( interval, numTiles );

		// set up multi-threading

		int adaptedThreadsPerRegion = threadsPerRegion;
		int adaptedRegionThreads = regionThreads;

		if ( tiles.size() == 1 )
		{
			adaptedThreadsPerRegion = Prefs.getThreads();
			adaptedRegionThreads = 1;
		}

		logger.info("Tile threads: " + adaptedRegionThreads);
		logger.info("Threads per tile: " + adaptedThreadsPerRegion);

		// submit tiles to executor service

		ExecutorService exe = Executors.newFixedThreadPool(
				adaptedRegionThreads );
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
									featureProvider )
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

	// TODO: make static!
	public void reportClassifierCharacteristics(
			FastRandomForest classifier,
			Instances instances)
	{

		int numDecisionNodes = classifier.getDecisionNodes();

		int[] usages = classifier.getAttributeUsages();
		ArrayList< Attribute > attributes =
				Collections.list( instances.enumerateAttributes() );

		NamesAndUsages[] namesAndUsages = new NamesAndUsages[usages.length];

		for ( int i = 0; i < usages.length; ++i )
		{
			namesAndUsages[i] = new NamesAndUsages(usages[ i ], attributes.get( i ).name() );
		}

		Arrays.sort( namesAndUsages, Collections.reverseOrder() );

		double avgRfTreeSize = numDecisionNodes / getNumTrees();
		double avgTreeDepth = 1.0 + Math.log(avgRfTreeSize) / Math.log(2.0);
		double randomUsage = 1.0 * numDecisionNodes / instances.numAttributes();
		int minFeatureUsage = (int) Math.ceil( minFeatureUsageFactor * randomUsage );

		logger.info("# Most used: ");
		for (int f = Math.min( 9, namesAndUsages.length - 1 ); f >= 0; f--)
		{
			logger.info(
					String.format("Relative usage: %.1f", namesAndUsages[f].usage / randomUsage ) +
							"; Absolute usage: " + namesAndUsages[f].usage +
							"; ID: " + "" + "; Name: " + namesAndUsages[f].name );
		}


		logger.info("Average number of decision nodes per tree: " +
				avgRfTreeSize);

		logger.info("Average tree depth: log2(numDecisionNodes) + 1 = " +
				avgTreeDepth);

		logger.info("Number of features: " + instances.numAttributes() );

		logger.info("Batch size [%]: " + batchSizePercent );

		logger.info("Number of instances: " + instances.size() );

		logger.info("Number of instances per tree = numInstances * batchSizePercent/100: "
				+ batchSizePercent / 100.0 * instances.size() );

		// Print classifier information
		logger.info( classifier.toString() );

	}


	private class NamesAndUsages implements Comparable<NamesAndUsages > {

		private int usage;
		private String name;

		public NamesAndUsages( int usage, String name) {
			this.usage = usage;
			this.name = name;
		}

		@Override
		public int compareTo(NamesAndUsages o) {
			return (int)(this.usage - o.usage);
		}

	}


	private ArrayList<FinalInterval> createTiles( FinalInterval interval, Integer numTiles )
	{

		IntervalUtils.logInterval( interval );

		ArrayList<FinalInterval> tiles = new ArrayList<>();

		long[] imgDims = getInputImageDimensions();
		long[] tileSizes = new long[5];

		for ( int d : IntervalUtils.XYZ )
		{

			if ( numTiles > 0 )
			{
				tileSizes[ d ] = (int) Math.ceil ( 1.0 * interval.dimension(d) / Math.pow( numTiles, 1.0/3.0 ) );
			}
			else if ( interval.dimension(d) <= getMaximalRegionSize() )
			{
				// everything can be computed at once
				tileSizes[d] = interval.dimension(d);
			}
			else
			{
				// we need to tile
				int n = (int) Math.ceil( (1.0 * interval.dimension(d)) / getMaximalRegionSize());
				tileSizes[ d ] = (int) Math.ceil ( 1.0 * interval.dimension(d) / n );
			}

			// make sure sizes fit into image
			tileSizes[d] = Math.min( tileSizes[d], imgDims[d] );
		}

		tileSizes[ IntervalUtils.T] = 1;

		logger.info("Tile sizes [x,y,z]: "
				+ tileSizes[ IntervalUtils.X]
				+ ", " + tileSizes[ IntervalUtils.Y]
				+ ", " + tileSizes[ IntervalUtils.Z]);


		for ( int t = (int) interval.min( IntervalUtils.T); t <= interval.max( IntervalUtils.T); t += 1)
		{
			for ( int z = (int) interval.min( IntervalUtils.Z); z <= interval.max( IntervalUtils.Z); z += tileSizes[ IntervalUtils.Z])
			{
				for ( int y = (int) interval.min( IntervalUtils.Y); y <= interval.max( IntervalUtils.Y); y += tileSizes[ IntervalUtils.Y])
				{
					for ( int x = (int) interval.min( IntervalUtils.X); x <= interval.max( IntervalUtils.X); x += tileSizes[ IntervalUtils.X])
					{
						long[] min = new long[5];
						min[ IntervalUtils.X ] = x;
						min[ IntervalUtils.Y ] = y;
						min[ IntervalUtils.Z ] = z;
						min[ IntervalUtils.T ] = t;

						long[] max = new long[5];
						max[ IntervalUtils.X ] = x + tileSizes[ IntervalUtils.X ] - 1;
						max[ IntervalUtils.Y ] = y + tileSizes[ IntervalUtils.Y ] - 1;
						max[ IntervalUtils.Z ] = z + tileSizes[ IntervalUtils.Z ] - 1;
						max[ IntervalUtils.T ] = t + tileSizes[ IntervalUtils.T ] - 1;

						// make sure to stay within image bounds
						for ( int d : IntervalUtils.XYZT )
						{
							max[ d ] = Math.min( interval.max( d ), max[ d ] );
						}

						tiles.add( new FinalInterval(min, max) );

					}
				}
			}
		}

		logger.info("Number of tiles: " + tiles.size());

		return (tiles);
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

		long timeUsed = (System.currentTimeMillis() - startTime);
		double timeUsedPerTile = 1.0 * timeUsed / tileCounter;
		long regionsLeft = tiles.size() - tileCounter;
		double minutesLeft = 1.0 * regionsLeft * timeUsedPerTile / (1000 * 60);
		double minutesCurr = 1.0 * timeUsed / (1000 * 60);
		double rate = 1.0 * pixelsClassified.get() /
				timeUsed;

		String timeInfo = String.format("Time (spent, left) [min]: " +
				"%.1f, %.1f", minutesCurr, minutesLeft);
		timeInfo += " (" + (int) (rate) + " kv/s)";


		long avgTreesUsed = (long) 1.0 * rfStatsTreesEvaluated.get() /
				pixelsClassified.get();

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
				+ ", " + rfStatsMaximumTreesUsed
				+ ", " + getNumTrees();

		logger.progress("Progress", tileInfo
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
	 * using as many threads as defined by the user.
	 *
	 * @param featureImages   feature stack array
	 * @param numThreads      The number of threads to use. Set to zero for auto-detection.
	 * @param probabilityMaps probability flag. Tue: probability maps are calculated, false: binary classification
	 * @return result image containing the probability maps or the binary classification
	 */
	public Runnable applyClassifier(
			String classifierKey,
			final FinalInterval tileInterval,
			final int numThreads,
			final int tileCounter,
			final int tileCounterMax,
			final FeatureProvider externalFeatureProvider)
	{

		return () ->
		{

			if ( ThreadUtils.stopThreads( logger, stopCurrentTasks,
					tileCounter, tileCounterMax ) ) return;

			if ( tileCounter <= regionThreads )
				waitMilliseconds( tileCounter * tilingDelay);

			boolean isLogging = (tileCounter <= regionThreads);

			// TODO: check whether this is a background region
			/*
			if ( settings.backgroundThreshold > 0 )
			{
				// check whether the region is background
				if ( isBackgroundRegion( imageToClassify, settings.backgroundThreshold) )
				{
					// don't classify, but leave all classification pixels as is, hopefully 0...
					pixelsClassified.addAndGet( nx * ny * nz  );
					return;
				}
			}*/


			// compute image features if necessary
			FeatureProvider featureProvider;

			if ( externalFeatureProvider == null )
			{
				featureProvider = new FeatureProvider( this );
				featureProvider.setActiveChannels( settings.activeChannels );
				featureProvider.setInterval( tileInterval );
				featureProvider.isLogging( isLogging );
				featureProvider.computeFeatures( numThreads );
			}
			else
			{
				featureProvider = externalFeatureProvider;
			}

			featureProvider.setFeatureListSubset(
					classifierManager.getClassifierAttributeNames( classifierKey ) );


			// determine chunking
			ArrayList< long[] > zChunks = getZChunks( numThreads, tileInterval );

			// create instances information (each instance needs a pointer to this)
			Instances dataInfo = new Instances("segment",
					classifierManager.getClassifierAttributesIncludingClass( classifierKey ),
					1);

			dataInfo.setClassIndex( dataInfo.numAttributes() - 1 );

			// get result image setter
			final ResultImageFrameSetter resultSetter = resultImage.getFrameSetter( tileInterval );

			// spawn threads
			ExecutorService exe = Executors.newFixedThreadPool( numThreads );
			ArrayList< Future > futures = new ArrayList<>();
			//ArrayList< UncertaintyRegion > uncertaintyRegions = new ArrayList<>();

			long start = System.currentTimeMillis();

			if ( isLogging ) logger.info("Classifying pixels...");

			// typically there are multiple chunks,
			// so we better split of the threads amongst them.
			int numThreadsPerZChunk = 1;

			if ( zChunks.size() == 1 )
			{
				// only one zChunk => use all threads for this one
				numThreadsPerZChunk = numThreads;
			}


			for (long[] zChunk : zChunks)
			{
				if ( ThreadUtils.stopThreads( logger, exe, stopCurrentTasks,
						tileCounter, tileCounterMax ) ) return;

				UncertaintyRegion uncertaintyRegion = new UncertaintyRegion();
				uncertaintyRegions.add(uncertaintyRegion);

				futures.add(
						exe.submit(
								classifyZChunk(
										featureProvider,
										resultSetter,
										zChunk[0], zChunk[1],
										//uncertaintyRegion,
										dataInfo,
										classifierManager.getClassifier( classifierKey ),
										accuracy,
										numThreadsPerZChunk
								)
						)
				);

			}

			// wait until done
			ThreadUtils.joinThreads( futures, logger );
			exe.shutdown();
			if ( isLogging ) logger.info("Classification computed in [ms]: " + (System.currentTimeMillis() - start) + ", using " + numThreads + " threads");

			// save classification results
			start = System.currentTimeMillis();
			resultSetter.close();
			if( isLogging ) logger.info("Saved classification results in [ms]: " + (System.currentTimeMillis() - start) );

			// store uncertainty information
			//storeUncertainties();

		};
	}


	// Configure z-chunking

	private ArrayList< long[] > getZChunks( int numThreads, FinalInterval tileInterval )
	{

		ArrayList< long[] > zChunks = new ArrayList<>();

		int sliceChunk;

		if ( tileInterval.dimension( IntervalUtils.Z ) < numThreads )
		{
			sliceChunk = 1;
		}
		else
		{
			sliceChunk = (int) Math.ceil ( 1.0 * tileInterval.dimension( IntervalUtils.Z ) / numThreads );
		}

		new ArrayList<>();

		for ( long zSlice = tileInterval.min( IntervalUtils.Z );
			  zSlice <= tileInterval.max( IntervalUtils.Z );
			  zSlice += sliceChunk )
		{

			long[] zChunk = new long[2];

			if ( zSlice + sliceChunk >= tileInterval.max( IntervalUtils.Z ) )
			{
				// last chunk can be smaller
				zChunk[0] = zSlice;
				zChunk[1] = tileInterval.max( IntervalUtils.Z );
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
	 * @param dataInfo empty set of instances containing the data structure (attributes and classes)
	 * @param first index of the first instance to classify (considering the feature stack array as a 1D array)
	 * @param numInstances number of instances to classify in this thread
	 * @param classifier current classifier
	 * @param counter auxiliary counter to be able to update the progress bar
	 * @param probabilityMaps if true return a probability map for each class instead of a classified image
	 * @return classification result
	 */
	private Runnable classifyZChunk(
			FeatureProvider featureProvider,
			ResultImageFrameSetter resultSetter,
			long zMin, long zMax,
			//UncertaintyRegion uncertaintyRegion,
			final Instances dataInfo,
			final FastRandomForest classifier,
			double accuracy,
			int numThreads )
	{

		final int FIRST_CLASS_ID = 0, SECOND_CLASS_ID = 1, FIRST_CLASS_PROB = 2,
				SECOND_CLASS_PROB = 3, NUM_TREES_EVALUATED = 4;

		if ( Thread.currentThread().isInterrupted() )
			return null;

		return () -> {

			// interval is the same for all channels, thus simply take from 0th
			FinalInterval interval = featureProvider.getInterval();

			// create empty reusable instance
			double[] featureValues = new double[ featureProvider.getNumActiveFeatures() + 1];
			final ReusableDenseInstance ins = new ReusableDenseInstance( 1.0, featureValues );
			ins.setDataset( dataInfo );

			// create empty reusable feature slice
			double[][][] reusableFeatureSlice = featureProvider.getReusableFeatureSlice();

			// create empty, reusable results array
			double[] result = new double[5];

			try
			{

				for ( long z = zMin; z <= zMax; ++z )
				{

					double[][][] featureSlice = featureProvider.getCachedFeatureSlice( (int) z );
					if ( featureSlice == null )
					{
						featureProvider.setFeatureSlicesValues( ( int ) z,
								reusableFeatureSlice,
								numThreads );
						featureSlice = reusableFeatureSlice;
					}

					int a = 1;


					for ( long y = interval.min( IntervalUtils.Y ); y <= interval.max( IntervalUtils.Y ); ++y )
					{
						for ( long x = interval.min( IntervalUtils.X ); x <= interval.max( IntervalUtils.X ); ++x )
						{

							// set instance values
							featureValues = featureProvider.getValuesFromFeatureSlice( (int) x, (int) y, featureSlice );
							ins.setValues( 1.0, featureValues );

							result = classifier.distributionForInstance( ins, accuracy );

							double certainty = ( result[ FIRST_CLASS_PROB ] - result [ SECOND_CLASS_PROB ] );

							resultSetter.set( x, y, z,
									(int) result[ FIRST_CLASS_ID ], certainty );

							pixelsClassified.incrementAndGet();

							// record tree usage stats
							rfStatsTreesEvaluated.addAndGet( (int) result[ NUM_TREES_EVALUATED ] );
							if ( result[ NUM_TREES_EVALUATED ] > rfStatsMaximumTreesUsed.get() )
							{
								rfStatsMaximumTreesUsed.set( (int) result[ NUM_TREES_EVALUATED ] );
							}

						}
					}
				}

			}
			catch(Exception e)
			{
					IJ.showMessage("Could not apply Classifier!");
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

	/**
	 * Set current number of trees (for random forest instances)
	 */
	public void setNumTrees(int n)
	{
		numOfTrees = n;
	}

	/**
	 * Get current number of trees (for random forest instances)
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
	 * Merge two datasets of Weka instancesComboBox in place
	 * @param first first (and destination) dataset
	 * @param second second dataset
	 */
	public void mergeDataInPlace(Instances first, Instances second)
	{
		for(int i=0; i<second.numInstances(); i++)
			first.add(second.get(i));
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
		return settings.classNames;
	}

}

