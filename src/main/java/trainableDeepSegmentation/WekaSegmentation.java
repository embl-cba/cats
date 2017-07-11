package trainableDeepSegmentation;

import java.awt.Point;
import java.awt.Rectangle;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import bigDataTools.Region5D;
import bigDataTools.VirtualStackOfStacks.VirtualStackOfStacks;
import bigDataTools.utils.Utils;
import bigDataTools.logging.Logger;
import bigDataTools.logging.IJLazySwingLogger;
import ij.gui.PolygonRoi;
import javafx.geometry.Point3D;
import org.scijava.vecmath.Point3f;

import hr.irb.fastRandomForest.FastRandomForest;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Prefs;
import ij.gui.Roi;
import ij.process.ImageProcessor;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.pmml.consumer.PMMLClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.pmml.PMMLFactory;
import weka.core.pmml.PMMLModel;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.Resample;
import weka.gui.explorer.ClassifierPanel;


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
	public static final int MAX_NUM_CLASSES = 100;
	/** array of lists of Rois for each slice (vector index)
	 * and each class (arraylist index) of the training image */
	private ArrayList<Example> examples;
	/** image to be used in the training */
	private ImagePlus trainingImage;
	/** result image after classification */
	private ImagePlus classifiedImage;
	/** features to be used in the training */
	private FeatureImagesMultiResolution  featureImages = null;
	/** set of instances from loaded data (previously saved segmentation) */
	private Instances loadedTrainingData = null;
	/** set of instances from the user's traces */
	private Instances traceTrainingData = null;
	/** current classifier */
	private AbstractClassifier classifier = null;
	/** train header */
	private Instances trainHeader = null;
	/** default classifier (Fast Random Forest) */
	private FastRandomForest rf;
	/** flag to update the feature stack (used when there is any change on the features) */
	private boolean updateFeatures = true;
	/** current number of classes */
	private int numOfClasses = 0;
	/** names of the current classes */
	private String[] classNames = new String[MAX_NUM_CLASSES];

	public double tileSizeFactor = 1.0;

	// Random Forest parameters
	/** current number of trees in the fast random forest classifier */
	private int numOfTrees = 200;
	/** number of random features per node in the fast random forest classifier */
	private int numRandomFeatures = 30;
	/** maximum depth per tree in the fast random forest classifier */
	private int maxDepth = 0;
	/** maximum depth per tree in the fast random forest classifier */
	private int batchSizePercent = 100;
	/** list of class names on the loaded data */
	private ArrayList<String> loadedClassNames = null;

	public int minTileSize = 81;

	/** maximum sigma to use on the filters */
	private int maximumFeatureScale = 3;

	private boolean computeFeatureImportance = false;

	public int numRegionThreads = (int) Math.sqrt(Prefs.getThreads());

	public int numThreadsPerRegion = (int) Math.sqrt(Prefs.getThreads());

	public int numRfTrainingThreads = Prefs.getThreads();

	public int regionSyncTime = 2000;

	private static Logger logger = new IJLazySwingLogger();

	/** flags of filters */
	private boolean[] enabledFeatures = new boolean[]{
			true, 	/* Hessian */
			true,	/* Structure */
			false,	/* Minimum */
			false,	/* Maximum */
			true,	/* Mean */
	};

	public void setComputeFeatureImportance(boolean b)
	{
		computeFeatureImportance = b;
	}

	public boolean getComputeFeatureImportance()
	{
		return computeFeatureImportance;
	}

	/** use neighborhood flag */
	private boolean useNeighbors = false;

	/** list of the names of features to use */
	public ArrayList<String> featureNames = null;

	public AtomicInteger totalThreadsExecuted = new AtomicInteger(0);

	/**
	 * flag to set the resampling of the training data in order to guarantee
	 * the same number of instances per class (class balance)
	 * */
	private boolean balanceClasses = false;

	/** Project folder name. It is used to stored temporary data if different from null */
	private String projectFolder = null;

	private ArrayList < String > featuresToShow = null;

	/** executor service to launch threads for the library operations */
	private ExecutorService exe = Executors.newFixedThreadPool(numThreadsPerRegion);

	public boolean stopCurrentThreads = false;

	/**
	 * Default constructor.
	 *
	 * @param trainingImage The image to be segmented/trained
	 */
	public WekaSegmentation(ImagePlus trainingImage)
	{
		initialize();
		this.trainingImage = trainingImage;
		featureImages = new FeatureImagesMultiResolution( trainingImage );
	}

	/**
	 * No-image constructor. If you use this constructor, the image has to be
	 * set using setTrainingImage().
	 */
	public WekaSegmentation()
	{
		initialize();
	}

	private void initialize()
	{
		// set class label names
		for(int i=0; i<MAX_NUM_CLASSES; i++)
			this.classNames[ i ] = new String("class " + (i+1));

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

		// start with two classes
		addClass();
		addClass();

		// choose feature image implementation
		featureImages = new FeatureImagesMultiResolution();

		// initialize the examples
		examples = new ArrayList<Example>();

	}

	/**
	 * Set the training image (single image or stack)
	 *
	 * @param imp training image
	 */
	public void setTrainingImage(ImagePlus imp)
	{
		this.trainingImage = imp;

		// Initialize feature stack (no features yet)
		//featureImages = new FeatureImagesMultiResolution( trainingImage );

		// update list of examples
		//examples = new ArrayList<Example>();
		/*
		examples = new Vector[trainingImage.getImageStackSize()];
		for(int i=0; i < trainingImage.getImageStackSize(); i++)
		{
			examples[i] = new Vector<ArrayList<Roi>>(MAX_NUM_CLASSES);

			for(int j=0; j<MAX_NUM_CLASSES; j++)
				examples[i].add(new ArrayList<Roi>());

		}*/
	}

	/**
	 * Adds a ROI to the list of examples for a certain class
	 * and slice.
	 *
	 * @param classNum the number of the class
	 * @param roi the ROI containing the new example
	 * @param n number of the current slice
	 */
	public void addExample(int classNum, Point[] points, int strokeWidth, int z, int t)
	{
		examples.add(new Example(classNum, points, strokeWidth, z, t,
				enabledFeatures, maximumFeatureScale, classNames));
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
			if ( (examples.get(iExample).classNum == classNum)
					&& (examples.get(iExample).t == t)
					&& (examples.get(iExample).z == z) )
			{
				if ( i == index )
				{
					examples.remove(iExample);
					return;
				}
				i++;
			}

		}

	}

	/**
	 * Return the list of examples for a certain class.
	 *
	 * @param classNum the number of the examples' class
	 * @param n the slice number
	 */
	public List<Roi> getExampleRois(int classNum, int z, int t)
	{
		List<Roi> rois = new ArrayList<>();

		for ( Example example : examples )
		{
			if ( (example.classNum == classNum)
					&& (example.t == t)
					&& (example.z == z) )
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
				rois.add( roi );
			}

		}
		return rois;
	}

	public ArrayList< Example > getExamples()
	{
		// update class names
		for ( Example example : examples )
		{
			example.classNames = classNames;
		}
		return examples;
	}

	public void setExamples( ArrayList< Example > examples )
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


	public int getNumClassesInExamples()
	{
		Set<Integer> classNums = new HashSet<>();

		for ( Example example : examples )
		{
			classNums.add(example.classNum);
		}
		return classNums.size();
	}

	/**
	 * Set flag to homogenize classes before training
	 *
	 * @param homogenizeClasses true to resample the classes before training
	 * @deprecated use setClassBalance
	 */
	public void setHomogenizeClasses(boolean homogenizeClasses)
	{
		this.balanceClasses = homogenizeClasses;
	}

	/**
	 * Set flag to balance classes before training
	 *
	 * @param balanceClasses true to resample the classes before training
	 */
	public void setClassBalance( boolean balanceClasses )
	{
		this.balanceClasses = balanceClasses;
	}

	/**
	 * Set the current number of classes. Should not be used to create new
	 * classes. Use {@link #addClass} instead.
	 *
	 * @param numOfClasses the new number of classes
	 */
	public void setNumOfClasses(int numOfClasses) {
		this.numOfClasses = numOfClasses;
	}

	/**
	 * Get the current number of classes.
	 *
	 * @return the current number of classes
	 */
	public int getNumClasses()
	{
		return numOfClasses;
	}

	/**
	 * Add new segmentation class.
	 */
	public void addClass()
	{
		// increase number of available classes
		numOfClasses++;
	}

	/**
	 * Set the name of a class.
	 *
	 * @param classNum class index
	 * @param label new name for the class
	 */
	public void setClassLabel(int classNum, String label)
	{
		getClassNames()[classNum] = label;
	}

	/**
	 * Get the label name of a class.
	 *
	 * @param classNum class index
	 */
	public String getClassLabel(int classNum)
	{
		return getClassNames()[classNum];
	}
	/**
	 * Get current classification result
	 * @return classified image
	 */
	public ImagePlus getClassifiedImage()
	{
		return classifiedImage;
	}

	public void setClassifiedImage(ImagePlus imp )
	{
		classifiedImage = imp;
	}

	/**
	 * Get the current training header
	 *
	 * @return training header (empty set of instances with the current attributes and classes)
	 */
	public Instances getTrainHeader()
	{
		return this.trainHeader;
	}


	/**
	 * bag class for getting the result of the loaded classifier
	 */
	private static class LoadedClassifier {
		private AbstractClassifier newClassifier = null;
		private Instances newHeader = null;
	}

	/**
	 * load a binary classifier
	 *
	 * @param classifierInputStream
	 * @throws Exception
	 *             exception is thrown if the reading is not properly done, the
	 *             caller has to handle this exception
	 */
	private LoadedClassifier internalLoadClassifier(
			InputStream classifierInputStream) throws Exception {
		ObjectInputStream objectInputStream = new ObjectInputStream(
				classifierInputStream);
		LoadedClassifier lc = new LoadedClassifier();
		lc.newClassifier = (AbstractClassifier) objectInputStream.readObject();
		try { // see if we can load the header
			lc.newHeader = (Instances) objectInputStream.readObject();
		} finally {
			objectInputStream.close();
		}
		return lc;
	}

	/**
	 * load a binary classifier from a stream
	 *
	 * @param classifierInputStream
	 *            the input stream
	 * @return true if properly read, false otherwise
	 */
	public boolean loadClassifier(InputStream classifierInputStream) {
		assert classifierInputStream != null;
		try {

			LoadedClassifier loadresult = internalLoadClassifier(classifierInputStream);

			try {
				// Check if the loaded information corresponds to current state
				// of
				// the segmentator
				// (the attributes can be adjusted, but the classes must match)
				if (!adjustSegmentationStateToData(loadresult.newHeader)) {
					IJ.log("Error: current segmentator state could not be updated to loaded data requirements (attributes and classes)");
					return false;
				}
			} catch (Exception e) {
				IJ.log("Error while adjusting data!");
				e.printStackTrace();
				return false;
			}

			this.classifier = loadresult.newClassifier;
			this.trainHeader = loadresult.newHeader;

			return true;

		} catch (Exception e) {
			IJ.error("Load Failed", "Error while loading classifier");
			e.printStackTrace();
			return false;
		}
	}

	/**
	 * Read header classifier from a .model file
	 * @param filename complete path and file name
	 * @return false if error
	 */
	public boolean loadClassifier(String filename)
	{
		AbstractClassifier newClassifier = null;
		Instances newHeader = null;
		File selected = new File(filename);
		try {
			InputStream is = new FileInputStream( selected );
			if (selected.getName().endsWith(ClassifierPanel.PMML_FILE_EXTENSION))
			{
				PMMLModel model = PMMLFactory.getPMMLModel(is, null);
				if (model instanceof PMMLClassifier)
					newClassifier = (PMMLClassifier)model;
				else
					throw new Exception("PMML model is not a classification/regression model!");
			}
			else
			{
				if (selected.getName().endsWith(".gz"))
					is = new GZIPInputStream(is);

				try {
					LoadedClassifier loadresult = internalLoadClassifier(is);
					newHeader = loadresult.newHeader;
					newClassifier = loadresult.newClassifier;
				} catch (Exception e) {
					IJ.error("Load Failed", "Error while loading train header");
					e.printStackTrace();
					return false;
				}

			}
		}
		catch (Exception e)
		{
			IJ.error("Load Failed", "Error while loading classifier");
			e.printStackTrace();
			return false;
		}

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

		this.classifier = newClassifier;
		this.trainHeader = newHeader;

		return true;
	}

	/**
	 * Returns the current classifier.
	 */
	public AbstractClassifier getClassifier() {
		return classifier;
	}

	/**
	 * Write current classifier into a file
	 *
	 * @param filename name (with complete path) of the destination file
	 * @return false if error
	 */
	public boolean saveClassifier(String filename)
	{
		File sFile = null;
		boolean saveOK = true;

		IJ.log("Saving model to file...");

		try {
			sFile = new File(filename);
			OutputStream os = new FileOutputStream(sFile);
			if (sFile.getName().endsWith(".gz"))
			{
				os = new GZIPOutputStream(os);
			}
			ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
			objectOutputStream.writeObject(classifier);
			trainHeader = trainHeader.stringFreeStructure();
			if (trainHeader != null)
				objectOutputStream.writeObject(trainHeader);
			objectOutputStream.flush();
			objectOutputStream.close();
		}
		catch (Exception e)
		{
			IJ.error("Save Failed", "Error when saving classifier into a file");
			saveOK = false;
		}
		if (saveOK)
			IJ.log("Saved model into " + filename );

		return saveOK;
	}


	/**
	 * Get the current feature stack array
	 * @return current feature stack array
	 */
	public FeatureImagesMultiResolution  getFeatureImages()
	{
		return this.featureImages;
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
     * Set current training header, and attempt to adjust segmentation state to it.
     * @param newHeader the header to set
     * @return true if adjustment was successful
     */
    public boolean setTrainHeader(final Instances newHeader)
    {
        if (adjustSegmentationStateToData(newHeader))
        {
            this.trainHeader = newHeader;
            return true;
        }
        else
        {
            return false;
        }
    }



	/**
	 * Add binary training data from input and label images.
	 * Input and label images can be 2D or stacks and their
	 * sizes must match.
	 *
	 * @param inputImage input grayscale image
	 * @param labelImage binary label image
	 * @param whiteClassName class name for the white pixels
	 * @param blackClassName class name for the black pixels
	 * @return false if error
	 */
	public boolean addBinaryData(
			ImagePlus inputImage,
			ImagePlus labelImage,
			String whiteClassName,
			String blackClassName)
	{

		// Check sizes
		if(labelImage.getWidth() != inputImage.getWidth()
				|| labelImage.getHeight() != inputImage.getHeight()
				|| labelImage.getImageStackSize() != inputImage.getImageStackSize())
		{
			IJ.log("Error: label and training image sizes do not fit.");
			return false;
		}

		final ImageStack inputSlices = inputImage.getImageStack();
		final ImageStack labelSlices = labelImage.getImageStack();

		for(int i=1; i <= inputSlices.getSize(); i++)
		{

			// Process label pixels
			final ImagePlus labelIP = new ImagePlus ("labels", labelSlices.getProcessor(i).duplicate());
			// Make sure it's binary
			labelIP.getProcessor().autoThreshold();

			/*
			final FeatureStack featureStack = new FeatureStack(new ImagePlus("slice " + i, inputSlices.getProcessor(i)));
			featureStack.setEnabledFeatures(enabledFeatures);
			featureStack.setMembranePatchSize(membranePatchSize);
			featureStack.setMembraneSize(this.membraneThickness);
			featureStack.setMaximumFeatureScale(this.maximumFeatureScale);
			featureStack.setMinimumSigma(this.minimumSigma);
			featureStack.updateFeaturesMT(computeFeatureImportance);
			filterFeatureStackByList(this.featureNames, featureStack);

			featureStack.setUseNeighbors(this.featureImages.useNeighborhood());

			if(!this.addBinaryData(labelIP, featureStack, whiteClassName, blackClassName))
			{
				IJ.log("Error while loading binary label data from slice " + i);
				return false;
			}
			*/
		}
		return true;
	}


	/**
	 * Add binary training data from input and label images in a
	 * random and balanced way (same number of samples per class).
	 * Input and label images can be 2D or stacks and their
	 * sizes must match.
	 *
	 * @param inputImage input grayscale image
	 * @param labelImage binary label image
	 * @param whiteClassName class name for the white pixels
	 * @param blackClassName class name for the black pixels
	 * @param numSamples number of samples to pick for each class
	 * @return false if error
	 */
	public boolean addRandomBalancedBinaryData(
			ImagePlus inputImage,
			ImagePlus labelImage,
			String whiteClassName,
			String blackClassName,
			int numSamples)
	{

		// Check sizes
		if(labelImage.getWidth() != inputImage.getWidth()
				|| labelImage.getHeight() != inputImage.getHeight()
				|| labelImage.getImageStackSize() != inputImage.getImageStackSize())
		{
			IJ.log("Error: label and training image sizes do not fit.");
			return false;
		}

		final ImageStack inputSlices = inputImage.getImageStack();
		final ImageStack labelSlices = labelImage.getImageStack();

		for(int i=1; i <= inputSlices.getSize(); i++)
		{
			// Process label pixels
			final ImagePlus labelIP = new ImagePlus ("labels", labelSlices.getProcessor(i).duplicate());
			// Make sure it's binary
			labelIP.getProcessor().autoThreshold();

			IJ.log("Feature stack is now updated.");

			/*
			if(!addRandomBalancedBinaryData(labelIP.getProcessor(), featureStack, whiteClassName, blackClassName,
					numSamples))
			{
				IJ.log("Error while loading binary label data from slice " + i);
				return false;
			}*/
		}
		return true;
	}

	/**
	 * Set pre-loaded training data (not from the user traces)
	 * @param data new data
	 */
	public void setLoadedTrainingData(Instances data)
	{
		this.loadedTrainingData = data;
	}

	/**
	 * Force segmentator to use all available features
	 */
	public void useAllFeatures()
	{
		/*
		boolean[] enableFeatures = this.enabledFeatures ;
		for (int i = 0; i < enableFeatures.length; i++)
			enableFeatures[i] = true;
		this.featureImages.setEnabledFeatures(enableFeatures);
		*/
	}

	/**
	 * Set the project folder
	 * @param projectFolder complete path name for project folder
	 */
	public void setProjectFolder(final String projectFolder)
	{
		this.projectFolder = projectFolder;
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
			filter.setInputFormat(this.loadedTrainingData);
			filter.setNoReplacement(false);
			filter.setSampleSizePercent(100);
			filteredIns = Filter.useFilter(this.loadedTrainingData, filter);
		} catch (Exception e) {
			IJ.log("Error when resampling input data!");
			e.printStackTrace();
		}
		this.loadedTrainingData = filteredIns;
	}

	/**
	 * Select attributes of current data by BestFirst search.
	 * The data is reduced to the selected attributes (features).
	 *
	 * @return false if the current dataset is empty
	 */
	public boolean selectAttributes()
	{
		if(null == loadedTrainingData)
		{
			IJ.error("There is no data so select attributes from.");
			return false;
		}
		// Select attributes by BestFirst
		loadedTrainingData = selectAttributes(loadedTrainingData);
		// Update list of features to use
		this.featureNames = new ArrayList<String>();
		IJ.log("Selected attributes:");
		for(int i = 0; i < loadedTrainingData.numAttributes(); i++)
		{
			this.featureNames.add(loadedTrainingData.attribute(i).name());
			IJ.log((i+1) + ": " + this.featureNames.get(i));
		}

		return true;
	}

	/**
	 * Select attributes using BestFirst search to reduce
	 * the number of parameters per instance of a dataset
	 *
	 * @param data input set of instances
	 * @return resampled set of instances
	 */
	public static Instances selectAttributes(Instances data)
	{
		final AttributeSelection filter = new AttributeSelection();
		Instances filteredIns = null;
		// Evaluator
		final CfsSubsetEval evaluator = new CfsSubsetEval();
		evaluator.setMissingSeparate(true);
		// Assign evaluator to filter
		filter.setEvaluator(evaluator);
		// Search strategy: best first (default values)
		final BestFirst search = new BestFirst();
		filter.setSearch(search);
		// Apply filter
		try {
			filter.setInputFormat(data);

			filteredIns = Filter.useFilter(data, filter);
		} catch (Exception e) {
			IJ.log("Error when resampling input data with selected attributes!");
			e.printStackTrace();
		}
		return filteredIns;

	}


	/**
	 * Get confusion matrix (binary images)
	 * @param proposedLabels proposed binary labels
	 * @param expectedLabels original binary labels
	 * @param whiteClassIndex index of white class
	 * @param blackClassIndex index of black class
	 * @return confusion matrix
	 */
	public int[][] getConfusionMatrix(
			ImagePlus proposedLabels,
			ImagePlus expectedLabels,
			int whiteClassIndex,
			int blackClassIndex)
	{
		int[][] confusionMatrix = new int[2][2];

		// Compare labels
		final int height = proposedLabels.getHeight();
		final int width = proposedLabels.getWidth();
		final int depth = proposedLabels.getStackSize();


		for(int z=1; z <= depth; z++)
			for(int y=0; y<height; y++)
				for(int x=0; x<width; x++)
				{
					if( expectedLabels.getImageStack().getProcessor(z).get(x, y) > 0)
					{
						if( proposedLabels.getImageStack().getProcessor(z).get(x, y) > 0 )
							confusionMatrix[whiteClassIndex][whiteClassIndex] ++;
						else
							confusionMatrix[whiteClassIndex][blackClassIndex] ++;
					}
					else
					{
						if( proposedLabels.getImageStack().getProcessor(z).get(x, y) > 0 )
							confusionMatrix[blackClassIndex][whiteClassIndex] ++;
						else
							confusionMatrix[blackClassIndex][blackClassIndex] ++;
					}
				}

		return confusionMatrix;
	}




	/**
	 * Calculate the confusion matrix of a slice (2 classes)
	 * @param proposal probability image (single 2D slice)
	 * @param expectedLabels original binary labels
	 * @param threshold threshold to apply to proposal
	 * @return confusion matrix (first row: black, second raw: white)
	 */
	public static Callable<int[][]> confusionMatrixBinarySlice(
			final ImageProcessor proposal,
			final ImageProcessor expectedLabels,
			final double threshold)
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return new Callable<int[][]>(){
			@Override
			public int[][] call()
			{
				int[][] confusionMatrix = new int[2][2];
				for(int y=0; y<proposal.getHeight(); y++)
					for(int x=0; x<proposal.getWidth(); x++)
					{
						double pix = proposal.getPixelValue(x, y) > threshold ? 1.0 : 0.0;

						if( expectedLabels.get(x, y) > 0)
						{
							if( pix > 0 )
								confusionMatrix[1][1] ++;
							else
								confusionMatrix[1][0] ++;
						}
						else
						{
							if( pix > 0 )
								confusionMatrix[0][1] ++;
							else
								confusionMatrix[0][0] ++;
						}
					}
				return confusionMatrix;
			}
		};
	}



	/**
	 * Update the class attribute of "loadedTrainingData" from
	 * the input binary labels. The number of instances of "loadedTrainingData"
	 * must match the size of the input labels image (or stack)
	 *
	 * @param labels input binary labels (single image or stack)
	 * @param className1 name of the white (different from 0) class
	 * @param className2 name of the black (0) class
	 */
	public void udpateDataClassification(
			ImagePlus labels,
			String className1,
			String className2)
	{

		// Detect class indexes
		int classIndex1 = 0;
		for(classIndex1 = 0 ; classIndex1 < this.getClassNames().length; classIndex1++)
			if(className1.equalsIgnoreCase(this.getClassNames()[classIndex1]))
				break;
		if(classIndex1 == this.getClassNames().length)
		{
			IJ.log("Error: class named '" + className1 + "' not found.");
			return;
		}
		int classIndex2 = 0;
		for(classIndex2 = 0 ; classIndex2 < this.getClassNames().length; classIndex2++)
			if(className2.equalsIgnoreCase(this.getClassNames()[classIndex2]))
				break;
		if(classIndex2 == this.getClassNames().length)
		{
			IJ.log("Error: class named '" + className2 + "' not found.");
			return;
		}

		updateDataClassification(this.loadedTrainingData, labels, classIndex1, classIndex2);
	}

	/**
	 * Update the class attribute of "data" from
	 * the input binary labels. The number of instances of "data"
	 * must match the size of the input labels image (or stack)
	 *
	 * @param data input instances
	 * @param labels binary labels
	 * @param classIndex1 index of the white (different from 0) class
	 * @param classIndex2 index of the black (0) class
	 */
	public static void updateDataClassification(
			Instances data,
			ImagePlus labels,
			int classIndex1,
			int classIndex2)
	{
		// Check sizes
		final int size = labels.getWidth() * labels.getHeight() * labels.getStackSize();
		if (size != data.numInstances())
		{
			IJ.log("Error: labels size does not match loaded training data set size.");
			return;
		}

		final int width = labels.getWidth();
		final int height = labels.getHeight();
		final int depth = labels.getStackSize();
		// Update class with new labels
		for(int n=0, z=1; z <= depth; z++)
		{
			final ImageProcessor slice = labels.getImageStack().getProcessor(z);
			for(int y=0; y<height; y++)
				for(int x=0; x<width; x++, n++)
					data.get(n).setClassValue(slice.getPixel(x, y) > 0 ? classIndex1 : classIndex2);

		}
	}

	/**
	 * Update the class attribute of "data" from
	 * the input binary labels. The number of instances of "data"
	 * must match the size of the input labels image (or stack)
	 *
	 * @param data input instances
	 * @param labels binary labels
	 * @param classIndex1 index of the white (different from 0) class
	 * @param classIndex2 index of the black (0) class
	 */
	public static void updateDataClassification(
			Instances data,
			ImagePlus labels,
			int classIndex1,
			int classIndex2,
			ArrayList<Point3f>[] mismatches)
	{
		// Check sizes
		final int size = labels.getWidth() * labels.getHeight() * labels.getStackSize();
		if (size != data.numInstances())
		{
			IJ.log("Error: labels size does not match loaded training data set size.");
			return;
		}

		final int width = labels.getWidth();
		final int height = labels.getHeight();
		final int depth = labels.getStackSize();
		// Update class with new labels
		for(int n=0, z=1; z <= depth; z++)
		{
			final ImageProcessor slice = labels.getImageStack().getProcessor(z);
			for(int y=0; y<height; y++)
				for(int x=0; x<width; x++, n++)
				{
					final double newValue = slice.getPixel(x, y) > 0 ? classIndex1 : classIndex2;
					/*
					// reward matching with previous value...
					if(data.get(n).classValue() == newValue)
					{
						double weight = data.get(n).weight();
						data.get(n).setWeight(++weight);
					}
					*/
					data.get(n).setClassValue(newValue);
				}

		}
		/*
		if(null !=  mismatches)
			for(int i=0; i<depth; i++)
			{
				IJ.log("slice " + i + ": " + mismatches[i].size() + " mismatches");

				for(Point3f p : mismatches[i])
				{
					//IJ.log("point = " + p);
					final int n = (int) p.x + ((int) p.y -1) * width + i * (width*height);
					double weight = data.get(n).weight();
					data.get(n).setWeight(++weight);
				}
			}
			*/
	}

	/**
	 * Read ARFF file
	 * @param filename ARFF file name
	 * @return set of instances read from the file
	 */
	public Instances readDataFromARFF(String filename){
		try{
			BufferedReader reader = new BufferedReader(
					new InputStreamReader(
							new FileInputStream(filename), StandardCharsets.UTF_8));
			try{
				Instances data = new Instances(reader);
				// setting class attribute
				data.setClassIndex(data.numAttributes() - 1);
				reader.close();
				return data;
			}
			catch(IOException e){
				IJ.showMessage("IOException: wrong file format!");
			}
		}
		catch(FileNotFoundException e){IJ.showMessage("File not found!");}
		return null;
	}

	/**
	 * Write current instances into an ARFF file
	 * @param data set of instances
	 * @param filename ARFF file name
	 */
	public boolean writeDataToARFF(Instances data, String filename)
	{
		BufferedWriter out = null;
		try{
			out = new BufferedWriter(
					new OutputStreamWriter(
							new FileOutputStream( filename ), StandardCharsets.UTF_8 ) );

			final Instances header = new Instances(data, 0);
			out.write(header.toString());

			for(int i = 0; i < data.numInstances(); i++)
			{
				out.write(data.get(i).toString()+"\n");
			}
		}
		catch(Exception e)
		{
			IJ.log("Error: couldn't write instances into .ARFF file.");
			IJ.showMessage("Exception while saving data as ARFF file");
			e.printStackTrace();
			return false;
		}
		finally{
			try {
				out.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		return true;

	}

	/**
	 * Adjust current segmentation state (attributes and classes) to
	 * loaded data
	 * @param data loaded instances
	 * @return false if error
	 */
	public boolean adjustSegmentationStateToData(Instances data)
	{
		// Check the features that were used in the loaded data
		boolean featuresChanged = false;
		Enumeration<Attribute> attributes = data.enumerateAttributes();
		final String[] availableFeatures =
					new FeatureImagesMultiResolution().availableFeatures;
		final int numFeatures = availableFeatures.length; 
		boolean[] usedFeatures = new boolean[numFeatures];

		// Initialize list of names for the features to use
		this.featureNames = new ArrayList<String>();

		float minSigma = Float.MAX_VALUE;
		float maxSigma = Float.MIN_VALUE;

		while(attributes.hasMoreElements())
		{
			final Attribute a = attributes.nextElement();
			this.featureNames.add(a.name());
			for(int i = 0 ; i < numFeatures; i++)
			{
				if(a.name().startsWith( availableFeatures[i] ))
				{
					usedFeatures[i] = true;
					String[] tokens;
					float sigma;

					tokens = a.name().split("_");
					for(int j=0; j<tokens.length; j++)
						if(tokens[j].indexOf(".") != -1)
						{
							sigma = Float.parseFloat(tokens[j]);
							if(sigma < minSigma)
								minSigma = sigma;
							if(sigma > maxSigma)
								maxSigma = sigma;
						}
				}
			}
		}

		// Check if classes match
		Attribute classAttribute = data.classAttribute();
		Enumeration<Object> classValues  = classAttribute.enumerateValues();

		// Update list of names of loaded classes
		loadedClassNames = new ArrayList<String>();

		int j = 0;
		setNumOfClasses(0);

		while(classValues.hasMoreElements())
		{
			final String className = ( (String) classValues.nextElement() ).trim();
			loadedClassNames.add(className);
		}

		for(String className : loadedClassNames)
		{
			IJ.log("Read class name: " + className);
			setClassLabel(j, className);
			addClass();
			j++;
		}

		if( null != featureImages)
		{
			final boolean[] oldEnableFeatures =
						featureImages.getEnabledFeatures();
			// Read checked features and check if any of them changed
			for(int i = 0; i < numFeatures; i++)
			{
				if (usedFeatures[i] != oldEnableFeatures[i])
					featuresChanged = true;
			}
		}
		else
			featuresChanged = true;

		// Update feature stack if necessary
		if(featuresChanged)
		{
			//this.setButtonsEnabled(false);
			this.setEnabledFeatures( usedFeatures );
			// Force features to be updated
			updateFeatures = true;
		}

		return true;
	}

	/**
	 * Create training instances out of the user markings
	 * @return set of instances (feature vectors in Weka format)
	 */
	public Instances createTrainingInstances()
	{

		int iExample = 0;
		boolean loggedAlready = false;

		ArrayList< Example > examplesWithoutFeatures = new ArrayList<>();

		for( Example example : examples )
		{
			// check if the example is up to date..
			if( example.instanceValuesArray == null ||
					! Arrays.equals( enabledFeatures, example.enabledFeatures ) ||
					! ( maximumFeatureScale == example.maximumFeatureScale ) )
			{
				example.featureNames = this.featureNames;
				example.enabledFeatures = this.enabledFeatures;
				example.maximumFeatureScale = this.maximumFeatureScale;
				examplesWithoutFeatures.add( example );
			}
		}

		ExecutorService exe = Executors.newFixedThreadPool( numRegionThreads );
		ArrayList<Future> futures = new ArrayList<>();

		for (int i = 0; i < examplesWithoutFeatures.size(); i ++ )
		{

			ArrayList < Example > closeExamples = new ArrayList<>();

			Point3D exampleLocation = new Point3D(
					examplesWithoutFeatures.get(i).getBounds().getX(),
					examplesWithoutFeatures.get(i).getBounds().getY(),
					examplesWithoutFeatures.get(i).z
					);

			closeExamples.add( examplesWithoutFeatures.get(i) );

			Boolean includeNextExample = true;

			i++;
			while ( includeNextExample && (i < examplesWithoutFeatures.size() ) )
			{
				Point3D nextExampleLocation = new Point3D(
						examplesWithoutFeatures.get(i).getBounds().getX(),
						examplesWithoutFeatures.get(i).getBounds().getY(),
						examplesWithoutFeatures.get(i).z
				);

				if ( exampleLocation.distance(nextExampleLocation) < getFeatureVoxelSizeAtMaximumScale() )
				{
					closeExamples.add(examplesWithoutFeatures.get(i));
					i++;
				}
				else
				{
					includeNextExample = false;
					i--;
				}

			}

			futures.add( exe.submit( setExamplesInstanceValues( closeExamples ) ) );
			this.totalThreadsExecuted.addAndGet(1);
		}

		trainableDeepSegmentation.utils.Utils.joinThreads( futures );
		exe.shutdown();

		int numFeatures = examples.get(0).instanceValuesArray.get(0).length - 1;

		// prepare training data
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int f = 0; f < numFeatures; f++)
		{
			String attString = examples.get(0).featureNames.get(f);
			attributes.add(new Attribute(attString));
		}
		attributes.add(new Attribute("class", getClassNamesAsArrayList()));

		// initialize set of instances
		Instances trainingData = new Instances("segment", attributes, 1);
		// Set the index of the class attribute
		trainingData.setClassIndex(numFeatures);


		// add training values
		int[] numExamplesPerClass = new int[ getNumClassesInExamples() ];

		for ( Example example : examples )
		{
			for ( double[] instanceValues : example.instanceValuesArray )
			{
				trainingData.add(new DenseInstance(1.0, instanceValues));
			}

			numExamplesPerClass[ example.classNum ] += example.instanceValuesArray.size();

		}

		int iClass = 0;
		for ( int n : numExamplesPerClass )
		{
			logger.info(getClassNames()[iClass++] + ": " + n + " pixels");
		}

		if (trainingData.numInstances() == 0)
			return null;

		logger.info("Memory usage [MB]: " + IJ.currentMemory()/1000000L + "/" + IJ.maxMemory()/1000000L);

		return trainingData;
	}

	private Runnable setExamplesInstanceValues( ArrayList < Example > examples )
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return () -> {

			logger.info("Computing image feature values for " + examples.size() + " examples..." );

			int[] bounds = getExamples3DBounds(examples);

			int maxFeatureVoxelSize = getFeatureVoxelSizeAtMaximumScale();
			// add one to width and height, as, e.g., a horizontal line has zero height.
			int sX = maxFeatureVoxelSize * (2 + (int) Math.ceil( (bounds[1] - bounds[0] + 1.0) / maxFeatureVoxelSize));
			int sY = maxFeatureVoxelSize * (2 + (int) Math.ceil( (bounds[3] - bounds[2] + 1.0) / maxFeatureVoxelSize));
			int sZ = maxFeatureVoxelSize * (2 + (int) Math.ceil( (bounds[5] - bounds[4] + 1.0) / maxFeatureVoxelSize));
			sX = sX < minTileSize ? minTileSize : sX;
			sY = sY < minTileSize ? minTileSize : sY;
			if ( trainingImage.getNSlices() == 1 ) // 2-D
			{
				sZ = 1;
			}
			else
			{
				sZ = sZ < minTileSize ? minTileSize : sZ;
			}


			Point3D center = new Point3D(
					( bounds[1] + bounds[0] ) / 2,
					( bounds[3] + bounds[2] ) / 2,
					( bounds[5] + bounds[4] ) / 2
			);

			final Region5D region5D = new Region5D();
			region5D.t = examples.get(0).t;
			region5D.c = 0;
			region5D.size = new Point3D(sX, sY, sZ);
			region5D.offset = Utils.computeOffsetFromCenterSize(center, region5D.size);
			region5D.subSampling = new Point3D(1, 1, 1);

			// compute feature values for pixels at and around the examples
			FeatureImagesMultiResolution  featureImages = new FeatureImagesMultiResolution();
			featureImages.setNumFeatureScales( maximumFeatureScale );
			featureImages.setOriginalImage( Utils.getDataCube(trainingImage, region5D, 0, 1) );
			featureImages.setEnabledFeatures(enabledFeatures);
			featureImages.wekaSegmentation = this;
			featureImages.updateFeaturesMT( false, featuresToShow, numThreadsPerRegion);
			featureNames = featureImages.getFeatureNames();

			// extract feature values at z-position of each example
			for ( Example example : examples )
			{
				int z = (int) (example.z - region5D.offset.getZ());
				int nf = featureImages.getNumFeatures();
				double[][][] featureSlice = new double
						[featureImages.getWidth()]
						[featureImages.getHeight()]
						[nf + 1]; // one extra for class label

				featureImages.setInterpolatedFeatureSlice(z, example.t, featureSlice);

				// get and set feature values at the x,y positions of the example
				//Point[] points = example.points;
				Point[] points = getPointsFromExample(example);

				// int randomNum = ThreadLocalRandom.current().nextInt(1, 100 + 1);

				//IJ.log("");
				//IJ.log("NEW EXAMPLE " + randomNum);

				example.instanceValuesArray = new ArrayList<>();

				for ( Point point : points )
				{

					double[] instanceValues = new double[nf + 1];

					int x = (int) (point.getX() - region5D.offset.getX());
					int y = (int) (point.getY() - region5D.offset.getY());

					//IJ.log(" x,y: " + x + ", " + y + "; max = " + featureSlice.length + ", " + featureSlice[0].length );

					/*
					IJ.log("");
					IJ.log(randomNum+" x,y,z global: " + point.getX() + "," + point.getY() + "," + example.z );
					IJ.log(randomNum+" x,y local: " + x + "," + y );

					String[] valueString = new String[4];
					valueString[0] = "";
					valueString[1] = "";
					valueString[2] = "";
					valueString[3] = "";
					*/

					// get the feature values at the x, y, z location
					for (int f = 0; f < nf; f++)
					{
						instanceValues[f] = featureSlice[x][y][f];


						/*if ( f < 7 )
							valueString[0] = valueString[0] + (int)Math.round(instanceValues[f]) + ", ";
						else if ( f < 7+24 )
							valueString[1] = valueString[1] + (int)Math.round(instanceValues[f]) + ", ";
						else if ( f < 7+24+27 )
							valueString[2] = valueString[2] + (int)Math.round(instanceValues[f]) + ", ";
						else
							valueString[3] = valueString[3] + (int)Math.round(instanceValues[f]) + ", ";
							*/
					}

					/*
					for ( String s : valueString )
						IJ.log(randomNum + " x,y,z global: " + point.getX() + "," + point.getY()+ "," + example.z + " values "+s);
						*/

					// Assign class
					instanceValues[instanceValues.length - 1] = (double) example.classNum;

					example.instanceValuesArray.add(instanceValues);

					example.featureNames = new ArrayList<>();
					for (String featureName : featureImages.getFeatureNames())
						example.featureNames.add(featureName);

				}
			}
		};
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


	public int[] getExamples3DBounds( ArrayList < Example > examples )
	{

		Rectangle bounds = examples.get(0).getBounds();

		int xMin = (int) bounds.getMinX();
		int xMax = (int) bounds.getMaxX();
		int yMin = (int) bounds.getMinY();
		int yMax = (int) bounds.getMaxY();
		int zMin = examples.get(0).z;
		int zMax = examples.get(0).z;


		for (Example example : examples)
		{
			bounds = example.getBounds();

			xMin = (int) bounds.getMinX() < xMin ? (int) bounds.getMinX() : xMin;
			xMax = (int) bounds.getMaxX() > xMax ? (int) bounds.getMaxX() : xMax;

			yMin = (int) bounds.getMinY() < yMin ? (int) bounds.getMinY() : yMin;
			yMax = (int) bounds.getMaxY() > yMax ? (int) bounds.getMaxY() : yMax;

			zMin = example.z < zMin ? example.z : zMin;
			zMax = example.z > zMax ? example.z : zMax;
		}

		return (new int[]{xMin, xMax, yMin, yMax, zMin, zMax});

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
		 return (int) Math.pow(3, maximumFeatureScale - 1);
	}

	public int getFeatureImagesBorderSize()
	{
		return (int) Math.ceil( getFeatureVoxelSizeAtMaximumScale() / 2) + 1;
	}

	public ArrayList<String> getClassNamesAsArrayList()
	{
		ArrayList<String> classes = new ArrayList<>();
		for(int i = 0; i < numOfClasses ; i++)
		{
			classes.add("class"+i);
		}
		return classes;
	}

	/**
	 * Train classifier with the current instances
	 * and current classifier settings
	 */
	public boolean trainClassifier()
	{
		if (Thread.currentThread().isInterrupted() )
		{
			IJ.log("Classifier training was interrupted.");
			return false;
		}

		rf.setSeed( (new Random()).nextInt() );
		rf.setMaxDepth(maxDepth);
		rf.setNumTrees(getNumTrees());
		rf.setNumThreads(numRfTrainingThreads);
		rf.setNumFeatures(getNumRandomFeatures());
		rf.setBatchSize("" + getBatchSizePercent());
		rf.setComputeImportances( getComputeFeatureImportance() );

		// At least two lists of different classes of examples need to be non empty
		int nonEmpty = 0;
		int sliceWithTraces = -1;
		for(int i = 0; i < numOfClasses; i++)
			for(int j=0; j< trainingImage.getImageStackSize(); j++)
				if( getNumExamples(i) > 0 )
				{
					nonEmpty++;
					sliceWithTraces = j; // store index of slice with traces
					break;
				}

		if (nonEmpty < 2 && null == loadedTrainingData)
		{
			IJ.showMessage("Cannot train without at least 2 sets of examples!");
			return false;
		}

		// Create feature stack if necessary (training from traces
		// and the features stack is empty or the settings changed)


		Instances data = null;
		if (nonEmpty < 1)
			IJ.log("Training from loaded data only...");
		else
		{
			final long start = System.currentTimeMillis();

			traceTrainingData = data = createTrainingInstances();

			final long end = System.currentTimeMillis();
			logger.info("Created training data in " + (end - start) + " ms");
		}

		if ( loadedTrainingData != null && data != null)
		{
			IJ.log("Merging data...");
			for (int i=0; i < loadedTrainingData.numInstances(); i++)
				data.add(loadedTrainingData.instance(i));
			IJ.log("Finished: total number of instances = " + data.numInstances());
		}
		else if (data == null)
		{
			data = loadedTrainingData;
			IJ.log("Taking loaded data as only data...");
		}

		if (null == data){
			IJ.log("WTF");
		}

		// Update train header
		this.trainHeader = new Instances(data, 0);

		// Resample data if necessary
		if( balanceClasses )
		{
			final long start = System.currentTimeMillis();
			IJ.showStatus("Balancing classes distribution...");
			IJ.log("Balancing classes distribution...");
			data = balanceTrainingData(data);
			final long end = System.currentTimeMillis();
			IJ.log("Done. Balancing classes distribution took: " + (end-start) + "ms");
		}

		logger.info("Training classifier...");

		if (Thread.currentThread().isInterrupted() )
		{
			IJ.log("Classifier training was interrupted.");
			return false;
		}


		/*
		if ( (featuresToShow != null) && (!featuresToShow.get(0).equals("None")))
		{
			int f = examples.get(0).featureNames.indexOf(featuresToShow.get(0));
			logger.info("Feature values for: "+featuresToShow.get(0));
			for ( Example example : examples )
			{
				logger.info("------------------");
				for ( double[] values : example.instanceValuesArray )
				{
					logger.info(" " + Math.round(values[f]) + "; class " + example.classNum );
				}
			}

		}*/


		// Train the classifier on the current data
		final long start = System.currentTimeMillis();
		try{
			classifier.buildClassifier(data);
			this.totalThreadsExecuted.addAndGet( numRfTrainingThreads );
		}
		catch (InterruptedException ie)
		{
			IJ.log("Classifier construction was interrupted.");
			return false;
		}
		catch(Exception e){
			IJ.showMessage(e.getMessage());
			e.printStackTrace();
			return false;
		}

		// Print classifier information
		IJ.log(classifier.toString());

		final long end = System.currentTimeMillis();

		logger.info("Trained classifier with " + featureNames.size() + " features in " + (end - start) + " ms; " +
					" total threads launched = " + totalThreadsExecuted.get() );
		return true;
	}


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
				imp = Utils.getDataCube(trainingImage, region5D, 0, 1);
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
				region5DThisSlice.c = region5D.c;
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

			trainableDeepSegmentation.utils.Utils.joinThreads(futures);
			exe.shutdown();
		};

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
	public Runnable applyClassifierRunnable(
				final Region5D region5DToClassify,
				final boolean updateFeatureImages,
				final int numThreads,
				final boolean probabilityMaps,
				final int counter,
				final int counterMax)
	{

		return () ->
		{

			if ( stopCurrentThreads || Thread.currentThread().isInterrupted())
			{
				logger.progress("Thread stopped:", "" + counter + "/" + counterMax);
				return;
			}

			// wait a bit to get threads out-of-sync
			if (counter < numRegionThreads)
			{
				try
				{
					Thread.sleep( counter * regionSyncTime );
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

			if (region5DToClassify != null)
			{
				imageToClassify = Utils.getDataCube(trainingImage, region5DToClassify, 0, numThreads);
			}
			else
			{
				imageToClassify = trainingImage; // classify whole image
			}

			FeatureImagesMultiResolution featureImages = new FeatureImagesMultiResolution();

			if ( updateFeatureImages )
			{
				//imageToClassify.show();
				featureImages.setNumFeatureScales( maximumFeatureScale );
				featureImages.setOriginalImage(imageToClassify);
				featureImages.setEnabledFeatures(enabledFeatures);
				featureImages.wekaSegmentation = this;
				featureImages.updateFeaturesMT( true, featuresToShow, numThreads );
			}

			start = System.currentTimeMillis();


			// border pixels cannot be classified,
			// because the interpolated features
			// cannot not be computed properly
			final int border = getFeatureImagesBorderSize();
			int nx = imageToClassify.getWidth() - 2 * border;
			int ny = imageToClassify.getHeight() - 2 * border;
			int nz = imageToClassify.getNSlices() > 1 ? imageToClassify.getNSlices() - 2 * border : 1;

			int nzPerThread  = nz / numThreads;
			int nzLastThread = nz - (numThreads - 1) * nzPerThread;

			// create instances information (each instance needs a pointer to this)
			Instances dataInfo = new Instances("segment", getAttributes( featureImages ), 1);
			dataInfo.setClassIndex( dataInfo.numAttributes() - 1 );

			int zs = imageToClassify.getNSlices() > 1 ? border : 0;

			int n = nz > numThreads ? numThreads : nz;
			ExecutorService exe = Executors.newFixedThreadPool( n );
			ArrayList<Future<byte[][]>> futures = new ArrayList<>();
			for ( int i = 0; i < n; i++ )
			{
				if (Thread.currentThread().isInterrupted())
					return;

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

				int nzThread = (i == (n - 1)) ? nzLastThread : nzPerThread;

				Region5D region5DThread = new Region5D();
				region5DThread.size = new Point3D(nx, ny, nzThread);
				region5DThread.offset = new Point3D(border, border, zs);
				region5DThread.t = 0;
				region5DThread.c = 0;
				region5DThread.subSampling = new Point3D(1, 1, 1);

				futures.add(
						exe.submit(
							classifyRegion(
								featureImages,
								region5DThread,
								region5DToClassify,
								dataInfo,
								classifierCopy
							)
						)
				); this.totalThreadsExecuted.addAndGet(1);

				zs += nzThread;

			}

			ArrayList<byte[]> classificationResults = new ArrayList<>();

			// Join classifications
			try
			{
				for (Future<byte[][]> future : futures)
				{
					byte[][] data = future.get();

					for (byte[] classifiedSlice : data )
					{
						classificationResults.add(classifiedSlice);
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
			futures = null;
			featureImages = null;
			exe.shutdown();
			final long end = System.currentTimeMillis();

			//
			// put (local) classification results into (big) results image
			//
			// TODO: make a version for having everything in RAM
			start = System.currentTimeMillis();
			exe = Executors.newFixedThreadPool(numThreads);
			ArrayList<Future> futures2 = new ArrayList<>();
			int iSlice = 0;
			for (byte[] classifiedSlice : classificationResults)
			{
				Region5D region5DThisSlice = new Region5D();
				region5DThisSlice.size = new Point3D(nx, ny, 1);
				int offsetZ;
				if ( imageToClassify.getNSlices() > 1 )
				{
					offsetZ = (int) region5DToClassify.offset.getZ() + border + iSlice;
				}
				else
				{
					offsetZ = iSlice;
				}
				region5DThisSlice.offset = new Point3D(
						(int) region5DToClassify.offset.getX() + border,
						(int) region5DToClassify.offset.getY() + border,
						offsetZ);
				region5DThisSlice.t = region5DToClassify.t;
				region5DThisSlice.c = region5DToClassify.c;
				region5DThisSlice.subSampling = new Point3D(1, 1, 1);

				futures2.add(
						exe.submit(
								setClassificationResult(
										classifiedImage,
										region5DThisSlice,
										classifiedSlice
								)
						)
				); this.totalThreadsExecuted.addAndGet(1);
				iSlice++;

			}

			trainableDeepSegmentation.utils.Utils.joinThreads( futures2 );
			exe.shutdown();

			//logger.info("Saved classification results in " + (System.currentTimeMillis() - start) + " ms");
		};

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
				classifiedImageStack.setAndSaveBytePixels(classifiedSlice, region5D);
			}
			else
			{
				int z = (int)region5D.offset.getZ();
				int xs = (int)region5D.offset.getX();
				int ys = (int)region5D.offset.getY();
				int nx = (int)region5D.size.getX();
				int ny = (int)region5D.size.getY();
				int ye = ys+ny-1;
				int xe = xs+nx-1;


				ImageProcessor ip = classifiedImage.getStack().getProcessor( z + 1 );

				int i = 0;
				for ( int y = ys ; y <= ye; y++ )
					for ( int x = xs ; x <= xe; x++ )
						ip.set(x, y, classifiedSlice[i++]);
			}

		};

	}

	public ArrayList<Attribute> getAttributes(FeatureImagesMultiResolution  featureImages)
	{
		ArrayList<Attribute> attributes = featureImages.getFeatureNamesAsAttributes();
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
	private static Callable<byte[][]> classifyRegion(
			final FeatureImagesMultiResolution  featureImages,
			final Region5D region5D, // region within the featureImages
			final Region5D region5Dglobal,
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

				int nf = featureImages.getNumFeatures();

				final byte[][] classificationResult = new byte[nz][nx*ny];

				// reusable array to be filled for each instance
				final double[] values = new double[ nf + 1 ];

				// create empty reusable instance
				final ReusableDenseInstance ins =
						new ReusableDenseInstance( 1.0, values );
				ins.setDataset(dataInfo);

				// create reusable feature slice
				double[][][] featureSlice = new double[nx][ny][nf+1]; // one extra for class label

				try
				{
					int iInstanceThisSlice = 0;
					int zPrevious = -1;
					for ( int z = 0; z < nz; z++ )
					{
						if (z != zPrevious)
						{
							if (Thread.currentThread().isInterrupted())
								return null;

							zPrevious = z;

							// set reusable featureSlice
							featureImages.setInterpolatedFeatureSliceRegion(zs + z, t, xs, xe, ys, ye, featureSlice);

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
								ins.setValues(1.0, featureSlice[x][y]);
								classificationResult[z][iInstanceThisSlice++] = (byte) classifier.classifyInstance(ins);

								/*
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



	/**
	 * Classify instances concurrently
	 *
	 * @param data set of instances to classify
	 * @param classifier current classifier
	 * @param counter auxiliary counter to be able to update the progress bar
	 * @param probabilityMaps return a probability map for each class instead of a
	 * classified image
	 * @return classification result
	 */
	private static Callable<double[][]> classifyInstances(
			final Instances data,
			final AbstractClassifier classifier,
			final AtomicInteger counter,
			final boolean probabilityMaps)
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return new Callable<double[][]>(){

			@Override
			public double[][] call(){

				final int numInstances = data.numInstances();
				final int numClasses   = data.numClasses();

				final double[][] classificationResult;

				if (probabilityMaps)
					classificationResult = new double[numClasses][numInstances];
				else
					classificationResult = new double[1][numInstances];

				for (int i=0; i<numInstances; i++)
				{
					try{

						if (0 == i % 4000)
						{
							if (Thread.currentThread().isInterrupted())
								return null;
							counter.addAndGet(4000);
						}

						if (probabilityMaps)
						{
							double[] prob = classifier.distributionForInstance(data.get(i));
							for(int k = 0 ; k < numClasses; k++)
								classificationResult[k][i] = prob[k];
						}
						else
						{
							classificationResult[0][i] = classifier.classifyInstance(data.get(i));
						}

					}catch(Exception e){

						IJ.showMessage("Could not apply Classifier!");
						e.printStackTrace();
						return null;
					}
				}
				return classificationResult;
			}
		};
	}



	/**
	 * Set the maximum sigma/radius to use in the features
	 * @param sigma maximum sigma to use in the features filters
	 */
	public void setMaximumFeatureScale(int sigma)
	{
		maximumFeatureScale = sigma;
		featureImages.setNumFeatureScales(sigma);
	}

	/**
	 * Get the maximum sigma/radius to use in the features
	 * @return maximum sigma/radius to use in the features
	 */
	public int getNumFeatureScales()
	{
		return maximumFeatureScale;
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
	/**
	 * Set number of random features (random forest training)
	 */
	public void setNumRandomFeatures(int n)
	{
		numRandomFeatures = n;
	}
	/**
	 * Get number of random features (random forest training)
	 * @return number of random feature per node of the random forest
	 */
	public int getNumRandomFeatures()
	{
		return numRandomFeatures;
	}

	public void setBatchSizePercent( int percent )
	{
		batchSizePercent = percent;
	}


	public int getBatchSizePercent()
	{
		return batchSizePercent;
	}


	public ArrayList < String > getFeaturesToShow()
	{
		return featuresToShow;
	}

	public String getFeaturesToShowAsOneString()
	{
		String ss = "";
		for ( String s : featuresToShow )
		{
			if ( !ss.equals("") )
				ss += ("," + s);
			else
				ss += s;
		}
		return ss;
	}

	public void setFeaturesToShow( String featuresToShow )
	{
		String[] ss = featuresToShow.split(",");
		this.featuresToShow = new ArrayList<>();
		for ( String s : ss)
			this.featuresToShow.add(s);
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
	public void setDoHomogenizeClasses(boolean homogenizeClasses)
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
	 * Set the new enabled features
	 * @param enabledFeatures new enabled feature flags
	 */
	public void setEnabledFeatures(boolean[] enabledFeatures)
	{
		this.enabledFeatures = enabledFeatures;
	}

	/**
	 * Get the current enabled features
	 * @return current enabled feature flags
	 */
	public boolean[] getEnabledFeatures()
	{
		return enabledFeatures;
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
	 * Shut down the executor service for training and feature creation
	 */
	public void shutDownNow()
	{
		featureImages.shutDownNow();
		exe.shutdownNow();
	}


	/**
	 * Set the list of loaded class names
	 * @param classNames new list of class names
	 */
	public void setLoadedClassNames(ArrayList<String> classNames)
	{
		this.loadedClassNames = classNames;
	}


	/**
	 * Set the labels for each class
	 * @param classNames array containing all the class labels
	 */
	public void setClassNames(String[] classNames)
	{
		this.classNames = classNames;
	}

	/**
	 * Get the current class labels
	 * @return array containing all the class labels
	 */
	public String[] getClassNames()
	{
		return classNames;
	}

	public boolean isProcessing3D() {
		return true;
	}

}

