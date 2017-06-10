package trainableSegmentation;

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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import bigDataTools.Region5D;
import bigDataTools.VirtualStackOfStacks.VirtualStackOfStacks;
import bigDataTools.utils.Utils;
import bigDataTools.logging.Logger;
import bigDataTools.logging.IJLazySwingLogger;
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
	private FeatureImages featureImages = null;
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
	private String[] classLabels = new String[MAX_NUM_CLASSES];

	// Random Forest parameters
	/** current number of trees in the fast random forest classifier */
	private int numOfTrees = 100;
	/** current number of random features per tree in the fast random forest classifier */
	private int randomFeatures = 2;
	/** maximum depth per tree in the fast random forest classifier */
	private int maxDepth = 0; // TODO: what does this do?
	/** list of class names on the loaded data */
	private ArrayList<String> loadedClassNames = null;

	/** maximum sigma to use on the filters */
	private int maximumFeatureScale = 3;

	private boolean showFeatureImages = false; // TODO: make this a button

	private static Logger logger = new IJLazySwingLogger();

	/** flags of filters */
	private boolean[] enabledFeatures = new boolean[]{
			true, 	/* Hessian */
			true,	/* Structure */
			true,	/* Minimum */
			true,	/* Maximum */
			true,	/* Mean */
	};

	/** use neighborhood flag */
	private boolean useNeighbors = false;

	/** list of the names of features to use */
	private ArrayList<String> featureNames = null;

	/**
	 * flag to set the resampling of the training data in order to guarantee
	 * the same number of instances per class (class balance)
	 * */
	private boolean balanceClasses = false;

	/** Project folder name. It is used to stored temporary data if different from null */
	private String projectFolder = null;

	/** executor service to launch threads for the library operations */
	private ExecutorService exe = Executors.newFixedThreadPool(  Prefs.getThreads() );

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
			this.classLabels[ i ] = new String("class " + (i+1));

		// Initialization of Fast Random Forest classifier
		rf = new FastRandomForest();
		rf.setNumTrees(numOfTrees);
		//this is the default that Breiman suggests
		//rf.setNumFeatures((int) Math.round(Math.sqrt(featureStack.getNumFeatures())));
		//but this seems to work better
		rf.setNumFeatures(randomFeatures);
		// Random seed
		rf.setSeed( (new Random()).nextInt() );
		// Set number of threads
		rf.setNumThreads( Prefs.getThreads() );

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
	public void addExample(int classNum, Roi roi, int z, int t)
	{
		examples.add(new Example(classNum, roi, z, t,
				enabledFeatures, maximumFeatureScale, classLabels));
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
		List<Roi> rois = new ArrayList<Roi>();

		for ( Example example : examples )
		{
			if ( (example.classNum == classNum) &&
					(example.t == t) && (example.z == z) )
			{
				rois.add( example.roi );
			}

		}
		return rois;
	}

	public ArrayList< Example > getExamples()
	{
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
		getClassLabels()[classNum] = label;
	}

	/**
	 * Get the label name of a class.
	 *
	 * @param classNum class index
	 */
	public String getClassLabel(int classNum)
	{
		return getClassLabels()[classNum];
	}

	/**
	 * Load training data
	 *
	 * @param pathname complete path name of the training data file (.arff)
	 * @return false if error
	 */
	public boolean loadTrainingData(String pathname)
	{
		IJ.log("Loading data from " + pathname + "...");
		loadedTrainingData = readDataFromARFF(pathname);
		if( null == loadedTrainingData )
		{
			IJ.log( "Unable to load training data from " + pathname );
			return false;
		}

		// Check the features that were used in the loaded data
		Enumeration<Attribute> attributes = loadedTrainingData.enumerateAttributes();
		final String[] availableFeatures = new FeatureImagesMultiResolution().availableFeatures;

		final int numFeatures = availableFeatures.length;
		boolean[] usedFeatures = new boolean[numFeatures];
		while(attributes.hasMoreElements())
		{
			final Attribute a = attributes.nextElement();
			for(int i = 0 ; i < numFeatures; i++)
				if(a.name().startsWith(FeatureStack.availableFeatures[i]))
					usedFeatures[i] = true;
		}

		// Check if classes match
		Attribute classAttribute = loadedTrainingData.classAttribute();
		Enumeration<Object> classValues  = classAttribute.enumerateValues();

		// Update list of names of loaded classes
		loadedClassNames = new ArrayList<String>();

		int j = 0;
		while(classValues.hasMoreElements())
		{
			final String className = ((String)classValues.nextElement()).trim();
			loadedClassNames.add(className);

			IJ.log("Read class name: " + className);
			if( !className.equals(this.getClassLabels()[j]))
			{
				String s = getClassLabels()[0];
				for(int i = 1; i < numOfClasses; i++)
					s = s.concat(", " + getClassLabels()[i]);
				IJ.error("ERROR: Loaded classes and current classes do not match!\nExpected: " + s);
				loadedTrainingData = null;
				return false;
			}
			j++;
		}

		if(j != numOfClasses)
		{
			IJ.error("ERROR: Loaded number of classes and current number do not match!");
			loadedTrainingData = null;
			return false;
		}

		boolean featuresChanged = false;
		final boolean[] oldEnableFeatures = enabledFeatures;
		// Read checked features and check if any of them chasetButtonsEnablednged
		for(int i = 0; i < numFeatures; i++)
		{
			if (usedFeatures[i] != oldEnableFeatures[i])
				featuresChanged = true;
		}
		// Update feature stack if necessary
		if( featuresChanged )
		{
			//this.setButtonsEnabled(false);
			this.setEnabledFeatures( usedFeatures );
			// Force features to be updated
			updateFeatures = true;

		}

		if (!adjustSegmentationStateToData(loadedTrainingData))
			loadedTrainingData = null;
		else
			IJ.log("Loaded data: " + loadedTrainingData.numInstances() + " instances (" + loadedTrainingData.numAttributes() + " attributes)");

		return true;
	}

	/**
	 * Returns a the loaded training data or null, if no training data was
	 * loaded.
	 */
	public Instances getLoadedTrainingData() {
		return loadedTrainingData;
	}

	/**
	 * Returns a the trace training data or null, if no examples have been
	 * given.
	 */
	public Instances getTraceTrainingData() {
		return traceTrainingData;
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
	 * Save training data into a file (.arff)
	 * @param pathname complete path name
	 * @return false if error
	 */
	public boolean saveData(final String pathname)
	{
		// TODO: implement this!
		boolean examplesEmpty = true;
		for(int i = 0; i < numOfClasses; i ++)
		{
			for(int n=0; n<trainingImage.getImageStackSize(); n++)
				if( getNumExamples(i) > 0 )
				{
					examplesEmpty = false;
					break;
				}
		}
		if (examplesEmpty && loadedTrainingData == null){
			IJ.log("There is no data to save");
			return false;
		}

		if( updateFeatures )
		{
			IJ.log("Creating feature stack...");

			if( ! featureImages.updateFeaturesMT(showFeatureImages) )
				return false;

			//filterFeatureStackByList();
			updateFeatures = false;
			IJ.log("Feature stack is now updated.");
		}

		Instances data = null;

		if(!examplesEmpty)
		{
			data = createTrainingInstances();
			data.setClassIndex(data.numAttributes() - 1);
		}
		if (null != loadedTrainingData && null != data){
			IJ.log("Merging data...");
			for (int i=0; i < loadedTrainingData.numInstances(); i++){
				// IJ.log("" + i)
				data.add(loadedTrainingData.instance(i));
			}
			IJ.log("Finished: total number of instances = " + data.numInstances());
		}
		else if (null == data)
			data = loadedTrainingData;


		IJ.log("Writing training data: " + data.numInstances() + " instances...");

		//IJ.log("Data: " + data.numAttributes() +" attributes, " + data.numClasses() + " classes");

		writeDataToARFF(data, pathname);
		IJ.log("Saved training data: " + pathname);

		return true;
	}

	/**
	 * Get the current feature stack array
	 * @return current feature stack array
	 */
	public FeatureImages getFeatureImages()
	{
		return this.featureImages;
	}

	/**
	 * Get loaded (or accumulated) training instances
	 *
	 * @return loaded/accumulated training instances
	 */
	public Instances getTrainingInstances()
	{
		return this.loadedTrainingData;
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
	 * Load a new image to segment (no GUI)
	 *
	 * @param newImage new image to segment
	 * @return false if error
	 */
	public boolean loadNewImage( ImagePlus newImage )
	{
		// Accumulate current data in "loadedTrainingData"
		IJ.log("Storing previous image instances...");

		if( updateFeatures )
		{
			IJ.log("Creating feature stack...");
			if ( !featureImages.updateFeaturesMT(showFeatureImages) )
				return false;
			updateFeatures = false;
			IJ.log("Feature stack is now updated.");
		}

		// Create instances
		Instances data = createTrainingInstances();
		if (null != loadedTrainingData && null != data)
		{
			data.setClassIndex(data.numAttributes() - 1);
			IJ.log("Merging data...");
			for (int i=0; i < loadedTrainingData.numInstances(); i++){
				// IJ.log("" + i)
				data.add(loadedTrainingData.instance(i));
			}
			IJ.log("Finished");
		}
		else if (null == data)
			data = loadedTrainingData;

		// Store merged data as loaded data
		loadedTrainingData = data;

		if(null != loadedTrainingData)
		{
			Attribute classAttribute = loadedTrainingData.classAttribute();
			Enumeration<Object> classValues  = classAttribute.enumerateValues();

			// Update list of names of loaded classes
			loadedClassNames = new ArrayList<String>();
			while(classValues.hasMoreElements())
			{
				final String className = ( (String) classValues.nextElement() ).trim();
				loadedClassNames.add(className);
			}
			IJ.log("Number of accumulated examples: " + loadedTrainingData.numInstances());
		}
		else
			IJ.log("Number of accumulated examples: 0");

		// Updating image
		IJ.log("Updating image...");

		// Set new image as training image
		trainingImage = new ImagePlus("Advanced Weka Segmentation", newImage.getImageStack());

		// Initialize feature stack array (no features yet)
		featureImages = new FeatureImagesMultiResolution( trainingImage );

		// Remove traces from the lists and ROI overlays and initialize each feature stack
		IJ.log("Removing previous markings...");
		examples = new ArrayList<>();

		/*
		examples = new Vector[trainingImage.getImageStackSize()];
		for(int i=0; i< trainingImage.getImageStackSize(); i++)
		{
			examples[i] = new Vector<ArrayList<Roi>>(MAX_NUM_CLASSES);

			for(int j=0; j<MAX_NUM_CLASSES; j++)
				examples[i].add(new ArrayList<Roi>());

		}*/

		updateFeatures = true;

		// Remove current classification result image
		classifiedImage = null;

		IJ.log("New image: " + newImage.getTitle() + " ("+trainingImage.getImageStackSize() + " slice(s))");

		IJ.log("Done");

		return true;
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
			featureStack.updateFeaturesMT(showFeatureImages);
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
	 * Get confusion matrix (2 classes)
	 * @param proposal probability image
	 * @param expectedLabels original labels
	 * @param threshold binary threshold to be applied to proposal
	 * @return confusion matrix
	 */
	public static int[][] getConfusionMatrix(
			ImagePlus proposal,
			ImagePlus expectedLabels,
			double threshold)
	{
		int[][] confusionMatrix = new int[2][2];

		final int depth = proposal.getStackSize();

		ExecutorService exe = Executors.newFixedThreadPool(Prefs.getThreads());
		ArrayList< Future <int[][]>  > fu = new ArrayList<Future <int[][]>>();

		// Compare labels
		for(int z=1; z <= depth; z++)
		{
			fu.add( exe.submit( confusionMatrixBinarySlice(proposal.getImageStack().getProcessor( z ), expectedLabels.getImageStack().getProcessor( z ), threshold)) );
		}

		for(int z=0; z < depth; z++)
		{
			try {
				int[][] temp = fu.get( z ).get();
				for(int i=0 ; i<2; i++)
					for(int j=0 ; j<2; j++)
						confusionMatrix[i][j] += temp[i][j];

			} catch (Exception e) {
				e.printStackTrace();
				return null;
			}
			finally{
				exe.shutdown();
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
		for(classIndex1 = 0 ; classIndex1 < this.getClassLabels().length; classIndex1++)
			if(className1.equalsIgnoreCase(this.getClassLabels()[classIndex1]))
				break;
		if(classIndex1 == this.getClassLabels().length)
		{
			IJ.log("Error: class named '" + className1 + "' not found.");
			return;
		}
		int classIndex2 = 0;
		for(classIndex2 = 0 ; classIndex2 < this.getClassLabels().length; classIndex2++)
			if(className2.equalsIgnoreCase(this.getClassLabels()[classIndex2]))
				break;
		if(classIndex2 == this.getClassLabels().length)
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

		logger.info("COMPUTING TRAINING FEATURES...");
		logger.info("Memory usage [MB]: " + IJ.currentMemory() / 1000000L + "/" + IJ.maxMemory() / 1000000L);

		ExecutorService executorService = Executors.newFixedThreadPool( Prefs.getThreads() );
		ArrayList<Future> futures = new ArrayList<>();

		int iExample = 0;
		for( Example example : examples )
		{
			// check if feature values are up to date..
			if( example.instanceValuesArray == null ||
					! Arrays.equals( enabledFeatures, example.enabledFeatures ) ||
					! ( maximumFeatureScale == example.maximumFeatureScale ) )
			{
				//logger.progress("Launching feature computation thread", ""+ ((iExample++)+1));
				// ..and if not, compute new ones
				futures.add( executorService.submit( setExampleInstanceValues(example) ) );
			}
		}

		trainableSegmentation.utils.Utils.joinThreads( futures );

		int numFeatures = examples.get(0).instanceValuesArray.get(0).length - 1;

		// prepare training data
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int f = 0; f < numFeatures; f++)
		{
			String attString = "feature_"+f;
			attributes.add(new Attribute(attString));
		}
		attributes.add(new Attribute("class", getClassNamesAsArrayList()));

		// initialize set of instances
		Instances trainingData = new Instances("segment", attributes, 1);
		// Set the index of the class attribute
		trainingData.setClassIndex( numFeatures );


		// add training values
		int[] numExamplesPerClass = new int[getNumClassesInExamples()];

		for ( Example example : examples )
		{
			for ( double[] instanceValues : example.instanceValuesArray )
			{
				trainingData.add(new DenseInstance(1.0, instanceValues));
			}

			numExamplesPerClass[example.classNum] += example.instanceValuesArray.size();

		}

		int iClass = 0;
		for ( int n : numExamplesPerClass )
		{
			logger.info(getClassLabels()[iClass++] + ": " + n + " pixels");
		}

		if (trainingData.numInstances() == 0)
			return null;

		logger.info("Memory usage [MB]: " + IJ.currentMemory()/1000000L + "/" + IJ.maxMemory()/1000000L);

		return trainingData;
	}

	private Runnable setExampleInstanceValues( Example example )
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return () -> {

			example.instanceValuesArray = new ArrayList<>();

			Rectangle bounds = example.roi.getBounds();

			// add one to width and height, as, e.g., a horizontal line has zero height.
			int sX = getVoxelSizeAtMaximumScale() * (2 + (int) Math.ceil((bounds.getWidth() + 1) / getVoxelSizeAtMaximumScale()));
			int sY = getVoxelSizeAtMaximumScale() * (2 + (int) Math.ceil((bounds.getHeight() + 1) / getVoxelSizeAtMaximumScale()));
			int sZ = getVoxelSizeAtMaximumScale() * 3;

			Point3D center = new Point3D(
					bounds.getX() + bounds.getWidth() / 2,
					bounds.getY() + bounds.getHeight() / 2,
					example.z);

			final Region5D region5D = new Region5D();
			region5D.t = example.t;
			region5D.c = 0;
			region5D.size = new Point3D(sX, sY, sZ);
			region5D.offset = Utils.computeOffsetFromCenterSize(center, region5D.size);
			region5D.subSampling = new Point3D(1, 1, 1);

			// compute feature values for pixels at and around the example
			VirtualStackOfStacks vss = (VirtualStackOfStacks) trainingImage.getStack();

			FeatureImages featureImages = new FeatureImagesMultiResolution();
			featureImages.setMaximumSigma( maximumFeatureScale );
			featureImages.setOriginalImage( vss.getDataCube(region5D, 0, 1) );
			featureImages.setEnabledFeatures( enabledFeatures );
			if ( showFeatureImages )
			{
				showFeatureImages = false; // immediately set to false for the other threads
				featureImages.updateFeaturesMT( true );
			}
			else
			{
				featureImages.updateFeaturesMT( false );
			}

			// extract feature values at the z-position of the example
			int z = (int) (example.z - region5D.offset.getZ());
			int nf = featureImages.getNumFeatures();
			double[][][] featureSlice = new double
					[featureImages.getWidth()]
					[featureImages.getHeight()]
					[nf + 1]; // one extra for class label
			featureImages.setInterpolatedFeatureSlice(z, example.t, featureSlice);

			// TODO:
			// - test that above really is the right z plane

			// get and set feature values at the x,y positions of the example
			Point[] points = example.roi.getContainedPoints();

			for (Point point : points)
			{
				double[] instanceValues = new double[nf + 1];

				int x = (int) (point.getX() - region5D.offset.getX());
				int y = (int) (point.getY() - region5D.offset.getY());

				// get the feature values at the x, y, z location
				for (int f = 0; f < nf; f++)
				{
					instanceValues[f] = featureSlice[x][y][f];
				}

				// Assign class
				instanceValues[instanceValues.length - 1] = (double) example.classNum;

				example.instanceValuesArray.add(instanceValues);

			}


		};

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

	public int getVoxelSizeAtMaximumScale()
	{
		 return (int) Math.pow(3, maximumFeatureScale - 1);
	}

	/**
	 * Create instances of a feature stack (to be submitted to an Executor Service)
	 *
	 * @param classNames names of the classes of data
	 * @param featureStack feature stack to create the instances from
	 * @return set of instances
	 */
	public Callable<Instances> createInstances(
			final ArrayList<String> classNames,
			final FeatureStack featureStack)
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return new Callable<Instances>(){
			@Override
			public Instances call()
			{
				return featureStack.createInstances(classNames);
			}
		};
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
	 */
	public boolean trainClassifier()
	{
		if (Thread.currentThread().isInterrupted() )
		{
			IJ.log("Classifier training was interrupted.");
			return false;
		}

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
			logger.info("Created training data in " + (end-start) + " ms");
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

		logger.info("# TRAINING CLASSIFIER...");

		if (Thread.currentThread().isInterrupted() )
		{
			IJ.log("Classifier training was interrupted.");
			return false;
		}

		// Train the classifier on the current data
		final long start = System.currentTimeMillis();
		try{
			classifier.buildClassifier(data);
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
		IJ.log( this.classifier.toString() );

		final long end = System.currentTimeMillis();

		logger.info("Finished training in "+(end-start)+" ms");
		return true;
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
	public void applyClassifierOld(
			final Region5D region5D,
			final boolean updateFeatureImages,
			final int numThreads,
			final boolean probabilityMaps)
	{

		// TODO: why should the training image be set globally?

		logger.info("# APPLYING CLASSIFIER...");

		long memBefore = IJ.currentMemory();
		IJ.log("Memory usage before [MB] :"
				+ memBefore/1000000L + "/" + IJ.maxMemory()/1000000L);


		ImagePlus imageToClassify = null;

		if ( region5D != null )
		{
			if (trainingImage.getImageStack() instanceof VirtualStackOfStacks)
			{
				VirtualStackOfStacks vss = (VirtualStackOfStacks) trainingImage.getStack();
				imageToClassify = vss.getDataCube(region5D, 0, numThreads);
			}
			else
			{
				IJ.showMessage("ERROR: no VSS");
				return;
			}
		}
		else
		{
			imageToClassify = trainingImage; // classify whole image
		}

		if ( updateFeatureImages )
		{
			imageToClassify.show();
			featureImages.setMaximumSigma(maximumFeatureScale);
			featureImages.setOriginalImage(imageToClassify);
			featureImages.setEnabledFeatures(enabledFeatures);
			featureImages.updateFeaturesMT( showFeatureImages );
		}

		int nx = imageToClassify.getWidth();
		int ny = imageToClassify.getHeight();
		int nz = imageToClassify.getNSlices();

		final int border = (int) Math.ceil( featureImages.getSizeLargestFeatureRegion() / 2 ) + 1;

		ArrayList<String> classNames = null;
		if(null != loadedClassNames)
			classNames = loadedClassNames;
		else
		{
			classNames = getClassNamesAsArrayList();

			//for(int i = 0; i < numOfClasses; i++)
			//		if(!classNames.contains(getClassLabels()[i]))
			//			classNames.add(getClassLabels()[i]);
		}

		// create instances information (each instance needs a pointer to this)
		//
		Instances dataInfo = new Instances("segment", getAttributes(), 1);
		dataInfo.setClassIndex(dataInfo.numAttributes()-1);

		// number of classes
		final int numClasses = classNames.size();

		// total number of instances (i.e. feature vectors)
		final int numInstances = nx * ny * nz;

		// number of channels of the result image
		final int numChannels  = (probabilityMaps ? numClasses : 1);
		// number of slices of the result image
		//final int numSlices    = (numChannels*numInstances)/(trainingImage.getWidth()*trainingImage.getHeight());
		final int numSlices  = numChannels * nz; // TODO

		final long start = System.currentTimeMillis();

		exe = Executors.newFixedThreadPool(numThreads);
		final byte[][][] results = new byte[numThreads][][];
		final int partialSize = numInstances / numThreads;
		Future<byte[][]>[] fu = new Future[numThreads];

		final AtomicInteger counter = new AtomicInteger();

		for(int i = 0; i < numThreads; i++)
		{
			if (Thread.currentThread().isInterrupted())
				return;

			int first = i*partialSize;
			int size = (i == numThreads - 1) ? numInstances - i*partialSize : partialSize;

			AbstractClassifier classifierCopy = null;
			try {
				// The Weka random forest classifiers do not need to be duplicated on each thread
				// (that saves much memory)
				if( classifier instanceof FastRandomForest || classifier instanceof RandomForest )
					classifierCopy = classifier;
				else
					classifierCopy = (AbstractClassifier) (AbstractClassifier.makeCopy( classifier ));
			} catch (Exception e) {
				IJ.log("Error: classifier could not be copied to classify in a multi-thread way.");
				e.printStackTrace();
			}

			fu[i] = exe.submit( classifyInstances( featureImages, region5D.t + 1, dataInfo, first, size, border, classifierCopy, counter, probabilityMaps ) );
		}

		ScheduledExecutorService monitor = Executors.newScheduledThreadPool(1);
		ScheduledFuture task = monitor.scheduleWithFixedDelay(new Runnable() {
			@Override
			public void run()
			{
				IJ.showProgress(counter.get(), numInstances);
			}
		}, 0, 1, TimeUnit.SECONDS);

		// Join threads
		try {
			for(int i = 0; i < numThreads; i++)
				results[i] = fu[i].get();
		} catch (InterruptedException e) {
			//e.printStackTrace();
			return;
		} catch (ExecutionException e) {
			e.printStackTrace();
			return;
		} catch ( OutOfMemoryError err ) {
			IJ.log( "ERROR: applyClassifier run out of memory. Please, "
					+ "use a smaller input image or fewer features." );
			err.printStackTrace();
			return;
		} finally {
			exe.shutdown();
			task.cancel(true);
			monitor.shutdownNow();
			IJ.showProgress(1);
		}

		// Create final array
		byte[][] classificationResult = new byte[numChannels][numInstances];

		for(int i = 0; i < numThreads; i++)
			for (int c = 0; c < numChannels; c++)
				System.arraycopy(results[i][c], 0, classificationResult[c], i*partialSize, results[i][c].length);

		IJ.showProgress(1.0);
		final long end = System.currentTimeMillis();
		IJ.log("Classification took: " + (end - start) + "ms");

		long memAfter = IJ.currentMemory();
		long memDiff = memAfter - memBefore;
		IJ.log("Memory usage after [MB]: " + (memAfter/1000000L));
		IJ.log("Memory usage delta [MB]: " + (memDiff / 1000000L));

		// put (local) classification results into  (big) results image
		long startSettingResults = System.currentTimeMillis();

		ExecutorService exe = Executors.newFixedThreadPool( Prefs.getThreads() );

		ArrayList<Future> futures = new ArrayList<>();

		for (int z = 0; z < nz; z++)
		{
			futures.add(
					exe.submit(
						setClassificationResultOld(
								classifiedImage,
								classificationResult,
								region5D,
								z, nx, ny, border)
					)
			);
		}

		trainableSegmentation.utils.Utils.joinThreads(futures); exe.shutdown();

		logger.info("Saved classification results in "
				+ (System.currentTimeMillis() - startSettingResults) + " ms");

		/*
		for (int i = 0; i < numSlices/numChannels; i++)
		{
			for (int c = 0; c < numChannels; c++)
			{
				ImageProcessor ip = classifiedImage.getProcessor();
				byte[] pixels = (byte[]) ip.getPixels();
				int offset = 0; // TODO: compute from region5D
				System.arraycopy( classificationResult[c], i * (nx * ny), pixels, offset, nx * ny);
			}
		}*/

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
	public void applyClassifier(
			final Region5D region5DToClassify,
			final boolean updateFeatureImages,
			final int numThreads,
			final boolean probabilityMaps)
	{

		// TODO: why should the training image be set globally?

		logger.info("# APPLYING CLASSIFIER...");
		logger.info("Memory usage [MB]: " + IJ.currentMemory() / 1000000L + "/" + IJ.maxMemory() / 1000000L);


		ImagePlus imageToClassify = null;

		if ( region5DToClassify != null )
		{
			if (trainingImage.getImageStack() instanceof VirtualStackOfStacks)
			{
				VirtualStackOfStacks vss = (VirtualStackOfStacks) trainingImage.getStack();
				imageToClassify = vss.getDataCube(region5DToClassify, 0, numThreads);
			}
			else
			{
				IJ.showMessage("ERROR: no VSS");
				return;
			}
		}
		else
		{
			imageToClassify = trainingImage; // classify whole image
		}

		if ( updateFeatureImages )
		{
			imageToClassify.show();
			featureImages.setMaximumSigma( maximumFeatureScale );
			featureImages.setOriginalImage( imageToClassify );
			featureImages.setEnabledFeatures( enabledFeatures );
			featureImages.updateFeaturesMT( showFeatureImages );
		}

		long start = System.currentTimeMillis();

		// border pixels cannot be classified because the interpolated features
		// cannot not be computed properly
		final int border = (int) Math.ceil( featureImages.getSizeLargestFeatureRegion() / 2 ) + 1;
		int nx = imageToClassify.getWidth() - 2 * border;
		int ny = imageToClassify.getHeight() - 2 * border;
		int nz = imageToClassify.getNSlices() - 2 * border;

		int nzPerThread = (int) nz / numThreads;
		int nzLastThread = nz - (numThreads - 1) * nzPerThread;

		// create instances information (each instance needs a pointer to this)
		//
		Instances dataInfo = new Instances("segment", getAttributes(), 1);
		dataInfo.setClassIndex(dataInfo.numAttributes()-1);

		exe = Executors.newFixedThreadPool(numThreads);

		Future<byte[][]>[] fu = new Future[numThreads];

		final AtomicInteger slicesClassified = new AtomicInteger();

		int zs = border;

		for(int i = 0; i < numThreads; i++)
		{
			if (Thread.currentThread().isInterrupted())
				return;

			AbstractClassifier classifierCopy = null;
			try {
				// The Weka random forest classifiers do not need to be duplicated on each thread
				// (that saves much memory)
				if( classifier instanceof FastRandomForest || classifier instanceof RandomForest )
					classifierCopy = classifier;
				else
					classifierCopy = (AbstractClassifier) (AbstractClassifier.makeCopy( classifier ));
			} catch (Exception e) {
				IJ.log("Error: classifier could not be copied to classify in a multi-thread way.");
				e.printStackTrace();
			}

			int nzThread = (i==(numThreads-1)) ? nzLastThread : nzPerThread;

			Region5D region5DThread = new Region5D();
			region5DThread.size = new Point3D( nx, ny, nzThread );
			region5DThread.offset = new Point3D( border, border, zs );
			region5DThread.t = 0;
			region5DThread.c = 0;
			region5DThread.subSampling = new Point3D( 1, 1, 1);

			boolean logging = true;
			fu[i] = exe.submit(
						classifyRegion(
							featureImages,
							region5DThread,
							dataInfo,
							classifierCopy,
							slicesClassified
						)
			);

			zs += nzThread;
		}

		ArrayList < byte[] > classificationResults = new ArrayList<>();

		// Join threads
		try {
			for(int i = 0; i < numThreads; i++)
			{
				for ( byte[] classifiedSlice : fu[i].get() )
				{
					classificationResults.add( classifiedSlice );
				}
			};
		} catch (InterruptedException e) {
			//e.printStackTrace();
			return;
		} catch (ExecutionException e) {
			e.printStackTrace();
			return;
		} catch ( OutOfMemoryError err ) {
			IJ.log( "ERROR: applyClassifier run out of memory. Please, "
					+ "use a smaller input image or fewer features." );
			err.printStackTrace();
			return;
		}

		final long end = System.currentTimeMillis();

		logger.info("Classified " + (nx * ny * nz) + " pixels in " + (end - start) + " ms");

		// put (local) classification results into  (big) results image
		start = System.currentTimeMillis();

		ExecutorService exe = Executors.newFixedThreadPool(Prefs.getThreads());

		ArrayList<Future> futures = new ArrayList<>();

		int iSlice = 0;
		for ( byte[] classifiedSlice : classificationResults )
		{
			Region5D region5DThisSlice = new Region5D();
			region5DThisSlice.size = new Point3D( nx, ny, 1 );
			region5DThisSlice.offset = new Point3D(
					(int)region5DToClassify.offset.getX() + border,
					(int)region5DToClassify.offset.getY() + border,
					(int)region5DToClassify.offset.getZ() + border + iSlice );
			region5DThisSlice.t = region5DToClassify.t;
			region5DThisSlice.c = region5DToClassify.c;
			region5DThisSlice.subSampling = new Point3D( 1, 1, 1);

			futures.add(
					exe.submit(
							setClassificationResult(
									classifiedImage,
									region5DThisSlice,
									classifiedSlice
							)
					)
			);

			iSlice++;

		}

		trainableSegmentation.utils.Utils.joinThreads( futures ); exe.shutdown();

		logger.info("Saved classification results in " + (System.currentTimeMillis() - start) + " ms");

	}


	private static Runnable setClassificationResultOld(
			ImagePlus classifiedImage,
			byte[][] classificationResult,
			Region5D region5D,
			int z, int nx, int ny, int border
	)
	{

		if (Thread.currentThread().isInterrupted())
			return null;

		return () -> {

			VirtualStackOfStacks classifiedImageStack = (VirtualStackOfStacks) classifiedImage.getStack();

			int sliceDest = (int) region5D.offset.getZ() + z + 1;
			byte[] pixels = (byte[]) classifiedImageStack.getProcessor(sliceDest).getPixels();

			for (int y = border; y < ny - border; y++)
			{
				int offsetDest =
						(int) region5D.offset.getY() * classifiedImage.getWidth()
						+ y * classifiedImage.getWidth()
						+ (int) region5D.offset.getX() + border;

				int offsetSource = z * (nx * ny) + y * nx + border;

				System.arraycopy(classificationResult[0], offsetSource, pixels, offsetDest, nx - 2 * border);
			}

			//classifiedImageStack.setAndSavePixels(pixels, sliceDest);

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

			VirtualStackOfStacks classifiedImageStack = (VirtualStackOfStacks) classifiedImage.getStack();
			classifiedImageStack.setAndSaveBytePixels( classifiedSlice, region5D );

		};

	}

	public ArrayList<Attribute> getAttributes()
	{
		ArrayList<Attribute> attributes = featureImages.getFeatureNamesAsAttributes();
		attributes.add(new Attribute("class", getClassNamesAsArrayList()));
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
	private static Callable<byte[][]> classifyInstances(
			final FeatureImages featureImages,
			final int frame,
			final Instances dataInfo,
			final int first,
			final int numInstances,
			final int border,
			final AbstractClassifier classifier,
			final AtomicInteger counter,
			final boolean probabilityMaps)
	{
		if (Thread.currentThread().isInterrupted())
			return null;

		return new Callable<byte[][]>(){

			@Override
			public byte[][] call(){

				final byte[][] classificationResult;

				final int nx = featureImages.getWidth();
				final int ny = featureImages.getHeight();
				final int nz = featureImages.getDepth();
				final int sliceSize = nx * ny;
				final int numClasses = dataInfo.numClasses();

				if (probabilityMaps)
					classificationResult = new byte[numClasses][numInstances];
				else
					classificationResult = new byte[1][numInstances];

				// auxiliary array to be filled for each instance
				final double[] values =
						new double[ featureImages.getNumFeatures() + 1 ];
				// create empty reusable instance
				final ReusableDenseInstance ins =
						new ReusableDenseInstance( 1.0, values );
				ins.setDataset(dataInfo);

				long tStart, durationFeatureFetching = 0 , durationRandomForest = 0;

				int zPrevious = -1;

				// create reusable feature slice
				double[][][] featureSlice = new double
						[featureImages.getWidth()]
						[featureImages.getHeight()]
						[featureImages.getNumFeatures() + 1]; // one extra for class label

				for ( int i = 0; i < numInstances; i++ )
				{

					try{

						tStart = System.nanoTime();

						final int absolutePos = first + i;
						final int z = absolutePos / sliceSize; // one-based
						final int localPos = absolutePos - z * sliceSize;
						final int x = localPos % nx;
						final int y = localPos / nx;

						if ( (x > border) && (y > border) && ( z > border)
								&& ( x < (nx-border) ) && ( y < (ny-border) ) && ( z  < (nz-border) ))
						{
							if (z != zPrevious)
							{
								if (zPrevious >= 0)
								{
									counter.getAndAdd(1);
									String[] logs = IJ.getLog().split("\n");
									if (logs[logs.length - 1].contains("Classifying slice"))
									{
										IJ.log("\\Update:Classifying slice: " + counter +
														"/"+nz+"; Feature fetching [ms]: " +
														Math.round((double) durationFeatureFetching / 1000000L) +
														"; Classification [ms]: " +
														Math.round((double) durationRandomForest / 1000000L)
										);

										// restart the timing
										tStart = System.nanoTime();
										durationFeatureFetching = 0;
										durationRandomForest = 0;
									}
									else
									{
										IJ.log("Classifying slice: " + counter);
									}
								}


								if (Thread.currentThread().isInterrupted())
									return null;

								zPrevious = z;

								// set reusable featureSlice
								featureImages.setInterpolatedFeatureSlice(z, frame - 1, featureSlice);

							}

							// set reusable instance
							ins.setValues(1.0, featureSlice[x][y]);

							durationFeatureFetching += (System.nanoTime() - tStart);

							// do random forest classification

							tStart = System.nanoTime();

							if ( probabilityMaps )
							{
								double[] prob = classifier.distributionForInstance(ins);
								for (int k = 0; k < numClasses; k++)
									classificationResult[k][i] = (byte) (100 * prob[k]);
							}
							else
							{
								classificationResult[0][i] = (byte) classifier.classifyInstance(ins);
							}

							durationRandomForest += ( System.nanoTime() - tStart );

						}


					}
					catch(Exception e)
					{

						IJ.showMessage("Could not apply Classifier!");
						e.printStackTrace();
						return null;
					}
				}

				//IJ.log( "Duration feature fetching [s]: " + Math.round( (double)durationFeatureFetching/1000000000L ) );
				//IJ.log( "Duration random forrest [s]: " + Math.round( (double) durationRandomForest/1000000000L ) );


				return classificationResult;
			}
		};
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
			final FeatureImages featureImages,
			final Region5D region5D, // region within the featureImages
			final Instances dataInfo,
			final AbstractClassifier classifier,
			final AtomicInteger counter)
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
							counter.getAndAdd(1);

							if (Thread.currentThread().isInterrupted())
								return null;

							zPrevious = z;

							// set reusable featureSlice
							featureImages.setInterpolatedFeatureSliceRegion(
									z + zs, t, xs, xe, ys, ye, featureSlice
							);

							iInstanceThisSlice = 0;

						}

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
	 * Set features to use during training
	 *
	 * @param featureNames list of feature names to use
	 * @return false if error
	 */
	public boolean setFeatures(ArrayList<String> featureNames)
	{
		if (null == featureNames)
			return false;

		this.featureNames = featureNames;

		final int numFeatures = FeatureStack.availableFeatures.length;
		boolean[] usedFeatures = new boolean[numFeatures];
		for(final String name : featureNames)
		{
			for(int i = 0 ; i < numFeatures; i++)
				if(name.startsWith(FeatureStack.availableFeatures[i]))
					usedFeatures[i] = true;
		}

		return true;
	}


	/**
	 * Set the maximum sigma/radius to use in the features
	 * @param sigma maximum sigma to use in the features filters
	 */
	public void setMaximumFeatureScale(int sigma)
	{
		maximumFeatureScale = sigma;
		featureImages.setMaximumSigma( sigma );
	}

	/**
	 * Get the maximum sigma/radius to use in the features
	 * @return maximum sigma/radius to use in the features
	 */
	public int getMaximumFeatureScale()
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
	 * Get current number of trees (for random forest training)
	 * @return number of trees
	 */
	public int getNumOfTrees()
	{
		return numOfTrees;
	}
	/**
	 * Get number of random features (random forest training)
	 * @return number of random feature per node of the random forest
	 */
	public int getNumRandomFeatures()
	{
		return randomFeatures;
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
		randomFeatures = newRandomFeatures;
		maxDepth = newMaxDepth;

		rf.setNumTrees(numOfTrees);
		rf.setNumFeatures(randomFeatures);
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
	 * Assign an arbitrary filter stack array
	 * @param fsa new filter stack array
	 */
	public void setFeatureImages(FeatureImages fsa)
	{
		/*
		this.featureImages = fsa;
		// set feature stacks to be updated during train and test to false
		// (since the feautures are set externally and expected to be up to date)
		featureStackToUpdateTrain = new boolean[featureImages.getNumFeatures()];
		featureStackToUpdateTest = new boolean[featureImages.getNumFeatures()];
		Arrays.fill(featureStackToUpdateTest, false);
		// set flag to not update features
		updateFeatures = false;
		*/
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
	 * Save specific slice feature stack
	 *
	 * @param slice slice number
	 * @param dir directory to save the stack(s)
	 * @param fileWithExt file name with extension for the file(s)
	 */
	public void saveFeatureStack(int slice, String dir, String fileWithExt)
	{

		if( featureImages.isEmpty() ||
				featureImages.getReferenceSliceIndex() == -1)
		{
			IJ.showStatus("Creating feature stack...");
			IJ.log("Creating feature stack...");

			featureImages.updateFeaturesMT( showFeatureImages );

		}

		if(null == dir || null == fileWithExt)
			return;


		final String fileName = dir + fileWithExt.substring(0, fileWithExt.length()-4)
									+ String.format("%04d", slice) + ".tif";
		/*
		if(!featureImages.saveStackAsTiff(fileName))
		{
			IJ.error("Error", "Feature stack could not be saved");
			return;
		}

		IJ.log("Saved feature stack for slice " + (slice) + " as " + fileName);
		*/
	}

	/**
	 * Set the labels for each class
	 * @param classLabels array containing all the class labels
	 */
	public void setClassLabels(String[] classLabels)
	{
		this.classLabels = classLabels;
	}

	/**
	 * Get the current class labels
	 * @return array containing all the class labels
	 */
	public String[] getClassLabels()
	{
		return classLabels;
	}

	public boolean isProcessing3D() {
		return true;
	}

}

