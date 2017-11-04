package trainableDeepSegmentation;

import bigDataTools.logging.Logger;
import fiji.util.gui.GenericDialogPlus;
import fiji.util.gui.OverlayedImageCanvas;
import hr.irb.fastRandomForest.FastRandomForest;
import ij.*;
import ij.gui.*;
import ij.io.OpenDialog;
import ij.io.SaveDialog;
import ij.measure.Calibration;
import ij.plugin.MacroInstaller;
import ij.plugin.PlugIn;
import ij.plugin.frame.Recorder;
import ij.process.*;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.GZIPOutputStream;

import javax.swing.*;

import net.imglib2.FinalInterval;
import trainableDeepSegmentation.classification.AttributeSelector;
import trainableDeepSegmentation.examples.Example;
import trainableDeepSegmentation.examples.ExamplesUtils;
import trainableDeepSegmentation.labels.LabelManager;
import trainableDeepSegmentation.results.ResultImage;
import trainableDeepSegmentation.results.ResultImageDisk;
import trainableDeepSegmentation.results.ResultImageGUI;
import trainableDeepSegmentation.results.ResultImageMemory;
import trainableDeepSegmentation.training.InstancesCreator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.gui.GUIChooserApp;

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
 * Authors: Ignacio Arganda-Carreras (iargandacarreras@gmail.com), Verena Kaynig,
 *          Albert Cardona
 */

// TODO:
// - button color should be class color

/**
 * Segmentation plugin based on the machine learning library Weka
 */
public class Weka_Deep_Segmentation implements PlugIn
{
	/** plugin's name */
	public static final String PLUGIN_NAME = "Trainable Deep Weka Segmentation";
	/** plugin's current version */
	public static final String PLUGIN_VERSION = "v" +
		Weka_Deep_Segmentation.class.getPackage().getImplementationVersion();

	/** reference to the segmentation backend */
	private WekaSegmentation wekaSegmentation = null;

	/** image to display on the GUI */
	private ImagePlus displayImage = null;
	/** image to be used in the training */
	private ImagePlus trainingImage = null;
	/** result image after classification */
	private CustomWindow win = null;
	/** number of classes in the GUI */
	private int numOfClasses = 2;
	/** array of number of traces per class */
	private int[] traceCounter = new int[WekaSegmentation.MAX_NUM_CLASSES];
	/** flag to display the overlay image */
	private boolean showColorOverlay = false;
	/** executor service to launch threads for the plugin methods and events */
	private final ExecutorService exec = Executors.newFixedThreadPool(1);

	/** train classifier button */
	private JButton trainClassifierButton = null;
	private JButton updateTrainingDataButton = null;

	private JCheckBox trainingRecomputeFeaturesCheckBox = null;
	private JComboBox trainingDataSource = null;

	private int X = 0, Y = 1, C = 2, Z = 3, T = 4;
	private int[] XYZ = new int[]{ X, Y, Z};

	/** toggle overlay button */
	private JButton overlayButton = null;

	/** create result button */
	//private JButton getResultButton = null;
	/** create result button */
	//private JButton setResultButton = null;

	private JButton assignResultImageButton = null;

	private JComboBox resultImageComboBox = null;

	private JButton exportResultImageButton = null;

	private JButton reviewLabelsButton = null;

	private JComboBox reviewLabelsClassComboBox = null;

	private JButton printProjectInfoButton = null;

	/** get probability maps button */
	private JButton probabilityButton = null;
	/** plot result button */
	private JButton plotButton = null;
	/** apply classifier button */
	private JButton applyButton = null;
	/** apply classifier button */
	private JButton postProcessButton = null;


	private JTextField classificationRangeTextField = null;
	private JTextField experimentTextField = null;

	private JTextField objectSizeRangeTextField = null;

	/** load annotations button */
	private JButton loadProjectButton = null;

	private JButton loadLabelImageButton = null;


	/** save annotations button */
	private JButton saveProjectButton = null;
	/** settings button */
	private JButton settingsButton = null;

	private JButton testThreadsButton = null;

	private JTextField uncertaintyTextField = new JTextField();

	private JComboBox trainingDataComboBox = new JComboBox( new String[] { } );

	private JComboBox classifiersComboBox = new JComboBox( new String[] { } );


	/** Weka button */
	private JButton wekaButton = null;
	/** create new class button */
	private JButton addClassButton = null;

	/** array of roi list overlays to paint the transparent rois of each class */
	private RoiListOverlay [] roiOverlay = null;

	/** available colors for available classes */
	private Color[] colors = new Color[]{
			Color.darkGray,
			Color.green,
			Color.magenta,
			Color.red,
			Color.cyan,
			Color.pink,
			Color.yellow,
			Color.black,
			Color.orange,
			Color.white
	};

	/** Lookup table for the result overlay image */
	private LUT overlayLUT = null;

	/** array of trace lists for every class */
	private java.awt.List[] exampleList = null;
	/** array of buttons for adding each trace class */
	private JButton [] addAnnotationButton = null;

	// Macro recording constants (corresponding to
	// static method names to be called)
	/** name of the macro method to add the current trace to a class */
	public static final String ADD_TRACE = "addTrace";
	/** name of the macro method to delete the current trace */
	public static final String DELETE_TRACE = "deleteTrace";
	/** name of the macro method to train the current classifier */
	public static final String TRAIN_CLASSIFIER = "createFastRandomForest";
	/** name of the macro method to toggle the overlay image */
	public static final String TOGGLE_OVERLAY = "toggleOverlay";
	/** name of the macro method to get the binary result */
	public static final String GET_RESULT = "getResult";
	/** name of the macro method to get the binary result */
	public static final String SET_RESULT = "setResult";
	/** name of the macro method to get the probability maps */
	public static final String GET_PROBABILITY = "getProbability";
	/** name of the macro method to plot the threshold curves */
	public static final String PLOT_RESULT = "plotResultGraphs";
	/** name of the macro method to apply the current classifier to an image or stack */
	public static final String APPLY_CLASSIFIER = "applyClassifier";
	/** name of the macro method to load a classifier from file */
	public static final String LOAD_CLASSIFIER = "loadProject";
	/** name of the macro method to save the current classifier into a file */
	public static final String SAVE_CLASSIFIER = "saveProject";
	/** name of the macro method to load data from an ARFF file */
	public static final String LOAD_DATA = "loadData";
	/** name of the macro method to save the current data into an ARFF file */
	public static final String SAVE_DATA = "saveData";
	/** name of the macro method to load data from an ARFF file */
	public static final String LOAD_ANNOTATIONS = "loadProject";
	/** name of the macro method to save the current data into an ARFF file */
	public static final String SAVE_ANNOTATIONS = "saveAnnotations";
	/** name of the macro method to create a new class */
	public static final String CREATE_CLASS = "createNewClass";
	/** name of the macro method to launch the Weka Chooser */
	public static final String LAUNCH_WEKA = "launchWeka";
	/** name of the macro method to enable/disable a feature */
	public static final String SET_FEATURE = "setFeature";
	/** name of the macro method to set the membrane thickness */
	public static final String SET_MEMBRANE_THICKNESS = "setMembraneThickness";
	/** name of the macro method to set the membrane patch size */
	public static final String SET_MEMBRANE_PATCH = "setMembranePatchSize";
	/** name of the macro method to set the minimum kernel radius */
	public static final String SET_MINIMUM_SIGMA = "setMinimumSigma";
	/** name of the macro method to set the maximum kernel radius */
	public static final String SET_MAXIMUM_SIGMA = "setMaxResolutionLevel";
	/**
	 * name of the macro method to enable/disable the class homogenization
	 * @deprecated use SET_BALANCE instead
	 **/
	public static final String SET_HOMOGENIZATION = "setClassHomogenization";
	/** name of the macro method to enable/disable the class balance */
	public static final String SET_BALANCE = "setClassBalance";
	/** name of the macro method to set a new classifier */
	public static final String SET_CLASSIFIER = "setClassifier";
	/** name of the macro method to save the feature stack into a file or files */
	public static final String SAVE_FEATURE_STACK = "saveFeatureStack";
	/** name of the macro method to change a class name */
	public static final String CHANGE_CLASS_NAME = "changeClassName";
	/** name of the macro method to set the overlay opacity */
	public static final String SET_OPACITY = "setOpacity";
	/** boolean flag set to true while training */
	private boolean trainClassifierFlag = false;
	private boolean updateTrainingDataFlag = false;


	static final String REVIEW_START = "Review labels";
	static final String REVIEW_END = "Done reviewing";

	private boolean reviewLabelsFlag = false;

	public static final String RESULT_IMAGE_DISK_SINGLE_TIFF = "Disk";
	public static final String RESULT_IMAGE_RAM = "RAM";

	public static final String TRAINING_DATA_TRACES = "Traces";
	public static final String TRAINING_DATA_LABEL_IMAGE = "Label image";

	private LabelManager labelManager = null;

	private boolean isFirstTime = true;

	private Logger logger;

	/**
	 * Basic constructor for graphical user interface use
	 */
	public Weka_Deep_Segmentation()
	{
		// check for image science
		// TODO: does this work??
		if ( ! isImageScienceAvailable() )
		{
			IJ.showMessage("Please install ImageScience: [Help > Update... > Manage Update Sites]: [X] ImageScience ");
			return;
		}

		// reserve shortcuts
		String macros = "macro 'shortcut 1 [1]' {};\n"
				+ "macro 'shortcut 2 [2]' {};"
				+ "macro 'shortcut 3 [3]' {};"
				+ "macro 'shortcut 4 [4]' {};"
				+ "macro 'shortcut 5 [5]' {};"
				+ "macro 'shortcut r [r]' {};"
				+ "macro 'shortcut p [p]' {};"
				+ "macro 'shortcut u [u]' {};"
				+ "macro 'shortcut g [g]' {};"
				+ "macro 'shortcut n [n]' {};"
				+ "macro 'shortcut b [b]' {};"
				+ "macro 'shortcut d [d]' {};"
				;
		new MacroInstaller().install(macros);

		// create overlay LUT
		final byte[] red = new byte[ 256 ];
		final byte[] green = new byte[ 256 ];
		final byte[] blue = new byte[ 256 ];

		// assign colors to classes
		for(int iClass = 0; iClass < WekaSegmentation.MAX_NUM_CLASSES; iClass++)
		{
			int offset = iClass * ResultImageDisk.CLASS_LUT_WIDTH;
			for( int i = 1; i <= ResultImageDisk.CLASS_LUT_WIDTH; i++ )
			{
				red[offset + i] = (byte) (1.0 * colors[iClass].getRed() * i / ( ResultImageDisk.CLASS_LUT_WIDTH - 1));
				green[offset + i] = (byte) (1.0 * colors[iClass].getGreen() * i / ( ResultImageDisk.CLASS_LUT_WIDTH - 1));
				blue[offset + i] = (byte) (1.0 * colors[iClass].getBlue()*i / ( ResultImageDisk.CLASS_LUT_WIDTH - 1));
			}
		}
		overlayLUT = new LUT(red, green, blue);

		exampleList = new java.awt.List[WekaSegmentation.MAX_NUM_CLASSES];
		addAnnotationButton = new JButton[ WekaSegmentation.MAX_NUM_CLASSES ];

		roiOverlay = new RoiListOverlay[WekaSegmentation.MAX_NUM_CLASSES];


		updateTrainingDataButton = new JButton("Update training data");

		trainClassifierButton = new JButton("Train classifier");

		trainingRecomputeFeaturesCheckBox = new JCheckBox("ReComp", false);
		trainingDataSource = new JComboBox( new String[]{
				TRAINING_DATA_TRACES, TRAINING_DATA_LABEL_IMAGE });

		overlayButton = new JButton("Toggle overlay [r][p][u]");
		overlayButton.setToolTipText("Toggle between current segmentation and original image");
		overlayButton.setEnabled(false);

		/*
		getResultButton = new JButton("Get result image");
		getResultButton.setToolTipText("Get result image. " +
				"It will either allocate memory in RAM; " +
				"or, if [X] disk, you can specify a " +
				"folder where the classification results should be saved.");
		getResultButton.setEnabled(false);

		setResultButton = new JButton("Set result image");
		setResultButton.setToolTipText("Set result image");
		setResultButton.setEnabled(true);
		*/

		assignResultImageButton = new JButton("Assign result image");
		assignResultImageButton.setToolTipText("Assign result image");
		assignResultImageButton.setEnabled(true);

		resultImageComboBox = new JComboBox( new String[]{ RESULT_IMAGE_DISK_SINGLE_TIFF ,
										RESULT_IMAGE_RAM} );

		exportResultImageButton = new JButton("Export results");
		exportResultImageButton.setToolTipText("Export results");
		exportResultImageButton.setEnabled(false);

		reviewLabelsClassComboBox = new JComboBox( new String[]{ "1" ,
				"2"} );

		reviewLabelsButton = new JButton(REVIEW_START);
		reviewLabelsButton.setEnabled(true);

		probabilityButton = new JButton("Get probability");
		probabilityButton.setToolTipText("Generate current probability maps");
		probabilityButton.setEnabled(false);

		plotButton = new JButton("Plot result");
		plotButton.setToolTipText("Plot result based on different metrics");
		plotButton.setEnabled(false);

		printProjectInfoButton = new JButton("Log project information");
		printProjectInfoButton.setToolTipText("Prints information " +
				"about the project into the Log window");
		printProjectInfoButton.setEnabled(true);

		applyButton = new JButton ("Apply classifier");
		applyButton.setToolTipText("Apply classifier");
		applyButton.setEnabled(false);

		postProcessButton = new JButton ("Post process");
		postProcessButton.setToolTipText("Post process");
		postProcessButton.setEnabled( false );

		classificationRangeTextField = new JTextField("None", 15);
		objectSizeRangeTextField     = new JTextField("300,100000");

		loadProjectButton = new JButton ("Load project");
		loadProjectButton.setEnabled(true);

		loadLabelImageButton = new JButton ("Load label image");
		loadLabelImageButton.setEnabled(true);

		saveProjectButton = new JButton ("Save project");
		saveProjectButton.setEnabled(false);

		addClassButton = new JButton ("Create new class");
		addClassButton.setToolTipText("Add one more label to mark different areas");

		settingsButton = new JButton ("Settings");
		settingsButton.setToolTipText("Display settings dialog");

		testThreadsButton = new JButton ("Thread test");
		testThreadsButton.setToolTipText("Tests how many threads this PC will concurrently handle.");

		/** The Weka icon image */
		ImageIcon icon = new ImageIcon( Weka_Deep_Segmentation.class.getResource("/trainableDeepSegmentation/images/weka.png"));
		wekaButton = new JButton( icon );
		wekaButton.setToolTipText("Launch Weka GUI chooser");

		showColorOverlay = false;
	}

	/** Thread that runs the training. We store it to be able to
	 * to interrupt it from the GUI */
	private Thread trainingTask = null;

	private Boolean stopClassificationThread = false;


	/**
	 * Button listener
	 */
	private ActionListener listener = new ActionListener() {

		public void actionPerformed(final ActionEvent e) {

			final String command = e.getActionCommand();

			// listen to the buttons on separate threads not to block
			// the event dispatch thread
			new Thread(new Runnable(){
			//exec.submit(new Runnable() {

				public void run()
				{
					if( e.getSource() == trainClassifierButton )
					{
						trainClassifier( command );
					}
					else if( e.getSource() == overlayButton){
						// Macro recording
						String[] arg = new String[] {};
						record(TOGGLE_OVERLAY, arg);
						win.toggleOverlay();
					}
					else if( e.getSource() == updateTrainingDataButton )
					{
						updateTrainingData( command );
					}
					/*
					else if(e.getSource() == getResultButton){
						// Macro recording
						String[] arg = new String[] {};
						record(GET_RESULT, arg);
						//showClassificationImage();
					}
					else if(e.getSource() == setResultButton){
						// Macro recording
						String[] arg = new String[] {};
						record(SET_RESULT, arg);
						setResultImage();
						postProcessButton.setEnabled( true );
					}*/
					else if(e.getSource() == assignResultImageButton )
					{
						assignResultImage( (String) resultImageComboBox.getSelectedItem() );

						if ( wekaSegmentation.hasResultImage() )
						{
							exportResultImageButton.setEnabled( true );
							applyButton.setEnabled( true );
						}
					}
					else if(e.getSource() == exportResultImageButton )
					{
						ResultImageGUI.showExportGUI(
								wekaSegmentation.getResultImage(),
								wekaSegmentation.getClassNames() );
					}
					else if(e.getSource() == reviewLabelsButton )
					{
						if ( reviewLabelsButton.getText().equals( REVIEW_START ) )
						{
							reviewLabelsFlag = true;
							reviewLabelsButton.setText( REVIEW_END );
							win.updateButtonsEnabling();
							reviewLabels( reviewLabelsClassComboBox.getSelectedIndex() );
						}
						else
						{
							labelManager.updateExamples();
							ArrayList< Example > approvedExamples = labelManager.getExamples();
							wekaSegmentation.setExamples( approvedExamples );
							labelManager.close();
							reviewLabelsFlag = false;
							reviewLabelsButton.setText( REVIEW_START );
							win.updateExampleLists();
							win.updateButtonsEnabling();
						}
					}

					else if( e.getSource() == probabilityButton )
					{
						// Macro recording
						String[] arg = new String[] {};
						record(GET_PROBABILITY, arg);
						//showProbabilityImage();
					}
					else if( e.getSource() == printProjectInfoButton)
					{
						// TODO: own function and more information
						logger.info("Active feature list, sorted according to usage in random forest:");

						if ( wekaSegmentation.settings.featureList != null )
						{

							ArrayList<Feature> sortedFeatureList = new ArrayList<>( wekaSegmentation.settings.featureList );
							sortedFeatureList.sort( Comparator.comparing( Feature::getUsageInRF ) );

							int sumFeatureUsage = 0;

							for ( Feature feature : sortedFeatureList )
							{
								sumFeatureUsage += feature.usageInRF;

								if ( feature.isActive )
								{
									int featureID = wekaSegmentation.settings.featureList.indexOf( feature );

									logger.info("ID: " + featureID +
											"; Name: " + feature.name +
											"; Usage: " + feature.usageInRF +
											"; Active: " + feature.isActive);
								}

							}

							logger.info( "Sum feature usage: " + sumFeatureUsage );

							// TODO:
							// - write a function that computes some stats about
							// which kind of features, e.g. from which resolution levels
							// were used and how much a.s.o.
							// - report the output of this function here

						}
						else
						{
							logger.info("  Feature list not yet known; please run a training first");
						}
					}
					else if(e.getSource() == applyButton)
					{
						applyClassifierToSelectedRegion( command );
					}
					else if(e.getSource() == postProcessButton)
					{
						/*
						postProcessSelectedRegion(
								command,
								classificationRangeTextField.getText(),
								objectSizeRangeTextField.getText());
								*/
					}
					else if(e.getSource() == loadProjectButton)
					{
						loadProject(null, null);
					}
					else if(e.getSource() == saveProjectButton)
					{
						saveProject();
					}
					else if(e.getSource() == addClassButton){
						addNewClass();
					}
					else if(e.getSource() == settingsButton){
						showSettingsDialog();
						win.updateButtonsEnabling();
					}
					else if(e.getSource() == loadLabelImageButton)
					{
						loadLabelImage();
					}
					else if(e.getSource() == testThreadsButton){
						testThreads();
					}
					else if(e.getSource() == wekaButton){
						// Macro recording
						String[] arg = new String[] {};
						record(LAUNCH_WEKA, arg);
						launchWeka();
					}
					else{
						for(int i = 0; i < wekaSegmentation.getNumClasses(); i++)
						{
							if(e.getSource() == exampleList[i])
							{
								deleteSelected(e);
								break;
							}
							if(e.getSource() == addAnnotationButton[i])
							{
								addAnnotation(i);
								break;
							}
						}
						win.updateButtonsEnabling();
					}

				}


			}).start();
		}
	};

	private void loadLabelImage()
	{

		ImagePlus labelImage = IJ.openImage();;

		if ( labelImage != null )
		{
			wekaSegmentation.setLabelImage( labelImage );
			logger.info( "...done." );
		}
	}

	/**
	 * Item listener for the trace lists
	 */
	private ItemListener itemListener = new ItemListener() {
		public void itemStateChanged(final ItemEvent e) {
			new Thread(new Runnable(){
			//exec.submit(new Runnable() {
				public void run() {
					for(int i = 0; i < wekaSegmentation.getNumClasses(); i++)
					{
						if( e.getSource() == exampleList[i] )
							listSelected(e, i);
					}
				}
			}).start();
		}
	};

	private void reviewLabels( int classNum )
	{
		labelManager = new LabelManager( displayImage );
		labelManager.setExamples( wekaSegmentation.getExamples() );
		labelManager.reviewLabelsInRoiManager( classNum );
	};


	/**
	 * Custom canvas to deal with zooming an panning
	 */
	private class CustomCanvas extends OverlayedImageCanvas
	{
		/**
		 * default serial version UID
		 */
		private static final long serialVersionUID = 1L;

		CustomCanvas(ImagePlus imp)
		{
			super(imp);
			Dimension dim = new Dimension(Math.min(512, imp.getWidth()), Math.min(512, imp.getHeight()));
			setMinimumSize(dim);
			setSize(dim.width, dim.height);
			setDstDimensions(dim.width, dim.height);
			addKeyListener(new KeyAdapter() {
				public void keyReleased(KeyEvent ke) {
					repaint();
				}
			});
		}
		//@Override
		public void setDrawingSize(int w, int h) {}

		public void setDstDimensions(int width, int height) {
			super.dstWidth = width;
			super.dstHeight = height;
			// adjust srcRect: can it grow/shrink?
			int w = Math.min((int)(width  / magnification), imp.getWidth());
			int h = Math.min((int)(height / magnification), imp.getHeight());
			int x = srcRect.x;
			if (x + w > imp.getWidth()) x = w - imp.getWidth();
			int y = srcRect.y;
			if (y + h > imp.getHeight()) y = h - imp.getHeight();
			srcRect.setRect(x, y, w, h);
			repaint();
		}

		//@Override
		public void paint(Graphics g) {
			Rectangle srcRect = getSrcRect();
			double mag = getMagnification();
			int dw = (int)(srcRect.width * mag);
			int dh = (int)(srcRect.height * mag);
			g.setClip(0, 0, dw, dh);

			super.paint(g);

			int w = getWidth();
			int h = getHeight();
			g.setClip(0, 0, w, h);

			// Paint away the outside
			g.setColor(getBackground());
			g.fillRect(dw, 0, w - dw, h);
			g.fillRect(0, dh, w, h - dh);
		}

		public void setImagePlus(ImagePlus imp)
		{
			super.imp = imp;
		}
	}

	private void testThreads()
	{
		logger.info("Testing maximum number of threads...");
		int i = 0;
		while(true){
			new Thread(new Runnable(){
				public void run() {
					try {
						Thread.sleep(10000000);
					} catch(InterruptedException e) { }
				}
			}).start();
			i++;
			if ( i%100 == 0 )
			{
				logger.info(""+i);
			}
		}
	}

	/**
	 * Custom window to define the Trainable Weka Segmentation GUI
	 */
	private class CustomWindow extends StackWindow
	{
		/** default serial version UID */
		private static final long serialVersionUID = 1L;
		/** layout for annotation panel */
		private GridBagLayout boxAnnotation = new GridBagLayout();
		/** constraints for annotation panel */
		private GridBagConstraints annotationsConstraints = new GridBagConstraints();

		/** scroll panel for the label/annotation panel */
		private JScrollPane scrollPanel = null;

		/** panel containing the annotations panel (right side of the GUI) */
		private JPanel labelsJPanel = new JPanel();
		/** Panel with class radio buttons and lists */
		private JPanel annotationsPanel = new JPanel();
		/** buttons panel (left side of the GUI) */
		private JPanel buttonsPanel = new JPanel();
		/** options panel (included in the left side of the GUI) */
		private JPanel trainingJPanel = new JPanel();
		/** main GUI panel (containing the buttons panel on the left,
		 *  the image in the center and the annotations panel on the right */
		private Panel all = new Panel();

		/** 50% alpha composite */
		private final Composite transparency050 = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.50f );
		/** 25% alpha composite */
		//final Composite transparency025 = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.25f );
		/** opacity (in %) of the result overlay image */
		private int overlayOpacity = 33;
		/** alpha composite for the result overlay image */
		private Composite overlayAlpha = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, overlayOpacity / 100f);
		/** current segmentation result overlay */
		private ImageOverlay resultOverlay;

		/** boolean flag set to true when training is complete */
		private boolean trainingComplete = false;

		/** boolean flag set to true when training is complete */
		private boolean classificationComplete = false;



		/**
		 * Construct the plugin window
		 *
		 * @param imp input image
		 */
		CustomWindow(ImagePlus imp)
		{
			super(imp, new CustomCanvas(imp));
			final CustomCanvas canvas = (CustomCanvas) getCanvas();

			// add roi list overlays (one per class)
			for(int i = 0; i < WekaSegmentation.MAX_NUM_CLASSES; i++)
			{
				roiOverlay[i] = new RoiListOverlay();
				roiOverlay[i].setComposite( transparency050 );
				((OverlayedImageCanvas)ic).addOverlay(roiOverlay[i]);
			}

			// add result overlay
			resultOverlay = new ImageOverlay();
			resultOverlay.setComposite( overlayAlpha );
			((OverlayedImageCanvas)ic).addOverlay(resultOverlay);

			// Remove the canvas from the window, to add it later
			removeAll();

			setTitle( Weka_Deep_Segmentation.PLUGIN_NAME + ": " + trainingImage.getTitle() );

			// Annotations panel
			annotationsConstraints.anchor = GridBagConstraints.NORTHWEST;
			annotationsConstraints.fill = GridBagConstraints.HORIZONTAL;
			annotationsConstraints.gridwidth = 1;
			annotationsConstraints.gridheight = 1;
			annotationsConstraints.gridx = 0;
			annotationsConstraints.gridy = 0;

			annotationsPanel.setBorder( BorderFactory.createTitledBorder("Labels") );
			annotationsPanel.setLayout( boxAnnotation );

			for(int i = 0; i < wekaSegmentation.getNumClasses(); i++)
			{
				exampleList[i].addActionListener(listener);
				exampleList[i].addItemListener(itemListener);

				addAnnotationButton[i] = createAnnotationButton( i );

				annotationsConstraints.insets = new Insets(5, 5, 6, 6);

				annotationsPanel.add( addAnnotationButton[i], annotationsConstraints );
				annotationsConstraints.gridy++;

				annotationsConstraints.insets = new Insets(0,0,0,0);

				annotationsPanel.add( exampleList[i], annotationsConstraints );
				annotationsConstraints.gridy++;
			}

			// Select first class
			//addAnnotationButton[0].setSelected(true);

			// Add listeners
			for(int i = 0; i < wekaSegmentation.getNumClasses(); i++)
				addAnnotationButton[i].addActionListener(listener);

			trainClassifierButton.addActionListener(listener);
			updateTrainingDataButton.addActionListener(listener);

			overlayButton.addActionListener(listener);
			//getResultButton.addActionListener(listener);
			//setResultButton.addActionListener(listener);
			assignResultImageButton.addActionListener(listener);
			exportResultImageButton.addActionListener(listener);
			reviewLabelsButton.addActionListener( listener );
			probabilityButton.addActionListener(listener);
			plotButton.addActionListener(listener);
			printProjectInfoButton.addActionListener(listener);
			applyButton.addActionListener(listener);
			postProcessButton.addActionListener(listener);
			loadProjectButton.addActionListener(listener);
			loadLabelImageButton.addActionListener(listener);
			saveProjectButton.addActionListener(listener);
			addClassButton.addActionListener(listener);
			settingsButton.addActionListener(listener);
			testThreadsButton.addActionListener(listener);

			wekaButton.addActionListener(listener);

			// add special listener if the training image is a stack
			if(null != zSelector)
			{
				// set slice selector to the correct number
				zSelector.setValue( imp.getSlice() );
				// add adjustment listener to the scroll bar
				zSelector.addAdjustmentListener(new AdjustmentListener() {

					public void adjustmentValueChanged(final AdjustmentEvent e)
					{
						new Thread(new Runnable() {
							//exec.submit(new Runnable() {
							public void run()
							{
								if (e.getSource() == zSelector)
								{
									//IJ.log("moving scroll");
									displayImage.killRoi();
									drawExamples();
									updateExampleLists();
									if (showColorOverlay)
									{
										updateResultOverlay();
										displayImage.updateAndDraw();
									}
								}

							}
						}).start();

					}
				});

				// add special listener if the training image is a movie
				if(null != tSelector)
				{
					// set slice selector to the correct number
					tSelector.setValue(imp.getFrame());
					// add adjustment listener to the scroll bar
					tSelector.addAdjustmentListener(new AdjustmentListener() {

						public void adjustmentValueChanged(final AdjustmentEvent e)
						{
							new Thread(new Runnable() {
								//exec.submit(new Runnable() {
								public void run()
								{
									if (e.getSource() == tSelector)
									{
										//IJ.log("moving scroll");
										displayImage.killRoi();
										drawExamples();
										updateExampleLists();
										if (showColorOverlay)
										{
											updateResultOverlay();
											displayImage.updateAndDraw();
										}
									}

								}
							}).start();

						}
					});
				}


				// mouse wheel listener to update the rois while scrolling
				addMouseWheelListener(new MouseWheelListener() {

					@Override
					public void mouseWheelMoved(final MouseWheelEvent e)
					{

						exec.submit(new Runnable() {
							public void run()
							{
								//IJ.log("moving scroll");
								displayImage.killRoi();
								drawExamples();
								updateExampleLists();
								if (showColorOverlay)
								{
									updateResultOverlay();
									displayImage.updateAndDraw();
								}
							}
						});

					}
				});
			}


			// key listener to repaint the display image and the traces
			// when using the keys to scroll the stack
			KeyListener keyListener = new KeyListener() {

				@Override
				public void keyTyped(KeyEvent e) {
					new Thread(new Runnable(){
						public void run()
						{
							if ( e.getKeyChar() == 'r' )
							{
								toggleOverlay("result");
							}

							if ( e.getKeyChar() == 'p' )
							{
								toggleOverlay("probability");
							}

							if ( e.getKeyChar() == 'u' )
							{
								toggleOverlay("uncertainty");
							}

							if ( e.getKeyChar() == 'g' )
							{
								int i = Integer.parseInt( uncertaintyTextField.getText().trim() );
								uncertaintyNavigation("go-to", i);
							}

							if ( e.getKeyChar() == 'n' )
							{
								int i = Integer.parseInt( uncertaintyTextField.getText().trim() );
								i++;

								if ( i >= wekaSegmentation.getNumUncertaintyRegions() )
								{
									return;
								}

								uncertaintyTextField.setText( ""+i );
								uncertaintyNavigation("go-to", i );
							}

							if ( e.getKeyChar() == 'b' )
							{
								int i = Integer.parseInt( uncertaintyTextField.getText().trim() );
								i--;

								if ( i < 0 )
								{
									return;
								}

								uncertaintyTextField.setText( ""+i );
								uncertaintyNavigation("go-to", i );
							}



							if ( e.getKeyChar() == 'd' )
							{
								int i = Integer.parseInt( uncertaintyTextField.getText() );
								uncertaintyNavigation("delete", i );
							}

							try
							{
								int iClass = Integer.parseInt("" + e.getKeyChar());
								addAnnotation(iClass - 1 );
							}
							catch (NumberFormatException e )
							{
								// do nothing
							}
						}
					}).start();
				}

				@Override
				public void keyReleased(final KeyEvent e) {
					new Thread(new Runnable(){
					//exec.submit(new Runnable() {
						public void run()
						{
							if(e.getKeyCode() == KeyEvent.VK_LEFT ||
									e.getKeyCode() == KeyEvent.VK_RIGHT ||
									e.getKeyCode() == KeyEvent.VK_LESS ||
									e.getKeyCode() == KeyEvent.VK_GREATER ||
									e.getKeyCode() == KeyEvent.VK_COMMA ||
									e.getKeyCode() == KeyEvent.VK_PERIOD)
							{
								//IJ.log("moving scroll " + e.getKeyCode());
								displayImage.killRoi();
								updateExampleLists();
								drawExamples();
								if( showColorOverlay )
								{
									updateResultOverlay();
									displayImage.updateAndDraw();
								}
							}
						}
					}).start();

				}

				@Override
				public void keyPressed(KeyEvent e) {}
			};
			// add key listener to the window and the canvas
			addKeyListener(keyListener);
			canvas.addKeyListener(keyListener);

			// Labels panel (includes annotations panel)
			GridBagLayout labelsLayout = new GridBagLayout();
			GridBagConstraints labelsConstraints = new GridBagConstraints();
			labelsJPanel.setLayout( labelsLayout );
			labelsConstraints.anchor = GridBagConstraints.NORTHWEST;
			labelsConstraints.fill = GridBagConstraints.HORIZONTAL;
			labelsConstraints.gridwidth = 1;
			labelsConstraints.gridheight = 1;
			labelsConstraints.gridx = 0;
			labelsConstraints.gridy = 0;
			labelsJPanel.add( annotationsPanel, labelsConstraints );

			// Scroll panel for the label panel
			scrollPanel = new JScrollPane( labelsJPanel );
			scrollPanel.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
			scrollPanel.setMinimumSize(labelsJPanel.getPreferredSize());

			// Training panel (left side of the GUI)
			//trainingJPanel.setBorder(BorderFactory.createEmptyBorder());
			GridBagLayout trainingLayout = new GridBagLayout();
			GridBagConstraints trainingConstraints = new GridBagConstraints();
			trainingConstraints.anchor = GridBagConstraints.NORTHWEST;
			trainingConstraints.fill = GridBagConstraints.HORIZONTAL;
			trainingConstraints.gridwidth = 1;
			trainingConstraints.gridheight = 1;
			trainingConstraints.gridx = 0;
			trainingConstraints.gridy = 0;
			trainingConstraints.insets = new Insets(5, 5, 6, 6);
			trainingJPanel.setLayout(trainingLayout);

			trainingJPanel.add(settingsButton, trainingConstraints);
			trainingConstraints.gridy++;

			trainingJPanel.add(addClassButton, trainingConstraints);
			trainingConstraints.gridy++;

			JPanel assignResultImagePanel = new JPanel();
			assignResultImagePanel.add( assignResultImageButton );
			assignResultImagePanel.add( resultImageComboBox );
			trainingJPanel.add(assignResultImagePanel, trainingConstraints);
			trainingConstraints.gridy++;

			JPanel experimentPanel = new JPanel();
			experimentPanel.add( new JLabel("Experiment:") );
			experimentTextField = new JTextField("image001", 10);
			experimentPanel.add( experimentTextField );
			trainingJPanel.add( experimentPanel, trainingConstraints );
			trainingConstraints.gridy++;

			JPanel trainClassifierPanel = new JPanel();
			trainClassifierPanel.add( updateTrainingDataButton );
			//trainClassifierPanel.add(trainingDataSource);
			//trainClassifierPanel.add(trainingRecomputeFeaturesCheckBox);
			trainingJPanel.add(trainClassifierPanel, trainingConstraints);
			trainingConstraints.gridy++;

			JPanel trainingDataPanel = new JPanel();
			trainingDataPanel.add( trainClassifierButton );
			trainingDataPanel.add( new JLabel("with") );
			trainingDataPanel.add( trainingDataComboBox );
			trainingJPanel.add(trainingDataPanel, trainingConstraints);
			trainingConstraints.gridy++;

			JPanel applyPanel = new JPanel();
			applyPanel.add( applyButton, trainingConstraints );
			applyPanel.add( classifiersComboBox, trainingConstraints );
			trainingJPanel.add(applyPanel, trainingConstraints);
			trainingConstraints.gridy++;

			JPanel panelZTRange = new JPanel();
			panelZTRange.add( new JLabel("Range") );
			panelZTRange.add( classificationRangeTextField );
			trainingJPanel.add( panelZTRange, trainingConstraints );
			trainingConstraints.gridy++;


			trainingJPanel.add(overlayButton, trainingConstraints);
			trainingConstraints.gridy++;

			/*
			TODO: make this work properly

			JPanel uncertaintyPanel = new JPanel();
			JLabel uncertaintyLabel = new JLabel(
					"Uncertainties: [g][n][b][d]"
			);
			uncertaintyPanel.add(uncertaintyLabel);
			uncertaintyTextField.setText("    0");
			uncertaintyPanel.add(uncertaintyTextField);
			trainingJPanel.add(uncertaintyPanel, trainingConstraints);
			trainingConstraints.gridy++;
			*/



			JPanel reviewLabelsPanel = new JPanel();
			reviewLabelsPanel.add( reviewLabelsButton );
			reviewLabelsPanel.add( reviewLabelsClassComboBox );
			trainingJPanel.add(reviewLabelsPanel, trainingConstraints);
			trainingConstraints.gridy++;


			trainingJPanel.add( exportResultImageButton, trainingConstraints );
			trainingConstraints.gridy++;


			//trainingJPanel.add(probabilityButton, trainingConstraints);
			//trainingConstraints.gridy++;
			//trainingJPanel.add(plotButton, trainingConstraints);
			//trainingConstraints.gridy++;

			//trainingJPanel.add(newImageButton, trainingConstraints);
			//trainingConstraints.gridy++;

			// Options panel
			/*
			trainingJPanel.setBorder(BorderFactory.createTitledBorder("Options"));
			GridBagLayout optionsLayout = new GridBagLayout();
			GridBagConstraints trainingConstraints = new GridBagConstraints();
			trainingConstraints.anchor = GridBagConstraints.NORTHWEST;
			trainingConstraints.fill = GridBagConstraints.HORIZONTAL;
			trainingConstraints.gridwidth = 1;
			trainingConstraints.gridheight = 1;
			trainingConstraints.gridx = 0;
			trainingConstraints.gridy = 0;
			trainingConstraints.insets = new Insets(5, 5, 6, 6);
			trainingJPanel.setLayout(optionsLayout);
			*/





			/*
			// Post processing
			trainingConstraints.gridy++;
			trainingJPanel.add(postProcessButton, trainingConstraints);
			trainingConstraints.gridy++;
			JPanel panelObjectSize = new JPanel();
			panelObjectSize.add(new JLabel("size:"));
			panelObjectSize.add(objectSizeRangeTextField);
			trainingJPanel.add(panelObjectSize, trainingConstraints);
			*/

			/*
			trainingConstraints.gridy++;
			trainingJPanel.add(loadClassifierButton, trainingConstraints);

			trainingConstraints.gridy++;
			trainingJPanel.add(saveClassifierButton, trainingConstraints);
			*/

			trainingJPanel.add(loadProjectButton, trainingConstraints);
			trainingConstraints.gridy++;

			trainingJPanel.add(saveProjectButton, trainingConstraints);
			trainingConstraints.gridy++;

			trainingJPanel.add(printProjectInfoButton, trainingConstraints);
			trainingConstraints.gridy++;

			trainingJPanel.add(loadLabelImageButton, trainingConstraints);
			trainingConstraints.gridy++;

			trainingJPanel.add(wekaButton, trainingConstraints);
			trainingConstraints.gridy++;

			//trainingConstraints.gridy++;

			// Buttons panel (including training and options)
			GridBagLayout buttonsLayout = new GridBagLayout();
			GridBagConstraints buttonsConstraints = new GridBagConstraints();
			buttonsPanel.setLayout(buttonsLayout);
			buttonsConstraints.anchor = GridBagConstraints.NORTHWEST;
			buttonsConstraints.fill = GridBagConstraints.HORIZONTAL;
			buttonsConstraints.gridwidth = 1;
			buttonsConstraints.gridheight = 1;
			buttonsConstraints.gridx = 0;
			buttonsConstraints.gridy = 0;
			buttonsPanel.add(trainingJPanel, buttonsConstraints);
			buttonsConstraints.gridy++;
			buttonsPanel.add(trainingJPanel, buttonsConstraints);
			buttonsConstraints.gridy++;
			buttonsConstraints.insets = new Insets(5, 5, 6, 6);

			GridBagLayout layout = new GridBagLayout();
			GridBagConstraints allConstraints = new GridBagConstraints();
			all.setLayout(layout);

			allConstraints.anchor = GridBagConstraints.NORTHWEST;
			allConstraints.fill = GridBagConstraints.BOTH;
			allConstraints.gridwidth = 1;
			allConstraints.gridheight = 2;
			allConstraints.gridx = 0;
			allConstraints.gridy = 0;
			allConstraints.weightx = 0;
			allConstraints.weighty = 0;

			all.add(buttonsPanel, allConstraints);

			allConstraints.gridx++;
			allConstraints.weightx = 1;
			allConstraints.weighty = 1;
			allConstraints.gridheight = 1;
			all.add(canvas, allConstraints);

			allConstraints.gridy++;
			allConstraints.weightx = 0;
			allConstraints.weighty = 0;
			if(null != zSelector)
				all.add( zSelector, allConstraints );
			//allConstraints.gridy--;

			allConstraints.gridy++;
			allConstraints.weightx = 0;
			allConstraints.weighty = 0;
			if(null != tSelector)
				all.add( tSelector, allConstraints );

			allConstraints.gridy--;
			allConstraints.gridy--;

			allConstraints.gridx++;
			allConstraints.anchor = GridBagConstraints.NORTHEAST;
			allConstraints.weightx = 0;
			allConstraints.weighty = 0;
			allConstraints.gridheight = 1;
			all.add( scrollPanel, allConstraints );

			GridBagLayout wingb = new GridBagLayout();
			GridBagConstraints winc = new GridBagConstraints();
			winc.anchor = GridBagConstraints.NORTHWEST;
			winc.fill = GridBagConstraints.BOTH;
			winc.weightx = 1;
			winc.weighty = 1;
			setLayout(wingb);
			add(all, winc);

			// Fix minimum size to the preferred size at this point
			pack();
			setMinimumSize( getPreferredSize() );


			// Propagate all listeners
			for (Component p : new Component[]{all, buttonsPanel}) {
				for (KeyListener kl : getKeyListeners()) {
					p.addKeyListener(kl);
				}
			}

			addWindowListener(new WindowAdapter() {
				public void windowClosing(WindowEvent e) {
					//IJ.log("closing window");
					// cleanup
					// Stop any thread from the segmentator
					if(null != trainingTask)
						trainingTask.interrupt();
					//wekaSegmentation.shutDownNow();
					exec.shutdownNow();

					for(int i = 0; i < wekaSegmentation.getNumClasses(); i++)
						addAnnotationButton[i].removeActionListener(listener);
					trainClassifierButton.removeActionListener(listener);
					updateTrainingDataButton.removeActionListener( listener );
					overlayButton.removeActionListener(listener);
					//getResultButton.removeActionListener(listener);
					//setResultButton.removeActionListener(listener);
					assignResultImageButton.removeActionListener(listener);
					exportResultImageButton.removeActionListener( listener );
					reviewLabelsButton.removeActionListener( listener );
					probabilityButton.removeActionListener(listener);
					plotButton.removeActionListener(listener);
					//newImageButton.removeActionListener(listener);
					applyButton.removeActionListener(listener);
					postProcessButton.removeActionListener(listener);

					loadProjectButton.removeActionListener(listener);
					saveProjectButton.removeActionListener(listener);
					addClassButton.removeActionListener(listener);
					settingsButton.removeActionListener(listener);
					wekaButton.removeActionListener(listener);


				}
			});

			canvas.addComponentListener(new ComponentAdapter() {
				public void componentResized(ComponentEvent ce) {
					Rectangle r = canvas.getBounds();
					canvas.setDstDimensions(r.width, r.height);
				}
			});

		}

		private JButton createAnnotationButton( int classNum )
		{
			JButton button = new JButton(
					wekaSegmentation.getClassName( classNum )
							+ " [" + (classNum+1) + "]" );
			button.setToolTipText("Add markings of label '"
					+ wekaSegmentation.getClassName( classNum ) + "'");
			button.setOpaque(true);
			button.setBackground(colors[classNum]);

			return ( button );
		}

		private void uncertaintyNavigation( String cmd, int iRegion )
		{
			if ( cmd.equals("go-to") )
			{
				UncertaintyRegion uncertaintyRegion = wekaSegmentation.getUncertaintyRegion( iRegion );
				if ( uncertaintyRegion != null )
				{
					displayImage.setT(uncertaintyRegion.xyzt[3] + 1);
					displayImage.setZ(uncertaintyRegion.xyzt[2] + 1);
					int x = uncertaintyRegion.xyzt[0] - wekaSegmentation.minTileSizes[0] / 2;
					int y = uncertaintyRegion.xyzt[1] - wekaSegmentation.minTileSizes[1] / 2;
					displayImage.setRoi(
							x, y,
							wekaSegmentation.minTileSizes[0],
							wekaSegmentation.minTileSizes[1]
					);
					// TODO: below does not work
					//zoomToSelection( displayImage, 3.0 );
					displayImage.updateAndDraw();
				}
				else
				{
					logger.error(" There are currently no known uncertainty regions; " +
							"please classify some regions to compute new uncertainties.");
				}
			}

			if ( cmd.equals("delete") )
			{
				wekaSegmentation.deleteUncertaintyRegion(iRegion);
			}
		}

		void zoomToSelection(ImagePlus imp, double marginFactor)
		{
			ImageCanvas ic = imp.getCanvas();
			Roi roi = imp.getRoi();
			ic.unzoom();
			if (roi==null) return;
			Rectangle w = imp.getWindow().getBounds();
			Rectangle r = roi.getBounds();
			double mag = ic.getMagnification();
			int marginw = (int)(marginFactor * (w.width - mag * imp.getWidth()));
			int marginh = marginw;
			//int marginh = (int)(marginFactor * (w.height - mag * imp.getHeight()));
			int x = r.x+r.width/2;
			int y = r.y+r.height/2;
			mag = ic.getHigherZoomLevel(mag);
			while( (r.width*mag < w.width-marginw)
					&& (r.height*mag<w.height-marginh) ) {
				ic.zoomIn(ic.screenX(x), ic.screenY(y));
				double cmag = ic.getMagnification();
				if (cmag==32.0) break;
				mag = ic.getHigherZoomLevel(cmag);
				w = imp.getWindow().getBounds();
			}
		}

		/**
		 * Get the Weka segmentation object. This tricks allows to
		 * extract the information from the plugin and use it from
		 * static methods.
		 *
		 * @return Weka segmentation data associated to the window.
		 */
		protected WekaSegmentation getWekaSegmentation()
		{
			return wekaSegmentation;
		}

		/**
		 * Get current label lookup table (used to color the results)
		 * @return current overlay LUT
		 */
		public LUT getOverlayLUT()
		{
			return overlayLUT;
		}

		/**
		 * Draw the painted traces on the display image
		 */
		protected void drawExamples()
		{
			final int frame = displayImage.getT();
			final int slice = displayImage.getZ();

			int numClasses = wekaSegmentation.getNumClasses();
			for(int iClass = 0; iClass < numClasses; iClass++)
			{
				roiOverlay[iClass].setColor( colors[iClass] );

				roiOverlay[iClass].setRoi(
						wekaSegmentation.getExampleRois(
								iClass, slice-1, frame-1));
			}

			displayImage.updateAndDraw();
		}

		/**
		 * Update the example lists in the GUI
		 */
		protected void updateExampleLists()
		{
			final int frame = displayImage.getT();
			final int slice = displayImage.getZ();

			for(int i = 0; i < wekaSegmentation.getNumClasses(); i++)
			{
				exampleList[i].removeAll();
				exampleList[i].addActionListener(listener);
				exampleList[i].addItemListener(itemListener);
				int numExamples = wekaSegmentation.getExampleRois(i, slice-1, frame-1).size();
				for(int j = 0; j < numExamples; j++)
				{
					exampleList[ i ].add( "trace " + j + " (Z=" + slice + " T=" + frame + ")" );
				}
			}

		}

		protected boolean isToogleEnabled()
		{
			return showColorOverlay;
		}

		/**
		 * Get the displayed image. This method can be used to
		 * extract the ROIs from the current image.
		 *
		 * @return image being displayed in the custom window
		 */
		protected ImagePlus getDisplayImage()
		{
			return this.getImagePlus();
		}

		/**
		 * Set the slice selector enable option
		 * @param b true/false to enable/disable the slice selector
		 */
		public void setSliceSelectorEnabled(boolean b)
		{
			if(null != zSelector)
				zSelector.setEnabled(b);
			if(null != tSelector)
				tSelector.setEnabled(b);

		}

		/**
		 * Repaint all panels
		 */
		public void repaintAll()
		{
			this.annotationsPanel.repaint();
			getCanvas().repaint();
			this.buttonsPanel.repaint();
			this.all.repaint();
		}

		/**
		 * Add new segmentation class (new label and new list on the right side)
		 */
		public void addClass()
		{
			int classNum = numOfClasses;

			exampleList[classNum] = new java.awt.List(5);
			//exampleList[classNum].setForeground(colors[classNum]);
			exampleList[classNum].setForeground( Color.black );

			exampleList[classNum].addActionListener(listener);
			exampleList[classNum].addItemListener(itemListener);

			addAnnotationButton[classNum] = createAnnotationButton( classNum );

			annotationsConstraints.fill = GridBagConstraints.HORIZONTAL;
			annotationsConstraints.insets = new Insets(5, 5, 6, 6);

			boxAnnotation.setConstraints( addAnnotationButton[classNum], annotationsConstraints);
			annotationsPanel.add( addAnnotationButton[classNum] );
			annotationsConstraints.gridy++;

			annotationsConstraints.insets = new Insets(0,0,0,0);

			boxAnnotation.setConstraints(exampleList[classNum], annotationsConstraints);
			annotationsPanel.add(exampleList[classNum]);
			annotationsConstraints.gridy++;

			// Add listener to the new button
			addAnnotationButton[classNum].addActionListener(listener);

			numOfClasses++;

			// recalculate minimum size of scroll panel
			scrollPanel.setMinimumSize( labelsJPanel.getPreferredSize() );

			repaintAll();
		}

		/**
		 * Set the image being displayed on the custom canvas
		 * @param imp new image
		 */
		public void setImagePlus(final ImagePlus imp)
		{
			super.imp = imp;
			((CustomCanvas) super.getCanvas()).setImagePlus(imp);
			Dimension dim = new Dimension(Math.min(512, imp.getWidth()), Math.min(512, imp.getHeight()));
			((CustomCanvas) super.getCanvas()).setDstDimensions(dim.width, dim.height);
			imp.setWindow(this);
			repaint();
		}

		/**
		 * Enable / disable buttons
		 * @param s enabling flag
		 */
		protected void setButtonsEnabled(Boolean s)
		{
			trainClassifierButton.setEnabled(s);
			updateTrainingDataButton.setEnabled(s);
			overlayButton.setEnabled(s);
			//getResultButton.setEnabled(s);
			//setResultButton.setEnabled(s);
			assignResultImageButton.setEnabled( s );
			exportResultImageButton.setEnabled( s );
			reviewLabelsButton.setEnabled( s );
			probabilityButton.setEnabled(s);
			probabilityButton.setEnabled(s);
			printProjectInfoButton.setEnabled(s);
			plotButton.setEnabled(s);
			//newImageButton.setEnabled(s);
			applyButton.setEnabled(s);
			postProcessButton.setEnabled(s);
			saveProjectButton.setEnabled(s);
			addClassButton.setEnabled(s);
			settingsButton.setEnabled(s);
			testThreadsButton.setEnabled(true);
			wekaButton.setEnabled(s);
			for(int i = 0 ; i < wekaSegmentation.getNumClasses(); i++)
			{
				exampleList[i].setEnabled(s);
				addAnnotationButton[i].setEnabled(s);
			}
			setSliceSelectorEnabled(s);
		}

		/**
		 * Update buttons enabling depending on the current status of the plugin
		 */
		protected void updateButtonsEnabling()
		{
			// While training, set disable all buttons except the train buttons,
			// which will be used to stop the training by the user.
			if( trainClassifierFlag )
			{
				setButtonsEnabled( false );
				trainClassifierButton.setEnabled( true );
			}
			else if( updateTrainingDataFlag )
			{
				setButtonsEnabled( false );
				updateTrainingDataButton.setEnabled( true );
			}
			else if ( reviewLabelsFlag )
			{
				setButtonsEnabled( false );
				reviewLabelsButton.setEnabled( true );
			}
			else // If the training is not going on
			{

				updateTrainingDataButton.setEnabled( true );

				trainClassifierButton.setEnabled(
						wekaSegmentation.getInstancesManager().getNames().size() > 0 );

				applyButton.setEnabled(
						wekaSegmentation.getClassifiersManager().getNames().size() > 0
						&& wekaSegmentation.hasResultImage() );

				postProcessButton.setEnabled( win.trainingComplete );
				overlayButton.setEnabled(wekaSegmentation.hasResultImage());
				//getResultButton.setEnabled(win.trainingComplete);
				//setResultButton.setEnabled( true) ;

				assignResultImageButton.setEnabled( true );
				exportResultImageButton.setEnabled( wekaSegmentation.hasResultImage());
				reviewLabelsButton.setEnabled( true );

				plotButton.setEnabled( win.trainingComplete );
				probabilityButton.setEnabled( win.trainingComplete );
				printProjectInfoButton.setEnabled( true );

				addClassButton.setEnabled(wekaSegmentation.getNumClasses() < WekaSegmentation.MAX_NUM_CLASSES);
				settingsButton.setEnabled(true);
				testThreadsButton.setEnabled(true);
				wekaButton.setEnabled(true);

				// Check if there are samples in any slice
				boolean examplesEmpty = true;
				for ( int t = 1; t <= displayImage.getNFrames(); t++)
					for( int z = 1; z <= displayImage.getNSlices(); z++)
						for(int i = 0; i < wekaSegmentation.getNumClasses(); i ++)
							if( wekaSegmentation.getExampleRois(i, z-1, t-1).size() > 0)
							{
								examplesEmpty = false;
								break;
							}
				saveProjectButton.setEnabled( wekaSegmentation.isTrainingCompleted );

				for(int i = 0 ; i < wekaSegmentation.getNumClasses(); i++)
				{
					exampleList[i].setEnabled(true);
					addAnnotationButton[i].setEnabled(true);
				}
				setSliceSelectorEnabled(true);
			}
		}


		synchronized void toggleOverlay()
		{
			toggleOverlay("result");
		}

		/**
		 * Toggle between overlay and original image with markings
		 */
		synchronized void toggleOverlay(String mode)
		{

			// create overlay LUT
			final byte[] red = new byte[ 256 ];
			final byte[] green = new byte[ 256 ];
			final byte[] blue = new byte[ 256 ];

			if ( mode.equals("result") )
			{
				// assign colors to classes
				for (int iClass = 0; iClass < WekaSegmentation.MAX_NUM_CLASSES; iClass++)
				{
					int offset = iClass * ResultImageDisk.CLASS_LUT_WIDTH;
					for ( int i = 1; i <= ResultImageDisk.CLASS_LUT_WIDTH; i++)
					{
						red[offset + i] = (byte) ( colors[iClass].getRed() );
						green[offset + i] = (byte) ( colors[iClass].getGreen() );
						blue[offset + i] = (byte) ( colors[iClass].getBlue() );
					}
				}
				overlayLUT = new LUT(red, green, blue);
			}


			if ( mode.equals("probability") )
			{
				// assign colors to classes
				for (int iClass = 0; iClass < WekaSegmentation.MAX_NUM_CLASSES; iClass++)
				{
					int offset = iClass * ResultImageDisk.CLASS_LUT_WIDTH;
					for ( int i = 1; i <= ResultImageDisk.CLASS_LUT_WIDTH; i++)
					{
						red[offset + i] = (byte) (1.0 * colors[iClass].getRed() * i / ( ResultImageDisk.CLASS_LUT_WIDTH - 1));
						green[offset + i] = (byte) (1.0 * colors[iClass].getGreen() * i / ( ResultImageDisk.CLASS_LUT_WIDTH - 1));
						blue[offset + i] = (byte) (1.0 * colors[iClass].getBlue() * i / ( ResultImageDisk.CLASS_LUT_WIDTH - 1));
					}
				}
				overlayLUT = new LUT(red, green, blue);
			}

			if ( mode.equals("uncertainty") )
			{
				// assign colors to classes
				for (int iClass = 0; iClass < WekaSegmentation.MAX_NUM_CLASSES; iClass++)
				{
					int offset = iClass * ResultImageDisk.CLASS_LUT_WIDTH;
					for ( int i = 1; i <= ResultImageDisk.CLASS_LUT_WIDTH; i++)
					{
						// TODO:
						// - check whether this is correct
						red[offset + i] = (byte) ( 255.0 * Math.exp( - wekaSegmentation.uncertaintyLUTdecay * i  ));
						green[offset + i] = (byte) ( 0 );
						blue[offset + i] = (byte) ( 255.0 * Math.exp( - wekaSegmentation.uncertaintyLUTdecay * i  ));
					}
				}
				overlayLUT = new LUT(red, green, blue);
			}

			showColorOverlay = !showColorOverlay;
			//IJ.log("toggle overlay to: " + showColorOverlay);
			if (showColorOverlay && null != wekaSegmentation.getResultImage())
			{
				updateResultOverlay();
			}
			else
				resultOverlay.setImage(null);

			displayImage.updateAndDraw();
		}

		/**
		 * Update the buttons to add classes with current information
		 */
		public void updateAddClassButtons()
		{
			int wekaNumOfClasses = wekaSegmentation.getNumClasses();
			while (numOfClasses < wekaNumOfClasses)
				win.addClass();
			for (int i = 0; i < numOfClasses; i++)
				addAnnotationButton[i].setText(wekaSegmentation.getClassName(i) + " [" + (i+1) + "]");

			win.updateButtonsEnabling();
			repaintWindow();
		}

		/**
		 * Set the flag to inform the the training has finished or not
		 *
		 * @param b tranining complete flag
		 */
		void setTrainingComplete(boolean b)
		{
			this.trainingComplete = b;
		}

		/**
		 * Get training image
		 * @return training image
		 */
		public ImagePlus getTrainingImage()
		{
			return trainingImage;
		}
	}// end class CustomWindow


	private boolean checkImageProperties()
	{

		GenericDialog gd = new NonBlockingGenericDialog("Check Image Properties");
		gd.addMessage(
				"Please consider checking the image meta-data in [Image > Properties].\n" +
				" \nFor this plugin it is very important that...\n" +
				"...the number of slices (z) and number of frames (t) are correct " +
						"(sometimes z and t are mixed up).\n" +
				"...the pixel width, height and depth are set properly " +
						"(sometimes the image calibration got lost and all is in units of 1 pixel).\n" +
						" \nYou can leave this dialog open. " +
						"Simply press [OK] once you checked/corrected the meta-data.\n ");
		gd.showDialog();

		if ( gd.wasCanceled() ) return false;
		return true;
	}

	/**
	 * Plugin run method
	 */
	public void run(String arg)
	{
		// START
		// instantiate segmentation backend
		wekaSegmentation = new WekaSegmentation();
		logger = wekaSegmentation.getLogger();

		//IJ.open( "  /Users/tischi/Desktop/brainiac2-mit-edu-SNEMI3D/train-labels/train-labels-binary-larger-borders.tif ");
		//wekaSegmentation.setLabelImage( IJ.getImage() );

		for(int i = 0; i < wekaSegmentation.getNumClasses() ; i++)
		{
			exampleList[i] = new java.awt.List( 5 );
			//exampleList[i].setForeground(colors[i]);
			exampleList[i].setForeground( Color.black );

		}
		numOfClasses = wekaSegmentation.getNumClasses();

		//get current image
		if (null == WindowManager.getCurrentImage())
		{
			trainingImage = IJ.openImage(); // this implicitely gets the open="..." filePath
			if (null == trainingImage) return; // user canceled open dialog
		}
		else
		{
			trainingImage = WindowManager.getCurrentImage(); //.duplicate();
			trainingImage.setSlice(WindowManager.getCurrentImage().getSlice());
		}

		if ( ! checkImageProperties() ) return;

		wekaSegmentation.setInputImage( trainingImage );

		Calibration calibration = trainingImage.getCalibration();
		wekaSegmentation.settings.anisotropy = 1.0 * calibration.pixelDepth / calibration.pixelWidth;

		if( calibration.pixelWidth != calibration.pixelHeight )
		{
			logger.error("Image calibration in x and y is not the same; currently cannot take this into " +
					"account; but you can still use this plugin, may work anyway...");
		}

		ArrayList< Integer > channelsToConsider = new ArrayList<>();
		for ( int c = 0; c < trainingImage.getNChannels(); c++ )
		{
			channelsToConsider.add(c); // zero-based
		}
		wekaSegmentation.settings.activeChannels = channelsToConsider;

		displayImage = trainingImage;

		ij.gui.Toolbar.getInstance().setTool(ij.gui.Toolbar.FREELINE);

		reviewLabelsClassComboBox = new JComboBox( wekaSegmentation.getClassNames().toArray() );


		//Build GUI
		SwingUtilities.invokeLater(
				new Runnable() {
					public void run()
					{
						win = new CustomWindow(displayImage);
						win.pack();
					}
				});

	}

	private void assignResultImage( String resultImageType )
	{
		if ( resultImageType.equals( RESULT_IMAGE_DISK_SINGLE_TIFF ) )
		{
			String directory = IJ.getDirectory("Select a directory");

			ResultImage resultImage = new ResultImageDisk( wekaSegmentation, directory,
					wekaSegmentation.getInputImageDimensions() );

			wekaSegmentation.setResultImage( resultImage );

			wekaSegmentation.setLogFile( directory );

			logger.info("Created disk-resident classification result image: " +
					directory);

		}
		else if ( resultImageType.equals( RESULT_IMAGE_RAM ))
		{

			ResultImage resultImage = new ResultImageMemory( wekaSegmentation,
					wekaSegmentation.getInputImageDimensions() );

			wekaSegmentation.setResultImage( resultImage );

			logger.info("Allocated memory for classification result image." );

		}

	}

	/**
	 * Add examples defined by the user to the corresponding list
	 * in the GUI and the example list in the segmentation object.
	 *
	 * @param classNum GUI list index
	 */
	private void addAnnotation( int classNum )
	{
		if ( (classNum >= numOfClasses) || (classNum < 0))
		{
			logger.error( " Class " + ( classNum + 1) + " does not exist." );
			return;
		}

		final Roi roi = displayImage.getRoi();
		if (null == roi)
			return;
		displayImage.killRoi();

		Point[] points = roi.getContainedPoints();

		final int z = displayImage.getZ() - 1;
		final int t = displayImage.getT() - 1;

		Example newExample = wekaSegmentation.createExample(classNum, points, (int)roi.getStrokeWidth(), z, t);

		wekaSegmentation.addExample( newExample );

		traceCounter[ classNum ]++;
		win.drawExamples();
		win.updateExampleLists();
		// Record
		final int n = displayImage.getCurrentSlice();
		String[] arg = new String[]{
				Integer.toString(classNum),
				Integer.toString(n)};
		record(ADD_TRACE, arg);

		String numLabelsPerClassString = "";

		int[] numLabelsPerClass = ExamplesUtils.getNumExamplesPerClass(
				wekaSegmentation.getExamples() );

		for ( int i = 0 ; i <  numLabelsPerClass.length; i++ )
		{
			numLabelsPerClassString += " "+(i+1)+":"+numLabelsPerClass[i];
		}
		logger.progress("Number of labels per class:", numLabelsPerClassString);

	}

	/**
	 * Update the result image overlay with the corresponding slice
	 */
	public void updateResultOverlay()
	{
		ImageProcessor overlay = wekaSegmentation.getResultImage().getSlice( displayImage.getZ(), displayImage.getT() );

		overlay = overlay.convertToByte( false );
		overlay.setColorModel(overlayLUT);

		win.resultOverlay.setImage(overlay);
	}

	/**
	 * Select a list and deselect the others
	 *
	 * @param e item event (originated by a list)
	 * @param iClass list index
	 */
	void listSelected( final ItemEvent e, final int iClass )
	{
		// i is the classnum

		// win.drawExamples();
		displayImage.setColor(Color.YELLOW);

		int selectedIndex = 0;

		for(int j = 0; j < wekaSegmentation.getNumClasses(); j++)
		{
			if (j == iClass)
			{
				selectedIndex = exampleList[iClass].getSelectedIndex();
				if ( selectedIndex != -1 )
				{

					ArrayList < Roi > exampleRois =
							wekaSegmentation.getExampleRois(iClass,
							displayImage.getZ() - 1,
							displayImage.getT() - 1);

					if ( selectedIndex < exampleRois.size() )
					{
						final Roi newRoi = exampleRois.get( selectedIndex );
						// Set selected trace as current ROI
						newRoi.setImage( displayImage );
						displayImage.setRoi( newRoi );
					}
				}
			}
			else
			{
				selectedIndex = exampleList[j].getSelectedIndex();
				if ( selectedIndex != -1 )
				{
					exampleList[j].deselect(selectedIndex);
				}
			}
		}

		displayImage.updateAndDraw();
	}

	/**
	 * Delete one of the ROIs
	 *
	 * @param e action event
	 */
	synchronized void deleteSelected(final ActionEvent e)
	{
		for(int iClass = 0; iClass < wekaSegmentation.getNumClasses(); iClass++)

			if ( e.getSource() == exampleList[iClass] )
			{
				//delete item from ROI
				int index = exampleList[iClass].getSelectedIndex();

				if ( index == -1 )
					return; // has been deleted already

				ArrayList<Roi> exampleRois = wekaSegmentation.getExampleRois(
						iClass,
						displayImage.getZ()-1,
						displayImage.getT()-1);

				Roi selectedRoi = exampleRois.get(index);
				Roi activeRoiOnImage = null;

				if ( displayImage.getRoi() != null )
					activeRoiOnImage = displayImage.getRoi();
				else
					return;

				if( activeRoiOnImage.getBounds().equals(selectedRoi.getBounds()) )
					displayImage.killRoi();
				else
					return;

				// delete item from the list of ROIs of that class and slice
				wekaSegmentation.removeExample(
						iClass,
						displayImage.getZ()-1,
						displayImage.getT()-1,
						index);

				//delete item from GUI list
				exampleList[iClass].remove( index );

				// Record
				String[] arg = new String[] {
						Integer.toString(iClass),
						Integer.toString( displayImage.getCurrentSlice() ),
						Integer.toString(index)};
				record(DELETE_TRACE, arg);
			}

		win.updateExampleLists();
		win.drawExamples();
	}

	private String experiment = "abc";

	private FinalInterval previousLabelImageInterval = null;

	void updateTrainingData( String command )
	{


		if ( command.equals( "STOP" ) )
		{
			try
			{
				updateTrainingDataFlag = false;
				win.setButtonsEnabled( false );
				updateTrainingDataButton.setText( "Update training data" );
				win.updateButtonsEnabling();
			}
			catch ( Exception ex )
			{
				ex.printStackTrace();
			}
		}
		else
		{
			updateTrainingDataFlag = true;
			updateTrainingDataButton.setText( "STOP" );

			// Disable rest of buttons until the training has finished
			win.updateButtonsEnabling();

			Thread newTask = new Thread() {
				public void run()
				{
					Instances instances = null;

					if ( wekaSegmentation.hasLabelImage() )
					{
						FinalInterval labelImageInterval = getIntervalFromGUI();

						if ( labelImageInterval == null ) return;

						// check if there really is something to be updated..
						if ( ( previousLabelImageInterval == null )
								|| trainingRecomputeFeaturesCheckBox.isSelected()
								|| !labelImageInterval.equals( previousLabelImageInterval ) )
						{
							// ..seems like there is => do it!
							instances = wekaSegmentation.getInstancesFromLabelImage(
									experimentTextField.getText(),
									labelImageInterval,
									wekaSegmentation.getNumLabelImageInstancesPerPlaneAndClass() );

							previousLabelImageInterval = new FinalInterval( labelImageInterval );

						}
						else
						{
							return;
						}

					}
					else
					{

						// compute training data for new labels
						wekaSegmentation.updateExamples( trainingRecomputeFeaturesCheckBox.isSelected() );

						if ( wekaSegmentation.getNumExamples() > 0 )
						{
							instances = InstancesCreator.createInstancesFromExamples(
									wekaSegmentation.getExamples(),
									experimentTextField.getText(),
									wekaSegmentation.getAllFeatureNames(),
									wekaSegmentation.getClassNames() );
						}
						else
						{
							return;
						}

					}

					wekaSegmentation.getInstancesManager().setInstances( instances );

					// update comboBox
					trainingDataComboBox.setModel(
							new DefaultComboBoxModel(
									wekaSegmentation.
											getInstancesManager().getNames().toArray()
							) );

					// switch on buttons
					updateTrainingDataFlag = false;
					win.setButtonsEnabled( false );
					updateTrainingDataButton.setText( "Update training data" );
					win.updateButtonsEnabling();
				}

			}; newTask.start();

		}
	}


	private static void showMemoryMonitor()
	{

		Thread thread = new Thread( new Runnable() {
			//exec.submit(new Runnable() {
			public void run()
			{
				IJ.run( "Monitor Memory...", "" );
			}
		} );
		thread.start();

	}


	/**
	 * Run/stop the classifier training
	 *
	 * @param command current text of the training button ("Train classifier" or "STOP")
	 */
	void trainClassifier( final String command )
	{
		if ( command.equals("Train classifier") )
		{
			trainClassifierFlag = true;
			trainClassifierButton.setText("STOP");
			final Thread oldTask = trainingTask;

			// Disable rest of buttons until the training has finished
			win.updateButtonsEnabling();

			showMemoryMonitor();

			// Thread to run the training
			Thread newTask = new Thread() {

				public void run()
				{

				try
				{

					FastRandomForest classifier = null;

					Instances instances = wekaSegmentation.
							getInstancesManager().getInstances(
							( String ) Weka_Deep_Segmentation.this.trainingDataComboBox.getSelectedItem() );

					if ( instances == null || this.isInterrupted() ) return;

					classifier = wekaSegmentation.createFastRandomForest( instances );

					if ( classifier == null || this.isInterrupted() ) return;

					// if wished for, train a second time
					// only with important features
					if ( wekaSegmentation.minFeatureUsageFactor > 0 )
					{

						ArrayList< Integer > goners = AttributeSelector.getGoners(
								classifier,
								instances,
								wekaSegmentation.minFeatureUsageFactor,
								wekaSegmentation.getLogger() );

						Instances instances2 = InstancesCreator.removeAttributes(
								instances, goners );

						logger.info ("\n# Second Training");

						classifier = wekaSegmentation.createFastRandomForest( instances2 );

						wekaSegmentation.getClassifiersManager().setClassifier(
								classifier, instances2);
					}
					else
					{
						wekaSegmentation.getClassifiersManager().setClassifier(
								classifier, instances);
					}

				}
				catch( Exception e )
				{
					e.printStackTrace();
				}
				catch( OutOfMemoryError err )
				{
					err.printStackTrace();
				}

				win.classificationComplete = true;

				// update comboBox
				classifiersComboBox.setModel(
						new DefaultComboBoxModel(
							wekaSegmentation.
								getClassifiersManager().getNames().toArray()
						));


				trainClassifierFlag = false;
				trainClassifierButton.setText("Train classifier");
				win.updateButtonsEnabling();
				}
			}; newTask.start();
		}
		else if ( command.equals("STOP") )
		{
			try{
				trainClassifierFlag = false;
				win.trainingComplete = false;
				IJ.log("Training was stopped by the user!");
				win.setButtonsEnabled( false );
				trainClassifierButton.setText("Train classifier");

				if(null != trainingTask)
					trainingTask.interrupt();
				else
					IJ.log("Error: interrupting training failed becaused the thread is null!");

				win.updateButtonsEnabling();
			}
			catch(Exception ex)
			{
				ex.printStackTrace();
			}
		}
	}

	/**
	 * Convert image to 8 bit in place without scaling it
	 *
	 * @param image input image
	 */
	static void convertTo8bitNoScaling( ImagePlus image )
	{
		boolean aux = ImageConverter.getDoScaling();

		ImageConverter.setDoScaling(false);

		if( image.getImageStackSize() > 1)
			(new StackConverter( image )).convertToGray8();
		else
			(new ImageConverter( image )).convertToGray8();

		ImageConverter.setDoScaling(aux);
	}

	public void applyClassifierToSelectedRegion( String command )
	{
		// TODO
		// move all of this into wekaSegmentation class

		if ( command.equals("STOP") )
		{
			logger.info("Stopping classification threads...");
			wekaSegmentation.stopCurrentThreads = true;
			applyButton.setText("Apply classifier");
			return;
		}

		if ( isFirstTime )
		{
			Thread thread = new Thread(new Runnable() {
				//exec.submit(new Runnable() {

				public void run()
				{
					IJ.run("Monitor Memory...", "");
					isFirstTime = false;
				}
			}); thread.start();
		}

		if ( ! wekaSegmentation.isTrainingCompleted )
		{
			logger.error("The training is not fully completed; " +
					"please (once more) click [Train Classifier].");
			return;
		}

		if ( wekaSegmentation.getResultImage() == null )
		{
			logger.error("Classification result image yet assigned.\n" +
					"Please [Assign result image].");
			return;
		}

		FinalInterval interval = getIntervalFromGUI( );
		if ( interval == null ) return;

		logger.info("# Classifying selected region...");
		applyButton.setText("STOP");

		Thread thread = new Thread() {
			public void run()
			{

				wekaSegmentation.stopCurrentThreads = false;
				wekaSegmentation.resetUncertaintyRegions();
				wekaSegmentation.applyClassifier( interval );

				applyButton.setText("Apply classifier");
				if (showColorOverlay)
					win.toggleOverlay();
				win.toggleOverlay();

			}
		}; thread.start();
	}

	// TODO: is there a on-final version of an interval?
	private FinalInterval getIntervalFromGUI( )
	{
		Roi roi = displayImage.getRoi();

		if (roi == null || !(roi.getType() == Roi.RECTANGLE))
		{
			IJ.showMessage("Please use ImageJ's rectangle selection tool to  image" +
					" in order to select the center of the region to be classified");
			return ( null );
		}

		Rectangle rectangle = roi.getBounds();

		long[] min = new long[ 5 ];
		long[] max = new long[ 5 ];

		min[ X ] = (int) rectangle.getX();
		max[ X ] = min[0] + (int) rectangle.getWidth() - 1;

		min[ Y ] = (int) rectangle.getY();
		max[ Y ] = min[1] + (int) rectangle.getHeight() - 1;

		min[ Z ] = max[ Z ] = displayImage.getZ() - 1;
		min[ T ] = max[ T ] = displayImage.getT() - 1;
		min[ C ] = max[ C ] = displayImage.getC() - 1;

		// potentially adapt z and t range to user selection

		String rangeString = classificationRangeTextField.getText();

		try
		{
			int[] range = bigDataTools.utils.Utils.delimitedStringToIntegerArray( rangeString, ",");

			if ( trainingImage.getNFrames() == 1 )
			{
				min[ Z ] = range[0] - 1;
				max[ Z ] = range[1] - 1;
			}
			else if ( trainingImage.getNSlices() == 1 )
			{
				min[ T ] = range[0] - 1;
				max[ T ] = range[1] - 1;
			}
			else
			{
				min[ Z ] = range[0] - 1;
				max[ Z ] = range[1] - 1;

				if ( range.length == 4 )
				{
					min[ T ] = range[2] - 1;
					max[ T ] = range[3] - 1;
				}
			}
			logger.info("Using selected z and t range: ");
			logger.info("...");
			// TODO: make function to print range
		}
		catch ( NumberFormatException e )
		{
			logger.info("No (or invalid) z and t range selected.");
		}

		FinalInterval interval = new FinalInterval( min, max );

		return ( interval );


	}

	/**
	 * Save current classifier into a file
	 */
	public void saveProject()
	{
		SaveDialog sd = new SaveDialog("Save project as...",
				"project",
				".tsp");

		if (sd.getFileName()==null)
			return;

		// Record
		String[] arg = new String[] { sd.getDirectory() + sd.getFileName() };
		record(SAVE_CLASSIFIER, arg);

		if( ! wekaSegmentation.saveProject( sd.getDirectory() + sd.getFileName()) )
		{
			IJ.error("Error while writing classifier into a file");
			return;
		}
	}

	/**
	 * Write classifier into a file
	 *
	 * @param classifier classifier
	 * @param trainHeader train header containing attribute and class information
	 * @param filename name (with complete path) of the destination file
	 * @return false if error
	 */
	public static boolean saveProject(
			AbstractClassifier classifier,
			Instances trainHeader,
			String filename)
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
			if (trainHeader != null)
				objectOutputStream.writeObject(trainHeader);
			objectOutputStream.flush();
			objectOutputStream.close();
		}
		catch (Exception e)
		{
			IJ.error("Save Failed", "Error when saving classifier into a file");
			saveOK = false;
			e.printStackTrace();
		}
		if (saveOK)
			IJ.log("Saved model into the file " + filename);

		return saveOK;
	}

	/**
	 * Write classifier into a file
	 *
	 * @param cls classifier
	 * @param filename name (with complete path) of the destination file
	 * @return false if error
	 */
	public static boolean writeClassifier(AbstractClassifier cls, String filename)
	{
		try {
			SerializationHelper.write(filename, cls);
		} catch (Exception e) {
			IJ.log("Error while writing classifier into a file");
			e.printStackTrace();
			return false;
		}
		return true;
	}

	/**
	 * Load a new image to segment
	 */
	public void loadNewImage()
	{
		OpenDialog od = new OpenDialog("Choose new image", OpenDialog.getLastDirectory());
		if (od.getFileName()==null)
			return;

		win.setButtonsEnabled(false);

		IJ.log("Loading image " + od.getDirectory() + od.getFileName() + "...");

		ImagePlus newImage = new ImagePlus(od.getDirectory() + od.getFileName());
		/*
		if( !wekaSegmentation.loadNewImage( newImage ) )
		{
			IJ.error("Error while loading new image!");
			win.updateButtonsEnabling();
			return;
		}*/

		// Remove traces from the lists and ROI overlays
		for(int i = 0; i < wekaSegmentation.getNumClasses(); i ++)
		{
			exampleList[i].removeAll();
			roiOverlay[i].setRoi(null);
		}

		// Updating image
		displayImage = new ImagePlus();
		displayImage.setProcessor( Weka_Deep_Segmentation.PLUGIN_NAME + " "
									+ Weka_Deep_Segmentation.PLUGIN_VERSION,
									trainingImage.getProcessor().duplicate());

		// Remove current classification result image
		win.resultOverlay.setImage(null);

		win.toggleOverlay();

		// Update GUI
		win.setImagePlus(displayImage);
		displayImage.updateAndDraw();
		win.pack();

		win.updateButtonsEnabling();
	}

	/**
	 * Load annotations from a file
	 */
	public void loadProject( String directory, String fileName )
	{

		if ( directory == null || fileName == null )
		{
			OpenDialog od = new OpenDialog("Choose project file",
					OpenDialog.getLastDirectory(), "myProject.tsproj");
			if (od.getFileName() == null)
				return;
			directory = od.getDirectory();
			fileName = od.getFileName();
		}

		// Macro recording
		String[] arg = new String[] { directory + fileName };
		record(SAVE_ANNOTATIONS, arg);

		win.setButtonsEnabled(false);

		try
		{

			wekaSegmentation.loadProject( directory + fileName );

			for ( int c = 0; c < wekaSegmentation.getNumClasses(); c++ )
			{
				if ( c == numOfClasses )
					win.addClass();
				changeClassName(c, wekaSegmentation.getClassName( c ));
			}

			if ( wekaSegmentation.getClassifier() != null )
			{
				win.trainingComplete = true;
			}

			updateReviewLabelComboBox();
			repaintWindow();
			win.updateExampleLists();

		}
		catch (IOException e)
		{
			IJ.showMessage(e.toString());
		}



		win.updateButtonsEnabling();
	}


	/**
	 * Add new class in the panel (up to MAX_NUM_CLASSES)
	 */
	private void addNewClass()
	{
		if( wekaSegmentation.getNumClasses() == WekaSegmentation.MAX_NUM_CLASSES )
		{
			IJ.showMessage("Trainable Weka Segmentation", "Sorry, maximum number of classes has been reached");
			return;
		}

		String inputName = JOptionPane.showInputDialog("Please input a new label name");

		if(null == inputName)
			return;


		if (null == inputName || 0 == inputName.length())
		{
			IJ.error("Invalid name for class");
			return;
		}
		inputName = inputName.trim();

		if (0 == inputName.toLowerCase().indexOf("add to "))
			inputName = inputName.substring(7);

		// Add new name to the list of labels
		wekaSegmentation.addClass( inputName );
		updateReviewLabelComboBox();

		// Add new class label and list
		win.addClass();

		repaintWindow();

		// Macro recording
		String[] arg = new String[] { inputName };
		record(CREATE_CLASS, arg);
	}

	private void updateReviewLabelComboBox()
	{
		reviewLabelsClassComboBox.setModel(
				new DefaultComboBoxModel( wekaSegmentation.getClassNames().toArray() ) );
	}

	/**
	 * Repaint whole window
	 */
	private void repaintWindow()
	{
		// Repaint window
		SwingUtilities.invokeLater(
				new Runnable() {
					public void run() {
						win.invalidate();
						win.validate();
						win.repaint();
					}
				});
	}

	/**
	 * Call the Weka chooser
	 */
	public static void launchWeka()
	{
		GUIChooserApp chooser = new GUIChooserApp();
		for (WindowListener wl : chooser.getWindowListeners())
		{
			chooser.removeWindowListener(wl);
		}
		chooser.setVisible(true);
	}

	/**
	 * Show advanced settings dialog
	 *
	 * @return false when canceled
	 */
	public boolean showSettingsDialog()
	{
		GenericDialogPlus gd = new GenericDialogPlus("Segmentation settings");

		/*
		final boolean[] oldEnableFeatures = wekaSegmentation.getEnabledFeatures();
		final String[] availableFeatures = FeatureImagesMultiResolution.availableFeatures;

		gd.addMessage("Training features:");
		final int rows = 1; //(int) Math.round( availableFeatures.length/2.0 );
		gd.addCheckboxGroup(rows, availableFeatures.length, availableFeatures, oldEnableFeatures);
		*/

		/*
		if(wekaSegmentation.getLoadedTrainingData() != null)
		{
			final Vector<Checkbox> v = gd.getCheckboxes();
			for(Checkbox c : v)
				c.setEnabled(false);
			gd.addMessage("WARNING: no features are selectable while using loaded data");
		}
		*/

		gd.addNumericField("Feature computation: Downsampling factor",
				wekaSegmentation.settings.downSamplingFactor, 0);
		gd.addNumericField("Feature computation: Maximum downsampling level",
				wekaSegmentation.settings.maxResolutionLevel, 0);
		gd.addNumericField("Feature computation: Maximal convolution depth",
				wekaSegmentation.settings.maxDeepConvolutionLevel, 0);
		gd.addNumericField("Feature computation: z/xy settings.anisotropy",
				wekaSegmentation.settings.anisotropy, 2);
		gd.addNumericField("Computation: Memory factor",
				wekaSegmentation.memoryFactor, 1);
		gd.addNumericField("Training: Label image: Num. trainingDataComboBox per class and plane",
				wekaSegmentation.getNumLabelImageInstancesPerPlaneAndClass(), 0);

		/*
		if(wekaSegmentation.getLoadedTrainingData() != null)
		{
			for(int i = 0; i < 4; i++)
				((TextField) gd.getNumericFields().get( i )).setEnabled(false);
		}
		*/

		gd.addNumericField("Classifier: Number of trees",
				wekaSegmentation.getNumTrees(), 0);
		gd.addNumericField("Classifier: Accuracy",
				wekaSegmentation.accuracy, 1);
		gd.addNumericField("Classifier: Fraction of random features per node",
				wekaSegmentation.fractionRandomFeatures, 2);
		gd.addNumericField("Classifier: Minimum feature usage factor",
				wekaSegmentation.minFeatureUsageFactor, 1);

		//gd.addNumericField("RF: Batch size per tree in percent", wekaSegmentation.getBatchSizePercent(), 0);
		//gd.addNumericField("RF: Maximum tree depth [0 = None]", wekaSegmentation.maxDepth, 0);

		gd.addStringField("Feature computation: Channels to consider (one-based) [ID,ID,..]",
				wekaSegmentation.getActiveChannelsAsString() );

		String featuresToShow = "None";
		if ( wekaSegmentation.getFeaturesToShow() != null )
			featuresToShow = wekaSegmentation.getFeaturesToShowAsString();

		gd.addStringField("Show features [ID,ID,..]", featuresToShow );
		gd.addStringField("Classification: Region tile size", wekaSegmentation.getTileSizeSetting());
		//gd.addNumericField("Number of region threads", wekaSegmentation.regionThreads, 0);
		//gd.addNumericField("Number of threads inside a region", wekaSegmentation.threadsPerRegion, 0);
		//gd.addNumericField("Number of RF training threads", wekaSegmentation.numRfTrainingThreads, 0);
		//gd.addNumericField("Tiling delay [ms]", wekaSegmentation.tilingDelay, 0);
		gd.addNumericField("Classification: Background threshold [gray values]", wekaSegmentation.settings.backgroundThreshold, 0);
		//gd.addStringField("Resolution level weights", wekaSegmentation.getResolutionWeightsAsString());
		//gd.addNumericField("Uncertainty LUT decay", wekaSegmentation.uncertaintyLUTdecay, 1);



		/*
		// Add Weka panel for selecting the classifier and its options
		GenericObjectEditor m_ClassifierEditor = new GenericObjectEditor();
		PropertyPanel m_CEPanel = new PropertyPanel(m_ClassifierEditor);
		m_ClassifierEditor.setClassType(Classifier.class);
		m_ClassifierEditor.setValue(wekaSegmentation.getClassifier());

		// add classifier editor panel
		gd.addComponent( m_CEPanel,  GridBagConstraints.HORIZONTAL , 1 );

		Object c = (Object)m_ClassifierEditor.getValue();
	    String originalOptions = "";
	    String originalClassifierName = c.getClass().getName();
	    if (c instanceof OptionHandler)
	    {
	    	originalOptions = Utils.joinOptions(((OptionHandler)c).getOptions());
	    }
	    */

		//gd.addMessage("Class names:");
		for(int i = 0; i < wekaSegmentation.getNumClasses(); i++)
			gd.addStringField("Class "+(i+1), wekaSegmentation.getClassName(i), 15);

		/*
		gd.addMessage("Advanced options:");

		gd.addCheckbox( "Balance classes", wekaSegmentation.doClassBalance() );
		gd.addButton("Save feature stack", new SaveFeatureStackButtonListener(
				"Select location to save feature stack", wekaSegmentation ) );
				*/

		// gd.addCheckbox("Compute feature importances during next training", wekaSegmentation.getComputeFeatureImportance());

		gd.addSlider("Result overlay opacity", 0, 100, win.overlayOpacity);

		gd.addHelp("http://fiji.sc/Trainable_Weka_Segmentation");

		gd.showDialog();

		if ( gd.wasCanceled() )
			return false;

		/*
		final int numOfFeatures = availableFeatures.length;

		final boolean[] newEnableFeatures = new boolean[numOfFeatures];

		boolean featuresChanged = false;

		// Read checked features and check if any of them changed
		for(int i = 0; i < numOfFeatures; i++)
		{
			newEnableFeatures[i] = gd.getNextBoolean();
			if (newEnableFeatures[i] != oldEnableFeatures[i])
			{
				featuresChanged = true;
				final String name = availableFeatures[ i ];
				// Macro recording
				record(SET_FEATURE, new String[]{ name + "=" + newEnableFeatures[ i ] });
			}
		}

		if(featuresChanged)
		{
			wekaSegmentation.setEnabledFeatures( newEnableFeatures );
		}
		*/

		wekaSegmentation.settings.downSamplingFactor = (int) gd.getNextNumber();
		wekaSegmentation.settings.maxResolutionLevel = (int) gd.getNextNumber();
		wekaSegmentation.settings.maxDeepConvolutionLevel = (int) gd.getNextNumber();
		wekaSegmentation.settings.anisotropy = gd.getNextNumber();
		wekaSegmentation.memoryFactor = gd.getNextNumber();
		wekaSegmentation.setLabelImageInstancesPerPlaneAndClass( (int) gd.getNextNumber());

		// Set classifier and options
		wekaSegmentation.setNumTrees((int) gd.getNextNumber());
		wekaSegmentation.accuracy = (double) gd.getNextNumber();
		wekaSegmentation.fractionRandomFeatures = (double) gd.getNextNumber();
		wekaSegmentation.minFeatureUsageFactor = (double) gd.getNextNumber();

		//wekaSegmentation.setBatchSizePercent((int) gd.getNextNumber());
		//wekaSegmentation.maxDepth = (int) gd.getNextNumber();

		wekaSegmentation.setActiveChannelsFromString(gd.getNextString());

		wekaSegmentation.setFeaturesToShowFromString(gd.getNextString());
		//wekaSegmentation.regionThreads = (int) gd.getNextNumber();
		//wekaSegmentation.threadsPerRegion = (int) gd.getNextNumber();
		//wekaSegmentation.numRfTrainingThreads = (int) gd.getNextNumber();

		wekaSegmentation.setTileSizeSetting( gd.getNextString() );
		//wekaSegmentation.tilingDelay = (int) gd.getNextNumber();
		wekaSegmentation.settings.backgroundThreshold = (int) gd.getNextNumber();
		//wekaSegmentation.setResolutionWeightsFromString( gd.getNextString() );
		//wekaSegmentation.uncertaintyLUTdecay = (double) gd.getNextNumber();

		/*
		c = (Object)m_ClassifierEditor.getValue();
	    String options = "";
	    final String[] optionsArray = ((OptionHandler)c).getOptions();
	    if (c instanceof OptionHandler)
	    {
	      options = Utils.joinOptions( optionsArray );
	    }
	    //System.out.println("Classifier after choosing: " + c.getClass().getName() + " " + options);
	    if( !originalClassifierName.equals( c.getClass().getName() )
	    		|| !originalOptions.equals( options ) )
	    {
	    	AbstractClassifier cls;
	    	try{
	    		cls = (AbstractClassifier) (c.getClass().newInstance());
	    		cls.setOptions( optionsArray );
	    	}
	    	catch(Exception ex)
	    	{
	    		ex.printStackTrace();
	    		return false;
	    	}

	    	// Assing new classifier
	    	wekaSegmentation.setClassifier( cls );

	    	// Set the training flag to false
	    	win.trainingComplete = false;

	    	// Macro recording
			record(SET_CLASSIFIER, new String[] { c.getClass().getName(), options} );

	    	IJ.log("Current classifier: " + c.getClass().getName() + " " + options);
	    }*/

		boolean classNameChanged = false;
		for(int i = 0; i < wekaSegmentation.getNumClasses(); i++)
		{
			String s = gd.getNextString();
			if (null == s || 0 == s.length()) {
				IJ.log("Invalid name for class " + (i+1));
				continue;
			}
			s = s.trim();
			if(!s.equals(wekaSegmentation.getClassName(i)))
			{
				if (0 == s.toLowerCase().indexOf("add to "))
					s = s.substring(7);

				wekaSegmentation.setClassLabel(i, s);
				classNameChanged = true;
				addAnnotationButton[i].setText( s + " [" + (i+1) + "]");
				// Macro recording
				record(CHANGE_CLASS_NAME, new String[]{Integer.toString(i), s});
			}
		}

		// adapt to changes in class names
		updateReviewLabelComboBox();

		// wekaSegmentation.setComputeFeatureImportance(gd.getNextBoolean());

		// Update flag to balance number of class trainingDataComboBox
		/*
		final boolean balanceClasses = gd.getNextBoolean();
		if( wekaSegmentation.doClassBalance() != balanceClasses )
		{
			wekaSegmentation.setClassBalance( balanceClasses );
			// Macro recording
			record( SET_BALANCE, new String[] { Boolean.toString( balanceClasses )});
		}
		*/
		// Update result overlay alpha
		final int newOpacity = (int) gd.getNextNumber();
		if( newOpacity != win.overlayOpacity )
		{
			win.overlayOpacity = newOpacity;
			win.overlayAlpha = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, win.overlayOpacity / 100f);
			win.resultOverlay.setComposite(win.overlayAlpha);

			// Macro recording
			record(SET_OPACITY, new String[] { Integer.toString( win.overlayOpacity )});

			if( showColorOverlay )
				displayImage.updateAndDraw();
		}


		// If there is a change in the class names,
		// the data set (trainingDataComboBox) must be updated.
		if(classNameChanged)
		{
			// Pack window to update buttons
			win.pack();
		}

		// Update feature stack if necessary
		/*
		if(featuresChanged)
		{
			// Force features to be updated
			wekaSegmentation.setFeaturesDirty();
		}
		else	// This checks if the feature stacks were updated while using the save feature stack button
			if( ! wekaSegmentation.getFeatureImages().isEmpty()
					&& wekaSegmentation.getFeatureImages().getReferenceSliceIndex() != -1)
				wekaSegmentation.setUpdateFeatures(false);
		*/

		return true;
	}

	// Quite of a hack from Johannes Schindelin:
	// use reflection to insert classifiersComboBox, since there is no other method to do that...
	// TODO: what is that good for??
	/*
	static {
		try {
			IJ.showStatus("Loading Weka properties...");
			IJ.log("Loading Weka properties...");
			Field field = GenericObjectEditor.class.getDeclaredField("EDITOR_PROPERTIES");
			field.setAccessible(true);
			Properties editorProperties = (Properties)field.get(null);
			String key = "weka.classifiersComboBox.Classifier";
			String value = editorProperties.getProperty(key);
			value += ",hr.irb.fastRandomForest.FastRandomForest";
			editorProperties.setProperty(key, value);
			//new Exception("insert").printStackTrace();
			//System.err.println("value: " + value);

			// add classifiersComboBox from properties (needed after upgrade to WEKA version 3.7.11)
			PluginManager.addFromProperties(editorProperties);
		} catch (Exception e) {
			IJ.error("Could not insert my own cool classifiersComboBox!");
		}
	}*/

	/* **********************************************************
	 * Macro recording related methods
	 * *********************************************************/

	/**
	 * Macro-record a specific command. The command names match the static
	 * methods that reproduce that part of the code.
	 *
	 * @param command name of the command including package info
	 * @param args set of arguments for the command
	 */
	public static void record(String command, String... args)
	{
		command = "call(\"trainableDeepSegmentation.Weka_Deep_Segmentation." + command;
		for(int i = 0; i < args.length; i++)
			command += "\", \"" + args[i];
		command += "\");\n";
		// in Windows systems, replace backslashes by double ones
		if( IJ.isWindows() )
			command = command.replaceAll( "\\\\", "\\\\\\\\" );
		if(Recorder.record)
			Recorder.recordString(command);
	}

	/**
	 * Toggle current result overlay image
	 */
	public synchronized static void toggleOverlay()
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			win.toggleOverlay();
		}
	}

	/**
	 * Save current project to disk
	 *
	 * @param projectPathName complete path name for the project file
	 */
	public void saveProject( String projectPathName )
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			final WekaSegmentation wekaSegmentation = win.getWekaSegmentation();
			if( ! wekaSegmentation.saveProject( projectPathName ) )
			{
				logger.error("Error while writing project to disk");
				return;
			}
		}
	}

	/**
	 * Change a class name
	 *
	 * @param classIndex index of the class to change
	 * @param className new class name
	 */
	public static void changeClassName(int classNum, String className)
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			final WekaSegmentation wekaSegmentation = win.getWekaSegmentation();

			//int classNum = Integer.parseInt(classIndex);
			wekaSegmentation.setClassLabel(classNum, className);
			win.updateAddClassButtons();
			win.pack();
		}

	}

	/**
	 * Set overlay opacity
	 * @param newOpacity string containing the new opacity value (integer 0-100)
	 */
	public static void setOpacity( String newOpacity )
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			win.overlayOpacity = Integer.parseInt(newOpacity);
			AlphaComposite alpha = AlphaComposite.getInstance(AlphaComposite.SRC_OVER,  win.overlayOpacity  / 100f);
			win.resultOverlay.setComposite(alpha);
		}
	}

	/**
	 * Check if ImageScience features are available
	 * @return true if ImageScience features are available
	 */
	private static boolean isImageScienceAvailable() {
		try {
			return ImageScience.isAvailable();
		}
		catch ( final NoClassDefFoundError err ) {
			return false;
		}
	}


	/**
	 * Create label image out of the current user traces. For convention, the
	 * label zero is used to define pixels with no class assigned. The rest of
	 * integer values correspond to the order of the classes (1 for the first
	 * class, 2 for the second class, etc.).
	 *
	 * @return label image containing user-defined traces (zero for undefined pixels)
	 */
	public static ImagePlus getLabelImage()
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			final WekaSegmentation wekaSegmentation = win.getWekaSegmentation();

			final int numClasses = wekaSegmentation.getNumClasses();
			final int width = win.getTrainingImage().getWidth();
			final int height = win.getTrainingImage().getHeight();
			final int depth = win.getTrainingImage().getNSlices();
			final int timepoints = win.getTrainingImage().getNFrames();
			final ImageStack labelStack;
			if( numClasses < 256)
				labelStack = ImageStack.create( width, height, depth, 8 );
			else if ( numClasses < 256 * 256 )
				labelStack = ImageStack.create( width, height, depth, 16 );
			else
				labelStack = ImageStack.create( width, height, depth, 32 );

			final ImagePlus labelImage = new ImagePlus( "Labels", labelStack );

			for( int t=1; t<=timepoints; t++ )
				for( int z=1; z<=depth; z++ )
				{
					labelImage.setT(t);
					labelImage.setZ(z);
					for( int j=0; j<numClasses; j++ )
					{
						 ArrayList<Roi> rois = wekaSegmentation.getExampleRois(j,z-1,t-1);
						 for( final Roi r : rois )
						 {
							 final ImageProcessor ip = labelImage.getProcessor();
							 ip.setValue( j+1 );
							 if( r.isLine() )
							 {
								 ip.setLineWidth( Math.round( r.getStrokeWidth() ) );
								 ip.draw( r );
							 }
							 else
								 ip.fill( r );
						 }
					}
				}
			labelImage.setSlice( 1 );
			labelImage.setDisplayRange( 0, numClasses );
			return labelImage;
		}
		return null;
	}



}// end of Weka_Deep_Segmentation class

