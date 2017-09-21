package trainableDeepSegmentation;

import bigDataTools.Region5D;
import bigDataTools.dataStreamingTools.DataStreamingTools;
import bigDataTools.logging.IJLazySwingLogger;
import bigDataTools.logging.Logger;
import bigDataTools.utils.ImageDataInfo;
import fiji.util.gui.GenericDialogPlus;
import fiji.util.gui.OverlayedImageCanvas;
import hr.irb.fastRandomForest.FastRandomForest;
import ij.*;
import ij.gui.ImageCanvas;
import ij.gui.ImageWindow;
import ij.gui.Roi;
import ij.gui.StackWindow;
import ij.io.FileSaver;
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
import java.lang.reflect.Field;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.zip.GZIPOutputStream;

import javax.swing.*;

import javafx.geometry.Point3D;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.gui.GUIChooserApp;
import weka.gui.GenericObjectEditor;
import weka.core.PluginManager;

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
	private ImagePlus classifiedImage = null;
	/** GUI window */
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
	private JButton trainButton = null;
	private JCheckBox trainingRecomputeFeaturesCheckBox = null;

	/** toggle overlay button */
	private JButton overlayButton = null;
	/** create result button */
	private JButton getResultButton = null;
	/** create result button */
	private JButton setResultButton = null;

	private JButton createResultButton = null;

	private JCheckBox resultOnDiskCheckBox = null;


	private JButton printFeatureNamesButton = null;

	/** get probability maps button */
	private JButton probabilityButton = null;
	/** plot result button */
	private JButton plotButton = null;
	/** apply classifier button */
	private JButton applyButton = null;
	/** apply classifier button */
	private JButton postProcessButton = null;

	private JTextField classificationRangeTextField = null;
	private JTextField objectSizeRangeTextField = null;

	/** load classifier button */
	private JButton loadClassifierButton = null;
	/** save classifier button */
	private JButton saveClassifierButton = null;
	/** load annotations button */
	private JButton loadAnnotationsButton = null;
	/** save annotations button */
	private JButton saveAnnotationsButton = null;
	/** settings button */
	private JButton settingsButton = null;

	private JButton testThreadsButton = null;

	private JTextField uncertaintyTextField = new JTextField();

	/** Weka button */
	private JButton wekaButton = null;
	/** create new class button */
	private JButton addClassButton = null;

	/** array of roi list overlays to paint the transparent rois of each class */
	private RoiListOverlay [] roiOverlay = null;

	/** available colors for available classes */
	private Color[] colors = new Color[]{
			Color.blue,
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
	public static final String TRAIN_CLASSIFIER = "trainClassifier";
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
	public static final String LOAD_CLASSIFIER = "loadClassifier";
	/** name of the macro method to save the current classifier into a file */
	public static final String SAVE_CLASSIFIER = "saveClassifier";
	/** name of the macro method to load data from an ARFF file */
	public static final String LOAD_DATA = "loadData";
	/** name of the macro method to save the current data into an ARFF file */
	public static final String SAVE_DATA = "saveData";
	/** name of the macro method to load data from an ARFF file */
	public static final String LOAD_ANNOTATIONS = "loadAnnotations";
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
	private boolean trainingFlag = false;


	private Logger logger;


	/**
	 * Basic constructor for graphical user interface use
	 */
	public Weka_Deep_Segmentation()
	{
		// check for image science
		// TODO: does not work
		if ( ! ImageScience.isAvailable() )
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
			int offset = iClass * WekaSegmentation.CLASS_LUT_WIDTH;
			for(int i = 0 ; i < WekaSegmentation.CLASS_LUT_WIDTH; i++)
			{
				red[offset + i] = (byte) (1.0*colors[iClass].getRed()*i/(WekaSegmentation.CLASS_LUT_WIDTH-1));
				green[offset + i] = (byte) (1.0*colors[iClass].getGreen()*i/(WekaSegmentation.CLASS_LUT_WIDTH-1));
				blue[offset + i] = (byte) (1.0*colors[iClass].getBlue()*i/(WekaSegmentation.CLASS_LUT_WIDTH-1));
			}
		}
		overlayLUT = new LUT(red, green, blue);

		exampleList = new java.awt.List[WekaSegmentation.MAX_NUM_CLASSES];
		addAnnotationButton = new JButton[WekaSegmentation.MAX_NUM_CLASSES];

		roiOverlay = new RoiListOverlay[WekaSegmentation.MAX_NUM_CLASSES];

		trainButton = new JButton("Train classifier");
		trainButton.setToolTipText("Start training the classifier");
		trainingRecomputeFeaturesCheckBox = new JCheckBox("Recompute", false);

		overlayButton = new JButton("Toggle overlay [r][p][u]");
		overlayButton.setToolTipText("Toggle between current segmentation and original image");
		overlayButton.setEnabled(false);

		getResultButton = new JButton("Create result");
		getResultButton.setToolTipText("Generate result image");
		getResultButton.setEnabled(false);

		setResultButton = new JButton("Set result image");
		setResultButton.setToolTipText("Set result image");
		setResultButton.setEnabled(true);

		createResultButton = new JButton("Create result image");
		createResultButton.setToolTipText("Create result image");
		createResultButton.setEnabled(true);

		resultOnDiskCheckBox = new JCheckBox("Disk", true);

		probabilityButton = new JButton("Get probability");
		probabilityButton.setToolTipText("Generate current probability maps");
		probabilityButton.setEnabled(false);

		plotButton = new JButton("Plot result");
		plotButton.setToolTipText("Plot result based on different metrics");
		plotButton.setEnabled(false);

		printFeatureNamesButton = new JButton("Feature information");
		printFeatureNamesButton.setToolTipText("Prints feature names to Log window");
		printFeatureNamesButton.setEnabled(true);

		applyButton = new JButton ("Apply classifier");
		applyButton.setToolTipText("Apply current classifier to a single image or stack");
		applyButton.setEnabled(false);

		postProcessButton = new JButton ("Post process");
		postProcessButton.setToolTipText("Post process results");
		postProcessButton.setEnabled( false );

		classificationRangeTextField = new JTextField("None", 15);
		objectSizeRangeTextField     = new JTextField("300,100000");

		loadClassifierButton = new JButton ("Load classifier");
		loadClassifierButton.setToolTipText("Load Weka classifier from a file");

		saveClassifierButton = new JButton ("Save classifier");
		saveClassifierButton.setToolTipText("Save current classifier into a file");
		saveClassifierButton.setEnabled(false);

		loadAnnotationsButton = new JButton ("Load labels");
		loadAnnotationsButton.setEnabled(true);

		saveAnnotationsButton = new JButton ("Save labels");
		saveAnnotationsButton.setEnabled(false);

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
					if(e.getSource() == trainButton)
					{
						runStopTraining(command);
					}
					else if(e.getSource() == overlayButton){
						// Macro recording
						String[] arg = new String[] {};
						record(TOGGLE_OVERLAY, arg);
						win.toggleOverlay();
					}
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
						setClassifiedImage();
						postProcessButton.setEnabled( true );
					}
					else if(e.getSource() == createResultButton){
						// Macro recording
						//String[] arg = new String[] {};
						//record(SET_RESULT, arg);

						createClassifiedImage( createResultButton,
								resultOnDiskCheckBox);


					}
					else if( e.getSource() == probabilityButton )
					{
						// Macro recording
						String[] arg = new String[] {};
						record(GET_PROBABILITY, arg);
						//showProbabilityImage();
					}
					else if( e.getSource() == printFeatureNamesButton )
					{
						logger.info("Feature list, sorted according to usage in random forest:");

						if ( wekaSegmentation.featureList != null )
						{

							ArrayList<Feature> sortedFeatureList = new ArrayList<>( wekaSegmentation.featureList );
							sortedFeatureList.sort( Comparator.comparing( Feature::getUsageInRF ) );

							int sumFeatureUsage = 0;

							for ( Feature feature : sortedFeatureList )
							{
								sumFeatureUsage += feature.usageInRF;

								if ( feature.isActive )
								{
									int featureID = wekaSegmentation.featureList.indexOf( feature );

									logger.info("ID: " + featureID +
											"; Name: " + feature.featureName +
											"; Usage: " + feature.usageInRF +
											"; Active: " + feature.isActive);
								}

							}

							logger.info( "Sum feature usage: " + sumFeatureUsage );

							int i = 0;
							for (int n : wekaSegmentation.numFeaturesPerResolution)
								logger.info("Number of features at resolution level " + (i++) + ": " + n);

						}
						else
						{
							logger.info("  Feature list not yet known; please run a training first");
						}
					}
					else if(e.getSource() == applyButton)
					{
						applyClassifierToSelectedRegion(
								command,
								classificationRangeTextField.getText());
					}
					else if(e.getSource() == postProcessButton)
					{
						postProcessSelectedRegion(
								command,
								classificationRangeTextField.getText(),
								objectSizeRangeTextField.getText());
					}
					else if(e.getSource() == loadClassifierButton)
					{
						loadClassifier();
						win.updateButtonsEnabling();
					}
					else if(e.getSource() == saveClassifierButton)
					{
						saveClassifier();
					}
					else if(e.getSource() == loadAnnotationsButton){
						loadAnnotations(null, null);
					}
					else if(e.getSource() == saveAnnotationsButton){
						saveAnnotations();
					}
					else if(e.getSource() == addClassButton){
						addNewClass();
					}
					else if(e.getSource() == settingsButton){
						showSettingsDialog();
						win.updateButtonsEnabling();
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
						if(e.getSource() == exampleList[i])
							listSelected(e, i);
					}
				}
			}).start();
		}
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
				exampleList[i].addActionListener( listener );
				exampleList[i].addItemListener( itemListener );
				addAnnotationButton[i] = new JButton(wekaSegmentation.getClassLabel(i) + " [" + (i+1) + "]" );
				addAnnotationButton[i].setToolTipText("Add markings of label '" + wekaSegmentation.getClassLabel(i) + "'");

				annotationsConstraints.insets = new Insets(5, 5, 6, 6);

				annotationsPanel.add( addAnnotationButton[i], annotationsConstraints );
				annotationsConstraints.gridy++;

				annotationsConstraints.insets = new Insets(0,0,0,0);

				annotationsPanel.add( exampleList[i], annotationsConstraints );
				annotationsConstraints.gridy++;
			}

			// Select first class
			addAnnotationButton[0].setSelected(true);

			// Add listeners
			for(int i = 0; i < wekaSegmentation.getNumClasses(); i++)
				addAnnotationButton[i].addActionListener(listener);
			trainButton.addActionListener(listener);
			overlayButton.addActionListener(listener);
			getResultButton.addActionListener(listener);
			setResultButton.addActionListener(listener);
			createResultButton.addActionListener(listener);
			probabilityButton.addActionListener(listener);
			plotButton.addActionListener(listener);
			printFeatureNamesButton.addActionListener(listener);
			applyButton.addActionListener(listener);
			postProcessButton.addActionListener(listener);
			loadClassifierButton.addActionListener(listener);
			saveClassifierButton.addActionListener(listener);
			loadAnnotationsButton.addActionListener(listener);
			saveAnnotationsButton.addActionListener(listener);
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
								addAnnotation(iClass-1);
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

			JPanel trainClassifierPanel = new JPanel();
			trainClassifierPanel.add(trainButton);
			trainClassifierPanel.add(trainingRecomputeFeaturesCheckBox);
			trainingJPanel.add(trainClassifierPanel, trainingConstraints);
			trainingConstraints.gridy++;

			trainingJPanel.add(overlayButton, trainingConstraints);
			trainingConstraints.gridy++;

			JPanel uncertaintyPanel = new JPanel();
			JLabel uncertaintyLabel = new JLabel(
					"Uncertainties: [g][n][b][d]"
			);
			uncertaintyPanel.add(uncertaintyLabel);
			uncertaintyTextField.setText("    0");
			uncertaintyPanel.add(uncertaintyTextField);
			trainingJPanel.add(uncertaintyPanel, trainingConstraints);
			trainingConstraints.gridy++;

			//trainingJPanel.add(setResultButton, trainingConstraints);
			//trainingConstraints.gridy++;

			JPanel resultPanel = new JPanel();
			resultPanel.add(createResultButton);
			resultPanel.add(resultOnDiskCheckBox);
			trainingJPanel.add(resultPanel, trainingConstraints);
			trainingConstraints.gridy++;

			//trainingJPanel.add(probabilityButton, trainingConstraints);
			//trainingConstraints.gridy++;
			//trainingJPanel.add(plotButton, trainingConstraints);
			//trainingConstraints.gridy++;
			trainingJPanel.add(printFeatureNamesButton, trainingConstraints);
			trainingConstraints.gridy++;

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

			trainingJPanel.add(applyButton, trainingConstraints);

			trainingConstraints.gridy++;
			JPanel panelZRange = new JPanel();
			panelZRange.add(new JLabel("Range"));
			panelZRange.add( classificationRangeTextField );
			trainingJPanel.add(panelZRange, trainingConstraints);

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

			trainingConstraints.gridy++;
			trainingJPanel.add(loadClassifierButton, trainingConstraints);

			trainingConstraints.gridy++;
			trainingJPanel.add(saveClassifierButton, trainingConstraints);

			trainingConstraints.gridy++;
			trainingJPanel.add(loadAnnotationsButton, trainingConstraints);

			trainingConstraints.gridy++;
			trainingJPanel.add(saveAnnotationsButton, trainingConstraints);

			trainingConstraints.gridy++;
			trainingJPanel.add(addClassButton, trainingConstraints);

			trainingConstraints.gridy++;
			trainingJPanel.add(settingsButton, trainingConstraints);

			//trainingConstraints.gridy++;
			//trainingJPanel.add(testThreadsButton, trainingConstraints);

			trainingConstraints.gridy++;
			trainingJPanel.add(wekaButton, trainingConstraints);

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
					trainButton.removeActionListener(listener);
					overlayButton.removeActionListener(listener);
					getResultButton.removeActionListener(listener);
					setResultButton.removeActionListener(listener);
					createResultButton.removeActionListener(listener);
					probabilityButton.removeActionListener(listener);
					plotButton.removeActionListener(listener);
					//newImageButton.removeActionListener(listener);
					applyButton.removeActionListener(listener);
					postProcessButton.removeActionListener(listener);

					loadClassifierButton.removeActionListener(listener);
					saveClassifierButton.removeActionListener(listener);
					loadAnnotationsButton.removeActionListener(listener);
					saveAnnotationsButton.removeActionListener(listener);
					addClassButton.removeActionListener(listener);
					settingsButton.removeActionListener(listener);
					wekaButton.removeActionListener(listener);

					// Set number of classes back to 2
					wekaSegmentation.setNumClasses(2);
				}
			});

			canvas.addComponentListener(new ComponentAdapter() {
				public void componentResized(ComponentEvent ce) {
					Rectangle r = canvas.getBounds();
					canvas.setDstDimensions(r.width, r.height);
				}
			});

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
					zoomToSelection( displayImage, 3.0 );
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


		void zoomToSelection(ImagePlus imp, double marginFactor) {

			ImageCanvas ic = imp.getCanvas();
			Roi roi = imp.getRoi();
			ic.unzoom();
			if (roi==null) return;
			Rectangle w = imp.getWindow().getBounds();
			Rectangle r = roi.getBounds();
			double mag = ic.getMagnification();
			int marginw = (int)(marginFactor * (w.width - mag * imp.getWidth()));
			int marginh = (int)(marginFactor * (w.height - mag * imp.getHeight()));
			int x = r.x+r.width/2;
			int y = r.y+r.height/2;
			mag = ic.getHigherZoomLevel(mag);
			while(r.width*mag < w.width-marginw && r.height*mag<w.height-marginh) {
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

			for(int iClass = 0; iClass < wekaSegmentation.getNumClasses(); iClass++)
			{
				roiOverlay[iClass].setColor(colors[iClass]);

				//roisThisClass = ;
				//final ArrayList< Roi > rois = new ArrayList<Roi>();
				//for (Roi r : roisThisClass ) rois.add( r );
				roiOverlay[iClass].setRoi(
						(ArrayList<Roi>) wekaSegmentation.getExampleRois(
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
				for(int j = 0; j < wekaSegmentation.getExampleRois(i, slice-1, frame-1).size(); j++)
					exampleList[i].add("trace " + j + " (Z=" + slice+" T="+frame+")");
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
			exampleList[classNum].setForeground(colors[classNum]);

			exampleList[classNum].addActionListener(listener);
			exampleList[classNum].addItemListener(itemListener);
			addAnnotationButton[classNum] = new JButton(wekaSegmentation.getClassLabel(classNum) + " [" + (classNum+1) + "]");

			annotationsConstraints.fill = GridBagConstraints.HORIZONTAL;
			annotationsConstraints.insets = new Insets(5, 5, 6, 6);

			boxAnnotation.setConstraints(addAnnotationButton[classNum], annotationsConstraints);
			annotationsPanel.add(addAnnotationButton[classNum]);
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
			trainButton.setEnabled(s);
			overlayButton.setEnabled(s);
			getResultButton.setEnabled(s);
			setResultButton.setEnabled(s);
			createResultButton.setEnabled(s);
			probabilityButton.setEnabled(s);
			probabilityButton.setEnabled(s);
			printFeatureNamesButton.setEnabled(s);
			plotButton.setEnabled(s);
			//newImageButton.setEnabled(s);
			applyButton.setEnabled(s);
			postProcessButton.setEnabled(s);
			loadClassifierButton.setEnabled(s);
			saveClassifierButton.setEnabled(s);
			saveAnnotationsButton.setEnabled(s);
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
			if( trainingFlag )
			{
				setButtonsEnabled( false );
				trainButton.setEnabled(true);
			}
			else // If the training is not going on
			{
				final boolean classifierExists =  null != wekaSegmentation.getClassifier();

				trainButton.setEnabled( classifierExists );
				applyButton.setEnabled( win.trainingComplete );
				postProcessButton.setEnabled( win.trainingComplete );
				final boolean resultExists = null != classifiedImage &&
											 null != classifiedImage.getProcessor();

				saveClassifierButton.setEnabled( win.trainingComplete );
				overlayButton.setEnabled(resultExists);
				getResultButton.setEnabled(win.trainingComplete);
				setResultButton.setEnabled( true) ;
				createResultButton.setEnabled( true) ;


				plotButton.setEnabled( win.trainingComplete );
				probabilityButton.setEnabled( win.trainingComplete );
				printFeatureNamesButton.setEnabled( true );

				//newImageButton.setEnabled(true);
				loadClassifierButton.setEnabled(true);
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

				saveAnnotationsButton.setEnabled( !examplesEmpty );

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
					int offset = iClass * WekaSegmentation.CLASS_LUT_WIDTH;
					for (int i = 0; i < WekaSegmentation.CLASS_LUT_WIDTH; i++)
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
					int offset = iClass * WekaSegmentation.CLASS_LUT_WIDTH;
					for (int i = 0; i < WekaSegmentation.CLASS_LUT_WIDTH; i++)
					{
						red[offset + i] = (byte) (1.0 * colors[iClass].getRed() * i / (WekaSegmentation.CLASS_LUT_WIDTH - 1));
						green[offset + i] = (byte) (1.0 * colors[iClass].getGreen() * i / (WekaSegmentation.CLASS_LUT_WIDTH - 1));
						blue[offset + i] = (byte) (1.0 * colors[iClass].getBlue() * i / (WekaSegmentation.CLASS_LUT_WIDTH - 1));
					}
				}
				overlayLUT = new LUT(red, green, blue);
			}

			if ( mode.equals("uncertainty") )
			{
				// assign colors to classes
				for (int iClass = 0; iClass < WekaSegmentation.MAX_NUM_CLASSES; iClass++)
				{
					int offset = iClass * WekaSegmentation.CLASS_LUT_WIDTH;
					for (int i = 0; i < WekaSegmentation.CLASS_LUT_WIDTH; i++)
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
			if (showColorOverlay && null != classifiedImage)
			{
				updateResultOverlay();
			}
			else
				resultOverlay.setImage(null);

			displayImage.updateAndDraw();
		}

		/**
		 * Set a new result (classified) image
		 * @param classifiedImage new result image
		 */
		protected void setClassfiedImage(ImagePlus classifiedImage)
		{
			updateClassifiedImage(classifiedImage);
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
				addAnnotationButton[i].setText(wekaSegmentation.getClassLabel(i) + " [" + (i+1) + "]");

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

	/**
	 * Plugin run method
	 */
	public void run(String arg)
	{
		// START
		// instantiate segmentation backend
		wekaSegmentation = new WekaSegmentation();
		logger = wekaSegmentation.getLogger();

		for(int i = 0; i < wekaSegmentation.getNumClasses() ; i++)
		{
			exampleList[i] = new java.awt.List(5);
			exampleList[i].setForeground(colors[i]);
		}
		numOfClasses = wekaSegmentation.getNumClasses();

		// TODO:
		// - what makes sense here??
		ArrayList< Integer > resolutionWeights = new ArrayList<>();
		resolutionWeights.add(1); // 9
		resolutionWeights.add(1); // 3
		resolutionWeights.add(1);
		resolutionWeights.add(1);
		resolutionWeights.add(1);
		resolutionWeights.add(1);
		wekaSegmentation.resolutionWeights = resolutionWeights;

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

		wekaSegmentation.setTrainingImage( trainingImage );

		Calibration calibration = trainingImage.getCalibration();
		wekaSegmentation.anisotropy = 1.0 * calibration.pixelDepth / calibration.pixelWidth;

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
		wekaSegmentation.activeChannels = channelsToConsider;

		displayImage = trainingImage;

		ij.gui.Toolbar.getInstance().setTool(ij.gui.Toolbar.FREELINE);

		//Build GUI
		SwingUtilities.invokeLater(
				new Runnable() {
					public void run()
					{
						win = new CustomWindow( displayImage );
						win.pack();
					}
				});
	}

	private void setClassifiedImage()
	{
		classifiedImage = WindowManager.getImage("classification_result");
		if ( classifiedImage == null )
		{
			IJ.showMessage("You need an open image with the name: classification_result");
			return;
		}
		overlayButton.setEnabled(true);
		classifiedImage.hide();
		wekaSegmentation.setClassifiedImage( classifiedImage );
	}

	private void createClassifiedImage(
			JButton createResultButton,
			JCheckBox resultOnDiskCheckBox)
	{
		String buttonText = createResultButton.getText();

		String directory = null;

		if ( buttonText.contains("Create") )
		{
			if ( resultOnDiskCheckBox.isSelected() )
			{
				directory = IJ.getDirectory("Select a directory");

				DataStreamingTools dst = new DataStreamingTools();
				String tMax = String.format("%05d", trainingImage.getNFrames());
				String zMax = String.format("%05d", trainingImage.getNSlices());

				String namingPattern = "classified--C<C01-01>--T<T00001-" +
						tMax + ">--Z<Z00001-"+zMax+">.tif";
				bigDataTools.utils.ImageDataInfo imageDataInfo = new ImageDataInfo();
				imageDataInfo.bitDepth = 8;
				int nIOthreads = 3;

				// create one image
				ImageStack stack = ImageStack.create(trainingImage.getWidth(),
						trainingImage.getHeight(), 1, 8);
				ImagePlus impC0T0Z0 = new ImagePlus("", stack);
				FileSaver fileSaver = new FileSaver( impC0T0Z0 );
				fileSaver.saveAsTiff( directory + "/" +
						"classified--C01--T00001--Z00001.tif");

				classifiedImage = dst.openFromDirectory(
						directory,
						namingPattern,
						"None",
						"None",
						imageDataInfo,
						nIOthreads,
						false,
						true);

				int nZ = trainingImage.getNSlices();
				int nT = trainingImage.getNFrames();
				classifiedImage.setDimensions(
						1,
						nZ,
						nT);

				classifiedImage.setOpenAsHyperStack(true);
				classifiedImage.setTitle("classification_result");
				overlayButton.setEnabled(true);

				wekaSegmentation.setClassifiedImage( classifiedImage );

				// set up logging to a file
				//
				String logFileDirectory = directory.substring(0, directory.length() - 1)
						+ "--log";
				String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").
						format(new Date());
				String logFileName = "log-" + timeStamp + ".txt";
				wekaSegmentation.setLogFileNameAndDirectory(logFileName,
						logFileDirectory);

				logger.info("Created disk-resident classification result image: " +
						directory);


			}
			else // create in RAM
			{
				ImageStack stack = ImageStack.create(
						trainingImage.getWidth(),
						trainingImage.getHeight(),
						trainingImage.getNSlices() * trainingImage.getNFrames(),
						8
				);

				ImagePlus classifiedImage = new ImagePlus("classification_result", stack);
				classifiedImage.setDimensions(1, trainingImage.getNSlices(), trainingImage.getNFrames());
				classifiedImage.setOpenAsHyperStack( true );
				//classifiedImage.show();
				this.classifiedImage = classifiedImage;
				overlayButton.setEnabled(true);
				wekaSegmentation.setClassifiedImage(classifiedImage);

				logger.info("Created memory-resident classification result image.");

			}

			createResultButton.setText("Get result image");

		}

		if ( buttonText.contains("Get") )
		{
			classifiedImage.show();
		}

	}

	/**
	 * Add examples defined by the user to the corresponding list
	 * in the GUI and the example list in the segmentation object.
	 *
	 * @param classNum GUI list index
	 */
	private void addAnnotation(int classNum)
	{
		if ( classNum >= numOfClasses ) return;

		final Roi roi = displayImage.getRoi();
		if (null == roi)
			return;
		displayImage.killRoi();

		Point[] points = roi.getContainedPoints();

		final int z = displayImage.getZ() - 1;
		final int t = displayImage.getT() - 1;

		Example newExample = wekaSegmentation.createExample(classNum, points, (int)roi.getStrokeWidth(), z, t);

		if ( wekaSegmentation.isValidExample( newExample ) )
		{
			wekaSegmentation.addExample(newExample);

			traceCounter[classNum]++;
			win.drawExamples();
			win.updateExampleLists();
			// Record
			final int n = displayImage.getCurrentSlice();
			String[] arg = new String[]{
					Integer.toString(classNum),
					Integer.toString(n)};
			record(ADD_TRACE, arg);
		}
		else
		{
			logger.error("Label is too close to image boundary.\n" +
					"Please:\n" +
					"- Put label more central.\n" +
					"- Go to [Settings] and reduce maximal feature scale.");
		}

	}


	/**
	 * Update the result image
	 *
	 * @param classifiedImage new result image
	 */
	public void updateClassifiedImage(ImagePlus classifiedImage)
	{
		this.classifiedImage = classifiedImage;
	}

	/**
	 * Update the result image overlay with the corresponding slice
	 */
	public void updateResultOverlay()
	{
		int n = classifiedImage.getStackIndex( 0, displayImage.getZ(), displayImage.getFrame() );
		ImageProcessor overlay = classifiedImage.getImageStack().getProcessor( n );

		overlay = overlay.convertToByte( false );
		overlay.setColorModel(overlayLUT);

		win.resultOverlay.setImage(overlay);
	}

	/**
	 * Select a list and deselect the others
	 *
	 * @param e item event (originated by a list)
	 * @param i list index
	 */
	void listSelected(final ItemEvent e, final int i)
	{
		// i is the classnum

		// win.drawExamples();
		displayImage.setColor(Color.YELLOW);

		int selectedIndex = 0;

		for(int j = 0; j < wekaSegmentation.getNumClasses(); j++)
		{
			if (j == i)
			{
				selectedIndex = exampleList[i].getSelectedIndex();
				if ( selectedIndex != -1 )
				{
					final Roi newRoi =
							wekaSegmentation.getExampleRois(i,
									displayImage.getZ() - 1,
									displayImage.getT() - 1)
									.get(selectedIndex);
					// Set selected trace as current ROI
					newRoi.setImage(displayImage);
					displayImage.setRoi(newRoi);
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
				wekaSegmentation.deleteExample(
						iClass,
						displayImage.getZ()-1,
						displayImage.getT()-1,
						index);

				//delete item from GUI list
				exampleList[iClass].remove(index);

				// Record
				String[] arg = new String[] {
						Integer.toString(iClass),
						Integer.toString( displayImage.getCurrentSlice() ),
						Integer.toString(index)};
				record(DELETE_TRACE, arg);
			}

		win.drawExamples();
		win.updateExampleLists();
	}

	/**
	 * Run/stop the classifier training
	 *
	 * @param command current text of the training button ("Train classifier" or "STOP")
	 */
	void runStopTraining(final String command)
	{
		// If the training is not going on, we start it
		if (command.equals("Train classifier"))
		{
			trainingFlag = true;
			trainButton.setText("STOP");
			final Thread oldTask = trainingTask;
			// Disable rest of buttons until the training has finished
			win.updateButtonsEnabling();

			// Set train button text to STOP
			trainButton.setText("STOP");

			// Thread to run the training
			Thread newTask = new Thread() {

				public void run()
				{
					// Wait for the old task to finish
					if (null != oldTask)
					{
						try {
							IJ.log("Waiting for old task to finish...");
							oldTask.join();
						}
						catch (InterruptedException ie)	{ /*IJ.log("interrupted");*/ }
					}

					try{
						// Macro recording
						String[] arg = new String[] {};
						record(TRAIN_CLASSIFIER, arg);

						logger.info("# Training classifier using all features ");
						wekaSegmentation.setAllFeaturesActive();
						boolean trainingFinished = wekaSegmentation.trainClassifier(
								trainingRecomputeFeaturesCheckBox.isSelected() );

						if( trainingFinished )
						{
							if( this.isInterrupted() )
							{
								win.trainingComplete = false;
								return;
							}

							if ( wekaSegmentation.minFeatureUsage > 0 )
							{

								wekaSegmentation.deactivateRarelyUsedFeatures();

								logger.info("# Training classifier again, " +
										"now only with useful features ");
								logger.info("Feature usage threshold: " +
										wekaSegmentation.minFeatureUsage);
								logger.info("Resulting active features: "
										+ wekaSegmentation.getNumActiveFeatures()
										+ "/" + wekaSegmentation.getNumFeatures());

								wekaSegmentation.trainClassifier( false );
							}

							win.trainingComplete = true;
						}
						else
						{
							IJ.log("The training did not finish.");
							win.trainingComplete = false;
						}

					}
					catch(Exception e)
					{
						e.printStackTrace();
					}
					catch( OutOfMemoryError err )
					{
						err.printStackTrace();
						IJ.log( "ERROR: plugin run out of memory. Please, "
								+ "use a smaller input image or fewer features." );
					}
					finally
					{
						trainingFlag = false;
						trainButton.setText("Train classifier");
						win.updateButtonsEnabling();
						trainingTask = null;
					}
				}

			};

			//IJ.log("*** Set task to new TASK (" + newTask + ") ***");
			trainingTask = newTask;
			newTask.start();
		}
		else if (command.equals("STOP"))
		{
			try{
				trainingFlag = false;
				win.trainingComplete = false;
				IJ.log("Training was stopped by the user!");
				win.setButtonsEnabled( false );
				trainButton.setText("Train classifier");

				if(null != trainingTask)
					trainingTask.interrupt();
				else
					IJ.log("Error: interrupting training failed becaused the thread is null!");

				//wekaSegmentation.shutDownNow();
				win.updateButtonsEnabling();
			}catch(Exception ex){
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

	public void applyClassifierToSelectedRegion( String command, String rangeString )
	{

		// TODO
		// move all of this into wekaSegmentation class

		if ( ! wekaSegmentation.isTrainingCompleted )
		{
			logger.error("The training is not fully completed; " +
					"please (once more) click [Train Classifier].");
			return;
		}

		if ( classifiedImage == null )
		{
			logger.error("classification_result image not set.\n" +
					"Please [Create result image].");
			return;
		}

		if ( classifiedImage.isVisible() )
		{
			classifiedImage.hide();
			wekaSegmentation.setClassifiedImage( classifiedImage );
		}

		if ( command.equals("STOP") )
		{
			logger.info("Stopping classification threads...");
			wekaSegmentation.stopCurrentThreads = true;
			applyButton.setText("Apply classifier");
			return;
		}
		else
		{
			Roi roi = displayImage.getRoi();
			if (roi == null || !(roi.getType() == Roi.RECTANGLE))
			{
				IJ.showMessage("Please use ImageJ's rectangle selection tool to  image" +
						" in order to select the center of the region to be classified");
				return;
			}

			logger.info("Classifying selected region...");
			applyButton.setText("STOP");
			wekaSegmentation.stopCurrentThreads = false;
			wekaSegmentation.setTrainingImage(displayImage);
			wekaSegmentation.setClassifiedImage(classifiedImage);
			wekaSegmentation.resetUncertaintyRegions();

			Thread thread = new Thread() {
				public void run()
				{
					wekaSegmentation.totalThreadsExecuted.addAndGet(1);
					ExecutorService exe = Executors.newFixedThreadPool(
							wekaSegmentation.numRegionThreads );
					ArrayList<Future> futures = new ArrayList<>();
					long startTime = System.currentTimeMillis();
					Rectangle rectangle = roi.getBounds();

					int[] sizes = new int[3];
					int[] borders = wekaSegmentation.getFeatureBorderSizes();
					int[] imgDims = wekaSegmentation.getImgDims();


					int[] xyztStart = new int[4];
					int[] xyztEnd = new int[4];
					int[] xyztNum = new int[4];


					xyztStart[0] = (int) rectangle.getX() - borders[0];
					xyztEnd[0] = xyztStart[0] + (int) rectangle.getWidth() - 1;

					xyztStart[1] = (int) rectangle.getY() - borders[1];
					xyztEnd[1] = xyztStart[1] + (int) rectangle.getHeight() - 1;

					xyztStart[2] = ( displayImage.getZ() - 1 ) - (int) ( 1.0 * sizes[2] / 2 );
					xyztStart[2] = xyztStart[2] < 0 ? 0 : xyztStart[2];
					xyztEnd[2] = xyztStart[2];

					xyztStart[3] = xyztEnd[3] = displayImage.getT() - 1;

					// potentially change z and t range to user selection

					try
					{
						int[] range = bigDataTools.utils.Utils.delimitedStringToIntegerArray(rangeString, ",");

						if ( trainingImage.getNFrames() == 1 )
						{
							xyztStart[2] = range[0] - 1;
							xyztEnd[2] = range[1] - 1;
						}
						else if ( trainingImage.getNSlices() == 1 )
						{
							// time
							xyztStart[3] = range[0] - 1;
							xyztEnd[3] = range[1] - 1;
						}
						else
						{
							xyztStart[2] = range[0] - 1;
							xyztEnd[2] = range[1] - 1;

							if ( range.length == 4 )
							{
								// time
								xyztStart[3] = range[2] - 1;
								xyztEnd[3] = range[3] - 1;
							}
						}
					}
					catch (NumberFormatException e)
					{
						// logger.info("No z or t range selected.");
					}

					// TODO:
					// - function that test whether current feature settings
					// are compatible with image dimensions!


					// tile sizes
					// TODO: put into extra function
					for ( int i = 0; i < 3; ++i )
					{
						if ( wekaSegmentation.tileSizeSetting.equals("auto") )
						{
							if ( xyztEnd[i] - xyztStart[i] < 3 * borders[i] )
							{
								sizes[i] = 3 * borders[i];
							}
							else
							{
								// TODO: could be even larger, but RAM!
								sizes[i] = Math.min ( 2 * 3 * borders[i],
										imgDims[i] );
							}
						}
					}


					int[] distances = new int[3];
					int iTotal = 0, nTotal = 0;
					int pixelsClassifiedPerRegion = 1;
					for ( int i = 0; i < 3; ++i )
					{
						distances[i] = sizes[i] - 2 * borders[i];
						distances[i] = distances[i] < 1 ? 1 : distances[i];
						xyztNum[i] = (xyztEnd[i] - xyztStart[i]) / distances[i] + 1;
						nTotal += xyztNum[i];
						pixelsClassifiedPerRegion *= (sizes[i]-2*borders[i]);
					}

					xyztNum[3] = (xyztEnd[3] - xyztStart[3]) + 1;
					nTotal += xyztNum[3];

					int numThreadsPerRegion = ( nTotal == 1 ) ? Prefs.getThreads() : wekaSegmentation.numThreadsPerRegion;

					logger.info("Selected region will be classified in " + nTotal + " tiles.");
					logger.info("Tile sizes are [x,y,z]: " + sizes[0]
									+ ", " + sizes[1]
									+ ", " + sizes[2]
					);

					ArrayList<int[]> positions = new ArrayList<>();

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
									region5D.c = displayImage.getC() - 1;
									region5D.offset = new Point3D(x, y, z);
									region5D.size = new Point3D(sizes[0], sizes[1], sizes[2]);
									region5D.offset = wekaSegmentation.shiftOffsetToStayInBounds( region5D.offset, region5D.size );
									region5D.subSampling = new Point3D(1, 1, 1);

									boolean updateFeatureImages = true;
									boolean computeProbabilityMaps = false;

									positions.add(new int[]{
											(int)region5D.offset.getX(),
											(int)region5D.offset.getY(),
											(int)region5D.offset.getZ() + 1,
											region5D.t + 1});

									try
									{
										if (!wekaSegmentation.stopCurrentThreads)
										{

											futures.add(
													exe.submit(
															wekaSegmentation.applyClassifierRunnable(
																	region5D,
																	numThreadsPerRegion,
																	++iTotal,
																	nTotal)
													)
											);
										}
									}
									catch (OutOfMemoryError e)
									{
										logger.error("Out of memory: " + e.toString());
									}


								}
							}
						}
					}


					int regionsClassified = 0;
					long nThreadsLast = wekaSegmentation.totalThreadsExecuted.get();
					long maximumMemoryUsage = 0L;
					long totalMemory = IJ.maxMemory();

					for ( Future future : futures )
					{
						try
						{
							if ( ! wekaSegmentation.stopCurrentThreads )
							{
								// collect all results and ask for garbage collection
								future.get();
								wekaSegmentation.totalThreadsExecuted.addAndGet(1);
								System.gc();
								regionsClassified++;
								long nThreadsNew = wekaSegmentation.totalThreadsExecuted.get() - nThreadsLast;
								long pixels = 1L * regionsClassified * pixelsClassifiedPerRegion;
								long milliSeconds = (System.currentTimeMillis() - startTime);
								double rate = 1.0 * pixels / (1.0 * milliSeconds);

								long currentMemoryUsage = IJ.currentMemory();

								if ( currentMemoryUsage > maximumMemoryUsage )
									maximumMemoryUsage = currentMemoryUsage;


								String memoryUsage = "; memory (curr/max/tot) [MB]: "
										+ currentMemoryUsage / 1000000L
										+ "/" + maximumMemoryUsage / 1000000L
										+ "/" + totalMemory / 1000000L;

								logger.progress("Classified ", "" + (regionsClassified)
												+ "/" + nTotal + " at "
												+ Arrays.toString( positions.get(regionsClassified-1) )
												+ "; " + (pixels / 1000) + " kv in "
												+ (int) (milliSeconds / 1000) + " s; " +
												"rate [kv/s]: " + (int) (rate) +
												memoryUsage +
												"; threads: " + nThreadsNew +
												", " + wekaSegmentation.numRegionThreads +
												", " + wekaSegmentation.numThreadsPerRegion
								);

								nThreadsLast = wekaSegmentation.totalThreadsExecuted.get();
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
					applyButton.setText("Apply classifier");

					// show results
					if (showColorOverlay)
						win.toggleOverlay();
					win.toggleOverlay();
				}
			};
			thread.start();
		}
	}

	public void postProcessSelectedRegion( String command, String zRange, String sizeRange )
	{

		if ( classifiedImage == null )
		{
			logger.error("classification_result image not set!");
			return;
		}

		if ( command.equals("STOP") )
		{
			logger.info("Stopping post-processing thread...");
			wekaSegmentation.stopCurrentThreads = true;
			postProcessButton.setText("Post process");
			return;
		}
		else
		{
			Roi roi = displayImage.getRoi();
			if (roi == null || !(roi.getType() == Roi.RECTANGLE))
			{
				IJ.showMessage("Please use ImageJ's rectangle selection tool" +
						" in order to select a region.");
				return;
			}

			postProcessButton.setText("STOP");
			wekaSegmentation.stopCurrentThreads = false;

			Thread thread = new Thread() {
				public void run()
				{

					int[] sizesMinMax = bigDataTools.utils.Utils.delimitedStringToIntegerArray(sizeRange,",");

					int zs, ze, zc;
					if ( zRange.equals("None") )
					{
						zc = displayImage.getZ();
						zs = zc - wekaSegmentation.getFeatureVoxelSizeAtMaximumScale();
						ze = zc + wekaSegmentation.getFeatureVoxelSizeAtMaximumScale();
					}
					else
					{
						int[] tmp = bigDataTools.utils.Utils.delimitedStringToIntegerArray(zRange,",");
						zs = tmp[0]; ze = tmp[1];
						zc = ( ze + zs ) / 2;
					}

					Rectangle rectangle = roi.getBounds();
					Region5D region5D = new Region5D();
					region5D.t = displayImage.getT() - 1;
					region5D.c = displayImage.getC() - 1;
					region5D.size = new Point3D( rectangle.width, rectangle.height, ze-zs);
					region5D.subSampling = new Point3D(1, 1, 1);
					region5D.offset = new Point3D(rectangle.x,  rectangle.y, zs);

					wekaSegmentation.postProcess(region5D, sizesMinMax).run();

					if (showColorOverlay)
						win.toggleOverlay();
					win.toggleOverlay();

					// we're done
					postProcessButton.setText("Post process");
				}
			};
			thread.start();
		}
	}

	/**
	 * Load a Weka classifier from a file
	 */
	public void loadClassifier()
	{
		OpenDialog od = new OpenDialog( "Choose Weka classifier file", "" );
		if (od.getFileName()==null)
			return;
		IJ.log("Loading Weka classifier from " + od.getDirectory() + od.getFileName() + "...");
		// Record
		String[] arg = new String[] { od.getDirectory() + od.getFileName() };
		record(LOAD_CLASSIFIER, arg);

		win.setButtonsEnabled(false);

		final AbstractClassifier oldClassifier = wekaSegmentation.getClassifier();


		// Try to load Weka model (classifier and train header)
		if(  !wekaSegmentation.loadClassifier(od.getDirectory() + od.getFileName()) )
		{
			IJ.error("Error when loading Weka classifier from file");
			IJ.log("Error: classifier could not be loaded.");
			win.updateButtonsEnabling();
			return;
		}

		IJ.log("Read header from " + od.getDirectory() + od.getFileName() + " (number of attributes = " +
				wekaSegmentation.getTrainHeader().numAttributes() + ")");

		if(wekaSegmentation.getTrainHeader().numAttributes() < 1)
		{
			IJ.error("Error", "No attributes were found on the model header");
			wekaSegmentation.setClassifier(oldClassifier);
			win.updateButtonsEnabling();
			return;
		}

		// Set the flag of training complete to true
		win.trainingComplete = true;

		// update GUI
		win.updateAddClassButtons();

		IJ.log("Loaded " + od.getDirectory() + od.getFileName());
	}

	/**
	 * Load a Weka model (classifier) from a file
	 * @param filename complete path and file name
	 * @return classifier
	 */
	public static AbstractClassifier readClassifier(String filename)
	{
		AbstractClassifier cls = null;
		// deserialize model
		try {
			cls = (AbstractClassifier) SerializationHelper.read(filename);
		} catch (Exception e) {
			IJ.log("Error when loading classifier from " + filename);
			e.printStackTrace();
		}
		return cls;
	}

	/**
	 * Save current classifier into a file
	 */
	public void saveClassifier()
	{
		SaveDialog sd = new SaveDialog("Save model as...", "classifier",".model");
		if (sd.getFileName()==null)
			return;

		// Record
		String[] arg = new String[] { sd.getDirectory() + sd.getFileName() };
		record(SAVE_CLASSIFIER, arg);

		if( !wekaSegmentation.saveClassifier(sd.getDirectory() + sd.getFileName()) )
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
	public static boolean saveClassifier(
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
	public void loadAnnotations(String directory, String fileName)
	{

		if ( directory == null || fileName == null )
		{
			OpenDialog od = new OpenDialog("Choose data file", OpenDialog.getLastDirectory(), "labels.ser");
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
			FileInputStream fin = new FileInputStream( directory + fileName );
			ObjectInputStream ois = new ObjectInputStream(fin);
			Examples examples = (Examples) ois.readObject();

			wekaSegmentation.setExamples( examples.exampleList );
			wekaSegmentation.featureList = examples.featureList;
			wekaSegmentation.maxResolutionLevel = examples.maxResolutionLevel;
			wekaSegmentation.enabledFeatures = examples.enabledFeatures;
			wekaSegmentation.anisotropy = examples.anisotropy;
			wekaSegmentation.downSamplingFactor = examples.downSamplingFactor;
			wekaSegmentation.maxDeepConvolutionLevel = examples.maxDeepConvolutionLevel;

			wekaSegmentation.setNumClasses(0);
			int numClassesInExamples = wekaSegmentation.getNumClassesInExamples();
			for ( int c = 0; c < numClassesInExamples; c++ )
			{
				if ( c == numOfClasses )
					win.addClass();
				wekaSegmentation.addClass();
				changeClassName(c, examples.classNames[c]);
			}
			repaintWindow();
			win.updateExampleLists();

		}
		catch (IOException e)
		{
			IJ.showMessage(e.toString());
		}
		catch (ClassNotFoundException e)
		{
			e.printStackTrace();
		}

		win.updateButtonsEnabling();
	}

	/**
	 * Save annotations into a file
	 */
	public void saveAnnotations()
	{
		SaveDialog sd = new SaveDialog("Choose save file", "annotations",".ser");
		if ( sd.getFileName()==null )
			return;

		// Macro recording
		String[] arg = new String[] { sd.getDirectory() + sd.getFileName() };
		record(SAVE_ANNOTATIONS, arg);

		win.setButtonsEnabled(false);

		try
		{
			FileOutputStream fout = new FileOutputStream(sd.getDirectory() + sd.getFileName());
			ObjectOutputStream oos = new ObjectOutputStream(fout);
			Examples examples = new Examples();
			examples.exampleList = wekaSegmentation.getExamples();
			examples.featureList = wekaSegmentation.featureList;
			examples.classNames = wekaSegmentation.getClassNames();
			examples.maxResolutionLevel = wekaSegmentation.maxResolutionLevel;
			examples.enabledFeatures = wekaSegmentation.enabledFeatures;
			examples.anisotropy = wekaSegmentation.anisotropy;
			examples.downSamplingFactor = wekaSegmentation.downSamplingFactor;
			examples.maxDeepConvolutionLevel = wekaSegmentation.maxDeepConvolutionLevel;
			oos.writeObject( examples );
		}
		catch (Exception e)
		{
			IJ.showMessage(e.toString());
		}


		//
		//if( ! saveAnnotations(sd.getDirectory() + sd.getFileName()) )
		//	IJ.showMessage("Saving failed");
		//

		win.updateButtonsEnabling();
	}

	/**
	 * Add new class in the panel (up to MAX_NUM_CLASSES)
	 */
	private void addNewClass()
	{
		if(wekaSegmentation.getNumClasses() == WekaSegmentation.MAX_NUM_CLASSES)
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
		wekaSegmentation.setClassLabel(wekaSegmentation.getNumClasses(), inputName);
		wekaSegmentation.addClass();

		// Add new class label and list
		win.addClass();

		repaintWindow();

		// Macro recording
		String[] arg = new String[] { inputName };
		record(CREATE_CLASS, arg);
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

		gd.addNumericField("Downsampling factor",
				wekaSegmentation.downSamplingFactor, 0);
		gd.addNumericField("Maximum downsampling level",
				wekaSegmentation.maxResolutionLevel, 0);
		gd.addNumericField("Maximal convolution depth",
				wekaSegmentation.maxDeepConvolutionLevel, 0);
		gd.addNumericField("z/xy anisotropy ",
				wekaSegmentation.anisotropy, 0);


		/*
		if(wekaSegmentation.getLoadedTrainingData() != null)
		{
			for(int i = 0; i < 4; i++)
				((TextField) gd.getNumericFields().get( i )).setEnabled(false);
		}
		*/

		gd.addNumericField("RF: Number of trees",
				wekaSegmentation.getNumTrees(), 0);
		gd.addNumericField("RF: Fraction of random features per node",
				wekaSegmentation.fractionRandomFeatures, 2);
		gd.addNumericField("RF: Minimum feature usage factor",
				wekaSegmentation.minFeatureUsageFactor, 1);

		//gd.addNumericField("RF: Batch size per tree in percent", wekaSegmentation.getBatchSizePercent(), 0);
		//gd.addNumericField("RF: Maximum tree depth [0 = None]", wekaSegmentation.maxDepth, 0);

		gd.addStringField("Channels to consider (one-based) [ID,ID,..]",
				wekaSegmentation.getActiveChannelsAsString() );

		String featuresToShow = "None";
		if ( wekaSegmentation.getFeaturesToShow() != null )
			featuresToShow = wekaSegmentation.getFeaturesToShowAsString();

		gd.addStringField("Show features [ID,ID,..]", featuresToShow );
		gd.addStringField("Region tile size", wekaSegmentation.getTileSizeSetting());
		//gd.addNumericField("Number of region threads", wekaSegmentation.numRegionThreads, 0);
		//gd.addNumericField("Number of threads inside a region", wekaSegmentation.numThreadsPerRegion, 0);
		//gd.addNumericField("Number of RF training threads", wekaSegmentation.numRfTrainingThreads, 0);
		//gd.addNumericField("Tiling delay [ms]", wekaSegmentation.tilingDelay, 0);
		gd.addNumericField("Background threshold [gray values]", wekaSegmentation.backgroundThreshold, 0);
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
			gd.addStringField("Class "+(i+1), wekaSegmentation.getClassLabel(i), 15);

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

		if (gd.wasCanceled())
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
				final String featureName = availableFeatures[ i ];
				// Macro recording
				record(SET_FEATURE, new String[]{ featureName + "=" + newEnableFeatures[ i ] });
			}
		}

		if(featuresChanged)
		{
			wekaSegmentation.setEnabledFeatures( newEnableFeatures );
		}
		*/

		wekaSegmentation.downSamplingFactor = (int) gd.getNextNumber();
		wekaSegmentation.maxResolutionLevel = (int) gd.getNextNumber();
		wekaSegmentation.maxDeepConvolutionLevel = (int) gd.getNextNumber();

		wekaSegmentation.anisotropy = (int) gd.getNextNumber();

		// Set classifier and options
		wekaSegmentation.setNumTrees((int) gd.getNextNumber());
		wekaSegmentation.fractionRandomFeatures = (double) gd.getNextNumber();
		wekaSegmentation.minFeatureUsageFactor = (double) gd.getNextNumber();

		//wekaSegmentation.setBatchSizePercent((int) gd.getNextNumber());
		//wekaSegmentation.maxDepth = (int) gd.getNextNumber();

		wekaSegmentation.setActiveChannelsFromString(gd.getNextString());

		wekaSegmentation.setFeaturesToShowFromString(gd.getNextString());
		//wekaSegmentation.numRegionThreads = (int) gd.getNextNumber();
		//wekaSegmentation.numThreadsPerRegion = (int) gd.getNextNumber();
		//wekaSegmentation.numRfTrainingThreads = (int) gd.getNextNumber();

		wekaSegmentation.setMinTileSizesFromString( gd.getNextString() );
		//wekaSegmentation.tilingDelay = (int) gd.getNextNumber();
		wekaSegmentation.backgroundThreshold = (int) gd.getNextNumber();
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
			if(!s.equals(wekaSegmentation.getClassLabel(i)))
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


		// wekaSegmentation.setComputeFeatureImportance(gd.getNextBoolean());

		// Update flag to balance number of class instances
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
		// the data set (instances) must be updated.
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
	}// end showSettingsDialog

	// Quite of a hack from Johannes Schindelin:
	// use reflection to insert classifiers, since there is no other method to do that...
	static {
		try {
			IJ.showStatus("Loading Weka properties...");
			IJ.log("Loading Weka properties...");
			Field field = GenericObjectEditor.class.getDeclaredField("EDITOR_PROPERTIES");
			field.setAccessible(true);
			Properties editorProperties = (Properties)field.get(null);
			String key = "weka.classifiers.Classifier";
			String value = editorProperties.getProperty(key);
			value += ",hr.irb.fastRandomForest.FastRandomForest";
			editorProperties.setProperty(key, value);
			//new Exception("insert").printStackTrace();
			//System.err.println("value: " + value);

			// add classifiers from properties (needed after upgrade to WEKA version 3.7.11)
			PluginManager.addFromProperties(editorProperties);
		} catch (Exception e) {
			IJ.error("Could not insert my own cool classifiers!");
		}
	}

	/**
	 * Button listener class to handle the button action from the
	 * settings dialog to save the feature stack
	 */
	static class SaveFeatureStackButtonListener implements ActionListener
	{
		private String title;
		private TextField text;
		private WekaSegmentation wekaSegmentation;

		/**
		 * Construct a listener for the save feature stack button
		 *
		 * @param title save dialog title
		 * @param wekaSegmentation reference to the segmentation backend
		 */
		public SaveFeatureStackButtonListener(
				String title,
				WekaSegmentation wekaSegmentation )
		{
			this.title = title;
			this.wekaSegmentation = wekaSegmentation;
		}

		/**
		 * Method to run when pressing the save feature stack button
		 */
		public void actionPerformed(ActionEvent e)
		{
			/*
			SaveDialog sd = new SaveDialog(title, "feature-stack", ".tif");
			final String dir = sd.getDirectory();
			final String fileWithExt = sd.getFileName();

			if(null == dir || null == fileWithExt)
				return;
			final FeatureImages featureImages
								= wekaSegmentation.getFeatureImages();
			for(int i=0; i< featureImages.getNumFeatures(); i++)
				wekaSegmentation.saveFeatureStack( i+1, dir, fileWithExt );

			// macro recording
			record( SAVE_FEATURE_STACK, new String[]{ dir, fileWithExt } );
			*/
		}
	}

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
	 * Load a new classifier
	 *
	 * @param newClassifierPathName classifier file name with complete path
	 */
	public static void loadClassifier(String newClassifierPathName)
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			final WekaSegmentation wekaSegmentation = win.getWekaSegmentation();

			IJ.log("Loading Weka classifier from " + newClassifierPathName + "...");

			win.setButtonsEnabled(false);

			final AbstractClassifier oldClassifier = wekaSegmentation.getClassifier();

			// Try to load Weka model (classifier and train header)
			if(  !wekaSegmentation.loadClassifier(newClassifierPathName) )
			{
				IJ.error("Error when loading Weka classifier from file");
				win.updateButtonsEnabling();
				return;
			}

			IJ.log("Read header from " + newClassifierPathName + " (number of attributes = " + wekaSegmentation.getTrainHeader().numAttributes() + ")");

			if(wekaSegmentation.getTrainHeader().numAttributes() < 1)
			{
				IJ.error("Error", "No attributes were found on the model header");
				wekaSegmentation.setClassifier(oldClassifier);
				win.updateButtonsEnabling();
				return;
			}

			// Set the flag of training complete to true
			win.trainingComplete = true;

			// update GUI
			win.updateAddClassButtons();

			IJ.log("Loaded " + newClassifierPathName);
		}
	}

	/**
	 * Save current classifier into a file
	 *
	 * @param classifierPathName complete path name for the classifier file
	 */
	public static void saveClassifier( String classifierPathName )
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			final WekaSegmentation wekaSegmentation = win.getWekaSegmentation();
			if( !wekaSegmentation.saveClassifier( classifierPathName ) )
			{
				IJ.error("Error while writing classifier into a file");
				return;
			}
		}
	}


	/**
	 * Create a new class
	 *
	 * @param inputName new class name
	 */
	public static void createNewClass( String inputName )
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			final WekaSegmentation wekaSegmentation = win.getWekaSegmentation();

			if (null == inputName || 0 == inputName.length())
			{
				IJ.error("Invalid name for class");
				return;
			}
			inputName = inputName.trim();

			if (0 == inputName.toLowerCase().indexOf("add to "))
				inputName = inputName.substring(7);

			// Add new name to the list of labels
			wekaSegmentation.setClassLabel(wekaSegmentation.getNumClasses(), inputName);
			wekaSegmentation.addClass();

			// Add new class label and list
			win.addClass();
			win.updateAddClassButtons();
		}
	}


	/**
	 * Set the class homogenization flag for training
	 *
	 * @param flagStr true/false if you want to balance the number of samples per class before training
	 */
	public static void setClassHomogenization(String flagStr)
	{
		setClassBalance( flagStr );
	}

	/**
	 * Set the class balance flag for training
	 *
	 * @param flagStr true/false if you want to balance the number of samples per class before training
	 */
	public static void setClassBalance( String flagStr )
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			boolean flag = Boolean.parseBoolean(flagStr);
			final WekaSegmentation wekaSegmentation = win.getWekaSegmentation();
			wekaSegmentation.setClassBalance( flag );
		}
	}

	/**
	 * Set classifier for current segmentation
	 *
	 * @param classifierName classifier name with complete package information
	 * @param options classifier options
	 */
	public static void setClassifier(String classifierName, String options)
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			final WekaSegmentation wekaSegmentation = win.getWekaSegmentation();

			try {
				AbstractClassifier cls = (AbstractClassifier)( Class.forName(classifierName).newInstance() );
				cls.setOptions( options.split(" "));
				wekaSegmentation.setClassifier(cls);
			} catch (Exception e) {
				e.printStackTrace();
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
	 * Enable/disable a single feature of the feature stack(s)
	 *
	 * @param feature name of the feature + "=" true/false to enable/disable
	 */
	public static void setFeature(String feature)
	{
		final ImageWindow iw = WindowManager.getCurrentImage().getWindow();
		if( iw instanceof CustomWindow )
		{
			final CustomWindow win = (CustomWindow) iw;
			final WekaSegmentation wekaSegmentation = win.getWekaSegmentation();
			final boolean isProcessing3D = wekaSegmentation.isProcessing3D();

			int index = feature.indexOf("=");
			String featureName = feature.substring(0, index);
			boolean featureValue = feature.contains("true");

			boolean[] enabledFeatures = wekaSegmentation.getEnabledFeatures();
			boolean forceUpdate = false;
			for(int i=0; i<enabledFeatures.length; i++)
			{
				final String availableFeature = new FeatureImagesMultiResolution().availableFeatures[i];

                if ( availableFeature.contains(featureName) && featureValue != enabledFeatures[i])
                {
                    enabledFeatures[i] = featureValue;
                    forceUpdate = true;
                }
			}
			wekaSegmentation.setEnabledFeatures(enabledFeatures);

			if(forceUpdate)
			{
				// Force features to be updated
				wekaSegmentation.setFeaturesDirty();
			}
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
		catch (final NoClassDefFoundError err) {
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

