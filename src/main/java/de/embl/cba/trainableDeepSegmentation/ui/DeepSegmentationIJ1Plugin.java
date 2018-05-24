package de.embl.cba.trainableDeepSegmentation.ui;

import de.embl.cba.trainableDeepSegmentation.classification.ClassificationRangeUtils;
import de.embl.cba.trainableDeepSegmentation.features.DownSampler;
import de.embl.cba.trainableDeepSegmentation.settings.FeatureSettings;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import de.embl.cba.utils.logging.Logger;


import fiji.util.gui.GenericDialogPlus;
import fiji.util.gui.OverlayedImageCanvas;
import ij.*;
import ij.gui.*;
import ij.plugin.PlugIn;
import ij.plugin.frame.Recorder;
import ij.plugin.frame.RoiManager;
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
import java.util.*;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.swing.*;

import net.imglib2.FinalInterval;
import de.embl.cba.trainableDeepSegmentation.*;
import de.embl.cba.trainableDeepSegmentation.labels.examples.Example;
import de.embl.cba.trainableDeepSegmentation.features.ImageScience;
import de.embl.cba.trainableDeepSegmentation.instances.InstancesAndMetadata;
import de.embl.cba.trainableDeepSegmentation.labels.LabelManager;
import de.embl.cba.trainableDeepSegmentation.results.ResultImage;
import de.embl.cba.trainableDeepSegmentation.results.ResultImageDisk;
import de.embl.cba.trainableDeepSegmentation.results.ResultImageExportGUI;
import de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils;
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

// TODO: make IJ2 plugin?!


public class DeepSegmentationIJ1Plugin implements PlugIn, RoiListener
{

	/** plugin's name */
	public static final String PLUGIN_NAME = "Trainable Deep Segmentation";
	/** plugin's current version */
	public static final String PLUGIN_VERSION = "v" + de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin.class.getPackage().getImplementationVersion();

	/** reference to the segmentation backend */
	private DeepSegmentation deepSegmentation = null;

	/** image to be used in the instances */
	public ImagePlus inputImage = null;
	/** resultImagePlus image after classification */
	private CustomWindow win = null;
	/** number of classes in the GUI */
	private int numOfClasses = 2;
	/** array of number of traces per class */
	private int[] traceCounter = new int[ DeepSegmentation.MAX_NUM_CLASSES];
	/** flag to display the overlay image */
	private boolean showColorOverlay = false;
	/** executor service to launch numWorkers for the plugin methods and events */
	private final ExecutorService exec = Executors.newFixedThreadPool(1);

	/** train classifier button */
	private JButton trainClassifierButton = null;
	private JButton updateTrainingDataButton = null;

	private JButton stopButton = new JButton( "   STOP   " );


	private int X = 0, Y = 1, C = 2, Z = 3, T = 4;
	private int[] XYZ = new int[]{ X, Y, Z};

	/** toggle overlay button */
	private JButton overlayButton = null;

	private JButton assignResultImageButton = null;

	private JComboBox resultImageComboBox = null;

	private JButton reviewLabelsButton = null;

    private JComboBox reviewLabelsClassComboBox = null;

	/** apply classifier button */
	private JButton applyClassifierButton = null;

	private JComboBox rangeComboBox = null;

	/** load annotations button */
	public static final String IO_LOAD_CLASSIFIER = "Load classifier";
	public static final String IO_SAVE_CLASSIFIER = "Save classifier";
	public static final String APPLY_CLASSIFIER = "Apply classifier";
	public static final String ADD_CLASS = "Add class";
	public static final String CHANGE_CLASS_NAMES = "Change class names";
	public static final String CHANGE_COLORS = "Change classColors";
	public static final String CHANGE_RESULT_OVERLAY_OPACITY = "Overlay opacity";
	public static final String UPDATE_LABELS_AND_TRAIN = "Update labels and train classifier";
    public static final String UPDATE_LABELS = "Update labels";
    public static final String TRAIN_CLASSIFIER = "Train classifier";
	public static final String IO_LOAD_LABEL_IMAGE = "Load label image";
	public static final String IO_LOAD_INSTANCES = "Load instances";
	public static final String IO_SAVE_INSTANCES = "Save instances of current image";
	public static final String IO_EXPORT_RESULT_IMAGE = "Export results";
	public static final String TRAIN_FROM_LABEL_IMAGE = "Train from label image";
	public static final String APPLY_CLASSIFIER_ON_SLURM = "Apply classifier on cluster";
	public static final String APPLY_BG_FG_CLASSIFIER = "Apply BgFg classifier (development)";
	public static final String DUPLICATE_RESULT_IMAGE_TO_RAM = "Show result image";
	public static final String GET_LABEL_IMAGE_TRAINING_ACCURACIES = "Label image training accuracies";
	public static final String CHANGE_CLASSIFIER_SETTINGS = "Change classifier settings";
	public static final String CHANGE_FEATURE_COMPUTATION_SETTINGS = "Change feature settings";
    public static final String CHANGE_ADVANCED_FEATURE_COMPUTATION_SETTINGS = "Change advanced feature settings";
    public static final String SEGMENT_OBJECTS = "Segment objects";
    public static final String REVIEW_OBJECTS = "Review objects";
    public static final String RECOMPUTE_LABEL_FEATURE_VALUES = "Recompute all feature values";
    public static final String CHANGE_DEBUG_SETTINGS = "Change development settings";

	public static final String NO_TRAINING_DATA = "No training data available";

	// TODO: how to know the featureSettings associated with instances data
	// ??
	private JButton doButton = null;

    // TODO: make this work:

	private JComboBox actionComboBox = new JComboBox(
			new String[] {
					ADD_CLASS,
					CHANGE_CLASS_NAMES,
					CHANGE_COLORS,
					UPDATE_LABELS_AND_TRAIN,
                    IO_SAVE_INSTANCES,
                    APPLY_CLASSIFIER,
					IO_SAVE_CLASSIFIER,
					APPLY_CLASSIFIER_ON_SLURM,
					TRAIN_CLASSIFIER,
                    IO_LOAD_INSTANCES,
                    SEGMENT_OBJECTS,
                    REVIEW_OBJECTS,
                    DUPLICATE_RESULT_IMAGE_TO_RAM,
                    IO_EXPORT_RESULT_IMAGE,
                    CHANGE_FEATURE_COMPUTATION_SETTINGS,
                    CHANGE_RESULT_OVERLAY_OPACITY,
					CHANGE_CLASSIFIER_SETTINGS,
					UPDATE_LABELS,
					IO_LOAD_LABEL_IMAGE,
					TRAIN_FROM_LABEL_IMAGE,
					GET_LABEL_IMAGE_TRAINING_ACCURACIES,
					IO_LOAD_CLASSIFIER,
                    RECOMPUTE_LABEL_FEATURE_VALUES,
                    CHANGE_DEBUG_SETTINGS,
                    CHANGE_ADVANCED_FEATURE_COMPUTATION_SETTINGS
			} );


	public static final String EM_IMAGING = "Electron microscopy";
	public static final String FLUORESCENCE_IMAGING = "Fluorescence microscopy";
	public static final String TRANSMISSION_LM_IMAGING = "Transmission microscopy";

	private JComboBox imagingModalityComboBox = new JComboBox(
			new String[] {
					FLUORESCENCE_IMAGING,
					EM_IMAGING,
					TRANSMISSION_LM_IMAGING
			} );


	private boolean computeLabelFeatureValuesInstantly = false;

	/** featureSettings button */
	private JButton settingsButton = null;

	private JButton testThreadsButton = null;

	private JTextField uncertaintyTextField = new JTextField();

	private JList instancesList = null;

	/** create new class button */
	private JButton addClassButton = null;

	/** array of roi list overlays to paint the transparent rois of each class */
	private RoiListOverlay[] roiOverlay = null;

	/** available classColors for available classes */
	private Color[] colors = new Color[]{
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

	/** Lookup table for the resultImagePlus overlay image */
	private LUT overlayLUT = null;

	/** array of trace lists for every class */
	private java.awt.List[] exampleList = null;
	/** array of buttons for adding each trace class */
	private JButton [] annotationButton = null;

	// Macro recording constants (corresponding to
	// static method names to be called)
	/** name of the macro method to add the current trace to a class */
	public static final String ADD_TRACE = "addTrace";
	/** name of the macro method to delete the current trace */
	public static final String DELETE_TRACE = "deleteTrace";
	/** name of the macro method to toggle the overlay image */
	public static final String TOGGLE_OVERLAY = "toggleOverlay";
	/** name of the macro method to getInstancesAndMetadata the binary resultImagePlus */
	public static final String GET_RESULT = "getResult";
	/** name of the macro method to getInstancesAndMetadata the binary resultImagePlus */
	public static final String SET_RESULT = "setResult";
	/** name of the macro method to getInstancesAndMetadata the probability maps */
	public static final String GET_PROBABILITY = "getProbability";
	/** name of the macro method to plot the threshold curves */
	public static final String PLOT_RESULT = "plotResultGraphs";
	/** name of the macro method to apply the current classifier to an image or stack */
	public static final String APPLY_CLASSIFIER_MACRO = "applyClassifier";
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
	/** name of the macro method to set the overlay opacity */
	public static final String SET_OPACITY = "setOpacity";
	/** boolean flag set to true while instances */

	// TODO: make an "I am busy flag"

    static final String REVIEW_LABELS_START = "Review labels";
    static final String REVIEW_OBJECTS_START = "Review objects3DPopulation";
    static final String REVIEW_END = "Stop reviewing!";

	public static final String TRAINING_DATA_TRACES = "Traces";
	public static final String TRAINING_DATA_LABEL_IMAGE = "Label image";

	public static final String SELECTION = "Selected roi";

    private LabelManager labelManager = null;

	private boolean isFirstTime = true;

	public boolean reviewRoisFlag = false;

	private Logger logger;

	/**
	 * Basic constructor for graphical user interface use
	 */
	public DeepSegmentationIJ1Plugin()
	{
		// check for image science
		// TODO: does this work??
		if ( ! isImageScienceAvailable() )
		{
			IJ.showMessage("Please install ImageScience: [Help > Update... > Manage Update Sites]: [X] ImageScience ");
			return;
		}

        DeepSegmentation.reserveKeyboardShortcuts();


        // create overlay LUT
		final byte[] red = new byte[ 256 ];
		final byte[] green = new byte[ 256 ];
		final byte[] blue = new byte[ 256 ];

		// assign classColors to classes
		for( int iClass = 0; iClass < DeepSegmentation.MAX_NUM_CLASSES; iClass++)
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

		exampleList = new java.awt.List[ DeepSegmentation.MAX_NUM_CLASSES];
		annotationButton = new JButton[ DeepSegmentation.MAX_NUM_CLASSES ];

		roiOverlay = new RoiListOverlay[ DeepSegmentation.MAX_NUM_CLASSES];

		updateTrainingDataButton = new JButton("Update label instances");

		trainClassifierButton = new JButton("Train classifier");

		overlayButton = new JButton("Toggle overlay [r][p][u]");
		overlayButton.setToolTipText("Toggle between current segmentation and original image");
		overlayButton.setEnabled(false);

		assignResultImageButton = new JButton("Assign resultImagePlus image");
		assignResultImageButton.setToolTipText("Assign resultImagePlus image");
		assignResultImageButton.setEnabled(true);

		resultImageComboBox = new JComboBox( new String[]{ DeepSegmentation.RESULT_IMAGE_RAM,
				DeepSegmentation.RESULT_IMAGE_DISK_SINGLE_TIFF } );

		reviewLabelsClassComboBox = new JComboBox( new String[]{ "1" , "2"} );

		reviewLabelsButton = new JButton( REVIEW_LABELS_START );
		reviewLabelsButton.setEnabled( true );

        applyClassifierButton = new JButton ("Apply classifier");
		applyClassifierButton.setToolTipText("Apply classifier");
		applyClassifierButton.setEnabled(false);


		rangeComboBox = new JComboBox( new String[]
				{ SELECTION,
						ClassificationRangeUtils.SELECTION_PM10Z,
						ClassificationRangeUtils.WHOLE_DATA_SET,
						"1,10,1,2"
				});

		rangeComboBox.setEditable( true );

		stopButton.setEnabled( false );

		doButton = new JButton ("Perform action");
		doButton.setEnabled(true);

		addClassButton = new JButton ("Create new class");
		addClassButton.setToolTipText("Add one more label to mark different areas");

		settingsButton = new JButton ("FeatureSettings");
		settingsButton.setToolTipText("Display featureSettings dialog");

		testThreadsButton = new JButton ("Thread test");
		testThreadsButton.setToolTipText("Tests how many numWorkers this PC will concurrently handle.");

		/** The Weka icon image */
		//ImageIcon icon = new ImageIcon( de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin.class.getResource("/trainableDeepSegmentation/images/weka.png"));
		//wekaButton = new JButton( icon );
		//wekaButton.setToolTipText("Launch Weka GUI chooser");

		showColorOverlay = false;

		Roi.addRoiListener( this );

		IJ.setTool("freeline");
	}

    /** Thread that runs the instances. We store it to be able to
	 * to interrupt it from the GUI */
	private Thread trainingTask = null;

	private Boolean stopClassificationThread = false;


	/**
	 * Button listener
	 */
	private ActionListener listener = new ActionListener() {

		public void actionPerformed(final ActionEvent e) {

			final String command = e.getActionCommand();

			// listen to the buttons on separate numWorkers not to block
			// the event dispatch thread
			new Thread( new Runnable(){
			//exec.submit(new Runnable() {

				public void run()
				{
					if( e.getSource() == overlayButton){
						// Macro recording
						String[] arg = new String[] {};
						record(TOGGLE_OVERLAY, arg);
						win.toggleOverlay();
					}
					else if( e.getSource() == stopButton )
					{
						stopCurrentWekaSegmentationTasks();
					}
					else if( e.getSource() == assignResultImageButton )
					{
						deepSegmentation.assignResultImage( (String) resultImageComboBox.getSelectedItem() );

						if ( deepSegmentation.hasResultImage() )
						{
							applyClassifierButton.setEnabled( true );
						}
					}
					else if(e.getSource() == reviewLabelsButton )
					{
						if ( reviewLabelsButton.getText().equals( REVIEW_LABELS_START ) )
						{
							win.setButtonsEnabled( false );
							reviewLabelsButton.setEnabled( true );
							reviewLabelsButton.setText( REVIEW_END );
							reviewRoisFlag = true;

							reviewLabels( reviewLabelsClassComboBox.getSelectedIndex() );
						}
						else
						{
							inputImage.setOverlay( new Overlay() );
							//IJ.run("Remove Overlay");

							labelManager.updateExamples();
							ArrayList< Example > approvedExamples = labelManager.getExamples();
							deepSegmentation.setExamples( approvedExamples );
							labelManager.close();
							reviewLabelsButton.setText( REVIEW_LABELS_START );
							reviewRoisFlag = false;

							// imageAroundCurrentSelection.close();

							win.setButtonsEnabled( true );
						}
					}
					else if( e.getSource() == applyClassifierButton )
					{
						applyClassifier();
					}
					else if( e.getSource() == doButton || e.getSource() == actionComboBox )
					{
						String action = (String) actionComboBox.getSelectedItem();

						switch ( action )
						{
							case SEGMENT_OBJECTS:
                                deepSegmentation.segmentObjects();
								break;
                            case REVIEW_OBJECTS:
                                deepSegmentation.reviewObjects();
                                break;
							case CHANGE_CLASSIFIER_SETTINGS:
								deepSegmentation.showClassifierSettingsDialog();
								break;
							case CHANGE_FEATURE_COMPUTATION_SETTINGS:
								showFeatureSettingsDialogIJ1Plugin( false );
								break;
                            case CHANGE_ADVANCED_FEATURE_COMPUTATION_SETTINGS:
                                showFeatureSettingsDialogIJ1Plugin( true );
                                break;
                            case IO_LOAD_CLASSIFIER:
								loadClassifierDialog();
								break;
							case ADD_CLASS:
								addNewClass();
								break;
							case CHANGE_CLASS_NAMES:
								showClassNamesDialog();
								break;
							case CHANGE_RESULT_OVERLAY_OPACITY:
								showResultsOverlayOpacityDialog();
								break;
							case CHANGE_COLORS:
								showColorsDialog();
								break;
							case IO_SAVE_CLASSIFIER:
								saveClassifier();
								break;
							case IO_LOAD_INSTANCES:
								loadInstances();
								break;
							case IO_SAVE_INSTANCES:
								saveInstances( inputImage.getTitle() );
								break;
							case IO_LOAD_LABEL_IMAGE:
								loadLabelImage();
								break;
							case RECOMPUTE_LABEL_FEATURE_VALUES:
								recomputeLabelFeaturesAndRetrainClassifier();
								break;
                            case CHANGE_DEBUG_SETTINGS:
                                showDebugSettingsDialog();
                                break;
							case IO_EXPORT_RESULT_IMAGE:
								ResultImageExportGUI.showExportGUI(
										getDeepSegmentation().getInputImage().getTitle(),
										deepSegmentation.getResultImage(),
										deepSegmentation.getInputImage(),
										deepSegmentation.getClassNames() );
								break;
							case APPLY_CLASSIFIER:
								applyClassifier();
								break;
							case APPLY_CLASSIFIER_ON_SLURM:
								applyClassifierOnSlurm();
								break;
							case APPLY_BG_FG_CLASSIFIER:
								applyBgFgClassification();
								break;
							case TRAIN_FROM_LABEL_IMAGE:
								trainFromLabelImage();
								break;
							case GET_LABEL_IMAGE_TRAINING_ACCURACIES:
								computeLabelImageBasedAccuracies();
								break;
                            case UPDATE_LABELS:
                                updateLabelsTrainingData();
                                break;
							case UPDATE_LABELS_AND_TRAIN:
								updateLabelsTrainingDataAndTrainClassifier();
								break;
							case DUPLICATE_RESULT_IMAGE_TO_RAM:
								ImagePlus imp = deepSegmentation.getResultImage().getWholeImageCopy();
								if ( imp != null ) imp.show();
								break;
							case TRAIN_CLASSIFIER:
                                // getInstancesAndMetadata instances instances
                                InstancesAndMetadata instancesAndMetadata = getCombinedSelectedInstancesFromGUI();
                                if ( instancesAndMetadata == null )
                                {
                                    logger.error( "Please select one or multiple training instances." );
                                    return;
                                }
								trainClassifier( instancesAndMetadata );
								break;
						}
					}
					else if(e.getSource() == imagingModalityComboBox )
					{
						String imagingModality = (String) imagingModalityComboBox.getSelectedItem();

						if ( imagingModality.equals( FLUORESCENCE_IMAGING ) )
						{
							deepSegmentation.featureSettings.log2 = true;
						}
						else
						{
							deepSegmentation.featureSettings.log2 = false;
						}
					}
					else if(e.getSource() == addClassButton){
						addNewClass();
					}
					else if(e.getSource() == testThreadsButton){
						testThreads();
					}
					else{
						for( int i = 0; i < deepSegmentation.getNumClasses(); i++)
						{
							if(e.getSource() == exampleList[i])
							{
								deleteSelected(e);
								break;
							}
							if(e.getSource() == annotationButton[i])
							{
								addAnnotation( i );
								break;
							}
						}
					}

				}


			}).start();
		}
	};

    public static void configureRoiManagerClosingEventListener( RoiManager manager,
                                                                DeepSegmentationIJ1Plugin deepSegmentationIJ1Plugin)
    {
        manager.addWindowListener( new WindowAdapter()
        {
            @Override
            public void windowClosing( WindowEvent we )
            {
                // IJ.log( "RoiManager closed.");
                deepSegmentationIJ1Plugin.reviewRoisFlag = false;
                deepSegmentationIJ1Plugin.inputImage.killRoi();
                deepSegmentationIJ1Plugin.inputImage.setOverlay( new Overlay(  ) );

            }
        });
    }


    private void stopCurrentWekaSegmentationTasks()
	{

		deepSegmentation.stopCurrentTasks = true;

		String dotDotDot = "...";

		while ( deepSegmentation.isBusy )
		{

			logger.progress( "Waiting for tasks to finish", dotDotDot );
			dotDotDot += ".";
			try
			{
				Thread.sleep( 3000 );
			} catch ( InterruptedException e )
			{
				e.printStackTrace();
			}
		}

		logger.info("...all tasks finished.");

		deepSegmentation.stopCurrentTasks = false;
		win.setButtonsEnabled( true );

	}


    public void makeTrainingImageTheActiveWindow()
    {
        sleep(); // otherwise below select window does not always work...

        IJ.selectWindow( inputImage.getID() );

        if ( ! inputImage.getWindow().isActive() )
        {
            sleep(); // otherwise below select window does not always work...
            IJ.selectWindow( inputImage.getID() );
        }
    }

	private void loadLabelImage()
	{

		ImagePlus labelImage = IJ.openImage();;

		if ( labelImage != null )
		{
			deepSegmentation.setLabelImage( labelImage );
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
					for( int i = 0; i < deepSegmentation.getNumClasses(); i++)
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
		// remove all drawing elements
		inputImage.killRoi();
		removeAllRoiOverlays();
		win.resultOverlay.setImage( null );
		inputImage.updateAndDraw();

		// show labels
		labelManager = new LabelManager( this );
		labelManager.setExamples( deepSegmentation.getExamples() );
		String order = labelManager.showOrderGUI();
		labelManager.reviewLabelsInRoiManager( classNum, order );
	};


	Roi currentlyDisplayedRoi;

	@Override
	public void roiModified( ImagePlus imagePlus, int actionId )
	{
		if ( ( imagePlus != null ) && ( imagePlus == inputImage ) )
		{
			if ( actionId == RoiListener.CREATED && reviewRoisFlag )
			{
				if ( currentlyDisplayedRoi == null)
				{
					currentlyDisplayedRoi = inputImage.getRoi();
					zoomToSelection();
				}
				else
				{
					int x = inputImage.getRoi().getBounds().x;
					int x2 = currentlyDisplayedRoi.getBounds().x;
					if ( x != x2 )
					{
						currentlyDisplayedRoi = inputImage.getRoi();
						zoomToSelection();
					}
				}

			}
		}
	}


	/**
	 * Custom canvas to deal with zooming an panning
	 */
	private class CustomCanvas extends OverlayedImageCanvas
	{
		/**
		 * default serial version UID
		 */
		private static final long serialVersionUID = 1L;

		CustomCanvas( ImagePlus imp )
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

	public DeepSegmentation getDeepSegmentation()
	{
		return deepSegmentation;
	}

    public void setDeepSegmentation( DeepSegmentation deepSegmentation )
    {
        this.deepSegmentation = deepSegmentation;
    }

    private void testThreads()
	{
		logger.info("Testing maximum number of numWorkers...");
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

	ImagePlus imageAroundCurrentSelection;

	private void zoomToSelection()
	{

		Roi roi = inputImage.getRoi(); if ( roi == null ) return;

		// makeTrainingImageTheActiveWindow();

		IJ.run("To Selection");

        // remove old overlay
		inputImage.setOverlay( new Overlay(  ) );

		// add new overlay
        IJ.run("Add Selection...");

        // remove roi
		inputImage.killRoi();

        updateResultOverlay();

        //int margin = getMargin( roi );

		//setImageAroundRoi( roi, margin );

		//showImageAroundRoi( roi, margin );

		LabelManager.zoomOut( 7 );

		// makeTrainingImageTheActiveWindow();

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




	/**
	 * Custom window to define the Trainable Deep Segmentation GUI
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
		/** opacity (in %) of the resultImagePlus overlay image */
		private int overlayOpacity = 33;
		/** alpha composite for the resultImagePlus overlay image */
		private Composite overlayAlpha = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, overlayOpacity / 100f);
		/** current segmentation resultImagePlus overlay */
		private ImageOverlay resultOverlay;

		/** boolean flag set to true when instances is complete */
		private boolean trainingComplete = false;

		/** boolean flag set to true when instances is complete */
		private boolean classificationComplete = false;



		/**
		 * Construct the plugin window
		 *
		 * @param imp input image
		 */

		CustomWindow( ImagePlus imp )
		{
			super( imp, new CustomCanvas( imp ) );

			final CustomCanvas canvas = ( CustomCanvas ) getCanvas();

			// add roi list overlays (one per class)
			for( int i = 0; i < DeepSegmentation.MAX_NUM_CLASSES; i++)
			{
				roiOverlay[i] = new RoiListOverlay();
				roiOverlay[i].setComposite( transparency050 );
				((OverlayedImageCanvas)ic).addOverlay(roiOverlay[i]);
			}

			addResultOverlay();

			removeAll();

			setTitle( de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin.PLUGIN_NAME + ": " + inputImage.getTitle() );

			setupAnnotationsPanel();

			addButtonListeners();

			addStackListeners( imp );

			addKeyListeners( canvas );


			// Labels panel (includes annotations panel)
			GridBagConstraints labelsConstraints = new GridBagConstraints();
			labelsConstraints.anchor = GridBagConstraints.NORTHWEST;
			labelsConstraints.fill = GridBagConstraints.HORIZONTAL;
			labelsConstraints.gridwidth = 1;
			labelsConstraints.gridheight = 1;
			labelsConstraints.gridx = 0;
			labelsConstraints.gridy = 0;
			labelsJPanel.setLayout( new GridBagLayout() );
			labelsJPanel.add( annotationsPanel, labelsConstraints );

			// Scroll panel for the label panel
			scrollPanel = new JScrollPane( labelsJPanel );
			scrollPanel.setHorizontalScrollBarPolicy( JScrollPane.HORIZONTAL_SCROLLBAR_NEVER );
			scrollPanel.setMinimumSize( labelsJPanel.getPreferredSize() );

			// Training panel (left side of the GUI)
			//trainingJPanel.setBorder(BorderFactory.createEmptyBorder());
			GridBagConstraints trainingConstraints = new GridBagConstraints();
			trainingConstraints.anchor = GridBagConstraints.NORTHWEST;
			trainingConstraints.fill = GridBagConstraints.HORIZONTAL;
			trainingConstraints.gridwidth = 1;
			trainingConstraints.gridheight = 1;
			trainingConstraints.gridx = 0;
			trainingConstraints.gridy = 0;
			trainingConstraints.insets = new Insets(5, 5, 6, 6);
			trainingJPanel.setLayout( new GridBagLayout() );

			/*
			JPanel imagingModality = new JPanel();
			imagingModality.add( new JLabel( "Image type" ), trainingConstraints);
			imagingModality.add( imagingModalityComboBox, trainingConstraints );
			trainingJPanel.add( imagingModality, trainingConstraints );
			trainingConstraints.gridy++;
			*/


			/*
			JPanel backgroundPanel = new JPanel();
			backgroundPanel.add( new JLabel("Image background") );
			backgroundPanel.add( imageBackgroundTextField );
			trainingJPanel.add( backgroundPanel, trainingConstraints );
			trainingConstraints.gridy++;
			*/


			/*
			JPanel assignResultImagePanel = new JPanel();
			assignResultImagePanel.add( assignResultImageButton );
			assignResultImagePanel.add( resultImageComboBox );
			trainingJPanel.add(assignResultImagePanel, trainingConstraints);
			trainingConstraints.gridy++;
			*/


			trainingConstraints.gridy++;

			trainingJPanel.add( doButton, trainingConstraints );
			trainingConstraints.gridy++;

			trainingJPanel.add( actionComboBox, trainingConstraints );
			actionComboBox.addActionListener(  this );
			trainingConstraints.gridy++;

			JPanel instancesPanel = new JPanel();
			DefaultListModel listModel = new DefaultListModel();
			listModel.addElement( "No training data available" );
			instancesList = new JList( listModel );
			instancesList.setVisibleRowCount(4);
			JViewport jv1 = new JViewport();
			jv1.setView( new JLabel("Training instances") );
			JScrollPane listScrollPane = new JScrollPane(instancesList);
			listScrollPane.setPreferredSize( new Dimension( 200,100 ) );
			listScrollPane.setColumnHeader(jv1);
			instancesPanel.add( listScrollPane );
			trainingJPanel.add(instancesPanel, trainingConstraints);
			trainingConstraints.gridy++;

			//trainingJPanel.add( applyClassifierButton, trainingConstraints);
			//trainingConstraints.gridy++;

			JPanel panelZTRange = new JPanel();
			panelZTRange.add( new JLabel("Range") );
			panelZTRange.add( rangeComboBox );
			trainingJPanel.add( panelZTRange, trainingConstraints );
			trainingConstraints.gridy++;

			trainingJPanel.add( overlayButton, trainingConstraints );
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

			//trainingJPanel.add(settingsButton, trainingConstraints);
			//trainingConstraints.gridy++;

			//trainingJPanel.add(addClassButton, trainingConstraints);
			//trainingConstraints.gridy++;

			trainingJPanel.add( stopButton, trainingConstraints );
			trainingConstraints.gridy++;



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
			panelObjectSize.add(imageBackgroundTextField);
			trainingJPanel.add(panelObjectSize, trainingConstraints);
			*/

			/*
			trainingConstraints.gridy++;
			trainingJPanel.add(loadClassifierButton, trainingConstraints);

			trainingConstraints.gridy++;
			trainingJPanel.add(saveClassifierButton, trainingConstraints);
			*/



			//trainingJPanel.add(printProjectInfoButton, trainingConstraints);
			//trainingConstraints.gridy++;

			//trainingJPanel.add(wekaButton, trainingConstraints);
			//trainingConstraints.gridy++;

			//trainingConstraints.gridy++;

			// Buttons panel (including instances and options)
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
			for ( Component p : new Component[]{all, buttonsPanel} ) {
				for ( KeyListener kl : getKeyListeners() ) {
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
					//deepSegmentation.shutDownNow();
					exec.shutdownNow();

					for( int i = 0; i < deepSegmentation.getNumClasses(); i++)
						annotationButton[i].removeActionListener(listener);
					trainClassifierButton.removeActionListener(listener);
					updateTrainingDataButton.removeActionListener( listener );
					overlayButton.removeActionListener(listener);
					//getResultButton.removeActionListener(listener);
					//setResultButton.removeActionListener(listener);
					assignResultImageButton.removeActionListener(listener);
					reviewLabelsButton.removeActionListener( listener );
					applyClassifierButton.removeActionListener(listener);
					doButton.removeActionListener(listener);
					stopButton.removeActionListener( listener );
					addClassButton.removeActionListener(listener);
					settingsButton.removeActionListener(listener);
				}
			});

			canvas.addComponentListener(new ComponentAdapter() {
				public void componentResized(ComponentEvent ce) {
					Rectangle r = canvas.getBounds();
					canvas.setDstDimensions(r.width, r.height);
				}
			});

		}

		private void addKeyListeners( CustomCanvas canvas )
		{
			KeyListener keyListener = new KeyListener() {

				@Override
				public void keyTyped(KeyEvent e) {
					new Thread(new Runnable(){
						public void run()
						{
							if ( e.getKeyChar() == 'r' )
							{
								toggleOverlay(OVERLAY_MODE_SEGMENTATION);
							}

							if ( e.getKeyChar() == 'p' )
							{
								toggleOverlay(OVERLAY_MODE_PROBABILITIES);
							}

							if ( e.getKeyChar() == 'u' )
							{
								toggleOverlay(OVERLAY_MODE_UNCERTAINTY);
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

								if ( i >= deepSegmentation.getNumUncertaintyRegions() )
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
								inputImage.killRoi();
								updateExampleLists();
								drawExamples();
								if( showColorOverlay )
								{
									updateResultOverlay();
									inputImage.updateAndDraw();
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
		}

		private void addStackListeners( ImagePlus imp )
		{
			if( null != zSelector )
			{
				// set slice selector to the correct number
				zSelector.setValue( imp.getSlice() );
				// add adjustment listener to the scroll bar
				zSelector.addAdjustmentListener( new AdjustmentListener() {

					public void adjustmentValueChanged( final AdjustmentEvent e )
					{
						new Thread(new Runnable() {
							//exec.submit(new Runnable() {
							public void run()
							{
								if (e.getSource() == zSelector)
								{
									//IJ.log("moving scroll");
									inputImage.killRoi();
									drawExamples();
									updateExampleLists();
									if (showColorOverlay)
									{
										updateResultOverlay();
										inputImage.updateAndDraw();
									}
								}

							}
						}).start();

					}
				});

				// add special listener if the instances image is a movie
				if( null != tSelector )
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
										inputImage.killRoi();
										drawExamples();
										updateExampleLists();
										if (showColorOverlay)
										{
											updateResultOverlay();
											inputImage.updateAndDraw();
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
								inputImage.killRoi();
								drawExamples();
								updateExampleLists();
								if (showColorOverlay)
								{
									updateResultOverlay();
									inputImage.updateAndDraw();
								}
							}
						});

					}
				});
			}
		}

		private void addResultOverlay()
		{
			resultOverlay = new ImageOverlay();
			resultOverlay.setComposite( overlayAlpha );
			((OverlayedImageCanvas )ic).addOverlay( resultOverlay );
		}

		public JButton getAnnotationButton( int iClass )
		{
			return annotationButton[ iClass ];
		}

		private void setupAnnotationsPanel()
		{
			annotationsConstraints.anchor = GridBagConstraints.NORTHWEST;
			annotationsConstraints.fill = GridBagConstraints.HORIZONTAL;
			annotationsConstraints.gridwidth = 1;
			annotationsConstraints.gridheight = 1;
			annotationsConstraints.gridx = 0;
			annotationsConstraints.gridy = 0;

			annotationsPanel.setBorder( BorderFactory.createTitledBorder("Labels") );
			annotationsPanel.setLayout( boxAnnotation );

			for( int i = 0; i < deepSegmentation.getNumClasses(); i++)
			{
				exampleList[i].addActionListener( listener );
				exampleList[i].addItemListener( itemListener );

				annotationButton[i] = createAnnotationButton( i );

				annotationsConstraints.insets = new Insets(5, 5, 6, 6);

				annotationsPanel.add( annotationButton[i], annotationsConstraints );
				annotationsConstraints.gridy++;

				annotationsConstraints.insets = new Insets(0,0,0,0);

				annotationsPanel.add( exampleList[i], annotationsConstraints );
				annotationsConstraints.gridy++;
			}
		}

		private void addButtonListeners()
		{
			// Add listeners
			for( int i = 0; i < deepSegmentation.getNumClasses(); i++)
			{
				annotationButton[ i ].addActionListener( listener );
			}

			trainClassifierButton.addActionListener( listener );
			updateTrainingDataButton.addActionListener(listener);

			overlayButton.addActionListener(listener);
			//getResultButton.addActionListener(listener);
			//setResultButton.addActionListener(listener);
			assignResultImageButton.addActionListener(listener);
			reviewLabelsButton.addActionListener( listener );
			applyClassifierButton.addActionListener(listener);
			doButton.addActionListener(listener);
			imagingModalityComboBox.addActionListener(listener);
			stopButton.addActionListener( listener );
			addClassButton.addActionListener(listener);
			settingsButton.addActionListener(listener);
			testThreadsButton.addActionListener(listener);
		}

		private JButton createAnnotationButton( int classNum )
		{
			JButton button = new JButton( deepSegmentation.getClassName( classNum ) + " [" + (classNum+1) + "]" );
			button.setToolTipText("Add markings of label '" + deepSegmentation.getClassName( classNum ) + "'");
			button.setOpaque( true );
			button.setBackground( colors[classNum] );

			return ( button );
		}

		private void uncertaintyNavigation( String cmd, int iRegion )
		{
			if ( cmd.equals("go-to") )
			{
				UncertaintyRegion uncertaintyRegion = deepSegmentation.getUncertaintyRegion( iRegion );
				if ( uncertaintyRegion != null )
				{
					inputImage.setT(uncertaintyRegion.xyzt[3] + 1);
					inputImage.setZ(uncertaintyRegion.xyzt[2] + 1);
					int x = uncertaintyRegion.xyzt[0] - deepSegmentation.minTileSizes[0] / 2;
					int y = uncertaintyRegion.xyzt[1] - deepSegmentation.minTileSizes[1] / 2;
					inputImage.setRoi(
							x, y,
							deepSegmentation.minTileSizes[0],
							deepSegmentation.minTileSizes[1]
					);
					// TODO: below does not work
					//zoomToSelection( inputImage, 3.0 );
					inputImage.updateAndDraw();
				}
				else
				{
					logger.error(" There are currently no known uncertainty regions; " +
							"please classify some regions to compute new uncertainties.");
				}
			}

			if ( cmd.equals("delete") )
			{
				deepSegmentation.deleteUncertaintyRegion(iRegion);
			}
		}


		void zoomToSelection(ImagePlus imp, ImageCanvas ic) {
			Roi roi = imp.getRoi();
			ic.unzoom();
			if (roi==null) return;
			Rectangle w = imp.getWindow().getBounds();
			Rectangle r = roi.getBounds();
			double mag = ic.getMagnification();
			int marginw = (int)((w.width - mag * imp.getWidth()));
			int marginh = (int)((w.height - mag * imp.getHeight()));
			int x = r.x+r.width/2;
			int y = r.y+r.height/2;
			mag = ic.getHigherZoomLevel(mag);
			while(r.width*mag<w.width - marginw && r.height*mag<w.height - marginh) {
				ic.zoomIn(ic.screenX(x), ic.screenY(y));
				double cmag = ic.getMagnification();
				if (cmag==32.0) break;
				mag = ic.getHigherZoomLevel(cmag);
				w = imp.getWindow().getBounds();
			}
		}

		private void zoomToSelectionTest( ImagePlus imp, double marginFactor )
		{
			//ImageCanvas ic = inputImage.getCanvas();
			Roi roi = imp.getRoi();

			ic.unzoom();
			if (roi==null) return;
			Rectangle w = imp.getWindow().getBounds();
			Rectangle r = roi.getBounds();

			double mag = ic.getMagnification();
			int marginw = (int)(marginFactor * (w.width - mag * imp.getWidth()));
			int marginh = marginw;
			//int marginh = (int)(marginFactor * (w.height - mag * inputImage.getHeight()));
			int x = r.x + r.width/2;
			int y = r.y + r.height/2;
			mag = ic.getHigherZoomLevel(mag);
			while( (r.width*mag < w.width-marginw) && (r.height*mag<w.height-marginh) )
			{
				ic.zoomIn(ic.screenX(x), ic.screenY(y));
				double cmag = ic.getMagnification();
				if (cmag==32.0) break;
				mag = ic.getHigherZoomLevel(cmag);
				w = imp.getWindow().getBounds();
			}
		}

		private void zoomToPointSelection( ImagePlus imp, double marginFactor )
		{
			//ImageCanvas ic = inputImage.getCanvas();
			Roi roi = imp.getRoi();

			ic.unzoom();
			if (roi==null) return;
			Rectangle w = imp.getWindow().getBounds();
			Rectangle r = roi.getBounds();

			double mag = ic.getMagnification();
			int marginw = (int)(marginFactor * (w.width - mag * imp.getWidth()));
			int marginh = marginw;
			//int marginh = (int)(marginFactor * (w.height - mag * inputImage.getHeight()));
			int x = r.x + r.width/2;
			int y = r.y + r.height/2;
			mag = ic.getHigherZoomLevel(mag);
			while( (r.width*mag < w.width-marginw) && (r.height*mag<w.height-marginh) )
			{
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
		protected DeepSegmentation getWekaSegmentation()
		{
			return deepSegmentation;
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
			final int frame = inputImage.getT();
			final int slice = inputImage.getZ();

			int numClasses = deepSegmentation.getNumClasses();
			for(int iClass = 0; iClass < numClasses; iClass++)
			{
				roiOverlay[iClass].setColor( colors[iClass] );

				roiOverlay[iClass].setRoi(
						deepSegmentation.getLabelRois(
								iClass, slice-1, frame-1));
			}

			inputImage.updateAndDraw();
		}

		/**
		 * Update the example lists in the GUI
		 */
		protected void updateExampleLists()
		{
			final int frame = inputImage.getT();
			final int slice = inputImage.getZ();

			for( int i = 0; i < deepSegmentation.getNumClasses(); i++)
			{
				exampleList[i].removeAll();
				exampleList[i].addActionListener(listener);
				exampleList[i].addItemListener(itemListener);
				int numExamples = deepSegmentation.getLabelRois(i, slice-1, frame-1).size();
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

			exampleList[classNum] = new java.awt.List( numShownLabelsPerClass );
			//exampleList[classNum].setForeground(classColors[classNum]);
			exampleList[classNum].setForeground( Color.black );

			exampleList[classNum].addActionListener(listener);
			exampleList[classNum].addItemListener(itemListener);

			annotationButton[classNum] = createAnnotationButton( classNum );

			annotationsConstraints.fill = GridBagConstraints.HORIZONTAL;
			annotationsConstraints.insets = new Insets(5, 5, 6, 6);

			boxAnnotation.setConstraints( annotationButton[classNum], annotationsConstraints);
			annotationsPanel.add( annotationButton[classNum] );
			annotationsConstraints.gridy++;

			annotationsConstraints.insets = new Insets(0,0,0,0);

			boxAnnotation.setConstraints(exampleList[classNum], annotationsConstraints);
			annotationsPanel.add(exampleList[classNum]);
			annotationsConstraints.gridy++;

			// Add listener to the new button
			annotationButton[classNum].addActionListener(listener);

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
		protected void setButtonsEnabled( Boolean s )
		{
			stopButton.setEnabled( !s );

			trainClassifierButton.setEnabled(s);
			updateTrainingDataButton.setEnabled(s);
			overlayButton.setEnabled(s);
			assignResultImageButton.setEnabled(s);
			reviewLabelsButton.setEnabled(s);
			assignResultImageButton.setEnabled( s );
			reviewLabelsButton.setEnabled( s );
			doButton.setEnabled( s );
			applyClassifierButton.setEnabled(s);
			addClassButton.setEnabled(s);
			settingsButton.setEnabled(s);
			testThreadsButton.setEnabled(s);
			for( int i = 0; i < deepSegmentation.getNumClasses(); i++)
			{
				exampleList[i].setEnabled(s);
				annotationButton[i].setEnabled(s);
			}
			setSliceSelectorEnabled(s);

			repaintWindow();
		}

		synchronized void updateOverlay()
		{
			if ( showColorOverlay ) win.toggleOverlay( recentOverlayMode );
			win.toggleOverlay( recentOverlayMode );
		}

		synchronized void toggleOverlay()
		{
			toggleOverlay( OVERLAY_MODE_PROBABILITIES );
		}

		public static final String OVERLAY_MODE_SEGMENTATION = "Segmentation";
        public static final String OVERLAY_MODE_PROBABILITIES = "Probabilities";
        public static final String OVERLAY_MODE_UNCERTAINTY = "Uncertainty";

        String recentOverlayMode = OVERLAY_MODE_PROBABILITIES;

		/**
		 * Toggle between overlay and original image with markings
		 */
		synchronized void toggleOverlay( String mode )
		{

			// create overlay LUT
			final byte[] red = new byte[ 256 ];
			final byte[] green = new byte[ 256 ];
			final byte[] blue = new byte[ 256 ];

			if ( mode.equals( OVERLAY_MODE_SEGMENTATION ) )
			{
				// assign classColors to classes
				for ( int iClass = 0; iClass < DeepSegmentation.MAX_NUM_CLASSES; iClass++)
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


			if ( mode.equals( OVERLAY_MODE_PROBABILITIES ) )
			{
				// assign classColors to classes
				for ( int iClass = 0; iClass < DeepSegmentation.MAX_NUM_CLASSES; iClass++)
				{
					int offset = iClass * ResultImageDisk.CLASS_LUT_WIDTH;

					for ( int i = 1; i <= ResultImageDisk.CLASS_LUT_WIDTH; i++)
					{
						red[offset + i] = (byte) (1.0 * colors[iClass].getRed() * i / ( ResultImageDisk.CLASS_LUT_WIDTH ));
						green[offset + i] = (byte) (1.0 * colors[iClass].getGreen() * i / ( ResultImageDisk.CLASS_LUT_WIDTH ));
						blue[offset + i] = (byte) (1.0 * colors[iClass].getBlue() * i / ( ResultImageDisk.CLASS_LUT_WIDTH ));
					}
				}
				overlayLUT = new LUT(red, green, blue);
			}

			if ( mode.equals( OVERLAY_MODE_UNCERTAINTY ) )
			{
				// assign classColors to classes
				for ( int iClass = 0; iClass < DeepSegmentation.MAX_NUM_CLASSES; iClass++)
				{
					int offset = iClass * ResultImageDisk.CLASS_LUT_WIDTH;
					for ( int i = 1; i <= ResultImageDisk.CLASS_LUT_WIDTH; i++)
					{
						// TODO:
						// - check whether this is correct
						red[offset + i] = (byte) ( 255.0 * Math.exp( - deepSegmentation.uncertaintyLutDecay * i  ));
						green[offset + i] = (byte) ( 0 );
						blue[offset + i] = (byte) ( 255.0 * Math.exp( - deepSegmentation.uncertaintyLutDecay * i  ));
					}
				}
				overlayLUT = new LUT(red, green, blue);
			}

			showColorOverlay = !showColorOverlay;
			//IJ.log("toggle overlay to: " + showColorOverlay);
			if (showColorOverlay && null != deepSegmentation.getResultImage())
			{
				updateResultOverlay();
			}
			else
				resultOverlay.setImage(null);

			inputImage.updateAndDraw();
		}

		/**
		 * Update the buttons to add classes with current information
		 */
		public void updateAddClassButtons()
		{
			int wekaNumOfClasses = deepSegmentation.getNumClasses();

			while (numOfClasses < wekaNumOfClasses)
            {
                win.addClass();
            }

            for (int i = 0; i < numOfClasses; i++)
            {
                annotationButton[ i ].setText( deepSegmentation.getClassName( i ) + " [" + ( i + 1 ) + "]" );
            }

			repaintWindow();
		}

		/**
		 * Set the flag to inform the the instances has finished or not
		 *
		 * @param b tranining complete flag
		 */
		void setTrainingComplete(boolean b)
		{
			this.trainingComplete = b;
		}

		/**
		 * Get instances image
		 * @return instances image
		 */
		public ImagePlus getTrainingImage()
		{
			return inputImage;
		}


	}// end class CustomWindow

	private boolean initialisation()
	{
		IJ.run( inputImage, "Properties...", "");

		GenericDialog gd = new NonBlockingGenericDialog("Set up");

		/*
		gd.addMessage(
				"IMAGE CALIBRATION\n \n" +
						"Please check your image calibration using [Image > Properties].\n" +
				"\nBelow points are very important:\n" +
				"1. The number of slices (z) and number of frames (t) have to be correct " +
						"(sometimes z and t are mixed up).\n" +
				"2. The pixel width, height and depth have to be set properly " +
						"(sometimes the image calibration got lost and all is in units of 1 pixel).\n" +
						"\n\nYou can leave this dialog open while changing the image properties!");
						*/

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
				new String[]
						{ DeepSegmentation.RESULT_IMAGE_DISK_SINGLE_TIFF,
								DeepSegmentation.RESULT_IMAGE_RAM },
				DeepSegmentation.RESULT_IMAGE_RAM);


		/*
		gd.addMessage( "IMAGING MODALITY\n \n" +
				"For optimal segmentation performance, please choose your imaging modality." );
				gd.addChoice( "Modality" ,
				new String[]
						{ FLUORESCENCE_IMAGING,
								EM_IMAGING,
								TRANSMISSION_LM_IMAGING },
						EM_IMAGING);
						*/

        /*
		IJ.run("Set Measurements...", "mean redirect=None decimal=4");

		gd.addMessage( "IMAGE BACKGROUND\n \n" +
				"This is currently only used for fluorescence images. " +
				"Setting the correct value here should help to achieve better segmentation results.\n" +
				"In order to measure the image background you could for instance put an ROI on " +
				"a background region of your image and then measure the mean intensity." );

		gd.addNumericField( "Value", 0, 0 );
		*/

		gd.showDialog();

        if ( gd.wasCanceled() ) return false;

        assignImageName( gd.getNextString() );
		deepSegmentation.assignResultImage( gd.getNextChoice() );
		// setImagingModality( gd.getNextChoice() );
		//deepSegmentation.setImageBackground( (int) gd.getNextNumber() );
        //deepSegmentation.setImagingModality( EM_IMAGING );

        deepSegmentation.setImageBackground( 0 );

		return true;
	}

	private void assignImageName( String imageName )
	{
		inputImage.setTitle( imageName );
		deepSegmentation.getInputImage().setTitle( imageName );
	}

	int numShownLabelsPerClass = 1;

	/**
	 * Plugin run method
	 */
	public void run( String arg )
	{
	    deepSegmentation = new DeepSegmentation(  );
	    deepSegmentation.deepSegmentationIJ1Plugin = this;
	    initialise( deepSegmentation, true );
    }


	public void initialise( DeepSegmentation deepSegmentation, boolean showUIs )
    {
        this.deepSegmentation = deepSegmentation;

        logger = deepSegmentation.getLogger();

        //IJ.open( "  /Users/de.embl.cba.trainableDeepSegmentation.weka/Desktop/brainiac2-mit-edu-SNEMI3D/train-labels/train-labels-binary-larger-borders.tif ");
        //deepSegmentation.setLabelImage( IJ.getImage() );

        for( int i = 0; i < deepSegmentation.getNumClasses() ; i++)
        {
            exampleList[i] = new java.awt.List( numShownLabelsPerClass );
            //exampleList[i].setForeground(classColors[i]);
            exampleList[i].setForeground( Color.black );

        }

        numOfClasses = deepSegmentation.getNumClasses();

        //getInstancesAndMetadata current image
        if ( null == WindowManager.getCurrentImage() )
        {
            inputImage = IJ.openImage(); // this implicitely gets the open="..." filePath
            if (null == inputImage ) return; // user canceled open dialog
        }
        else
        {
            inputImage = WindowManager.getCurrentImage(); //.duplicate();
            inputImage.setSlice(WindowManager.getCurrentImage().getSlice());
        }

        deepSegmentation.setInputImage( inputImage );

        if ( showUIs ) if ( ! initialisation() ) return;

        reviewLabelsClassComboBox = new JComboBox( deepSegmentation.getClassNames().toArray() );

        if ( showUIs ) showFeatureSettingsDialogIJ1Plugin( false );

        //Build GUI
        SwingUtilities.invokeLater(
                new Runnable() {
                    public void run()
                    {
						win = new CustomWindow( inputImage );
                        win.pack();
                    }
                });

    }

    /**
	 * Add examples defined by the user to the corresponding list
	 * in the GUI and the example list in the segmentation object.
	 *
	 * @param classNum GUI list index
	 */
	private void addAnnotation( int classNum )
	{
		if ( classNum >= deepSegmentation.getNumClasses() )
		{
			logger.error( "Class number " + classNum + " does not exist; cannot add label.");
			return;
		}

		if ( deepSegmentation.isBusy )
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

		Example newExample = deepSegmentation.createExample( classNum, points, (int)roi.getStrokeWidth(), z, t );

		deepSegmentation.addExample( newExample );

		if ( computeLabelFeatureValuesInstantly )
        {
            Thread thread = new Thread()
            {
                public void run()
                {
                    deepSegmentation.updateLabelInstancesAndMetadata();
                    updateComboBoxes();
                }
            }; thread.start();
        }

		traceCounter[ classNum ]++;
		win.drawExamples();
		win.updateExampleLists();

		// Record
		final int n = inputImage.getCurrentSlice();
		String[] arg = new String[]{ Integer.toString(classNum), Integer.toString(n)};
		record(ADD_TRACE, arg);

		/*
		String numLabelsPerClassString = "";

		int[] numLabelsPerClass = ExamplesUtils.getNumExamplesPerClass(
				deepSegmentation.getExamples() );

		for ( int i = 0 ; i <  numLabelsPerClass.length; i++ )
		{
			numLabelsPerClassString += " "+(i+1)+":"+numLabelsPerClass[i];
		}

		logger.progress("Number of labels per class:", numLabelsPerClassString);
		*/

	}

	/**
	 * Update the resultImagePlus image overlay with the corresponding slice
	 */
	public void updateResultOverlay()
	{
		ImageProcessor overlay = deepSegmentation.getResultImage().getSlice( inputImage.getZ(), inputImage.getT() );

		overlay = overlay.convertToByte( false );
		overlay.setColorModel( overlayLUT );

		win.resultOverlay.setImage( overlay );
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

		// win.updateLabels();
		inputImage.setColor(Color.YELLOW);

		int selectedIndex = 0;

		for( int j = 0; j < deepSegmentation.getNumClasses(); j++)
		{
			if (j == iClass)
			{
				selectedIndex = exampleList[iClass].getSelectedIndex();
				if ( selectedIndex != -1 )
				{

					ArrayList < Roi > exampleRois =
							deepSegmentation.getLabelRois(iClass,
							inputImage.getZ() - 1,
							inputImage.getT() - 1);

					if ( selectedIndex < exampleRois.size() )
					{
						final Roi newRoi = exampleRois.get( selectedIndex );
						// Set selected trace as current ROI
						newRoi.setImage( inputImage );
						inputImage.setRoi( newRoi );
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

		inputImage.updateAndDraw();
	}

	/**
	 * Delete one of the ROIs
	 *
	 * @param e action event
	 */
	synchronized void deleteSelected(final ActionEvent e)
	{
		for( int iClass = 0; iClass < deepSegmentation.getNumClasses(); iClass++)

			if ( e.getSource() == exampleList[iClass] )
			{
				//delete item from ROI
				int index = exampleList[iClass].getSelectedIndex();

				if ( index == -1 )
					return; // has been deleted already

				ArrayList<Roi> exampleRois = deepSegmentation.getLabelRois(
						iClass,
						inputImage.getZ()-1,
						inputImage.getT()-1);

				Roi selectedRoi = exampleRois.get(index);
				Roi activeRoiOnImage = null;

				if ( inputImage.getRoi() != null )
					activeRoiOnImage = inputImage.getRoi();
				else
					return;

				if( activeRoiOnImage.getBounds().equals(selectedRoi.getBounds()) )
					inputImage.killRoi();
				else
					return;

				// delete item from the list of ROIs of that class and slice
				deepSegmentation.removeExample(
						iClass,
						inputImage.getZ()-1,
						inputImage.getT()-1,
						index);

				//delete item from GUI list
				exampleList[iClass].remove( index );

				// Record
				String[] arg = new String[] {
						Integer.toString(iClass),
						Integer.toString( inputImage.getCurrentSlice() ),
						Integer.toString(index)};
				record(DELETE_TRACE, arg);
			}

		win.updateExampleLists();
		win.drawExamples();
	}


	void updateLabelsTrainingDataAndTrainClassifier()
	{
		if ( deepSegmentation.isExampleInstanceValuesAreCurrentlyBeingComputed() )
		{
			logger.error( "Feature value computation is ongoing (see log window); please try again later." );
			return;
		}

		// Disable rest of buttons until the instances has finished
		win.setButtonsEnabled( false );

		Thread task = new Thread() {
			public void run()
			{

				deepSegmentation.updateLabelInstancesAndMetadata();

				updateComboBoxes();

				InstancesAndMetadata updatedInstancesAndMetadata = deepSegmentation.getCurrentLabelInstancesAndMetadata();

				deepSegmentation.trainClassifierWithFeatureSelection( updatedInstancesAndMetadata );

				win.setButtonsEnabled( true );

			}

		}; task.start();

	}


    void updateLabelsTrainingData()
    {
        // Disable rest of buttons until the instances has finished
        win.setButtonsEnabled( false );

        Thread task = new Thread() {
            public void run()
            {

                deepSegmentation.updateLabelInstancesAndMetadata();

                updateComboBoxes();

                win.setButtonsEnabled( true );

            }

        }; task.start();

    }


    void updateLabelInstancesAndTrainClassifier()
	{
	    win.setButtonsEnabled( false );

		Thread task = new Thread() {
			public void run()
			{
				deepSegmentation.updateLabelInstancesAndMetadata();
                trainClassifier( deepSegmentation.getCurrentLabelInstancesAndMetadata() );
				updateComboBoxes( );
				win.setButtonsEnabled( true );
			}

		}; task.start();
	}

    private static void showMemoryMonitor()
	{
		Thread thread = new Thread( new Runnable() {
			//exec.submit(new Runnable() {
			public void run()
			{
				// IJ.run( "Monitor Memory...", "" );
			}
		} ); thread.start();
	}


    /**
	 * Run/stop the classifier instances
	 *
	 * @param command current text of the instances button ("Train classifier" or "STOP")
	 */
	public void trainFromLabelImage()
	{
		FinalInterval interval = getIntervalFromGUI();

		if ( interval == null ) return;

		if ( ! deepSegmentation.hasLabelImage() )
		{
			logger.error( "No label image loaded." );
			return;
		}

		if ( ! deepSegmentation.hasResultImage() )
		{
			logger.error( "No resultImagePlus image assigned." );
			return;
		}

		win.setButtonsEnabled( false );
		deepSegmentation.isBusy = true;

		// Run the instances
		Thread newTask = new Thread() {

			public void run()
			{

				try
				{

					GenericDialog gd = new GenericDialogPlus( "Training from label image" );
					gd.addStringField( "Training data name", "labelImage", 15 );
					gd.addChoice( "Modality", new String[]
							{ deepSegmentation.START_NEW_INSTANCES,
									deepSegmentation.APPEND_TO_PREVIOUS_INSTANCES },
									deepSegmentation.START_NEW_INSTANCES );
					//gd.addNumericField( "Number of iterations", 3.0, 0 );
					gd.addNumericField( "Z chunk size", Prefs.getThreads(), 0 );
					gd.addNumericField( "(nx,ny) for tiling", 3, 0 );
					gd.addNumericField( "Radius for local instances pairs", 5, 0 );
					gd.addNumericField( "Maximum number of instances in total", 400000, 0 );
					gd.addNumericField( "Maximum number of instance pairs per plane and tile", 20, 0 );

					gd.addCheckbox( "Auto save instances", true );

					gd.showDialog();

					if ( gd.wasCanceled() ) return;

					String instancesKey = gd.getNextString();
					String modality = gd.getNextChoice();
					int numIterations = 1000; // let's rather control it via the maxNumInstances
					int zChunkSize = (int) gd.getNextNumber();
					int nxyTiles = (int) gd.getNextNumber();
					int localRadius = (int) gd.getNextNumber();
					long maxNumInstances = (long) gd.getNextNumber();
					int numInstanceSetsPerTilePlaneClass = (int) gd.getNextNumber();
					boolean autoSaveInstances = gd.getNextBoolean();

					String directory = null;

					if ( ! showFeatureSettingsDialogIJ1Plugin( false ) ) return;
					//if ( ! showClassifierSettingsDialog() ) return;

					if ( autoSaveInstances )
					{
						directory = IJ.getDirectory( "Choose directory for saving the instances" );
					}

					ArrayList< Double > classWeights = new ArrayList<>(  );
					classWeights.add( 1.05 );
					classWeights.add( 1.0 );

					deepSegmentation.trainFromLabelImage(
							instancesKey,
							modality,
							numIterations,
							zChunkSize,
							nxyTiles,
							localRadius,
							numInstanceSetsPerTilePlaneClass,
							maxNumInstances,
							100, // TODO
							100, // TODO
							400,
							100, // TODO
							classWeights,
							directory,
							interval,
							interval
					);


				}
				catch( Exception e )
				{
					e.printStackTrace();
				}
				catch( OutOfMemoryError err )
				{
					err.printStackTrace();
				}
				finally
				{
					updateComboBoxes();
					win.classificationComplete = true;
					deepSegmentation.isBusy = false;
					win.setButtonsEnabled( true );
				}

			}
		}; newTask.start();


	}

	public void computeLabelImageBasedAccuracies()
	{
		win.setButtonsEnabled( false );
		deepSegmentation.isBusy = true;

		Thread task = new Thread() {

			public void run()
			{

				FinalInterval interval = IntervalUtils.getIntervalWithChannelsDimensionAsSingleton( deepSegmentation.getInputImage() );

				deepSegmentation.computeLabelImageBasedAccuracies( "accuracies", 1, interval);

				deepSegmentation.isBusy = false;
				win.setButtonsEnabled( true );

			}
		}; task.start();

	}


	public InstancesAndMetadata getCombinedSelectedInstancesFromGUI()
	{
		List<String> selectedInstances = (List< String >) instancesList.getSelectedValuesList();

		if ( selectedInstances.size() == 0 ) return null;

		if ( selectedInstances.size() == 1 )
		{
			return deepSegmentation.getInstancesManager().getInstancesAndMetadata( selectedInstances.get(0) );
		}
		else
		{
			logger.info("# Combining instances:");
			for ( String key : selectedInstances )
			{
				logger.info( key );
			}
			return deepSegmentation.getInstancesManager().getCombinedInstancesAndMetadata( selectedInstances );
		}

	}

	void trainClassifier( InstancesAndMetadata instancesAndMetadata )
	{

		win.setButtonsEnabled( false );

		// Thread to run the instances
		Thread newTask = new Thread() {

			public void run()
			{

			try
			{
				deepSegmentation.trainClassifierWithFeatureSelection(  instancesAndMetadata );
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
			win.setButtonsEnabled( true );

			}
		}; newTask.start();

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


	private void ensureExistenceOfBgFgResultImage()
	{

		if ( deepSegmentation.getResultImageBgFg() != null ) return;

		ResultImage resultImage = null;

		if ( resultImageComboBox.getSelectedItem().equals( DeepSegmentation.RESULT_IMAGE_DISK_SINGLE_TIFF ) )
		{
			String directory = IJ.getDirectory("Select a directory");


			deepSegmentation.setResultImageBgFgDisk( directory );

		}
		else if ( resultImageComboBox.getSelectedItem().equals( DeepSegmentation.RESULT_IMAGE_RAM ))
		{
			deepSegmentation.setResultImageBgFgRAM( );
		}

	}


	public void applyBgFgClassification()
	{
		FinalInterval interval = getIntervalFromGUI( );
		if ( interval == null ) return;

		Thread thread = new Thread() {
			public void run()
			{

				win.setButtonsEnabled( false );
				deepSegmentation.stopCurrentTasks = false;
				deepSegmentation.isBusy = true;

				InstancesAndMetadata iam = getCombinedSelectedInstancesFromGUI();
				deepSegmentation.applyBgFgClassification( interval, iam );

				win.updateOverlay();
				win.setButtonsEnabled( true );
				deepSegmentation.isBusy = false;

			}
		}; thread.start();


	}

	public void applyClassifier( )
	{

		showMemoryMonitor();

		String classifierKey = deepSegmentation.getClassifierManager().getMostRecentClassifierKey();

		if ( classifierKey == null )
		{
			logger.error( "No classifier trained yet..." );
		}

		if ( deepSegmentation.getResultImage() == null )
		{
			logger.error("Classification resultImagePlus image yet assigned.\n" +
					"Please [Assign resultImagePlus image].");
			return;
		}

		FinalInterval interval = getIntervalFromGUI( );
		if ( interval == null ) return;

		logger.info("\n# Classifying selected region...");

		Thread thread = new Thread() {
			public void run()
			{

				win.setButtonsEnabled( false );
				deepSegmentation.stopCurrentTasks = false;
				deepSegmentation.isBusy = true;
				deepSegmentation.resetUncertaintyRegions();

				deepSegmentation.applyClassifierWithTiling( classifierKey, interval );

				if (showColorOverlay)
					win.toggleOverlay();
				win.toggleOverlay();

				win.setButtonsEnabled( true );
				deepSegmentation.isBusy = false;

			}
		}; thread.start();

	}


	public void applyClassifierOnSlurm()
	{

		/*
		String mostRecentClassifierKey = deepSegmentation.getClassifierManager().getMostRecentClassifierKey();

		if ( mostRecentClassifierKey == null )
		{
			logger.error( "No classifier trained yet..." );
		}
		*/

		if ( deepSegmentation.getResultImage() == null )
		{
			logger.error("Classification resultImagePlus image yet assigned.\n" + "Please [Assign resultImagePlus image].");
			return;
		}


		/*
		if ( ! (deepSegmentation.getResultImage() instanceof ResultImageDisk) )
		{
			logger.error( "Result image type not supported: must be disk resident." );
			return;
		}
		*/

		FinalInterval interval = getIntervalFromGUI( );

		if ( interval == null )
		{
			logger.error("Could not determine the interval to be classified.");
			return;
		}

		logger.info("\n# Classifying selected region...");



		Thread thread = new Thread() {
			public void run()
			{

				win.setButtonsEnabled( false );
				deepSegmentation.stopCurrentTasks = false;
				deepSegmentation.isBusy = true;
				deepSegmentation.resetUncertaintyRegions();

				deepSegmentation.applyClassifierOnSlurm( new HashMap<>(), interval );

				if (showColorOverlay)
					win.toggleOverlay();

				win.toggleOverlay();

				win.setButtonsEnabled( true );
				deepSegmentation.isBusy = false;

			}
		}; thread.start();

	}


	private FinalInterval getIntervalFromGUI( )
	{

		String rangeString = (String) rangeComboBox.getSelectedItem();

		if ( rangeString.equals( ClassificationRangeUtils.WHOLE_DATA_SET) )
		{
			return ( IntervalUtils.getIntervalWithChannelsDimensionAsSingleton( deepSegmentation.getInputImage() ) );
		}
		else
		{
			return ( ClassificationRangeUtils.getIntervalFromRoi( inputImage, rangeString ) );
		}

	}



	/**
	 * Save instances to a file
	 */
	public void saveInstances( String key )
	{
		String[] dirFile = IOUtils.getSaveDirFile( "Save instance file", ".ARFF" );

		if ( dirFile != null )
		{
			win.setButtonsEnabled( false );
			deepSegmentation.isBusy = true;

			deepSegmentation.saveInstances( key, dirFile[ 0 ], dirFile[ 1 ] );

			win.setButtonsEnabled( true );
			deepSegmentation.isBusy = false;
		}

	}


    private void removeAllRoiOverlays()
	{
		for ( int i = 0; i < roiOverlay.length; ++i )
		{
			roiOverlay[ i ].setRoi( null );
		}
	}

	/**
	 * Save instances to a file
	 */
	public void loadInstances()
	{

	    String[] dirFile = IOUtils.getOpenDirFile( "Please choose instance file" );

	    if ( dirFile != null )
		{
			win.setButtonsEnabled( false );
			deepSegmentation.isBusy = true;

			deepSegmentation.loadInstancesAndMetadata( dirFile[ 0 ], dirFile[ 1 ] );

			updateUI();

			win.setButtonsEnabled( true );
			deepSegmentation.isBusy = false;

			logger.info( "Loading instances finished." );
		}

	}

    private void updateUI()
    {
        updateClassNamesInUI();
        updateComboBoxes();
        repaintWindow();
        win.updateExampleLists();
    }


    public void loadClassifierDialog()
	{

		String[] dirFile = IOUtils.getOpenDirFile( "Please choose a classifier file" );

		if ( dirFile != null )
		{
			win.setButtonsEnabled( false );
			deepSegmentation.isBusy = true;

			deepSegmentation.loadClassifier( dirFile[ 0 ], dirFile[ 1 ] );

			updateClassNamesInUI();

			win.setButtonsEnabled( true );
			deepSegmentation.isBusy = false;
		}

	}

    private void updateClassNamesInUI()
    {

        for ( int c = 0; c < deepSegmentation.getNumClasses(); c++ )
        {
            if ( c == numOfClasses )
            {
                win.addClass();
            }

            changeClassName( c, deepSegmentation.getClassName( c ) );
        }
    }

    /**
	 * Save instances to a file
	 */
	public void saveClassifier()
	{

		String[] dirFile = IOUtils.getSaveDirFile("Please choose a output file", ".classifier" );

		if ( dirFile != null )
		{
			win.setButtonsEnabled( false );
			deepSegmentation.isBusy = true;

			deepSegmentation.saveClassifier( dirFile[ 0 ], dirFile[ 1 ] );

			win.setButtonsEnabled( true );
			deepSegmentation.isBusy = false;
		}

	}

	/**
	 * Add new class in the panel (up to MAX_NUM_CLASSES)
	 */
	private void addNewClass()
	{

        String inputName = IOUtils.classNameDialog();
        if ( inputName == null ) return;

		// Add new name to the list of labels
		deepSegmentation.addClass( inputName );
		updateComboBoxes();

		// Add new class label and list
		win.addClass();

		repaintWindow();

		// Macro recording
		String[] arg = new String[] { inputName };
		record(CREATE_CLASS, arg);
	}

    private void updateComboBoxes()
	{
		reviewLabelsClassComboBox.setModel( new DefaultComboBoxModel( deepSegmentation.getClassNames().toArray() ) );

		DefaultListModel listModel = new DefaultListModel();

		for ( String instancesKey : deepSegmentation.getInstancesManager().getKeys() )
		{
			listModel.addElement( instancesKey );
		}

		instancesList.setModel( listModel );
		instancesList.setSelectedIndex( 0 );

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


	public boolean showResultsOverlayOpacityDialog()
	{
		GenericDialogPlus gd = new GenericDialogPlus("Result overlay opacity");
		gd.addSlider("Result overlay opacity", 0, 100, win.overlayOpacity);

		gd.showDialog();

		if ( gd.wasCanceled() )
			return false;

		final int newOpacity = (int) gd.getNextNumber();
		if( newOpacity != win.overlayOpacity )
		{
			win.overlayOpacity = newOpacity;
			win.overlayAlpha = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, win.overlayOpacity / 100f);
			win.resultOverlay.setComposite(win.overlayAlpha);

			if( showColorOverlay )
				inputImage.updateAndDraw();
		}

		return true;


	}

	/**
	 * Show class featureSettings dialog
	 *
	 * @return false when canceled
	 */
	public boolean showClassNamesDialog()
	{
		GenericDialogPlus gd = new GenericDialogPlus("Class names");

		for( int i = 0; i < deepSegmentation.getNumClasses(); i++)
			gd.addStringField("Class "+(i+1), deepSegmentation.getClassName(i), 15);

		gd.showDialog();

		if ( gd.wasCanceled() )
			return false;

		boolean classNameChanged = false;

		for( int i = 0; i < deepSegmentation.getNumClasses(); i++)
		{
			String s = gd.getNextString();
			if (null == s || 0 == s.length()) {
				IJ.log("Invalid name for class " + (i+1));
				continue;
			}
			s = s.trim();
			if(!s.equals( deepSegmentation.getClassName(i)))
			{
				if (0 == s.toLowerCase().indexOf("add to "))
					s = s.substring(7);

				deepSegmentation.setClassLabel(i, s);
				classNameChanged = true;
				annotationButton[i].setText( s + " [" + (i+1) + "]");
			}
		}

		// adapt to changes in class names
		updateComboBoxes();

		if(classNameChanged)
		{
			// Pack window to update buttons
			win.pack();
		}


		return true;
	}


	/**
	 * Show class featureSettings dialog
	 *
	 * @return false when canceled
	 */
	public boolean showColorsDialog()
	{
		GenericDialogPlus gd = new GenericDialogPlus("Change class color");

		String[] classNames = new String[ deepSegmentation.getClassNames().size()];

		gd.addChoice("Class", deepSegmentation.getClassNames().toArray( classNames ), deepSegmentation.getClassName( 0 ));

		gd.showDialog();

		if ( gd.wasCanceled() )
			return false;

		changeClassColor( gd.getNextChoiceIndex() );

		return true;
	}

	private void changeClassColor( int iClass )
	{
		ColorChooser colorChooser = new ColorChooser( "" + deepSegmentation.getClassName( iClass ), colors[ iClass ], false );

		colors[ iClass ] = colorChooser.getColor();

		win.getAnnotationButton( iClass ).setBackground( colors[ iClass ] );
	}


    public boolean showFeatureSettingsDialogIJ1Plugin( boolean showAdvancedSettings )
	{
		GenericDialogPlus gd = new GenericDialogPlus("Segmentation featureSettings");

		for ( int i = 0; i < 5; ++i )
		{
			gd.addNumericField( "Binning " + ( i + 1 ), deepSegmentation.featureSettings.binFactors.get( i ), 0 );
		}

		gd.addNumericField("Maximal convolution depth", deepSegmentation.featureSettings.maxDeepConvLevel, 0);
		gd.addNumericField("z/xy featureSettings.anisotropy", deepSegmentation.featureSettings.anisotropy, 10);
		gd.addStringField("Feature computation: Channels to consider (one-based) [ID,ID,..]",
				FeatureSettings.getAsCSVString( deepSegmentation.featureSettings.activeChannels, 1 ) );

		if ( showAdvancedSettings )
		{
			gd.addStringField( "Bounding box offsets ",
					FeatureSettings.getAsCSVString( deepSegmentation.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels, 0 ) );

			gd.addChoice( "Downsampling method", new String[]{
							DownSampler.BIN_AVERAGE,
							DownSampler.BIN_MAXIMUM,
							DownSampler.TRANSFORMJ_SCALE_LINEAR,
							DownSampler.TRANSFORMJ_SCALE_CUBIC },
							DownSampler.getString( deepSegmentation.featureSettings.downSamplingMethod ) );

			gd.addStringField( "Smoothing sigmas [pixels x/y] ",
					FeatureSettings.getAsCSVString( deepSegmentation.featureSettings.smoothingScales, 0 ) );

			gd.addCheckbox( "Compute Gaussian filter", deepSegmentation.featureSettings.computeGaussian );

            gd.addCheckbox( "Use log2 transformation", deepSegmentation.featureSettings.log2 );

            gd.addCheckbox( "Consider multiple bounding box offsets during loading", deepSegmentation.considerMultipleBoundingBoxOffsetsDuringInstancesLoading );

        }

		gd.showDialog();

		if ( gd.wasCanceled() )
			return false;

        FeatureSettings newFeatureSettings = deepSegmentation.getFeatureSettingsFromGenericDialog( gd, showAdvancedSettings );
        boolean settingsChanged = ! deepSegmentation.featureSettings.equals( newFeatureSettings );
        deepSegmentation.featureSettings = newFeatureSettings;

        if ( settingsChanged )
        {
        	if ( deepSegmentation.getInstancesManager().getNumInstancesSets() > 0 )
			{
				recomputeLabelFeaturesAndRetrainClassifier();
			}
        }

		return true;
	}


    public boolean showDebugSettingsDialog()
    {
        GenericDialogPlus gd = new GenericDialogPlus("Debug featureSettings");

        gd.addCheckbox("Log label pixel values convolution depth", deepSegmentation.debugLogLabelPixelValues );
        gd.addCheckbox("Compute feature values instantly", computeLabelFeatureValuesInstantly );
        gd.addCheckbox("Compute example feature values at multiple offsets",  deepSegmentation.computeExampleFeatureValuesAtMultipleRegionOffest );
        gd.addCheckbox("Compute feature values in whole image interval", deepSegmentation.debugUseWholeImageForFeatureComputation);
        gd.addStringField( "Show feature image: ", deepSegmentation.featureImageToBeShown, 100 );
        gd.showDialog();

        if ( gd.wasCanceled() )
            return false;

        deepSegmentation.debugLogLabelPixelValues = gd.getNextBoolean();
        computeLabelFeatureValuesInstantly = gd.getNextBoolean();
        deepSegmentation.computeExampleFeatureValuesAtMultipleRegionOffest = gd.getNextBoolean();
        deepSegmentation.debugUseWholeImageForFeatureComputation = gd.getNextBoolean();
        deepSegmentation.featureImageToBeShown = gd.getNextString();

        return true;
    }

    private void recomputeLabelFeaturesAndRetrainClassifier()
    {
        logger.warning( "Feature computation featureSettings have been changed.\n" +
                "Thus, all label features will be recomputed \n" +
                "and the classifier will be retrained. \n" +
                "This might take some time." );

        deepSegmentation.recomputeLabelInstances = true;

        updateLabelInstancesAndTrainClassifier();


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
		command = "call(\"de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin." + command;
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
	 * Toggle current resultImagePlus overlay image
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
			final DeepSegmentation deepSegmentation = win.getWekaSegmentation();

			//int classNum = Integer.parseInt(classIndex);
			deepSegmentation.setClassLabel(classNum, className);
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
			final DeepSegmentation deepSegmentation = win.getWekaSegmentation();

			final int numClasses = deepSegmentation.getNumClasses();
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
						 ArrayList<Roi> rois = deepSegmentation.getLabelRois(j,z-1,t-1);
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



}// end of DeepSegmentationIJ1Plugin class

