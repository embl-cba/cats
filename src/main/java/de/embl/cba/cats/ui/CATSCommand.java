package de.embl.cba.cats.ui;

import de.embl.cba.cats.CATS;
import de.embl.cba.cats.classification.ClassificationRangeUtils;
import de.embl.cba.cats.features.ImageScience;
import de.embl.cba.cats.instances.InstancesAndMetadata;
import de.embl.cba.cats.results.ResultImageExportGUI;
import de.embl.cba.cats.utils.IOUtils;
import de.embl.cba.cats.utils.IntervalUtils;
import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import net.imglib2.FinalInterval;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.command.Interactive;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.Button;

import java.util.ArrayList;
import java.util.Set;

import static de.embl.cba.cats.CATS.logger;
import static de.embl.cba.cats.utils.IOUtils.getOpenDirFile;
import static de.embl.cba.cats.utils.IOUtils.getSaveDirFile;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>CATS", initializer = "init")
public class CATSCommand implements Command, Interactive
{
    public static final String ARFF = ".ARFF";
    public static final String CLASSIFIER = ".classifier";

    @Parameter ( required = true )
    public ImagePlus inputImage;

    /** actions */
    public static final String IO_LOAD_CLASSIFIER = "Load classifier";
    public static final String IO_SAVE_CLASSIFIER = "Save classifier";
    public static final String APPLY_CLASSIFIER = "Apply classifier";
    public static final String ADD_CLASS = "Add class";
    public static final String CHANGE_CLASS_NAMES = "Change class names";
    public static final String CHANGE_COLORS = "Change class colors";
    public static final String CHANGE_RESULT_OVERLAY_OPACITY = "Overlay opacity";
    public static final String UPDATE_LABEL_INSTANCES = "Update labels";
    public static final String UPDATE_LABELS = "Update labels";
    public static final String TRAIN_CLASSIFIER = "Train classifier";
    public static final String IO_LOAD_LABEL_IMAGE = "Load label image";
    public static final String IO_LOAD_LABEL_INSTANCES = "Load labels";
    public static final String STOP_CLASSIFICATION = "Stop classification";
    public static final String IO_SAVE_LABELS = "Save label instances of current image";
    public static final String IO_EXPORT_RESULT_IMAGE = "Export results";
    public static final String TRAIN_FROM_LABEL_IMAGE = "Train from label image";
    public static final String APPLY_CLASSIFIER_ON_SLURM = "Apply classifier on cluster";
    public static final String APPLY_BG_FG_CLASSIFIER = "Apply BgFg classifier (development)";
    public static final String DUPLICATE_RESULT_IMAGE_TO_RAM = "Show result image";
    public static final String GET_LABEL_IMAGE_TRAINING_ACCURACIES = "Label image training accuracies";
    public static final String CHANGE_CLASSIFIER_SETTINGS = "Change classifier settings";
    public static final String CHANGE_FEATURE_SETTINGS = "Change feature settings";
    public static final String CHANGE_ADVANCED_FEATURE_SETTINGS = "Change advanced feature settings";
    public static final String SEGMENT_OBJECTS = "Segment objects";
    public static final String REVIEW_OBJECTS = "Review objects";
    public static final String REVIEW_LABELS = "Review labels";
	public static final String CREATE_OBJECTS_IMAGE = "Create object image";
    public static final String UPDATE_LABELS_AND_TRAIN_CLASSIFIER =
            "Update labels and train classifier";
    public static final String LOAD_LABELS_AND_TRAIN_CLASSIFIER = "Load labels and train classifier";
    public static final String RECOMPUTE_LABEL_FEATURE_VALUES = "Recompute all feature values";
    public static final String CHANGE_DEBUG_SETTINGS = "Change development featuresettings";

    @Parameter( label = "Execute Action", callback = "executeAction" )
    private Button executeActionButton;

    @Parameter(label = "Actions", persist = false,
            choices = {
                    ADD_CLASS,
                    CHANGE_CLASS_NAMES,
                    CHANGE_COLORS,
                    UPDATE_LABEL_INSTANCES,
                    REVIEW_LABELS,
                    IO_LOAD_LABEL_INSTANCES,
                    TRAIN_CLASSIFIER,
                    IO_LOAD_CLASSIFIER,
                    IO_SAVE_CLASSIFIER,
                    APPLY_CLASSIFIER_ON_SLURM,
                    SEGMENT_OBJECTS,
                    REVIEW_OBJECTS,
                    CREATE_OBJECTS_IMAGE,
                    IO_EXPORT_RESULT_IMAGE,
                    CHANGE_FEATURE_SETTINGS,
                    CHANGE_CLASSIFIER_SETTINGS,
                    STOP_CLASSIFICATION
//                    CHANGE_RESULT_OVERLAY_OPACITY,
//                    UPDATE_LABELS,
//                    IO_LOAD_LABEL_IMAGE,
//                    TRAIN_FROM_LABEL_IMAGE,
//                    GET_LABEL_IMAGE_TRAINING_ACCURACIES,
//                    IO_LOAD_CLASSIFIER
//                    RECOMPUTE_LABEL_FEATURE_VALUES,
//                    CHANGE_DEBUG_SETTINGS,
//                    CHANGE_ADVANCED_FEATURE_SETTINGS
            } )

    private String action = ADD_CLASS;

    @Parameter( visibility = ItemVisibility.MESSAGE )
    private String emptySpace01 = "\n\n";

    @Parameter( label = "Train Classifier",
            callback = "updateLabelsAndTrainClassifier" )
    private Button updateLabelsAndTrainClassifierButton;

    @Parameter( label = "Apply Classifier",
            callback = "applyClassifier" )
    private Button applyClassifierButton;

    @Parameter(label = "Classification Range", persist = false,
            choices = {
                    ClassificationRangeUtils.SELECTION_ROI,
                    ClassificationRangeUtils.WHOLE_DATA_SET,
                    ClassificationRangeUtils.SELECTION_PM10Z })
    private String range = ClassificationRangeUtils.SELECTION_ROI;


    /*
    @Parameter( visibility = ItemVisibility.MESSAGE )
    private String classificationToggleMessage =
            "<html> " +
            "[ p ] toggle probability overlay <br>" +
                    "[ r ] toggle result overlay <br>";
                    */

    private CATS cats;

    private Overlays overlays;

    private Listeners listeners;

    private LabelButtonsPanel labelButtonsPanel;

    @Override
    public void run()
    {

    }

    public void init()
    {
        try
        {
            ImageScience.isAvailable();
        }
        catch (final NoClassDefFoundError err)
        {
            IJ.showMessage( "Please install ImageScience! [ Help > Update > Manage Update Sites ]" );
        }

        IJ.setTool( "freeline" );

        cats = new CATS( );

        cats.setInputImage( inputImage );

        cats.imageCalibrationDialog();

        cats.initialisationDialog();

        overlays = new Overlays( cats );

        labelButtonsPanel = new LabelButtonsPanel(
                cats,
                overlays,
                inputImage.getWindow().getLocation(),
                inputImage.getWindow().getWidth()
                );

        listeners = new Listeners( cats, overlays, labelButtonsPanel );

        cats.reserveKeyboardShortcuts();

    }

    protected void executeAction()
    {

        Thread thread = new Thread( () -> {

			FinalInterval interval;
			String[] dirFile;
			GenericDialogPlus gdWait;

			switch ( action )
			{
				case SEGMENT_OBJECTS:
					cats.segmentObjects();
					break;
				case REVIEW_OBJECTS:
					cats.reviewObjects();
					break;
				case CREATE_OBJECTS_IMAGE:
					createObjectsImage();
					break;
				case REVIEW_LABELS:
					overlays.reviewLabelsInRoiManagerUI( listeners );
					break;
				case CHANGE_CLASSIFIER_SETTINGS:
					cats.showClassifierSettingsDialog();
					break;
				case CHANGE_FEATURE_SETTINGS:
					boolean settingsChanged =
							cats.featureSettingsDialog( false );
					if ( settingsChanged ) saveLabelInstances();
					break;
				case CHANGE_ADVANCED_FEATURE_SETTINGS:
					settingsChanged =
							cats.featureSettingsDialog( true );
					if ( settingsChanged ) saveLabelInstances();
					break;
				case IO_LOAD_CLASSIFIER:
					dirFile = getOpenDirFile( "Please choose a classifier file" );
					cats.loadClassifier( dirFile[ 0 ], dirFile[ 1 ] );
					break;
				case ADD_CLASS:
					String inputName = IOUtils.classNameDialog();
					if ( inputName == null ) return;
					cats.addClass( inputName );
					labelButtonsPanel.updateButtons();
					break;
				case CHANGE_CLASS_NAMES:
					cats.changeClassNamesDialog();
					labelButtonsPanel.updateButtons();
					break;
				case CHANGE_RESULT_OVERLAY_OPACITY:
					//showResultsOverlayOpacityDialog();
					break;
				case CHANGE_COLORS:
					overlays.changeClassColorViaGUI( labelButtonsPanel );
					break;
				case IO_SAVE_CLASSIFIER:
					dirFile = getSaveDirFile( "Please choose output file", ".classifier" );
					cats.saveClassifier( dirFile[ 0 ], dirFile[ 1 ] );
					break;
				case STOP_CLASSIFICATION:

					cats.stopCurrentTasks = true;

					String dotDotDot = "...";

					while ( cats.isBusy )
					{
						logger.progress( "Waiting for tasks to finish", dotDotDot );
						dotDotDot += ".";
						try { Thread.sleep( 3000 ); }
						catch ( InterruptedException e ) { e.printStackTrace(); };
					}

					logger.info("...all tasks finished.");

					cats.stopCurrentTasks = false;
					break;

				case IO_LOAD_LABEL_INSTANCES:
					loadLabelInstances();
					break;

				case CHANGE_DEBUG_SETTINGS:
					//showDebugSettingsDialog();
					break;

				case IO_EXPORT_RESULT_IMAGE:
					ResultImageExportGUI.showExportGUI(
							cats.getInputImage().getTitle(),
							cats.getResultImage(),
							cats.getInputImage(),
							cats.getClassNames() );
					break;

				case APPLY_CLASSIFIER_ON_SLURM:

					if ( ! isSelectionFullWidthAndHeight() )
					{
						logger.error( "Classification on cluster is only " +
                                "possible when the whole xy range of the image " +
                                "is selected. " +
								"You could use Ctrl+A on the image to do so. " +
                                "Or choose to classify the whole data set." );
						break;
					}

					dirFile = getSaveDirFile( "Save classifier", ".classifier" );
					cats.saveClassifier( dirFile[ 0 ], dirFile[ 1 ] );
					cats.applyClassifierOnSlurm( getIntervalFromUI() );
					break;

				case UPDATE_LABEL_INSTANCES:
					updateLabelInstances( );
					break;

				case TRAIN_CLASSIFIER:
					trainClassifier();
					break;

                //                    case APPLY_BG_FG_CLASSIFIER:
//                        //applyBgFgClassification();
//                        break;
//                    case TRAIN_FROM_LABEL_IMAGE:
//                        //trainFromLabelImage();
//                        break;
//                    case GET_LABEL_IMAGE_TRAINING_ACCURACIES:
//                        //computeLabelImageBasedAccuracies();
//                        break;
//                    case UPDATE_LABELS:
//                        //updateLabelsTrainingData();
//                        break;
//                    case IO_SAVE_LABELS:
//                        dirFile = getSaveDirFile( "Save file", instancesFilename, ".ARFF" );
//                        instancesFilename = dirFile[ 1 ];
//                        gdWait = showWaitDialog( "Saving...\nPlease wait until this window disappears!" );
//                        cats.saveInstances( inputImage.getTitle(), dirFile[ 0 ], dirFile[ 1 ] );
//                        gdWait.dispose();
//                        break;
//                    case IO_LOAD_LABEL_IMAGE:
//                        //loadLabelImage();
//                        break;
//                    case RECOMPUTE_LABEL_FEATURE_VALUES:
//                        //recomputeLabelFeaturesAndRetrainClassifier();
//                        break;
			}
		} );

        thread.start();
    }

    private void applyClassifier()
    {
        if ( cats.hasClassifier() )
		{
			cats.applyClassifierWithTiling( getIntervalFromUI() );
			overlays.showProbabilities();
		}
		else
		{
			IJ.showMessage("Please train a classifier first." );
		}
    }

    private boolean isSelectionFullWidthAndHeight()
    {
        final FinalInterval selectedInterval = getIntervalFromUI();
        final FinalInterval imageInterval = IntervalUtils.getInterval( cats.getInputImage() );
        boolean fullWidthAndHeight = true;
        for ( int d = 0; d < 2; ++d )
		{
			if ( selectedInterval.min(d) != imageInterval.min( d ) ) fullWidthAndHeight = false;
			if ( selectedInterval.max(d) != imageInterval.max( d ) ) fullWidthAndHeight = false;
		}
        return fullWidthAndHeight;
    }

    private void loadLabelInstances()
    {
        String[] dirFile;
        dirFile = IOUtils.getOpenDirFile( "Please choose instances file" );
        final String key = cats.loadInstancesAndMetadata( dirFile[ 0 ], dirFile[ 1 ] );
        labelButtonsPanel.updateButtons();
        if ( key.equals( inputImage.getTitle() ) )
		{
			labelButtonsPanel.setLabellingInformations();
		}
    }

    private void updateLabelInstances( )
    {
        cats.updateLabelInstances();

        labelButtonsPanel.setLabellingInformations();

		saveLabelInstances();
    }

	private void saveLabelInstances()
	{
		String[] dirFile = IOUtils.getSaveDirFile( "Save instances...", inputImage.getTitle() + ARFF, ARFF );

		if ( dirFile != null )
		{
			GenericDialogPlus gdWait;
			gdWait = showWaitDialog( "I/O operation in progress...\nPlease wait until this window disappears!" );
			cats.saveInstances( inputImage.getTitle(), dirFile[ 0 ], dirFile[ 1 ] );
			gdWait.dispose();
		}
	}

	private void trainAndSaveClassifier()
    {
        String[] dirFile;
        String classifierFilename = trainClassifier();

        if ( classifierFilename == null ) return;

        dirFile = getSaveDirFile( "Save classifier...", classifierFilename, ".classifier" );

        if ( dirFile != null )
		{
			cats.saveClassifier( dirFile[ 0 ], dirFile[ 1 ] );
		}
    }

    boolean[] instancesSelection;

    private String trainClassifier()
    {
        final Set< String > keySet = cats.getInstancesManager().getKeys();
        String[] keys = keySet.toArray( new String[keySet.size()] );
        final int numInstances = keys.length;

        if ( numInstances == 0 )
		{
			logger.error( "Please create or load some instances first!" );
            return null;
		}
		else if ( numInstances == 1 )
		{
			final InstancesAndMetadata instancesAndMetadata = cats.getInstancesManager().getInstancesAndMetadata( keys[ 0 ] );
			cats.trainClassifierWithFeatureSelection( instancesAndMetadata );
		}
		else
		{
			final GenericDialog gd = new GenericDialog( "Instances selection" );

			if ( instancesSelection == null || instancesSelection.length != numInstances )
			{
				instancesSelection = new boolean[ numInstances ];
			}

			gd.addCheckboxGroup( numInstances, 1, keys, instancesSelection );
			gd.showDialog();

			if ( gd.wasCanceled() ) return null;

			ArrayList< String > selectedInstances = new ArrayList<>(  );

			for( int i = 0; i < numInstances; ++i )
			{
				if ( gd.getNextBoolean() )
				{
					selectedInstances.add( keys[ i ] );
					instancesSelection[ i ] = true;
				}
				else
				{
					instancesSelection[ i ] = false;
				}
			}

			if ( selectedInstances.size() == 0 )
            {
                return null;
            }
            else if ( selectedInstances.size() == 1 )
            {
                String key = selectedInstances.get( 0 );
                final InstancesAndMetadata instancesAndMetadata = cats.getInstancesManager().getInstancesAndMetadata( selectedInstances.get( 0 ) );
                cats.trainClassifierWithFeatureSelection( instancesAndMetadata );
                return key + ".classifier";
            }
            else
            {
                final InstancesAndMetadata combinedInstancesAndMetadata = cats.getInstancesManager().getCombinedInstancesAndMetadata( selectedInstances );
                cats.trainClassifierWithFeatureSelection( combinedInstancesAndMetadata );
                return "combined.classifier";
            }
		}

		return null;
    }

    private void updateLabelsAndTrainClassifier()
    {
        updateLabelInstances();
        trainClassifier();
    }

    private void createObjectsImage()
    {
        final ImagePlus objectsImage = cats.getObjectReview().getObjectsInRoiManagerAsImage();
        objectsImage.setTitle( "objects" );
        objectsImage.show();
    }

    private GenericDialogPlus showWaitDialog( String text )
    {
        GenericDialogPlus gd = new GenericDialogPlus( "Save labels"  );
        gd.setModal( false );
        gd.hideCancelButton();
        gd.addMessage(  text );
        gd.showDialog();
        return gd;
    }

    private FinalInterval getIntervalFromUI()
    {
        if ( range.equals( ClassificationRangeUtils.WHOLE_DATA_SET) )
        {
            return ( IntervalUtils.getIntervalWithChannelsDimensionAsSingleton( cats.getInputImage() ) );
        }
        else
        {
            return ( ClassificationRangeUtils.getIntervalFromRoi( inputImage, range ) );
        }
    }


}
