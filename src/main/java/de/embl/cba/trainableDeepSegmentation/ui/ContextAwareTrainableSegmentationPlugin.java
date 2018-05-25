package de.embl.cba.trainableDeepSegmentation.ui;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.classification.ClassificationRangeUtils;
import de.embl.cba.trainableDeepSegmentation.results.ResultImageExportGUI;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils;
import ij.IJ;
import ij.ImagePlus;
import net.imglib2.FinalInterval;
import org.scijava.command.Command;
import org.scijava.command.Interactive;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.Button;

import static de.embl.cba.trainableDeepSegmentation.utils.IOUtils.getOpenDirFile;
import static de.embl.cba.trainableDeepSegmentation.utils.IOUtils.getSaveDirFile;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Development>CATS", initializer = "init")
public class ContextAwareTrainableSegmentationPlugin implements Command, Interactive
{
    @Parameter ( required = true )
    public ImagePlus inputImage;

    /** actions */
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


    @Parameter(label = "Perform Action", callback = "performAction")
    private Button performActionButton;

    @Parameter(label = "Actions", persist = false,
            choices = {
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
                    CHANGE_ADVANCED_FEATURE_COMPUTATION_SETTINGS } )
    private String actionInput = ADD_CLASS;

    @Parameter(label = "Range", persist = false,
        choices = { ClassificationRangeUtils.WHOLE_DATA_SET, ClassificationRangeUtils.SELECTION_PM10Z })
    private String range = ClassificationRangeUtils.SELECTION_PM10Z;

    private DeepSegmentation deepSegmentation;

    @Override
    public void run()
    {

    }

    public void init()
    {
        IJ.setTool("freeline");

        deepSegmentation = new DeepSegmentation(  );

        deepSegmentation.setInputImage( inputImage );

        deepSegmentation.initialisationDialog();

        deepSegmentation.featureSettingsDialog( false );

        Overlays overlays = new Overlays( deepSegmentation.classColors, deepSegmentation.getInputImage(), deepSegmentation.getResultImage(), deepSegmentation );

        Listeners listeners = new Listeners( inputImage, overlays, deepSegmentation );

        DeepSegmentation.reserveKeyboardShortcuts();

    }

    protected void performAction()
    {

        FinalInterval interval;
        String[] dirFile;

        switch ( actionInput )
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
                deepSegmentation.featureSettingsDialog( false );
                break;
            case CHANGE_ADVANCED_FEATURE_COMPUTATION_SETTINGS:
                deepSegmentation.featureSettingsDialog( true );
                break;
            case IO_LOAD_CLASSIFIER:
                dirFile = getOpenDirFile( "Please choose a classifier file" );
                deepSegmentation.loadClassifier( dirFile[ 0 ], dirFile[ 1 ] );
                break;
            case ADD_CLASS:
                String inputName = IOUtils.classNameDialog();
                if ( inputName == null ) return;
                deepSegmentation.addClass( inputName );
                break;
            case CHANGE_CLASS_NAMES:
                deepSegmentation.showClassNamesDialog();
                break;
            case CHANGE_RESULT_OVERLAY_OPACITY:
                //showResultsOverlayOpacityDialog();
                break;
            case CHANGE_COLORS:
                //showColorsDialog();
                break;
            case IO_SAVE_CLASSIFIER:
                dirFile = getSaveDirFile("Please choose a output file", ".classifier" );
                deepSegmentation.saveClassifier( dirFile[ 0 ], dirFile[ 1 ] );
                break;
            case IO_LOAD_INSTANCES:
                dirFile = IOUtils.getOpenDirFile( "Please choose instances file" );
                deepSegmentation.loadInstancesAndMetadata( dirFile[ 0 ], dirFile[ 1 ] );
                break;
            case IO_SAVE_INSTANCES:
                dirFile = IOUtils.getSaveDirFile( "Save instance file", ".ARFF" );
                deepSegmentation.saveInstances( inputImage.getTitle(), dirFile[0], dirFile[1] );
                break;
            case IO_LOAD_LABEL_IMAGE:
                //loadLabelImage();
                break;
            case RECOMPUTE_LABEL_FEATURE_VALUES:
                //recomputeLabelFeaturesAndRetrainClassifier();
                break;
            case CHANGE_DEBUG_SETTINGS:
                //showDebugSettingsDialog();
                break;
            case IO_EXPORT_RESULT_IMAGE:
                ResultImageExportGUI.showExportGUI(
                        deepSegmentation.getInputImage().getTitle(),
                        deepSegmentation.getResultImage(),
                        deepSegmentation.getInputImage(),
                        deepSegmentation.getClassNames() );
                break;
            case APPLY_CLASSIFIER:
                deepSegmentation.applyClassifierWithTiling( getIntervalFromUI() );
                break;
            case APPLY_CLASSIFIER_ON_SLURM:
                deepSegmentation.applyClassifierOnSlurm( getIntervalFromUI() );
                break;
            case APPLY_BG_FG_CLASSIFIER:
                //applyBgFgClassification();
                break;
            case TRAIN_FROM_LABEL_IMAGE:
                //trainFromLabelImage();
                break;
            case GET_LABEL_IMAGE_TRAINING_ACCURACIES:
                //computeLabelImageBasedAccuracies();
                break;
            case UPDATE_LABELS:
                //updateLabelsTrainingData();
                break;
            case UPDATE_LABELS_AND_TRAIN:
                deepSegmentation.updateLabelInstancesAndTrainClassifier();
                break;
            case DUPLICATE_RESULT_IMAGE_TO_RAM:
                ImagePlus imp = deepSegmentation.getResultImage().getWholeImageCopy();
                if ( imp != null ) imp.show();
                break;
            case TRAIN_CLASSIFIER:
                // getInstancesAndMetadata instances instances
                //InstancesAndMetadata instancesAndMetadata = getCombinedSelectedInstancesFromGUI();
                //if ( instancesAndMetadata == null )
                //{
                //    logger.error( "Please select one or multiple training instances." );
                //    return;
                //}
                //trainClassifier( instancesAndMetadata );
                break;
        }
    }


    private FinalInterval getIntervalFromUI()
    {
        if ( range.equals( ClassificationRangeUtils.WHOLE_DATA_SET) )
        {
            return ( IntervalUtils.getIntervalWithChannelsDimensionAsSingleton( deepSegmentation.getInputImage() ) );
        }
        else
        {
            return ( ClassificationRangeUtils.getIntervalFromRoi( inputImage, range ) );
        }
    }


}
