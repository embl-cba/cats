package de.embl.cba.trainableDeepSegmentation.ui;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import ij.IJ;
import ij.ImagePlus;
import org.scijava.command.Command;
import org.scijava.command.Interactive;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.widget.Button;

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
    public String actionInput = ADD_CLASS;


    private DeepSegmentation deepSegmentation;

    @Override
    public void run()
    {

    }

    public void init()
    {
        deepSegmentation.setInputImage( inputImage );

        deepSegmentation.initialisationDialog();

        deepSegmentation.featureSettingsDialog( false );

        Overlays overlays = new Overlays( deepSegmentation.classColors, deepSegmentation.getInputImage(), deepSegmentation.getResultImage() );

        Listeners listeners = new Listeners( inputImage, overlays );

        DeepSegmentation.reserveKeyboardShortcuts();

    }

    protected void performAction()
    {
        IJ.log( actionInput );
    }


}
