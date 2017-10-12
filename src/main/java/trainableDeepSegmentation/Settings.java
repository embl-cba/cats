package trainableDeepSegmentation;

import java.io.Serializable;
import java.util.ArrayList;

import static trainableDeepSegmentation.WekaSegmentation.MAX_NUM_CLASSES;

public class Settings implements Serializable {

    public double anisotropy = 1.0;

    public int maxResolutionLevel = 3; // 3

    public int downSamplingFactor = 3; // 3

    public int maxDeepConvolutionLevel = 3; // 3

    public double memoryFactor = 10;

    public int backgroundThreshold = 0; // gray-values

    public ArrayList< String > activeFeatureNames = null;

    /** List of all possible features for current settings,
     * including active as well as inactive features.
     * It is necessary to keep this list, because the training
     * annotations always have all features, because the
     * active features could change each training and one does not
     * want to recompute the features for the training data each time.*/
    public ArrayList< Feature > featureList = null;

    public ArrayList< Integer > activeChannels = null;

    public ArrayList < String > classNames = new ArrayList<>();


}
