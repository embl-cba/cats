package trainableDeepSegmentation;

import java.io.Serializable;
import java.util.ArrayList;

import static trainableDeepSegmentation.WekaSegmentation.MAX_NUM_CLASSES;

public class Settings implements Serializable {

    public final static String ANISOTROPY = "anisotropy";
    public final static String BIN_FACTOR = "binFactor";
    public final static String MAX_BIN_LEVEL = "maxBinLevel";
    public final static String MAX_DEEP_CONV_LEVEL = "maxDeepConvLevel";

    public double anisotropy = 1.0;

    public int maxBinLevel = 3; // 3

    public int binFactor = 3; // 3

    public int maxDeepConvLevel = 3; // 3

    public int imageBackground = 0; // gray-values

    public String batchSizePercent = "66";

    //public ArrayList< String > activeFeatureNames = null;

    /** List of all possible features for current settings,
     * including active as well as inactive features.
     * It is necessary to keep this list, because the instances
     * 
     * annotations always have all features, because the
     * active features could change each instances and one does not
     * want to recompute the features for the instances data each time.*/
    public ArrayList< Feature > featureList = null;

    public ArrayList< Integer > activeChannels = null;

    public ArrayList < String > classNames = new ArrayList<>();


}
