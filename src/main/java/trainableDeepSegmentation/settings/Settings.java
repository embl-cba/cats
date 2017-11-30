package trainableDeepSegmentation.settings;

import trainableDeepSegmentation.WekaSegmentation;

import java.io.Serializable;
import java.util.ArrayList;

import static trainableDeepSegmentation.WekaSegmentation.MAX_NUM_CLASSES;

public class Settings {

    public final static String ANISOTROPY = "anisotropy";
    public final static String BIN_FACTOR = "binFactor";
    public final static String MAX_BIN_LEVEL = "maxBinLevel";
    public final static String MAX_DEEP_CONV_LEVEL = "maxDeepConvLevel";

    public double anisotropy = 1.0;

    public int[] binFactors = new int[]{1,2,3,4,-1,-1,-1,-1,-1,-1,-1};

    public int maxDeepConvLevel = 3; // 3

    public int imageBackground = 0; // gray-values

    public boolean log2 = false;

    public String batchSizePercent = "66";

    public ArrayList< Integer > activeChannels = null;

    public ArrayList < String > classNames = new ArrayList<>();


}
