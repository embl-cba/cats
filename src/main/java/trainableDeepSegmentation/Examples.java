package trainableDeepSegmentation;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by tischi on 01/09/17.
 */
public class Examples implements Serializable
{
    ArrayList< Example > exampleList = null;
    ArrayList< Feature > featureList = null;

    public double anisotropy = 1.0;
    public int maxResolutionLevel = 3;
    public int downSamplingFactor = 3;
    public int maxDeepConvolutionLevel = 3;

    public boolean[] enabledFeatures = null;
    String[] classNames = null;
}
