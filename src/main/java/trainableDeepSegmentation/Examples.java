package trainableDeepSegmentation;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by tischi on 01/09/17.
 */
public class Examples implements Serializable {
    ArrayList< Example > exampleList = null;
    ArrayList< Feature > featureList = null;
    public int maxResolutionLevel = 0;
    public boolean[] enabledFeatures = null;
    String[] classNames = null;

}
