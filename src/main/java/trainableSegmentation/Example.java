package trainableSegmentation;

import ij.gui.Roi;
import weka.core.Instance;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by tischi on 29/05/17.
 */
public class Example implements Serializable {

    public int classNum;
    public Roi roi = null;
    public int z; // zero-based
    public int t; // zero-based
    public ArrayList<double[]> instanceValuesArray = null;
    public int maximumFeatureScale = 0;
    public boolean[] enabledFeatures = null;
    String[] classNames = null;

    public Example(int classNum, Roi roi, int z, int t,
                   boolean[] enabledFeatures, int maximumFeatureScale,
                   String[] classNames)
    {
        this.classNum = classNum;
        this.roi = roi;
        this.t = t;
        this.z = z;
        this.enabledFeatures = enabledFeatures;
        this.maximumFeatureScale = maximumFeatureScale;
        this.classNames = classNames;
    }

}
