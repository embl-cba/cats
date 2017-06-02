package trainableSegmentation;

import ij.gui.Roi;

import java.io.Serializable;

/**
 * Created by tischi on 29/05/17.
 */
public class Example implements Serializable {

    public int classNum;
    public Roi roi;
    public int z;
    public int t;

    public Example(int classNum, Roi roi, int z, int t)
    {
        this.classNum = classNum;
        this.roi = roi;
        this.t = t;
        this.z = z;
    }

}
