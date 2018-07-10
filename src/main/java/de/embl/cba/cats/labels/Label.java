package de.embl.cba.cats.labels;

import java.awt.*;
import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by de.embl.cba.cats.weka on 29/05/17.
 */
public class Label implements Serializable {

    public int classNum;
    public Point[] points = null;
    public int strokeWidth = 0;
    public int z; // zero-based
    public int t; // zero-based
    public ArrayList< ArrayList< double[] > > instanceValuesArrays = null;
    public boolean instanceValuesAreCurrentlyBeingComputed = false;

    public Label( int classNum, Point[] points, int strokeWidth, int z, int t )
    {
        this.classNum = classNum;
        this.points = points;
        this.strokeWidth = strokeWidth;
        this.t = t;
        this.z = z;
    }

}
