package trainableDeepSegmentation;

import ij.gui.PolygonRoi;
import ij.gui.Roi;

import java.awt.*;
import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by tischi on 29/05/17.
 */
public class Example implements Serializable {

    public int classNum;
    public Point[] points = null;
    public int strokeWidth = 0;
    public int z; // zero-based
    public int t; // zero-based
    public ArrayList<double[]> instanceValuesArray = null;
    public int maximumFeatureScale = 0;
    public boolean[] enabledFeatures = null;
    String[] classNames = null;
    ArrayList<String> featureNames;

    public Example(int classNum, Point[] points, int strokeWidth, int z, int t,
                   boolean[] enabledFeatures, int maximumFeatureScale,
                   String[] classNames)
    {
        this.classNum = classNum;
        this.points = points;
        this.strokeWidth = strokeWidth;
        this.t = t;
        this.z = z;
        this.enabledFeatures = enabledFeatures;
        this.maximumFeatureScale = maximumFeatureScale;
        this.classNames = classNames;
    }

    public Rectangle getBounds()
    {
        int xMin = points[0].x;
        int xMax = points[0].x;
        int yMin = points[0].y;
        int yMax = points[0].y;

        for ( Point point : points  )
        {
            xMin = point.x < xMin ? point.x : xMin;
            xMax = point.x > xMax ? point.x : xMax;
            yMin = point.y < yMin ? point.y : yMin;
            yMax = point.y > yMax ? point.y : yMax;
        }

        xMin -= strokeWidth;
        xMax += strokeWidth;
        yMin -= strokeWidth;
        yMax += strokeWidth;

        return ( new Rectangle( xMin, yMin, xMax-xMin+1, yMax-yMin+1 ) );
    }

}
