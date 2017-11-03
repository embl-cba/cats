package trainableDeepSegmentation;

import ij.ImagePlus;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;

import java.awt.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class LabelManager {

    ImagePlus imp;

    ArrayList< Example > examples;
    Map < String, Example > exampleMap;


    public LabelManager ( ImagePlus imp )
    {
        this.imp = imp;
    }

    public void setExamples( ArrayList< Example > examples )
    {

        this.examples = examples;

        exampleMap = new HashMap<>();

        for ( int i = 0; i < examples.size(); ++i )
        {
            Example example = examples.get( i );
            String key = "c" + example.classNum
                    + "-t" + (example.t+1)
                    + "-z" + (example.z+1)
                    + "-i" + i;
            exampleMap.put( key, example );
        }
    }

    public void showExamplesInRoiManager( int classNum )
    {
        ArrayList< Roi > rois = getRoisFromExamples( classNum );

        RoiManager manager = new RoiManager();

        for ( Roi roi : rois )
        {
            int n = imp.getStackIndex(  roi.getCPosition(), roi.getZPosition(), roi.getTPosition());
            imp.setSliceWithoutUpdate( n );
            manager.addRoi( roi );
        }

    }

    private ArrayList< Roi > getRoisFromExamples( int classNum )
    {
        ArrayList< Roi > rois = new ArrayList<>();

        final Set< String > strings = exampleMap.keySet();

        for ( String key : strings )
        {
            if ( key.contains( "c"+ classNum + "-" ) )
            {
                Roi roi = getRoiFromExample( exampleMap.get( key ) );
                roi.setProperty( "key", key );
                roi.setName( key );
                rois.add( roi );
            }
        }

        return rois;
    }

    private Roi getRoiFromExample( Example example )
    {
        float[] x = new float[example.points.length];
        float[] y = new float[example.points.length];
        for (int iPoint = 0; iPoint < example.points.length; iPoint++)
        {
            x[iPoint] = (float) example.points[iPoint].getX();
            y[iPoint] = (float) example.points[iPoint].getY();
        }
        Roi roi = new PolygonRoi(x, y, PolygonRoi.FREELINE);

        roi.setStrokeWidth((double) example.strokeWidth);
        roi.setPosition( 1, example.z + 1, example.t + 1 );
        roi.setProperty( "classNum", "" + example.classNum );

        return ( roi );
    }

}
