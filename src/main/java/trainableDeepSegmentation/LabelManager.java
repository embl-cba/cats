package trainableDeepSegmentation;

import ij.ImagePlus;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;

import java.util.*;

public class LabelManager {

    ImagePlus imp;

    Map < String, Example > examples;
    ArrayList < String > underReview;

    RoiManager manager = null;

    private final static String KEY = "key";

    public LabelManager ( ImagePlus imp )
    {
        this.imp = imp;
    }

    public void setExamples( ArrayList< Example > examples )
    {

        this.examples = new HashMap<>();

        for ( int i = 0; i < examples.size(); ++i )
        {
            Example example = examples.get( i );
            String key = "c" + example.classNum
                    + "-t" + (example.t+1)
                    + "-z" + (example.z+1)
                    + "-i" + i;
            this.examples.put( key, example );
        }
    }

    public void updateExamples()
    {

        ArrayList< String > approved = getKeysFromRoiManager( manager );
        ArrayList< String > rejected = getRejectedKeys( underReview, approved );
        removeRejectedExamples ( rejected );
    }

    public ArrayList< Example > getExamples()
    {
        ArrayList< Example > exampleArrayList = new ArrayList<>(  examples.values() );

        return ( exampleArrayList );
    }

    public void reviewLabelsInRoiManager( int classNum )
    {
        ArrayList< Roi > rois = getRoisFromExamples( classNum );
        manager = new RoiManager();
        underReview = new ArrayList<>();

        for ( Roi roi : rois )
        {
            addRoiToManager( manager, imp, roi );
            underReview.add ( roi.getProperty( KEY ) ) ;
        }
    }

    private static void addRoiToManager( RoiManager manager, ImagePlus imp, Roi roi )
    {


        int nSave = imp.getSlice();
        int n = imp.getStackIndex(  roi.getCPosition(), roi.getZPosition(), roi.getTPosition());
        imp.setSliceWithoutUpdate( n );

        int tSave = 0, zSave = 0, cSave = 0;
        if ( imp.isHyperStack() )
        {
            tSave = imp.getT();
            zSave = imp.getZ();
            cSave = imp.getC();
            imp.setPositionWithoutUpdate( roi.getCPosition(), roi.getZPosition(), roi.getTPosition() );
        }

        manager.addRoi( roi );

        if ( imp.isHyperStack() )
        {
            imp.setPositionWithoutUpdate( cSave, zSave, tSave );
        }

        imp.setSliceWithoutUpdate( nSave );


    }


    private void removeRejectedExamples ( ArrayList< String > rejected )
    {
        for ( String key : rejected )
        {
            examples.remove( key );
        }
    }

    private static ArrayList< String > getRejectedKeys( ArrayList< String > underReview,
                                                              ArrayList< String > approved )
    {
        ArrayList< String > rejected = new ArrayList<>();

        for ( String key : underReview )
        {
            if ( ! ( approved.contains( key )))
            {
                rejected.add( key );
            }
        }

        return ( rejected );
    }

    private static ArrayList< String > getKeysFromRoiManager( RoiManager manager )
    {
        Roi[] approvedRois = manager.getRoisAsArray();
        ArrayList< String > approved = getKeyListFromRois( approvedRois );
        return ( approved );
    }

    private static ArrayList< String > getKeyListFromRois( Roi[] rois )
    {
        ArrayList< String > keys = new ArrayList<>();

        for ( Roi roi : rois )
        {
            keys.add( roi.getProperty( KEY ) );
        }

        return ( keys );
    }

    private ArrayList< Roi > getRoisFromExamples( int classNum )
    {
        ArrayList< Roi > rois = new ArrayList<>();

        final Set< String > strings = examples.keySet();

        for ( String key : strings )
        {
            if ( key.contains( "c"+ classNum + "-" ) )
            {
                Roi roi = getRoiFromExample( examples.get( key ) );
                roi.setProperty( KEY, key );
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

    public void close()
    {
        manager.close();
    }

}
