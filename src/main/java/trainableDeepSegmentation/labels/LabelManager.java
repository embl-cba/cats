package trainableDeepSegmentation.labels;

import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import trainableDeepSegmentation.examples.Example;

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

        this.examples = new LinkedHashMap<>();

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

    public String showOrderGUI()
    {
        GenericDialog gd = new GenericDialog("Label ordering");
        gd.addChoice( "Ordering of labels:",
                new String[]{ ORDER_TIME_ADDED, ORDER_Z}, ORDER_TIME_ADDED);
        gd.showDialog();
        if(gd.wasCanceled())
        {
            return null;
        }
        else
        {
            return gd.getNextChoice();
        }

    }

    public void reviewLabelsInRoiManager( int classNum,
                                          String order)
    {

        ArrayList< Roi > rois = getRoisFromExamples( classNum, order );
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

    final static String ORDER_Z = "z-position";
    final static String ORDER_TIME_ADDED = "time added";

    private ArrayList< Roi > getRoisFromExamples( int classNum,
                                                  String ordering)
    {
        ArrayList< Roi > rois = new ArrayList<>();

        final Set< String > keys = examples.keySet();
        final ArrayList< String > keysRequestedClass = new ArrayList<>();
        final ArrayList< Integer > zPositions = new ArrayList<>();

        for ( String key : keys )
        {
            if ( key.contains( "c"+ classNum + "-" ) )
            {
                keysRequestedClass.add( key );
            }
        }

        if ( ordering == ORDER_Z )
        {

            Collections.sort( keysRequestedClass, new Comparator< String >() {
                public int compare( String o1, String o2 )
                {
                    return extractInt( o1 ) - extractInt( o2 );
                }

                int extractInt( String s )
                {
                    int z = Integer.parseInt(
                            s.split( "z" )[ 1 ].split( "-" )[ 0 ] );
                    return z;
                }
            } );
        }

        for ( String key : keysRequestedClass )
        {
            Roi roi = getRoiFromExample( examples.get( key ) );
            roi.setProperty( KEY, key );
            roi.setName( key );
            rois.add( roi );
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
