package de.embl.cba.cats.labels;

import de.embl.cba.cats.CATS;
import de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin;
import de.embl.cba.cats.ui.Overlays;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.Overlay;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.plugin.Duplicator;
import ij.plugin.frame.RoiManager;

import java.awt.*;
import java.util.*;

public class LabelReview
{

    DeepSegmentationIJ1Plugin catsIJ1Plugin;
    CATS CATS;

    ImagePlus inputImage;

    ImagePlus imageAroundCurrentSelection;

    Map < String, Label > labelMap;
    ArrayList < String > underReview;

    RoiManager manager = null;

    private final static String KEY = "key";

    public LabelReview( DeepSegmentationIJ1Plugin catsIJ1Plugin )
    {
        this.catsIJ1Plugin = catsIJ1Plugin;
        this.inputImage = catsIJ1Plugin.inputImage;
    }

    public LabelReview( ArrayList< Label > labelMap, CATS CATS )
    {
        this.inputImage = CATS.getInputImage();
        setLabelsAsMap( labelMap );

    }

    public void setLabelsAsMap( ArrayList< Label > labels )
    {

        this.labelMap = new LinkedHashMap<>();

        for ( int i = 0; i < labels.size(); ++i )
        {
            Label label = labels.get( i );
            String key = "c" + label.classNum
                    + "-t" + ( label.t+1)
                    + "-z" + ( label.z+1)
                    + "-i" + i;
            this.labelMap.put( key, label );
        }
    }

    public void updateLabelsAccordingToWhatHasBeenRemovedInRoiManager(  )
    {
        updateLabelsAccordingToWhatHasBeenRemovedInRoiManager( manager );
    }

    public void updateLabelsAccordingToWhatHasBeenRemovedInRoiManager( RoiManager roiManager )
    {
        ArrayList< String > approved = getKeysFromRoiManager( roiManager );
        ArrayList< String > rejected = getRejectedKeys( underReview, approved );
        removeRejectedLabels ( rejected );
    }

    public ArrayList< Label > getApprovedLabelList( RoiManager roiManager )
    {
        updateLabelsAccordingToWhatHasBeenRemovedInRoiManager( roiManager );

        ArrayList< Label > approvedLabels = new ArrayList<>( labelMap.values() );

        return ( approvedLabels );
    }

    public ArrayList< Label > getLabelList()
    {
        ArrayList< Label > labelArrayList = new ArrayList<>(  labelMap.values() );

        return ( labelArrayList );
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

    public void reviewLabelsInRoiManager( int classNum, String order )
    {

        ArrayList< Roi > rois = getRoisFromLabels( classNum, order );

        manager = new RoiManager();

        Overlays overlays = new Overlays( CATS );

        for ( Roi roi : rois )
        {
            overlays.addRoiToRoiManager( manager, inputImage, roi );
        }

        setLabelsCurrentlyBeingReviewed( rois );

        overlays.zoomInOnRois( true );
        overlays.cleanUpOverlaysAndRoisWhenRoiManagerIsClosed( manager );

    }

    public void setLabelsCurrentlyBeingReviewed( ArrayList< Roi > rois )
    {
        underReview = new ArrayList<>();

        for ( Roi roi : rois )
        {
            underReview.add ( roi.getProperty( KEY ) ) ;
        }
    }

    private void removeRejectedLabels ( ArrayList< String > rejected )
    {
        for ( String key : rejected )
        {
            labelMap.remove( key );
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

    public final static String ORDER_Z = "z-position";
    public final static String ORDER_TIME_ADDED = "time added";


    public ArrayList< Roi > getRoisFromLabels(  int classNum, String ordering )
    {
        ArrayList< Roi > rois = new ArrayList<>();

        final Set< String > keys = labelMap.keySet();
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
                    int z = Integer.parseInt( s.split( "z" )[ 1 ].split( "-" )[ 0 ] );
                    return z;
                }
            } );
        }

        for ( String key : keysRequestedClass )
        {
            Roi roi = getRoiFromLabel( labelMap.get( key ) );
            roi.setProperty( KEY, key );
            roi.setName( key );
            roi.setProperty( Overlays.REVIEW, "" );
            rois.add( roi );
        }

        return rois;
    }

    private Roi getRoiFromLabel( Label label )
    {
        float[] x = new float[ label.points.length];
        float[] y = new float[ label.points.length];
        for ( int iPoint = 0; iPoint < label.points.length; iPoint++)
        {
            x[iPoint] = (float) label.points[iPoint].getX();
            y[iPoint] = (float) label.points[iPoint].getY();
        }
        Roi roi = new PolygonRoi(x, y, PolygonRoi.FREELINE);

        roi.setStrokeWidth((double) label.strokeWidth);

        if ( inputImage.isHyperStack() )
        {
            roi.setPosition( 1, label.z + 1, label.t + 1 );
        }
        else
        {
            roi.setPosition( label.z + 1 );
        }

        roi.setProperty( "classNum", "" + label.classNum );

        return ( roi );
    }

    public void close()
    {
        manager.close();
    }

    private static int getMargin( Roi roi )
    {
        return (int) ( Math.max( roi.getBounds().width, roi.getBounds().height ) * 2 );
    }

    private void showImageAroundRoi( Roi roi, int margin )
    {
        if ( ! imageAroundCurrentSelection.isVisible() )
        {
            imageAroundCurrentSelection.show();
        }
        imageAroundCurrentSelection.updateAndDraw();

        Roi zoomedInROI = (Roi) roi.clone();
        zoomedInROI.setLocation( margin, margin );
        imageAroundCurrentSelection.setRoi( zoomedInROI );
    }

    private void setImageAroundRoi( ImagePlus trainingImage,  Roi roi, int margin )
    {
        Rectangle r = roi.getBounds();

        Roi rectangleRoi = new Roi( r.x - margin, r.y - margin, r.width + 2 * margin, r.height + 2 * margin  );

        trainingImage.setRoi( rectangleRoi );

        Duplicator duplicator = new Duplicator();

        if ( imageAroundCurrentSelection != null )
        {
            imageAroundCurrentSelection.close();
        }

        imageAroundCurrentSelection = duplicator.run( trainingImage, trainingImage.getC(), trainingImage.getC(), trainingImage.getZ(), trainingImage.getZ(), trainingImage.getT(), trainingImage.getT() );

        trainingImage.setRoi( roi );


    }




}
