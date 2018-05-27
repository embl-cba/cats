package de.embl.cba.trainableDeepSegmentation.labels;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin;
import de.embl.cba.trainableDeepSegmentation.ui.Overlays;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.plugin.Duplicator;
import ij.plugin.frame.RoiManager;
import de.embl.cba.trainableDeepSegmentation.labels.examples.Example;

import java.awt.*;
import java.util.*;

public class LabelReviewManager
{

    DeepSegmentationIJ1Plugin deepSegmentationIJ1Plugin;
    DeepSegmentation deepSegmentation;

    ImagePlus inputImage;

    ImagePlus imageAroundCurrentSelection;

    Map < String, Example > labelMap;
    ArrayList < String > underReview;

    RoiManager manager = null;

    private final static String KEY = "key";

    public LabelReviewManager( DeepSegmentationIJ1Plugin deepSegmentationIJ1Plugin )
    {
        this.deepSegmentationIJ1Plugin = deepSegmentationIJ1Plugin;
        this.inputImage = deepSegmentationIJ1Plugin.inputImage;
    }

    public LabelReviewManager( ArrayList< Example > labelMap, DeepSegmentation deepSegmentation )
    {
        this.inputImage = deepSegmentation.getInputImage();
        setLabelsAsMap( labelMap );

    }

    public void setLabelsAsMap( ArrayList< Example > examples )
    {

        this.labelMap = new LinkedHashMap<>();

        for ( int i = 0; i < examples.size(); ++i )
        {
            Example example = examples.get( i );
            String key = "c" + example.classNum
                    + "-t" + (example.t+1)
                    + "-z" + (example.z+1)
                    + "-i" + i;
            this.labelMap.put( key, example );
        }
    }

    public void updateExamplesAccordingToWhatHasBeenRemovedInRoiManager(  )
    {
        updateExamplesAccordingToWhatHasBeenRemovedInRoiManager( manager );
    }

    public void updateExamplesAccordingToWhatHasBeenRemovedInRoiManager( RoiManager roiManager )
    {
        ArrayList< String > approved = getKeysFromRoiManager( roiManager );
        ArrayList< String > rejected = getRejectedKeys( underReview, approved );
        removeRejectedExamples ( rejected );
    }

    public ArrayList< Example > getApprovedLabelList( RoiManager roiManager )
    {
        updateExamplesAccordingToWhatHasBeenRemovedInRoiManager( roiManager );

        ArrayList< Example > approvedLabels = new ArrayList<>( labelMap.values() );

        return ( approvedLabels );
    }

    public ArrayList< Example > getExampleList()
    {
        ArrayList< Example > exampleArrayList = new ArrayList<>(  labelMap.values() );

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

    public void reviewLabelsInRoiManager( int classNum, String order )
    {

        ArrayList< Roi > rois = getRoisFromLabels( classNum, order );

        manager = new RoiManager();

        Overlays overlays = new Overlays( deepSegmentation );

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

   private void removeRejectedExamples ( ArrayList< String > rejected )
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
            roi.setName( key + "-" + Overlays.REVIEW );
            rois.add( roi );
        }

        return rois;
    }


    private Roi getRoiFromLabel( Example example )
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

        if ( inputImage.isHyperStack() )
        {
            roi.setPosition( 1, example.z + 1, example.t + 1 );
        }
        else
        {
            roi.setPosition( example.z + 1 );
        }

        roi.setProperty( "classNum", "" + example.classNum );

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
