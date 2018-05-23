package de.embl.cba.trainableDeepSegmentation.postprocessing;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.labels.LabelManager;
import de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin;
import fiji.util.gui.GenericDialogPlus;
import ij.gui.GenericDialog;
import ij.gui.OvalRoi;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;

import java.util.ArrayList;

public class ObjectsReview
{
    RoiManager manager;
    DeepSegmentation deepSegmentation;
    DeepSegmentationIJ1Plugin deepSegmentationIJ1Plugin;
    String objectsName;


    public ObjectsReview( DeepSegmentation deepSegmentation,
                          DeepSegmentationIJ1Plugin deepSegmentationIJ1Plugin)
    {
        this.deepSegmentation = deepSegmentation;
        this.deepSegmentationIJ1Plugin = deepSegmentationIJ1Plugin;
    }

    public void runUI( )
    {
        GenericDialog gd = openGenericDialog();
        if ( gd == null ) return;
        setSettingsFromUI( gd );

        reviewObjectsUsingRoiManager( deepSegmentation.getSegmentedObjectsMap().get( objectsName ) );
    }

    private GenericDialog openGenericDialog()
    {
        GenericDialog gd = new GenericDialogPlus("Objects Review");

        gd.addChoice( "Objects", deepSegmentation.getSegmentedObjectsNames().toArray( new String[0] ), deepSegmentation.getClassNames().get( 0 ) );

        gd.showDialog();

        if ( gd.wasCanceled() ) return null;
        return gd;
    }

    private void setSettingsFromUI( GenericDialog gd )
    {
        objectsName = gd.getNextChoice();
    }


    public void reviewObjectsUsingRoiManager( SegmentedObjects objects )
    {

        deepSegmentationIJ1Plugin.reviewRoisFlag = true;

        ArrayList< Roi > rois = getCentroidRoisFromObjects( objects );

        deepSegmentationIJ1Plugin.makeTrainingImageTheActiveWindow();

        manager = new RoiManager();

        for ( Roi roi : rois )
        {
            LabelManager.addRoiToManager( manager, deepSegmentation.getInputImage(), roi );
        }

        DeepSegmentationIJ1Plugin.configureRoiManagerClosingEventListener( manager, deepSegmentationIJ1Plugin );
    }


    public static ArrayList< Roi > getCentroidRoisFromObjects( SegmentedObjects objects )
    {

        ArrayList< double[] > centroids = objects.objects3DPopulation.getMeasureCentroid();

        ArrayList< Roi > rois = new ArrayList<>();

        double scaleXY = objects.objects3DPopulation.getScaleXY();
        double scaleZ = objects.objects3DPopulation.getScaleZ();

        for ( double[] centroid : centroids )
        {
            Roi roi = new PointRoi( centroid[ 1 ] * scaleXY, centroid[ 2 ] * scaleXY );
            roi.setPosition( 1, (int) ( centroid[ 3 ] * scaleZ ) + 1, objects.t + 1 );
            rois.add( roi );
        }

        return rois;
    }


    public static ArrayList< Roi > getOvalRoisFromObjects( SegmentedObjects objects )
    {
        double width = 10.0;

        ArrayList< double[] > centroids = objects.objects3DPopulation.getMeasureCentroid();

        ArrayList< Roi > rois = new ArrayList<>();

        for ( double[] centroid : centroids )
        {
            Roi roi = new OvalRoi( centroid[ 1 ], centroid[ 2 ], width, width );
            roi.setPosition( 1, (int) centroid[ 3 ] + 1, objects.t + 1 );
            rois.add( roi );
        }

        return rois;
    }


}
