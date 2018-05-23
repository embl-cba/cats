package de.embl.cba.trainableDeepSegmentation.postprocessing;

import de.embl.cba.bigDataTools.utils.Utils;
import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.labels.LabelManager;
import fiji.util.gui.GenericDialogPlus;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.OvalRoi;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import mcib3d.geom.Objects3DPopulation;
import mcib3d.image3d.ImageShort;
import mcib_plugins.tools.RoiManager3D_2;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Set;

public class ObjectsReview
{
    RoiManager manager;
    DeepSegmentation deepSegmentation;
    int objectsId;

    public ObjectsReview( DeepSegmentation deepSegmentation )
    {
        this.deepSegmentation = deepSegmentation;
    }

    public void runUI( )
    {
        GenericDialog gd = openGenericDialog();
        if ( gd == null ) return;
        setSettingsFromUI( gd );

        deepSegmentation.makeInputImageTheActiveWindow();

        reviewObjectsUsingRoiManager( deepSegmentation.getSegmentedObjectsList().get( objectsId ) );
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
        objectsId = gd.getNextChoiceIndex();
    }


    public void reviewObjectsUsingRoiManager( SegmentedObjects objects )
    {
        ArrayList< Roi > rois = getCentroidRoisFromObjects( objects );

        manager = new RoiManager();

        for ( Roi roi : rois )
        {
            LabelManager.addRoiToManager( manager, deepSegmentation.getInputImage(), roi );
        }
    }


    public static ArrayList< Roi > getCentroidRoisFromObjects( SegmentedObjects objects )
    {
        ArrayList< double[] > centroids = objects.objects3DPopulation.getMeasureCentroid();

        ArrayList< Roi > rois = new ArrayList<>();

        for ( double[] centroid : centroids )
        {
            Roi roi = new PointRoi( centroid[ 1 ], centroid[ 2 ] );
            roi.setPosition( 1, (int) centroid[ 3 ] + 1, objects.t + 1 );
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
