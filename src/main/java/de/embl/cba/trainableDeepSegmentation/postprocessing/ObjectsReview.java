package de.embl.cba.trainableDeepSegmentation.postprocessing;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.labels.LabelManager;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
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

    public ObjectsReview( DeepSegmentation deepSegmentation )
    {
        this.deepSegmentation = deepSegmentation;
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

}
