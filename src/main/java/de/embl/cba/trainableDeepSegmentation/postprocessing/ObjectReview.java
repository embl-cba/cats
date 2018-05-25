package de.embl.cba.trainableDeepSegmentation.postprocessing;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.labels.LabelReviewManager;
import de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin;
import fiji.util.gui.GenericDialogPlus;
import ij.gui.GenericDialog;
import ij.gui.OvalRoi;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import mcib3d.geom.Object3D;

import java.util.ArrayList;

public class ObjectReview
{
    RoiManager manager;
    DeepSegmentation deepSegmentation;
    DeepSegmentationIJ1Plugin deepSegmentationIJ1Plugin;

    String objectsName;
    public int minVolumeInPixels;


    public ObjectReview( DeepSegmentation deepSegmentation )
    {
        this.deepSegmentation = deepSegmentation;
        this.deepSegmentationIJ1Plugin = deepSegmentation.deepSegmentationIJ1Plugin;
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

        gd.addNumericField( "Minimum number of voxels ", 10, 0);

        gd.showDialog();

        if ( gd.wasCanceled() ) return null;
        return gd;
    }

    private void setSettingsFromUI( GenericDialog gd )
    {

        objectsName = gd.getNextChoice();

        minVolumeInPixels = (int) gd.getNextNumber();

    }


    public void reviewObjectsUsingRoiManager( SegmentedObjects objects )
    {

        ArrayList< Roi > rois = getCentroidRoisFromObjects( objects );

        deepSegmentationIJ1Plugin.makeTrainingImageTheActiveWindow();

        manager = new RoiManager();

        for ( Roi roi : rois )
        {
            LabelReviewManager.addRoiToManager( manager, deepSegmentation.getInputImage(), roi );
        }

        deepSegmentation.logger.info( "\nReviewing objects: " + rois.size() );

        DeepSegmentationIJ1Plugin.configureRoiManagerClosingEventListener( manager, deepSegmentationIJ1Plugin );
    }


    public ArrayList< Roi > getCentroidRoisFromObjects( SegmentedObjects objects )
    {

        ArrayList< Object3D > objects3D = objects.objects3DPopulation.getObjectsList();

        ArrayList< Roi > rois = new ArrayList<>();

        double scaleXY = objects.objects3DPopulation.getScaleXY();
        double scaleZ = objects.objects3DPopulation.getScaleZ();

        for ( Object3D object3D : objects3D )
        {
            if ( object3D.getVolumePixels() * scaleXY * scaleZ > minVolumeInPixels  )
            {
                Roi roi = new PointRoi( object3D.getCenterX() * scaleXY, object3D.getCenterY() * scaleXY );
                roi.setPosition( 1, ( int ) ( object3D.getCenterZ() * scaleZ ) + 1, objects.t + 1 );
                rois.add( roi );
            }
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
