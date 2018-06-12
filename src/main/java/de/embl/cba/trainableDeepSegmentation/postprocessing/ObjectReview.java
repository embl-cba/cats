package de.embl.cba.trainableDeepSegmentation.postprocessing;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.ui.DeepSegmentationIJ1Plugin;
import de.embl.cba.trainableDeepSegmentation.ui.Overlays;
import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.OvalRoi;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import mcib3d.geom.Object3D;
import mcib3d.geom.Objects3DPopulation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

public class ObjectReview
{
    RoiManager roiManager;
    DeepSegmentation deepSegmentation;
    DeepSegmentationIJ1Plugin deepSegmentationIJ1Plugin;

    String objectsName;
    public double minCalibratedVolume;


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

        String volumeUnit = deepSegmentation.getInputImage().getCalibration().getUnit();
        gd.addNumericField( "Minimum volume [" + volumeUnit +"^3] ", 10, 0);

        gd.showDialog();

        if ( gd.wasCanceled() ) return null;
        return gd;
    }

    private void setSettingsFromUI( GenericDialog gd )
    {

        objectsName = gd.getNextChoice();

        minCalibratedVolume = gd.getNextNumber();

    }


    public void reviewObjectsUsingRoiManager( SegmentedObjects objects )
    {
        ArrayList< Roi > rois = getCentroidRoisFromObjects( objects, SORT_BY_VOLUME );

        makeImageTheActiveWindow( deepSegmentation.getInputImage() );

        roiManager = new RoiManager();

        Overlays overlays = new Overlays( deepSegmentation );

        for ( Roi roi : rois )
        {
            overlays.addRoiToRoiManager( roiManager, deepSegmentation.getInputImage(), roi );
        }

        deepSegmentation.logger.info( "\nReviewing objects: " + rois.size() );

        overlays.zoomInOnRois( true );
        overlays.cleanUpOverlaysAndRoisWhenRoiManagerIsClosed( roiManager );
    }

    private static void makeImageTheActiveWindow( ImagePlus imp )
    {
        sleep( 300 ); // otherwise below select window does not always work...

        IJ.selectWindow( imp.getID() );

        if ( ! imp.getWindow().isActive() )
        {
            sleep( 300 ); // otherwise below select window does not always work...
            IJ.selectWindow( imp.getID() );
        }
    }


    private static void sleep( long millis )
    {
        try
        {
            Thread.sleep( millis );
        }
        catch ( InterruptedException e )
        {
            e.printStackTrace();
        }
    }



    public static final String SORT_BY_VOLUME = "volume";

    public ArrayList< Roi > getCentroidRoisFromObjects( SegmentedObjects objects, String sorting )
    {

        TreeMap< Double, Object3D > sortedObjectMap = getVolumeSortedAndVolumeFilteredObjectMap( objects );

        ArrayList< Roi > rois = new ArrayList<>();

        double scaleXY = objects.objects3DPopulation.getScaleXY();
        double scaleZ = objects.objects3DPopulation.getScaleZ();

        for ( Map.Entry< Double, Object3D > object3D : sortedObjectMap.entrySet() )
        {

            double x = object3D.getValue().getCenterX() * scaleXY;
            double y = object3D.getValue().getCenterY() * scaleXY;
            double z = object3D.getValue().getCenterZ() * scaleZ;

            PointRoi roi = new PointRoi( x, y );

            if ( deepSegmentation.getInputImage().isHyperStack() )
            {
                roi.setPosition( 1, ( int ) ( z ) + 1, objects.t + 1 );
            }
            else
            {
                roi.setPosition( ( int ) ( z ) + 1 );
            }

            roi.setName( "" + object3D.getKey().intValue() + "-" + (int) x + "-" + (int) y + "-" + (int) z + "-" + Overlays.REVIEW );
            roi.setSize( 4 );

            rois.add( roi );

        }

        return rois;
    }


    private double getCalibratedVolume(  Object3D object3D, Objects3DPopulation objects3DPopulation, ImagePlus imp )
    {
        double pixelVolumeInBinnedImage = object3D.getVolumePixels();
        double scaleXY = objects3DPopulation.getScaleXY();
        double scaleZ = objects3DPopulation.getScaleZ();

        double pixelVolumeInOriginalImage = pixelVolumeInBinnedImage * scaleXY * scaleZ;

        double calibrationX = imp.getCalibration().pixelWidth;
        double calibrationY = imp.getCalibration().pixelHeight;
        double calibrationZ = imp.getCalibration().pixelDepth;

        double calibratedVolume = pixelVolumeInOriginalImage * calibrationX * calibrationY * calibrationZ;

        return calibratedVolume;

    }

    private TreeMap< Double, Object3D > getVolumeSortedAndVolumeFilteredObjectMap( SegmentedObjects objects )
    {
        ArrayList< Object3D > objects3D = objects.objects3DPopulation.getObjectsList();

        TreeMap< Double, Object3D > sortedObjectMap = new TreeMap<>( Collections.reverseOrder() );

        for ( Object3D object3D : objects3D )
        {
            double calibratedVolume = getCalibratedVolume( object3D, objects.objects3DPopulation, deepSegmentation.getInputImage() );

            if ( calibratedVolume > minCalibratedVolume )
            {
                sortedObjectMap.put( calibratedVolume, object3D );
            }
        }

        return sortedObjectMap;
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
