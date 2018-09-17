package de.embl.cba.cats.objects;

import de.embl.cba.cats.CATS;
import de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin;
import de.embl.cba.cats.ui.Overlays;
import de.embl.cba.cats.utils.Utils;
import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.*;
import ij.plugin.frame.RoiManager;
import mcib3d.geom.Object3D;
import mcib3d.geom.Objects3DPopulation;
import mcib3d.geom.Objects3DPopulationAnalysis;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class ObjectReview
{
	public static final String OBJECT_ID = "ObjectId";
	RoiManager roiManager;
    CATS cats;
    DeepSegmentationIJ1Plugin catsIJ1Plugin;

    String objectsName;
    public double minCalibratedVolume;
    int zoomLevel;
	private ArrayList< Object3D > reviewedObjects;


	public ObjectReview( CATS cats )
    {
        this.cats = cats;
        this.catsIJ1Plugin = cats.catsIJ1Plugin; // TODO: ?
    }

    public void runUI( )
    {
        GenericDialog gd = openObjectReviewDialog();

        if ( gd == null ) return;

        setSettingsFromUI( gd );

        reviewObjectsUsingRoiManager( cats.getSegmentedObjectsMap().get( objectsName ) );
    }

    private GenericDialog openObjectReviewDialog()
    {
        GenericDialog gd = new GenericDialogPlus( "Objects Review" );

        gd.addChoice( "Objects", cats.getSegmentedObjectsNames().toArray( new String[0] ), cats.getClassNames().get( 0 ) );

        gd.addNumericField( "Minimum volume [ " + cats.getInputImage().getCalibration().getUnit() + "^3 ] ", 10, 0);

        gd.addNumericField( "Zoom level [ 0 - " + Overlays.ZOOM_LEVEL_MAX + " ]", 5, 0);

        gd.showDialog();

        if ( gd.wasCanceled() ) return null;

        return gd;
    }

    private void setSettingsFromUI( GenericDialog gd )
    {
        objectsName = gd.getNextChoice();
        minCalibratedVolume = gd.getNextNumber();
        zoomLevel = ( int ) gd.getNextNumber();
    }

    public void reviewObjectsUsingRoiManager( SegmentedObjects objects )
    {
        ArrayList< Roi > rois = getCentroidRoisOfVolumeFilteredObjects( objects, SORT_BY_VOLUME );

        makeImageTheActiveWindow( cats.getInputImage() );

        configureObjectOverlayAndRoiManager( rois );
    }

    private void configureObjectOverlayAndRoiManager( ArrayList< Roi > rois )
    {
        roiManager = new RoiManager();

        Overlays overlays = new Overlays( cats );

        for ( Roi roi : rois )
        {
            overlays.addRoiToRoiManager( roiManager, cats.getInputImage(), roi );
        }

        cats.logger.info( "\nReviewing objects: " + rois.size() );

        overlays.setZoomLevel( zoomLevel );

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

    public ArrayList< Roi > getCentroidRoisOfVolumeFilteredObjects( SegmentedObjects objects, String sorting )
    {
        reviewedObjects = getVolumeFilteredObjects( objects, sorting );

		ArrayList< Roi > rois = new ArrayList<>();

        double scaleXY = objects.objects3DPopulation.getScaleXY();
        double scaleZ = objects.objects3DPopulation.getScaleZ();

        int objectID = 0;

        for ( Object3D object3D : reviewedObjects )
        {

            double x = object3D.getCenterX() * scaleXY;
            double y = object3D.getCenterY() * scaleXY;
            double z = object3D.getCenterZ() * scaleZ;

            PointRoi roi = new PointRoi( x, y );

            if ( cats.getInputImage().isHyperStack() )
            {
                roi.setPosition( 1, ( int ) ( z ) + 1, objects.t + 1 );
            }
            else
            {
                roi.setPosition( ( int ) ( z ) + 1 );
            }

			double calibratedVolume = getCalibratedVolume( object3D,
					objects.objects3DPopulation.getScaleXY(),
					objects.objects3DPopulation.getScaleZ(),
					cats.getInputImage() );

            roi.setName( "#" + ( objectID + 1) + " volume " + calibratedVolume );
            roi.setProperty( Overlays.REVIEW, "ObjectReview" );
			roi.setProperty( OBJECT_ID,  "" + objectID );
			roi.setSize( 4 );
            rois.add( roi );

			++objectID;

        }

        return rois;
    }

    private double getCalibratedVolume(  Object3D object3D, double scaleXY, double scaleZ, ImagePlus imp )
    {
        double pixelVolumeInBinnedImage = object3D.getVolumePixels();

        double pixelVolumeInOriginalImage = pixelVolumeInBinnedImage * scaleXY * scaleZ;

        double calibrationX = imp.getCalibration().pixelWidth;
        double calibrationY = imp.getCalibration().pixelHeight;
        double calibrationZ = imp.getCalibration().pixelDepth;

        double calibratedVolume = pixelVolumeInOriginalImage * calibrationX * calibrationY * calibrationZ;

        return calibratedVolume;

    }


    private class ObjectAndSize
    {
        Object3D object;
        double size;

        public double getSize()
        {
            return size;
        }

    }


    private ArrayList< Object3D > getVolumeFilteredObjects( SegmentedObjects segmentedObjects, String sorting )
    {
        ArrayList< Object3D > allObjects = segmentedObjects.objects3DPopulation.getObjectsList();

        cats.getLogger().info( "Number of all objects: " + allObjects.size() );

        ArrayList< ObjectAndSize > objectsAndSizes = new ArrayList<>( );

        for ( Object3D object3D : allObjects )
        {
            double calibratedVolume = getCalibratedVolume( object3D,
					segmentedObjects.objects3DPopulation.getScaleXY(),
					segmentedObjects.objects3DPopulation.getScaleZ(),
					cats.getInputImage() );

            if ( calibratedVolume > minCalibratedVolume )
            {
                final ObjectAndSize objectAndSize = new ObjectAndSize();
                objectAndSize.object = object3D;
                objectAndSize.size = calibratedVolume;
                objectsAndSizes.add( objectAndSize );
            }
        }

        cats.getLogger().info( "Number of objects after volume filter: " + objectsAndSizes.size() );

        if ( sorting.equals( SORT_BY_VOLUME ) )
        {
            objectsAndSizes.sort( Comparator.comparing( ObjectAndSize::getSize ).reversed() );
        }

        ArrayList< Object3D > sortedAndFilteredObjects = new ArrayList<>(  );

        for ( ObjectAndSize objectAndSize : objectsAndSizes )
        {
            sortedAndFilteredObjects.add( objectAndSize.object );
        }

        return sortedAndFilteredObjects;
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

    public ImagePlus getObjectsInRoiManagerAsImage( )
    {
        Objects3DPopulation objects3DPopulation = new Objects3DPopulation( getObjectsFromRoiManager() );

        ImagePlus imp = Utils.create8bitImagePlus( cats.getInputImageDimensions() );

        objects3DPopulation.draw( imp.getImageStack(), 255 );

        return imp;
    }

	public ArrayList< Object3D > getObjectsFromRoiManager( )
	{
		Roi[] approvedRois = roiManager.getRoisAsArray();

		final ArrayList< Object3D > approvedObjects = new ArrayList<>();

		for ( Roi roi : approvedRois )
		{
			final int objectId = Integer.parseInt( roi.getProperty( OBJECT_ID ) );
			approvedObjects.add( reviewedObjects.get( objectId ) );
		}

		return approvedObjects;
	}


}
