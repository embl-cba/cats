package de.embl.cba.trainableDeepSegmentation.ui;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.labels.LabelReviewManager;
import de.embl.cba.trainableDeepSegmentation.labels.examples.Example;
import de.embl.cba.trainableDeepSegmentation.results.ResultImage;
import de.embl.cba.trainableDeepSegmentation.results.ResultImageDisk;
import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.*;
import ij.plugin.frame.RoiManager;
import ij.process.ImageProcessor;
import ij.process.LUT;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;

import static de.embl.cba.trainableDeepSegmentation.labels.LabelReviewManager.ORDER_TIME_ADDED;
import static de.embl.cba.trainableDeepSegmentation.labels.LabelReviewManager.ORDER_Z;

public class Overlays implements RoiListener
{
    public static final String OVERLAY_MODE_SEGMENTATION = "Segmentation";
    public static final String OVERLAY_MODE_PROBABILITIES = "Probabilities";
    public static final String OVERLAY_MODE_UNCERTAINTY = "Uncertainty";
    public static final String RESULT_OVERLAY = "result overlay";

    Color[] colors;
    final ResultImage resultImage;
    final ImagePlus inputImage;
    final DeepSegmentation deepSegmentation;

    private LUT overlayLUT = null;
    private int overlayOpacity = 33;
    private Roi probabilities;


    public Overlays( DeepSegmentation deepSegmentation )
    {

        this.deepSegmentation = deepSegmentation;
        this.colors = deepSegmentation.classColors;
        this.resultImage = deepSegmentation.getResultImage();
        this.inputImage = deepSegmentation.getInputImage();

        setProbabilityOverlayLut( OVERLAY_MODE_PROBABILITIES );
        inputImage.setOverlay( new Overlay(  ) );
    }


    /**
     * Toggle between overlay and original image with markings
     */
    public synchronized void toggleOverlay( String mode )
    {
        if ( inputImage.getOverlay().contains( probabilities ) )
        {
            inputImage.getOverlay().remove( probabilities );
        }
        else
        {
            setProbabilityOverlayLut( mode );
            addProbabilities();
        }

    }

    private void setProbabilityOverlayLut( String mode )
    {
        // create overlay LUT
        final byte[] red = new byte[ 256 ];
        final byte[] green = new byte[ 256 ];
        final byte[] blue = new byte[ 256 ];

        if ( mode.equals( OVERLAY_MODE_SEGMENTATION ) )
        {
            for ( int iClass = 0; iClass < DeepSegmentation.MAX_NUM_CLASSES; iClass++)
            {
                int offset = iClass * ResultImageDisk.CLASS_LUT_WIDTH;

                for ( int i = 1; i <= ResultImageDisk.CLASS_LUT_WIDTH; i++)
                {
                    red[offset + i] = (byte) ( colors[iClass].getRed() );
                    green[offset + i] = (byte) ( colors[iClass].getGreen() );
                    blue[offset + i] = (byte) ( colors[iClass].getBlue() );
                }
            }

            overlayLUT = new LUT(red, green, blue);

        }


        if ( mode.equals( OVERLAY_MODE_PROBABILITIES ) )
        {
            for ( int iClass = 0; iClass < DeepSegmentation.MAX_NUM_CLASSES; iClass++)
            {
                int offset = iClass * ResultImageDisk.CLASS_LUT_WIDTH;

                for ( int i = 1; i <= ResultImageDisk.CLASS_LUT_WIDTH; i++)
                {
                    red[offset + i] = (byte) (1.0 * colors[iClass].getRed() * i / ( ResultImageDisk.CLASS_LUT_WIDTH ));
                    green[offset + i] = (byte) (1.0 * colors[iClass].getGreen() * i / ( ResultImageDisk.CLASS_LUT_WIDTH ));
                    blue[offset + i] = (byte) (1.0 * colors[iClass].getBlue() * i / ( ResultImageDisk.CLASS_LUT_WIDTH ));
                }
            }

            overlayLUT = new LUT(red, green, blue);
        }

    }

    public void clearAllOverlays()
    {
        inputImage.setOverlay( new Overlay( ) );
    }

    public void clearAllOverlaysAndRois()
    {
        inputImage.setOverlay( new Overlay( ) );
        inputImage.killRoi();
    }

    public void updateProbabilities()
    {
        if ( inputImage.getOverlay().contains( probabilities ) )
        {
            inputImage.getOverlay().remove( probabilities );
            addProbabilities();
        }
    }

    public void addProbabilities()
    {
        ImageProcessor overlayImage = resultImage.getSlice( inputImage.getZ(), inputImage.getT() );
        overlayImage = overlayImage.convertToByte( false );
        overlayImage.setColorModel( overlayLUT );

        probabilities = new ImageRoi( 0, 0, overlayImage );
        ( ( ImageRoi ) probabilities ).setOpacity( overlayOpacity / 100.0 );
        probabilities.setName( RESULT_OVERLAY );
        inputImage.getOverlay().add( probabilities );
        inputImage.updateAndDraw();
    }

    public void showProbabilities()
    {
        if ( inputImage.getOverlay().contains( probabilities ) )
        {
            updateProbabilities();
        }
        else
        {
            addProbabilities();
        }

    }

    public void updateLabels()
    {
        Overlay overlay = inputImage.getOverlay();

        Roi[] rois = overlay.toArray();

        for ( Roi roi : rois )
        {
            if ( roi != probabilities )
            {
                overlay.remove( roi );
            }
        }

        addLabels();

    }

    protected void addLabels()
    {
        final int frame = inputImage.getT();
        final int slice = inputImage.getZ();

        int numClasses = deepSegmentation.getNumClasses();

        for(int iClass = 0; iClass < numClasses; iClass++)
        {
            ArrayList< Roi > classRois = deepSegmentation.getLabelRois( iClass, slice-1, frame-1);
            for ( Roi roi : classRois )
            {
                roi.setStrokeColor( colors[ iClass ] );
                roi.setStrokeWidth( 1.0 );

                if ( ! inputImage.getOverlay().contains( roi  ) )
                {
                    inputImage.getOverlay().add( roi );
                }
            }
        }

        inputImage.updateAndDraw();
    }



    Roi currentlyDisplayedRoi;

    public boolean reviewRoisFlag = false;

    RoiManager roiManager;
    LabelReviewManager labelReviewManager;


    private void reviewLabels( int classNum )
    {
        labelReviewManager = new LabelReviewManager( deepSegmentation.getExamples() );
        String order = LabelReviewManager.ORDER_TIME_ADDED; // showOrderGUI();
        reviewLabelsInRoiManager( classNum, order );
    };


    public String showOrderGUI()
    {
        GenericDialog gd = new GenericDialog("Label ordering");
        gd.addChoice( "Ordering of labels:", new String[]{ ORDER_TIME_ADDED, ORDER_Z}, ORDER_TIME_ADDED);
        gd.showDialog();

        if( gd.wasCanceled() )
        {
            return null;
        }
        else
        {
            return gd.getNextChoice();
        }

    }

    public void reviewLabelsInRoiManagerUI(  )
    {
        int classId = getClassIdFromUI();

        if ( classId > 0 )
        {
            reviewLabelsInRoiManager( classId, ORDER_TIME_ADDED );
        }
    }

    private int getClassIdFromUI()
    {
        GenericDialog gd = new GenericDialogPlus("Labels Review");
        gd.addChoice( "Classes", deepSegmentation.getClassNames().toArray( new String[0] ), deepSegmentation.getClassNames().get( 0 ) );
        gd.showDialog();
        if ( gd.wasCanceled() ) return -1;
        int classId = gd.getNextChoiceIndex();
        return classId;
    }

    public void reviewLabelsInRoiManager( int classNum, String order )
    {
        ArrayList< Roi > rois = labelReviewManager.getRoisFromLabels( classNum, order );

        labelReviewManager.setLabelsCurrentlyBeingReviewed( rois );

        addRoisToRoiManager( rois );

        updateLabelsWhenRoiManagerIsClosed( );

    }

    private void addRoisToRoiManager( ArrayList< Roi > rois )
    {
        roiManager = new RoiManager();

        for ( Roi roi : rois )
        {
            addRoiToRoiManager( roiManager, inputImage, roi );
        }
    }

    public static void addRoiToRoiManager( RoiManager manager, ImagePlus imp, Roi roi )
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


    @Override
    public void roiModified( ImagePlus imagePlus, int actionId )
    {
        if ( ( imagePlus != null ) && ( imagePlus == inputImage ) )
        {
            if ( actionId == RoiListener.CREATED && reviewRoisFlag )
            {
                if ( currentlyDisplayedRoi == null)
                {
                    currentlyDisplayedRoi = inputImage.getRoi();
                    zoomToSelection();
                }
                else
                {
                    int x = inputImage.getRoi().getBounds().x;
                    int x2 = currentlyDisplayedRoi.getBounds().x;
                    if ( x != x2 )
                    {
                        currentlyDisplayedRoi = inputImage.getRoi();
                        zoomToSelection();
                    }
                }

            }
        }
    }


    private void zoomToSelection()
    {

        Roi roi = inputImage.getRoi(); if ( roi == null ) return;

        // makeInputImageTheActiveWindow();

        IJ.run("To Selection");

        // remove old overlay
        inputImage.setOverlay( new Overlay(  ) );

        // add new overlay
        IJ.run("Add Selection...");

        // remove roi
        inputImage.killRoi();

        updateProbabilities();

        zoomOut( 7 );

    }

    public void updateLabelsWhenRoiManagerIsClosed(  )
    {
        roiManager.addWindowListener( new WindowAdapter()
        {
            @Override
            public void windowClosing( WindowEvent we )
            {

                ArrayList< Example > approvedLabelList = labelReviewManager.getApprovedLabelList( roiManager );
                deepSegmentation.setExamples( approvedLabelList );
                clearAllOverlaysAndRois();

            }
        });
    }

    public static void removeAllOverlaysAndRoisWhenRoiManagerIsClosed( RoiManager manager, ImagePlus imp  )
    {
        manager.addWindowListener( new WindowAdapter()
        {
            @Override
            public void windowClosing( WindowEvent we )
            {
                // IJ.log( "RoiManager closed.");
                imp.killRoi();
                imp.setOverlay( new Overlay(  ) );
            }
        });
    }

    private static void zoomIn( int n )
    {
        for ( int i = 0; i < n; ++i )
        {
            IJ.run( "In [+]", "" );
        }
    }

    public static void zoomOut( int n )
    {
        for ( int i = 0; i < n; ++i )
        {
            IJ.run( "Out [-]", "" );
        }
    }


}
