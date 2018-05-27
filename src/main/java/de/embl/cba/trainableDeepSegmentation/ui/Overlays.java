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
    public static final String REVIEW = "review";

    Color[] colors;
    final ResultImage resultImage;
    final ImagePlus inputImage;
    final DeepSegmentation deepSegmentation;

    private LUT overlayLUT = null;
    private int overlayOpacity = 33;
    private Roi probabilities;
    Roi currentlyDisplayedRoi;
    public boolean zoomInOnRois = false;
    RoiManager roiManager;
    LabelReviewManager labelReviewManager;



    public Overlays( DeepSegmentation deepSegmentation )
    {

        this.deepSegmentation = deepSegmentation;
        this.colors = deepSegmentation.classColors;
        this.resultImage = deepSegmentation.getResultImage();
        this.inputImage = deepSegmentation.getInputImage();

        setProbabilityOverlayLut( OVERLAY_MODE_PROBABILITIES );
        inputImage.setOverlay( new Overlay(  ) );

        Roi.addRoiListener( this );
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

        if ( classId >= 0 )
        {
            reviewLabelsInRoiManager( classId, ORDER_TIME_ADDED );
        }
    }

    public void zoomInOnRois( boolean zoomIn )
    {
        zoomInOnRois = zoomIn;
    }

    private int getClassIdFromUI()
    {
        GenericDialog gd = new GenericDialogPlus("Labels Review");
        gd.addChoice( "Classes", deepSegmentation.getClassNames().toArray( new String[0] ), deepSegmentation.getClassNames().get( 0 ) );
        gd.showDialog();
        if ( gd.wasCanceled() ) return -1;
        int classId = gd.getNextChoiceIndex() ;
        return classId;
    }

    public void reviewLabelsInRoiManager( int classNum, String order )
    {
        labelReviewManager = new LabelReviewManager( deepSegmentation.getExamples(), deepSegmentation );

        ArrayList< Roi > rois = labelReviewManager.getRoisFromLabels( classNum, order );

        labelReviewManager.setLabelsCurrentlyBeingReviewed( rois );

        zoomInOnRois = true;

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
        manager.add( imp, roi, -1 );
    }


    @Override
    public void roiModified( ImagePlus imagePlus, int actionId )
    {
        if ( ( imagePlus != null ) && ( imagePlus == inputImage ) )
        {
            if ( actionId == RoiListener.CREATED && zoomInOnRois )
            {
                Roi roi = inputImage.getRoi();

                if ( isNewRoi( roi ) )
                {
                    if ( roi.getName() != null && roi.getName().contains( REVIEW ) )
                    {
                        currentlyDisplayedRoi = roi;
                        zoomToSelection();
                    }
                }

            }
        }
    }


    private boolean isNewRoi( Roi roi )
    {
        if ( roi == null )
        {
            return false;
        }
        else if ( currentlyDisplayedRoi == null )
        {
            return true;
        }
        else
        {
            int x = roi.getBounds().x;
            int x2 = currentlyDisplayedRoi.getBounds().x;
            return x != x2;
         }
    }

    private void zoomToSelection()
    {

        Roi roi = inputImage.getRoi();

        if ( roi == null ) return;

        IJ.run("To Selection");

        // add roi as overlay to make it persists when the user clicks on the image
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
                addLabels();
                zoomInOnRois = false;

            }
        });
    }

    public void cleanUpOverlaysAndRoisWhenRoiManagerIsClosed( RoiManager manager  )
    {
        manager.addWindowListener( new WindowAdapter()
        {
            @Override
            public void windowClosing( WindowEvent we )
            {
                Roi[] rois = inputImage.getOverlay().toArray();
                for ( Roi roi : rois )
                {
                    if ( roi.getName() != RESULT_OVERLAY )
                    {
                        inputImage.getOverlay().remove( roi );
                    }
                }

                addLabels();
                zoomInOnRois = false;
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
