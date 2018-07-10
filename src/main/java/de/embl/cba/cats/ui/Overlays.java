package de.embl.cba.cats.ui;

import de.embl.cba.cats.CATS;
import de.embl.cba.cats.labels.LabelReview;
import de.embl.cba.cats.labels.Label;
import de.embl.cba.cats.results.ResultImage;
import de.embl.cba.cats.results.ResultImageDisk;
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

import static de.embl.cba.cats.labels.LabelReview.ORDER_TIME_ADDED;
import static de.embl.cba.cats.labels.LabelReview.ORDER_Z;

public class Overlays implements RoiListener
{
    public static final String OVERLAY_MODE_SEGMENTATION = "Segmentation";
    public static final String OVERLAY_MODE_PROBABILITIES = "Probabilities";
    public static final String OVERLAY_MODE_UNCERTAINTY = "Uncertainty";
    public static final String RESULT_OVERLAY = "result overlay";
    public static final String REVIEW = "review";
    public static final int ZOOM_LEVEL_MAX = 10;

    Color[] colors;
    final ResultImage resultImage;
    final ImagePlus inputImage;
    final CATS CATS;

    private LUT overlayLUT = null;
    private int overlayOpacity = 33;
    private Roi probabilities;
    Roi currentlyDisplayedRoi;
    public boolean zoomInOnRois = false;
    RoiManager roiManager;
    LabelReview labelReview;
    private Listeners listeners;

    private int roiStrokeWidthDuringLabelReview = 4;
    private int zoomLevel;

    public Overlays( CATS CATS )
    {
        this.CATS = CATS;
        this.colors = CATS.classColors;
        this.resultImage = CATS.getResultImage();
        this.inputImage = CATS.getInputImage();

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
            inputImage.updateAndDraw();
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
            for ( int iClass = 0; iClass < CATS.MAX_NUM_CLASSES; iClass++)
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
            for ( int iClass = 0; iClass < CATS.MAX_NUM_CLASSES; iClass++)
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
        removeLabels();
        addLabels();

    }

    public void removeLabels()
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
    }


    protected void addLabels()
    {
        final int frame = inputImage.getT();
        final int slice = inputImage.getZ();

        int numClasses = CATS.getNumClasses();

        for(int iClass = 0; iClass < numClasses; iClass++)
        {
            ArrayList< Roi > classRois = CATS.getLabelManager().getLabelsAsRois( iClass, slice-1, frame-1);
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


    public Color changeClassColorViaGUI( LabelButtonsPanel labelButtonsPanel )
    {

        GenericDialogPlus gd = new GenericDialogPlus("Change class color");

        String[] classNames = new String[ CATS.getClassNames().size()];

        gd.addChoice("Class", CATS.getClassNames().toArray( classNames ), CATS.getClassName( 0 ));

        gd.showDialog();

        if ( gd.wasCanceled() ) return null;

        int classIndex = gd.getNextChoiceIndex();

        ColorChooser colorChooser = new ColorChooser( "" + CATS.getClassName( classIndex ), colors[ classIndex ], false );

        Color color = colorChooser.getColor();

        setClassColor( classIndex, color );

        labelButtonsPanel.setClassColor( classIndex, color );

        return color;
    }

    public void setClassColor( int classIndex, Color color )
    {
        colors[ classIndex ] = color;
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

    public void reviewLabelsInRoiManagerUI( Listeners listeners  )
    {

        this.listeners = listeners;

        listeners.updateLabelsWhenImageSliceIsChanged( false );

        int[] classIdAndLineSize = getClassIdAndLineSizeFromUI();

        if ( classIdAndLineSize == null ) return;

        roiStrokeWidthDuringLabelReview = classIdAndLineSize[ 1 ];

        reviewLabelsInRoiManager( classIdAndLineSize[ 0 ], ORDER_TIME_ADDED );

    }

    public void zoomInOnRois( boolean zoomIn )
    {
        zoomInOnRois = zoomIn;
    }

    public void setZoomLevel( int zoomLevel )
    {
        this.zoomLevel = zoomLevel;
    }

    private int[] getClassIdAndLineSizeFromUI()
    {
        int[] classIdAndLineSize = new int[ 2 ];
        GenericDialog gd = new GenericDialogPlus("Label Review");
        gd.addChoice( "Review labels of class", CATS.getClassNames().toArray( new String[0] ), CATS.getClassNames().get( 0 ) );
        gd.addNumericField( "Roi stroke width during review", roiStrokeWidthDuringLabelReview, 0 );
        gd.showDialog();
        if ( gd.wasCanceled() ) return null;
        classIdAndLineSize[ 0 ] = gd.getNextChoiceIndex() ;
        classIdAndLineSize[ 1 ] = (int) gd.getNextNumber() ;
        return classIdAndLineSize;
    }

    public void reviewLabelsInRoiManager( int classNum, String order )
    {
        labelReview = new LabelReview( CATS.getLabelManager().getLabels(), CATS );

        ArrayList< Roi > rois = labelReview.getRoisFromLabels( classNum, order );

        labelReview.setLabelsCurrentlyBeingReviewed( rois );

        zoomInOnRois = true;

        addRoisToRoiManager( rois );

        removeLabels();

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
        //IJ.run("Add Selection...");

        // remove roi
        //inputImage.killRoi();

        updateProbabilities();

        zoomOut( ZOOM_LEVEL_MAX - zoomLevel );

    }

    public void updateLabelsWhenRoiManagerIsClosed(  )
    {
        roiManager.addWindowListener( new WindowAdapter()
        {
            @Override
            public void windowClosing( WindowEvent we )
            {

                listeners.updateLabelsWhenImageSliceIsChanged( true );
                ArrayList< Label > approvedLabelList = labelReview.getApprovedLabelList( roiManager );
                CATS.getLabelManager().setLabels( approvedLabelList );
                clearAllOverlaysAndRois();
                updateLabels();
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

                updateLabels();
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

    @Override
    public void roiModified( ImagePlus imagePlus, int actionId )
    {
        if ( ( imagePlus != null ) && ( imagePlus == inputImage ) )
        {
            if ( actionId == RoiListener.CREATED && zoomInOnRois )
            {
                Roi roi = inputImage.getRoi();

                roi.setStrokeWidth( roiStrokeWidthDuringLabelReview );

                if ( isNewRoi( roi ) )
                {
                    if ( roi.getName() != null && roi.getProperty( REVIEW ) != null )
                    {
                        currentlyDisplayedRoi = roi;
                        zoomToSelection();
                    }
                }

            }

        }
    }



}
