package de.embl.cba.trainableDeepSegmentation.ui;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.results.ResultImage;
import de.embl.cba.trainableDeepSegmentation.results.ResultImageDisk;
import ij.ImagePlus;
import ij.gui.ImageRoi;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.process.ImageProcessor;
import ij.process.LUT;

import java.awt.*;
import java.util.ArrayList;

public class Overlays
{


    public static final String OVERLAY_MODE_SEGMENTATION = "Segmentation";
    public static final String OVERLAY_MODE_PROBABILITIES = "Probabilities";
    public static final String OVERLAY_MODE_UNCERTAINTY = "Uncertainty";

    Color[] colors;
    final ResultImage resultImage;
    final ImagePlus inputImage;
    final DeepSegmentation deepSegmentation;

    private LUT overlayLUT = null;
    private int overlayOpacity = 33;
    boolean isOverlayShown = false;


    public Overlays( Color[] colors, ImagePlus inputImage, ResultImage resultImage, DeepSegmentation deepSegmentation )
    {
        this.colors = colors;
        this.resultImage = resultImage;
        this.inputImage = inputImage;
        this.deepSegmentation = deepSegmentation;
    }


    /**
     * Toggle between overlay and original image with markings
     */
    public synchronized void toggleOverlay( String mode )
    {

        // create overlay LUT
        final byte[] red = new byte[ 256 ];
        final byte[] green = new byte[ 256 ];
        final byte[] blue = new byte[ 256 ];

        if ( isOverlayShown )
        {
            inputImage.setOverlay( new Overlay( ) );
            isOverlayShown = false;
            return;
        }

        if ( mode.equals( OVERLAY_MODE_SEGMENTATION ) )
        {
            // assign classColors to classes
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
            // assign classColors to classes
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

        isOverlayShown = true;
        updateProbabilities();
        inputImage.updateAndDraw();

    }

    public void updateProbabilities()
    {
        if ( isOverlayShown )
        {
            ImageProcessor overlayImage = resultImage.getSlice( inputImage.getZ(), inputImage.getT() );
            overlayImage = overlayImage.convertToByte( false );
            overlayImage.setColorModel( overlayLUT );

            Roi roi = new ImageRoi( 0, 0, overlayImage );
            ( ( ImageRoi ) roi ).setOpacity( overlayOpacity / 100.0 );
            roi.setName( "result overlay" );
            Overlay overlay = new Overlay( roi );

            inputImage.setOverlay( overlay );
        }

    }

    protected void updateLabels()
    {
        final int frame = inputImage.getT();
        final int slice = inputImage.getZ();

        int numClasses = deepSegmentation.getNumClasses();

        inputImage.setOverlay( new Overlay( ) );

        for(int iClass = 0; iClass < numClasses; iClass++)
        {
            ArrayList< Roi > classRois = deepSegmentation.getLabelRois( iClass, slice-1, frame-1);
            for ( Roi roi : classRois )
            {
                roi.setStrokeColor( colors[ iClass ] );
                roi.setStrokeWidth( 2.0 );
                inputImage.getOverlay().add( roi );
            }
        }

        inputImage.updateAndDraw();
    }


}
