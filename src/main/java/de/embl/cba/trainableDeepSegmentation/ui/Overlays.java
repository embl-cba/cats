package de.embl.cba.trainableDeepSegmentation.ui;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.results.ResultImage;
import de.embl.cba.trainableDeepSegmentation.results.ResultImageDisk;
import fiji.util.gui.OverlayedImageCanvas;
import ij.ImagePlus;
import ij.gui.ImageRoi;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.process.ImageProcessor;
import ij.process.LUT;

import java.awt.*;

public class Overlays
{


    public static final String OVERLAY_MODE_SEGMENTATION = "Segmentation";
    public static final String OVERLAY_MODE_PROBABILITIES = "Probabilities";
    public static final String OVERLAY_MODE_UNCERTAINTY = "Uncertainty";

    Color[] colors;
    final ResultImage resultImage;
    final ImagePlus imp;


    OverlayedImageCanvas overlayedImageCanvas;

    private LUT overlayLUT = null;
    private int overlayOpacity = 33;
    boolean isOverlayShown = false;




    public Overlays( Color[] colors, ImagePlus imp, ResultImage resultImage )
    {
        this.colors = colors;
        this.resultImage = resultImage;
        this.imp = imp;
        this.overlayedImageCanvas = new OverlayedImageCanvas( imp );


    }

    public OverlayedImageCanvas getOverlayedImageCanvas()
    {
        return overlayedImageCanvas;
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
            imp.setOverlay( new Overlay( ) );
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

        /*
        if ( mode.equals( OVERLAY_MODE_UNCERTAINTY ) )
        {
            // assign classColors to classes
            for ( int iClass = 0; iClass < DeepSegmentation.MAX_NUM_CLASSES; iClass++)
            {
                int offset = iClass * ResultImageDisk.CLASS_LUT_WIDTH;
                for ( int i = 1; i <= ResultImageDisk.CLASS_LUT_WIDTH; i++)
                {
                    // TODO:
                    // - check whether this is correct
                    red[offset + i] = (byte) ( 255.0 * Math.exp( - deepSegmentation.uncertaintyLutDecay * i  ));
                    green[offset + i] = (byte) ( 0 );
                    blue[offset + i] = (byte) ( 255.0 * Math.exp( - deepSegmentation.uncertaintyLutDecay * i  ));
                }
            }
            overlayLUT = new LUT(red, green, blue);
        }*/

        //showColorOverlay = !showColorOverlay;
        //IJ.log("toggle overlay to: " + showColorOverlay);
        //if (showColorOverlay && null != deepSegmentation.getResultImage())
     //   {
  ///      }
//        else
        //    resultOverlay.setImage(null);

        isOverlayShown = true;
        updateResultOverlay();
        imp.updateAndDraw();

    }

    public void updateResultOverlay()
    {
        if ( isOverlayShown )
        {
            ImageProcessor overlayImage = resultImage.getSlice( imp.getZ(), imp.getT() );
            overlayImage = overlayImage.convertToByte( false );
            overlayImage.setColorModel( overlayLUT );

            Roi roi = new ImageRoi( 0, 0, overlayImage );
            ( ( ImageRoi ) roi ).setOpacity( overlayOpacity / 100.0 );
            roi.setName( "result overlay" );
            Overlay overlay = new Overlay( roi );

            imp.setOverlay( overlay );
        }

    }


}
