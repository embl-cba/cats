package de.embl.cba.trainableDeepSegmentation.ui;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import ij.ImageListener;
import ij.ImagePlus;

import java.awt.event.*;

import static de.embl.cba.trainableDeepSegmentation.ui.Overlays.OVERLAY_MODE_PROBABILITIES;
import static de.embl.cba.trainableDeepSegmentation.ui.Overlays.OVERLAY_MODE_SEGMENTATION;
import static de.embl.cba.trainableDeepSegmentation.ui.Overlays.OVERLAY_MODE_UNCERTAINTY;

public class Listeners
{

    final Overlays overlays;
    final ImagePlus inputImage;
    final DeepSegmentation deepSegmentation;
    final LabelButtonsPanel labelButtonsPanel;

    private boolean updateLabelsWhenImageSliceIsChanged = true;

    int c;
    int t;
    int z;

    public Listeners( DeepSegmentation deepSegmentation, Overlays overlays, LabelButtonsPanel labelButtonsPanel )
    {
        this.overlays = overlays;
        this.inputImage = deepSegmentation.getInputImage();
        this.deepSegmentation = deepSegmentation;
        this.labelButtonsPanel = labelButtonsPanel;

        addKeyListeners( );
        addImageListeners( );
        updatePosition( );

    }

    private void addKeyListeners( )
    {
        KeyListener keyListener = new KeyListener() {

            @Override
            public void keyTyped(KeyEvent e) {
                new Thread(new Runnable(){
                    public void run()
                    {
                        if ( e.getKeyChar() == 'r' )
                        {
                            overlays.toggleOverlay( OVERLAY_MODE_SEGMENTATION );
                        }

                        if ( e.getKeyChar() == 'p' )
                        {
                            overlays.toggleOverlay( OVERLAY_MODE_PROBABILITIES );
                        }

                        if ( e.getKeyChar() == 'u' )
                        {
                            overlays.toggleOverlay( OVERLAY_MODE_UNCERTAINTY);
                        }

                        try
                        {
                            int classIndex = Integer.parseInt("" + e.getKeyChar() ) - 1;
                            deepSegmentation.addLabelFromImageRoi(  classIndex  );
                            overlays.updateLabels( );
                        }
                        catch (NumberFormatException e )
                        {
                            // do nothing
                        }
                    }
                }).start();
            }

            @Override
            public void keyReleased(final KeyEvent e) {
                new Thread(new Runnable(){
                    //exec.submit(new Runnable() {
                    public void run()
                    {
                        /*
                        if(e.getKeyCode() == KeyEvent.VK_LEFT ||
                                e.getKeyCode() == KeyEvent.VK_RIGHT ||
                                e.getKeyCode() == KeyEvent.VK_LESS ||
                                e.getKeyCode() == KeyEvent.VK_GREATER ||
                                e.getKeyCode() == KeyEvent.VK_COMMA ||
                                e.getKeyCode() == KeyEvent.VK_PERIOD)
                        {
                            //IJ.log("moving scroll " + e.getKeyCode());
                            inputImage.killRoi();
                            updateExampleLists();
                            updateLabelsWhenImageSliceIsChanged();
                            if( showColorOverlay )
                            {
                                updateProbabilities();
                                inputImage.updateAndDraw();
                            }
                        }*/
                    }
                }).start();

            }

            @Override
            public void keyPressed(KeyEvent e) {}
        };
        // add key listener to the window and the canvas
        // addKeyListener( keyListener );


        inputImage.getWindow().addKeyListener( keyListener );
        inputImage.getWindow().getCanvas().addKeyListener( keyListener );

    }

    private void addImageListeners(  )
    {

        ImageListener imageListener = new ImageListener()
        {
            @Override
            public void imageOpened( ImagePlus imp )
            {

            }

            @Override
            public void imageClosed( ImagePlus imp )
            {

            }

            @Override
            public void imageUpdated( ImagePlus imp )
            {
                if ( updatePosition() )
                {
                    overlays.updateProbabilities();

                    if ( updateLabelsWhenImageSliceIsChanged )
                    {
                        overlays.updateLabels();
                    }
                }
            }
        };

        inputImage.addImageListener( imageListener );

    }

    public boolean updatePosition( )
    {
        boolean positionChanged = false;

        int c = inputImage.getC();
        int t = inputImage.getT();
        int z = inputImage.getZ();

        if ( this.c != c ) positionChanged = true;
        if ( this.t != t ) positionChanged = true;
        if ( this.z != z ) positionChanged = true;

        this.c = c;
        this.t = t;
        this.z = z;

        return positionChanged;

    }


    public void updateLabelsWhenImageSliceIsChanged( boolean trueFalse )
    {
        this.updateLabelsWhenImageSliceIsChanged = trueFalse;
    }


}
