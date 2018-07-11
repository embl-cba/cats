package de.embl.cba.cats.ui;

import de.embl.cba.cats.CATS;
import ij.ImageListener;
import ij.ImagePlus;

import java.awt.event.*;

import static de.embl.cba.cats.ui.Overlays.OVERLAY_MODE_PROBABILITIES;
import static de.embl.cba.cats.ui.Overlays.OVERLAY_MODE_SEGMENTATION;
import static de.embl.cba.cats.ui.Overlays.OVERLAY_MODE_UNCERTAINTY;

public class Listeners
{

    final Overlays overlays;
    final ImagePlus inputImage;
    final CATS CATS;
    final LabelButtonsPanel labelButtonsPanel;

    private boolean updateLabelsWhenImageSliceIsChanged = true;

    int c;
    int t;
    int z;

    public Listeners( CATS CATS, Overlays overlays, LabelButtonsPanel labelButtonsPanel )
    {
        this.overlays = overlays;
        this.inputImage = CATS.getInputImage();
        this.CATS = CATS;
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
                            CATS.addLabelFromImageRoi(  classIndex  );
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
                            updateLabelLists();
                            updateLabelsWhenImageSliceIsChanged();
                            if( showColorOverlay )
                            {
                                updateProbabilitiesOverlay();
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
                    overlays.updateProbabilitiesOverlay();

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
