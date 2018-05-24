package de.embl.cba.trainableDeepSegmentation.ui;

import ij.ImageListener;
import ij.ImagePlus;

import java.awt.event.*;

import static de.embl.cba.trainableDeepSegmentation.ui.Overlays.OVERLAY_MODE_PROBABILITIES;
import static de.embl.cba.trainableDeepSegmentation.ui.Overlays.OVERLAY_MODE_SEGMENTATION;
import static de.embl.cba.trainableDeepSegmentation.ui.Overlays.OVERLAY_MODE_UNCERTAINTY;

public class Listeners
{

    Overlays overlays;
    final ImagePlus imp;

    int c;
    int t;
    int z;

    public Listeners( ImagePlus imagePlus, Overlays overlays)
    {
       this.overlays = overlays;
       this.imp = imagePlus;
       addKeyListeners( );
       addStackListeners( );

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


                        /*
                        try
                        {
                            int iClass = Integer.parseInt("" + e.getKeyChar());
                            addAnnotation( iClass - 1  );
                        }
                        catch (NumberFormatException e )
                        {
                            // do nothing
                        }
                        */

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
                            drawExamples();
                            if( showColorOverlay )
                            {
                                updateResultOverlay();
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


        imp.getWindow().addKeyListener( keyListener );
        imp.getWindow().getCanvas().addKeyListener( keyListener );

    }

    private void addStackListeners(  )
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
                    overlays.updateResultOverlay();
                }
            }
        };

        imp.addImageListener( imageListener );

    }

    public boolean updatePosition( )
    {
        boolean positionChanged = false;

        int c = imp.getC();
        int t = imp.getT();
        int z = imp.getZ();

        if ( this.c != c ) positionChanged = true;
        if ( this.t != t ) positionChanged = true;
        if ( this.z != z ) positionChanged = true;

        this.c = c;
        this.t = t;
        this.z = z;

        return positionChanged;

    }


}
