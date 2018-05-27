package de.embl.cba.trainableDeepSegmentation.ui;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

public class LabelButtonsPanel extends JPanel implements ActionListener
{
    protected ArrayList< JButton > buttons;
    final DeepSegmentation deepSegmentation;
    final Overlays overlays;
    JFrame frame;

    public LabelButtonsPanel( DeepSegmentation deepSegmentation, Overlays overlays )
    {
        this.deepSegmentation = deepSegmentation;
        this.overlays = overlays;

        buttons = new ArrayList<>(  );

        for ( int i = 0; i < 2; ++i )
        {
            buttons.add( createLabelButton(  i ) );
        }

        createAndShowGUI( );
    }

    private JButton createLabelButton( int classNum )
    {

        JButton button = new JButton( deepSegmentation.getClassName( classNum ) + " [" + ( classNum + 1 ) + "]" );
        button.setToolTipText("Add markings of label '" + deepSegmentation.getClassName( classNum ) + "'");
        button.setOpaque( true );
        button.setBackground( deepSegmentation.getColors()[ classNum ] );
        button.addActionListener( this );
        button.setAlignmentX( Component.CENTER_ALIGNMENT );

        add( button );

        return ( button );
    }

    public void addButton()
    {
        int classNum = buttons.size();
        buttons.add( createLabelButton(  classNum ) );

        this.revalidate();
        this.repaint( );
        frame.pack();
    }

    @Override
    public void actionPerformed( ActionEvent e )
    {

        for ( int i = 0; i < buttons.size(); ++i )
        {
            if ( e.getSource() == buttons.get( i ) )
            {
                deepSegmentation.addLabelFromImageRoi(  i  );
                overlays.updateLabels( );
                break;
            }
        }

    }


    /**
     * Create the GUI and show it.  For thread safety,
     * this method should be invoked from the
     * event-dispatching thread.
     */
    private void createAndShowGUI( )
    {

        //Create and set up the window.
        frame = new JFrame( "Labels" );
        frame.setDefaultCloseOperation( JFrame.EXIT_ON_CLOSE );

        //Create and set up the content pane.
        setOpaque( true ); //content panes must be opaque
        setLayout( new BoxLayout(this, BoxLayout.Y_AXIS ) );

        frame.setContentPane( this );

        //Display the window.
        frame.pack();
        frame.setVisible( true );
    }


}