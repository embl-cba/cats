package de.embl.cba.cats.ui;

import de.embl.cba.cats.CATS;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

public class LabelButtonsPanel extends JPanel implements ActionListener
{
    protected ArrayList< JButton > buttons;
    final CATS cats;
    final Overlays overlays;
    JFrame frame;

    public LabelButtonsPanel( CATS cats, Overlays overlays, Point location, int width )
    {
        this.cats = cats;
        this.overlays = overlays;

        addButtons();
        setFrameAndLocation( location, width );
        createAndShowGUI( );
    }

    private void addButtons()
    {
        buttons = new ArrayList<>(  );

        for ( int i = 0; i < 2; ++i )
        {
            addLabelButton( i );
        }
    }

    private void setFrameAndLocation( Point location, int width )
    {
        this.frame = new JFrame( "Labels" );
        location.x = location.x + width;
        frame.setLocation( location );
    }

    private JButton addLabelButton( int classIndex )
    {

        JButton button = new JButton( getClassText( classIndex ) );
        button.setToolTipText("Add label for class '" + cats.getClassName( classIndex ) + "'");
        button.setOpaque( true );
        button.setBackground( cats.getColors()[ classIndex ] );
        button.addActionListener( this );
        button.setAlignmentX( Component.CENTER_ALIGNMENT );

        add( button ); // add to GUI

        buttons.add( button ); // add to List

        return ( button );
    }

    private String getClassText( int classIndex )
    {
        return cats.getClassName( classIndex ) + " [" + ( classIndex + 1 ) + "]";
    }

    public void setClassColor( int classIndex, Color color )
    {
        buttons.get( classIndex ).setBackground( color );

        refreshGui();
    }

    public void setLabellingInformations()
    {
        final long[] numInstancesPerClass = cats.getNumInstancesPerClass();
        final int[] numLabelsPerClass = cats.getLabelManager().getNumLabelsPerClass();

        for ( int classIndex = 0; classIndex < cats.getNumClasses(); ++classIndex )
        {
            setLabellingInformation( classIndex, numLabelsPerClass[ classIndex ], numInstancesPerClass[ classIndex ] );
        }
    }

    public void setLabellingInformation( int classIndex, long numLabels, long numInstances )
    {
        buttons.get( classIndex ).setText( getClassText( classIndex )
                + "  Labels: " + numLabels + " "
                + " Points: " + numInstances);

        refreshGui();
    }

    public void updateButtons( )
    {

        clearButtons();

        int numClasses = cats.getNumClasses();

        for ( int classNum = 0; classNum < numClasses; ++classNum )
        {
            addLabelButton( classNum );
        }

        refreshGui();

    }

    private void clearButtons()
    {
        for ( JButton button : buttons )
        {
            remove( button );
        }

        buttons = new ArrayList<>(  );
    }

    @Override
    public void actionPerformed( ActionEvent e )
    {

        for ( int i = 0; i < buttons.size(); ++i )
        {
            if ( e.getSource() == buttons.get( i ) )
            {
                cats.addLabelFromImageRoi(  i  );
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
        frame.setDefaultCloseOperation( JFrame.DISPOSE_ON_CLOSE );

        //Create and set up the content pane.
        setOpaque( true ); //content panes must be opaque
        setLayout( new BoxLayout(this, BoxLayout.Y_AXIS ) );

        frame.setContentPane( this );

        //Display the window.
        frame.pack();
        frame.setVisible( true );
    }

    private void refreshGui()
    {
        this.revalidate();
        this.repaint();
        frame.pack();
    }



}