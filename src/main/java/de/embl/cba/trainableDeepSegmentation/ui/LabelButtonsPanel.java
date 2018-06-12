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
            addLabelButton( i );
        }

        createAndShowGUI( );
    }

    private JButton addLabelButton( int classIndex )
    {

        JButton button = new JButton( getClassButtonText( classIndex ) );
        button.setToolTipText("Add markings of label '" + deepSegmentation.getClassName( classIndex ) + "'");
        button.setOpaque( true );
        button.setBackground( deepSegmentation.getColors()[ classIndex ] );
        button.addActionListener( this );
        button.setAlignmentX( Component.CENTER_ALIGNMENT );

        add( button ); // add to GUI

        buttons.add( button ); // add to List

        return ( button );
    }

    private String getClassButtonText( int classIndex )
    {
        return deepSegmentation.getClassName( classIndex ) + " [" + ( classIndex + 1 ) + "]";
    }

    public void setClassColor( int classIndex, Color color )
    {
        buttons.get( classIndex ).setBackground( color );

        refreshGui();
    }

    public void addNumInstancesToButtonTexts( long[] numLabelInstancesPerClass )
    {
        for ( int c = 0; c < numLabelInstancesPerClass.length; ++c )
        {
            addNumInstancesToButtonText( c, numLabelInstancesPerClass[ c ] );
        }
    }

    public void addNumInstancesToButtonText( int classIndex, long numInstances )
    {

        buttons.get( classIndex ).setText( getClassButtonText( classIndex ) + " ( " + numInstances + " )" );

        refreshGui();
    }

    public void updateButtons( )
    {

        clearButtons();

        int numClasses = deepSegmentation.getNumClasses();

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

    private void refreshGui()
    {
        this.revalidate();
        this.repaint();
        frame.pack();
    }



}