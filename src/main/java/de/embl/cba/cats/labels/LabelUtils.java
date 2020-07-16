package de.embl.cba.cats.labels;

import de.embl.cba.cats.instances.InstancesAndMetadata;
import de.embl.cba.log.Logger;
import weka.core.Instances;

import java.awt.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import static de.embl.cba.cats.instances.InstancesAndMetadata.Metadata.*;

public abstract class LabelUtils
{

    public static int getNumClassesInLabels( ArrayList< de.embl.cba.cats.labels.Label > labels )
    {
        Set<Integer> classNums = new HashSet<>();

        for ( de.embl.cba.cats.labels.Label label : labels )
        {
            classNums.add( label.classNum );
        }

        return classNums.size();
    }


    public static Label createLabel( int classNum, Point[] points, int strokeWidth, int z, int t)
	{
		Label label = new Label( classNum, points, strokeWidth, z, t );
		return ( label );
	}

    public static Rectangle getLabelRectangleBounds( Label label )
	{
		int xMin = label.points[0].x;
		int xMax = label.points[0].x;
		int yMin = label.points[0].y;
		int yMax = label.points[0].y;

		for (Point point : label.points)
		{
			xMin = point.x < xMin ? point.x : xMin;
			xMax = point.x > xMax ? point.x : xMax;
			yMin = point.y < yMin ? point.y : yMin;
			yMax = point.y > yMax ? point.y : yMax;
		}

		xMin -= label.strokeWidth + 2; // +2 just to be on the save side
		xMax += label.strokeWidth + 2;
		yMin -= label.strokeWidth + 2;
		yMax += label.strokeWidth + 2;

		return (new Rectangle(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1));
	}

    public static void setLabelInstanceValuesAreCurrentlyBeingComputed( ArrayList< Label > labels, boolean b )
    {
        for ( Label label : labels )
        {
            label.instanceValuesAreCurrentlyBeingComputed = b;
        }
    }

    public static void clearInstancesValues( ArrayList< Label > labels )
	{
		for ( Label label : labels )
{
label.instanceValuesArrays = new ArrayList<>();
}
	}

    public void logLabelsInfo( ArrayList< de.embl.cba.cats.labels.Label > labels,
                                 ArrayList< String > classNames,
                                 Logger logger )
    {

        // add and report instances values
        int[] numLabelsPerClass = new int[classNames.size()];
        int[] numLabelPixelsPerClass = new int[classNames.size()];

        for ( de.embl.cba.cats.labels.Label label : labels )
        {
            numLabelsPerClass[ label.classNum ] += 1;
            numLabelPixelsPerClass[ label.classNum ] += label.instanceValuesArrays.get( 0 ).size();
        }

        logger.info("## Annotation information: ");
        for ( int iClass = 0; iClass < getNumClassesInLabels( labels ); iClass++)
        {
            logger.info(classNames.get(iClass) + ": "
                    + numLabelsPerClass[iClass] + " labels; "
                    + numLabelPixelsPerClass[iClass] + " pixels");
        }

    }



    public static ArrayList< de.embl.cba.cats.labels.Label > getLabelsFromInstancesAndMetadata( InstancesAndMetadata instancesAndMetadata,
                                                                                                                     boolean considerMultipleBoundingBoxOffsetsDuringInstancesLoading )
    {
        ArrayList< de.embl.cba.cats.labels.Label > labels = new ArrayList<>(  );

        Instances instances = instancesAndMetadata.getInstances();
        int iInstance = 0;

        while ( iInstance < instances.size() )
        {
            int label_id = ( int ) instancesAndMetadata.getMetadata( Metadata_Label_Id, iInstance );

            de.embl.cba.cats.labels.Label label = new Label(
                    ( int ) instances.get( iInstance ).classValue(),
                    null,
                    1,
                    ( int ) instancesAndMetadata.getMetadata( Metadata_Position_Z, iInstance ),
                    ( int ) instancesAndMetadata.getMetadata( Metadata_Position_T, iInstance )
            );

            label.instanceValuesArrays = new ArrayList<>();

            ArrayList< Point > points = new ArrayList<>();

            int iBoundingBoxOffset = 0;

            do
            {
                // TODO: this assumes that the instances are sorted
                // according to their label id...maybe this should be
                // ensured during loading

                Point point = new Point( ( int ) instancesAndMetadata.getMetadata( Metadata_Position_X, iInstance ), ( int ) instancesAndMetadata.getMetadata( Metadata_Position_Y, iInstance ) );

                if (  points.contains( point ) )
                {
                    if ( considerMultipleBoundingBoxOffsetsDuringInstancesLoading ) // TODO: fix this...
                    {
                        iBoundingBoxOffset++;
                    }
                    else
                    {
                        points.add( point );
                    }
                }
                else
                {
                    points.add( point );
                    iBoundingBoxOffset = 0;
                }

                if ( label.instanceValuesArrays.size() < iBoundingBoxOffset + 1 )
                {
                    label.instanceValuesArrays.add( new ArrayList<>( ) );
                }

                label.instanceValuesArrays.get( iBoundingBoxOffset ).add( instances.get( iInstance ).toDoubleArray() );
                iInstance++;

            } while ( iInstance < instances.size() && ( int ) instancesAndMetadata.getMetadata( Metadata_Label_Id, iInstance ) == label_id );

            label.points = points.toArray( new Point[ points.size() ]);
            labels.add( label );
        }

        return ( labels );
    }


}
