package de.embl.cba.trainableDeepSegmentation.labels.examples;

import de.embl.cba.utils.logging.Logger;
import de.embl.cba.trainableDeepSegmentation.instances.InstancesAndMetadata;
import weka.core.Instances;

import java.awt.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

import static de.embl.cba.trainableDeepSegmentation.instances.InstancesAndMetadata.Metadata.*;
import static de.embl.cba.trainableDeepSegmentation.instances.InstancesAndMetadata.Metadata.Metadata_Label_Id;

public abstract class ExamplesUtils {


    public static int getNumClassesInExamples( ArrayList< Example > examples )
    {
        Set<Integer> classNums = new HashSet<>();

        for ( Example example : examples )
        {
            classNums.add( example.classNum );
        }

        return classNums.size();
    }

    public static int[] getNumExamplesPerClass( ArrayList< Example > examples )
    {
        int[] numExamplesPerClass = new int[ getNumClassesInExamples( examples ) ];

        for (Example example : examples )
        {
            numExamplesPerClass[ example.classNum ]++;
        }

        return ( numExamplesPerClass );
    }

    public void logExamplesInfo( ArrayList< Example > examples,
                                 ArrayList< String > classNames,
                                 Logger logger )
    {

        // add and report instances values
        int[] numExamplesPerClass = new int[classNames.size()];
        int[] numExamplePixelsPerClass = new int[classNames.size()];

        for ( Example example : examples )
        {
            numExamplesPerClass[ example.classNum ] += 1;
            numExamplePixelsPerClass[ example.classNum ] += example.instanceValuesArrays.get( 0 ).size();
        }

        logger.info("## Annotation information: ");
        for (int iClass = 0; iClass < getNumClassesInExamples( examples ); iClass++)
        {
            logger.info(classNames.get(iClass) + ": "
                    + numExamplesPerClass[iClass] + " labels; "
                    + numExamplePixelsPerClass[iClass] + " pixels");
        }

    }



    public static ArrayList< Example > getExamplesFromInstancesAndMetadata( InstancesAndMetadata instancesAndMetadata )
    {
        ArrayList< Example > examples = new ArrayList<>(  );

        Instances instances = instancesAndMetadata.getInstances();
        int iInstance = 0;

        while ( iInstance < instances.size() )
        {
            int label_id = ( int ) instancesAndMetadata.getMetadata( Metadata_Label_Id, iInstance );

            Example example = new Example(
                    ( int ) instances.get( iInstance ).classValue(),
                    null,
                    1,
                    ( int ) instancesAndMetadata.getMetadata( Metadata_Position_Z, iInstance ),
                    ( int ) instancesAndMetadata.getMetadata( Metadata_Position_T, iInstance )
            );

            example.instanceValuesArrays = new ArrayList<>();

            Set< Point > points = new LinkedHashSet<>() ;

            int iBoundingBoxOffset = 0;

            do
            {
                // TODO: this assumes that the instances are sorted
                // according to their label id...maybe this should be
                // ensured during loading

                Point point = new Point( ( int ) instancesAndMetadata.getMetadata( Metadata_Position_X, iInstance ), ( int ) instancesAndMetadata.getMetadata( Metadata_Position_Y, iInstance ) );

                if (  points.contains( point ) )
                {
                    iBoundingBoxOffset++;
                }
                else
                {
                    points.add( point );
                    iBoundingBoxOffset = 0;
                }

                if ( example.instanceValuesArrays.size() < iBoundingBoxOffset + 1 )
                {
                    example.instanceValuesArrays.add( new ArrayList<>( ) );
                }


                example.instanceValuesArrays.get( iBoundingBoxOffset ).add( instances.get( iInstance ).toDoubleArray() );
                iInstance++;

            } while ( iInstance < instances.size() && ( int ) instancesAndMetadata.getMetadata( Metadata_Label_Id, iInstance ) == label_id );

            example.points = points.toArray( new Point[ points.size() ]);
            examples.add( example );
        }

        return ( examples );
    }


}
