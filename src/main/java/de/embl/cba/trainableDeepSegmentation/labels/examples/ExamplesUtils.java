package de.embl.cba.trainableDeepSegmentation.labels.examples;

import de.embl.cba.bigDataTools.logging.Logger;
import de.embl.cba.trainableDeepSegmentation.instances.InstancesAndMetadata;
import weka.core.Instances;

import java.awt.*;
import java.util.ArrayList;
import java.util.HashSet;
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
            numExamplesPerClass[example.classNum] += 1;
            numExamplePixelsPerClass[example.classNum] += example.instanceValuesArray.size();
        }

        logger.info("## Annotation information: ");
        for (int iClass = 0; iClass < getNumClassesInExamples( examples ); iClass++)
        {
            logger.info(classNames.get(iClass) + ": "
                    + numExamplesPerClass[iClass] + " labels; "
                    + numExamplePixelsPerClass[iClass] + " pixels");
        }

    }



    public static ArrayList< Example > getExamplesFromInstancesAndMetadata(
            InstancesAndMetadata instancesAndMetadata )
    {
        ArrayList< Example > examples = new ArrayList<>(  );

        Instances instances = instancesAndMetadata.getInstances();
        int i = 0;

        while ( i < instances.size() )
        {
            int label_id = ( int ) instancesAndMetadata.getMetadata( Metadata_Label_Id, i );

            Example example = new Example(
                    ( int ) instances.get( i ).classValue(),
                    null,
                    1,
                    ( int ) instancesAndMetadata.getMetadata( Metadata_Position_Z, i ),
                    ( int ) instancesAndMetadata.getMetadata( Metadata_Position_T, i )
            );

            example.instanceValuesArray = new ArrayList<>();
            ArrayList< Point > points = new ArrayList<>();

            do
            {
                // TODO: this assumes that the instances are sorted
                // according to their label id...maybe this should be
                // ensured during loading
                example.instanceValuesArray.add( instances.get( i ).toDoubleArray() );
                points.add( new Point(
                        ( int ) instancesAndMetadata.getMetadata( Metadata_Position_X, i ),
                        ( int ) instancesAndMetadata.getMetadata( Metadata_Position_Y, i )
                ) );

                i++;
            } while (
                    i < instances.size() &&
                    ( int ) instancesAndMetadata.getMetadata( Metadata_Label_Id, i ) == label_id );

            example.points = points.toArray( new Point[ points.size() ]);
            examples.add( example );
        }

        return ( examples );
    }


}
