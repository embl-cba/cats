package trainableDeepSegmentation.examples;

import bigDataTools.logging.Logger;
import ij.IJ;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public abstract class ExamplesUtils {


    public static int getNumClassesInExamples( ArrayList< Example > examples )
    {
        Set<Integer> classNums = new HashSet<>();

        for (Example example : examples )
        {
            classNums.add( example.classNum );
        }
        return classNums.size();
    }

    public static int[] getNumExamplesPerClass( ArrayList< Example > examples )
    {
        int[] numExamplesPerClass = new int[ getNumClassesInExamples(examples) ];

        for (Example example : examples )
        {
            numExamplesPerClass[example.classNum]++;
        }

        return (numExamplesPerClass);
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





}
