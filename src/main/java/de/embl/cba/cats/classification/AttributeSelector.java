package de.embl.cba.cats.classification;

import de.embl.cba.classifiers.weka.FastRandomForest;
import de.embl.cba.log.Logger;
import weka.core.Instances;

import java.util.ArrayList;

import static de.embl.cba.cats.classification.ClassifierUtils.getAttributesSortedByUsage;

public abstract class AttributeSelector {

    public static ArrayList< Integer > getGonersBasedOnUsage(
            FastRandomForest classifier,
            Instances instances,
            double minRelativeUsage,
            int minAbsoluteUsage,
            Logger logger)
    {

        double randomUsage = 1.0 * classifier.getDecisionNodes() / instances.numAttributes();

        int usageThreshold = 0;

        if ( minAbsoluteUsage > 0 )
        {
            usageThreshold = minAbsoluteUsage;
        }
        else if ( minRelativeUsage > 0 )
        {
            usageThreshold = (int) Math.ceil( minRelativeUsage * randomUsage );
        }

        ArrayList< Integer > goners = new ArrayList<>();

        int[] actualUsage = classifier.getAttributeUsages();

        // last attribute is class attribute => -1
        for ( int i = 0; i < instances.numAttributes() - 1 ; ++i )
        {
            if ( actualUsage[i] < usageThreshold )
            {
                goners.add( i );
            }
        }

        if ( logger != null )
        {
            logger.info("\n# AttributeSelector: ");
            logger.info("Required usage factor: " + minRelativeUsage);
            logger.info("Attributes: " + instances.numAttributes());
            logger.info("Nodes: " + classifier.getDecisionNodes());
            logger.info("Random usage = nodes / attributes: " + randomUsage);
            logger.info("Usage threshold = random * factor: " + usageThreshold);
            logger.info("Resulting number of removed attributes: " + goners.size());
            logger.info("Resulting number of kept attributes: " + (instances.numAttributes() - goners.size()));

        }

        return ( goners );
    }


    public static ArrayList< Integer > getMostUsedAttributes(
            FastRandomForest classifier,
            Instances instances,
            int n,
            Logger logger)
    {


        ClassifierUtils.NamesAndUsages[] namesAndUsages = getAttributesSortedByUsage( classifier, instances );

        ArrayList< Integer > mostUsed = new ArrayList<>(  );

        for ( int i = 0; i < n; ++i )
        {
            mostUsed.add( namesAndUsages[i].index );
        }

        return mostUsed;
    }


}
