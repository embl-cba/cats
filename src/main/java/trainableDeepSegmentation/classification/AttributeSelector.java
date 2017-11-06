package trainableDeepSegmentation.classification;

import bigDataTools.logging.Logger;
import hr.irb.fastRandomForest.FastRandomForest;
import trainableDeepSegmentation.Feature;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Comparator;

public abstract class AttributeSelector {


    public static ArrayList< Integer > getGoners( FastRandomForest classifier,
                                           Instances instances,
                                           double minUsageFactor,
                                           Logger logger)
    {

        double randomUsage = 1.0 *
                classifier.getDecisionNodes() / instances.numAttributes();

        int usageThreshold = (int) Math.ceil( minUsageFactor * randomUsage );

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
            logger.info("Required usage factor: " + minUsageFactor);
            logger.info("Attributes: " + instances.numAttributes());
            logger.info("Nodes: " + classifier.getDecisionNodes());
            logger.info("Random usage = nodes / attributes: " + randomUsage);
            logger.info("Usage threshold = random * factor: " + usageThreshold);
            logger.info("Resulting number of removed attributes: " + goners.size());
            logger.info("Resulting number of kept attributes: " + (instances.numAttributes() - goners.size()));

        }

        return ( goners );
    }
}
