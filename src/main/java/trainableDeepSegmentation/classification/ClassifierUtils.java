package trainableDeepSegmentation.classification;

import hr.irb.fastRandomForest.FastRandomForest;
import trainableDeepSegmentation.WekaSegmentation;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class ClassifierUtils {

    public static void reportClassifierCharacteristics(
            FastRandomForest classifier,
            Instances instances)
    {

        WekaSegmentation.logger.info( "\n# Classifier characteristics");

        int numDecisionNodes = classifier.getDecisionNodes();

        int[] usages = classifier.getAttributeUsages();
        ArrayList< Attribute > attributes =
                Collections.list( instances.enumerateAttributes() );

        NamesAndUsages[] namesAndUsages = new NamesAndUsages[usages.length];

        for ( int i = 0; i < usages.length; ++i )
        {
            namesAndUsages[i] = new NamesAndUsages(usages[ i ], attributes.get( i ).name() );
        }

        Arrays.sort( namesAndUsages, Collections.reverseOrder() );

        double avgRfTreeSize = 1.0 * numDecisionNodes / classifier.getNumTrees();

        double avgTreeDepth = 1.0 + Math.log( avgRfTreeSize ) / Math.log( 2.0 );
        if ( avgRfTreeSize < 1 ) avgTreeDepth = 1.0;

        double randomUsage = 1.0 * numDecisionNodes / instances.numAttributes();

        int batchSizePercent = Integer.parseInt( classifier.getBatchSize() );

        WekaSegmentation.logger.info("## Most used features: ");
        for (int f = Math.min( 9, namesAndUsages.length - 1 ); f >= 0; f--)
        {
            WekaSegmentation.logger.info(
                    String.format("Relative usage: %.1f", namesAndUsages[f].usage / randomUsage ) +
                            "; Absolute usage: " + namesAndUsages[f].usage +
                            "; ID: " + "" + "; Name: " + namesAndUsages[f].name );
        }

        WekaSegmentation.logger.info("## Further information: ");

        WekaSegmentation.logger.info("Average number of decision nodes per tree: " +
                avgRfTreeSize);

        WekaSegmentation.logger.info( "Average tree depth: log2(numDecisionNodes) + 1 = " +
                avgTreeDepth );


        WekaSegmentation.logger.info("Number of features: " + instances.numAttributes() );

        WekaSegmentation.logger.info("Batch size [%]: " + batchSizePercent );

        WekaSegmentation.logger.info("Number of instances: " + instances.size() );

        WekaSegmentation.logger.info("Average number of instances per tree = numInstances * batchSizePercent/100: "
                + batchSizePercent / 100.0 * instances.size() );

        // Print classifier information
        WekaSegmentation.logger.info( classifier.toString() );

    }

    public static class NamesAndUsages implements Comparable< NamesAndUsages > {

        public int usage;
        public String name;

        public NamesAndUsages( int usage, String name) {
            this.usage = usage;
            this.name = name;
        }

        @Override
        public int compareTo(NamesAndUsages o) {
            return (int)(this.usage - o.usage);
        }
    }
}
