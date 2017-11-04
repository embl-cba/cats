package trainableDeepSegmentation.training;

import bigDataTools.logging.IJLazySwingLogger;
import bigDataTools.logging.Logger;
import ij.IJ;
import trainableDeepSegmentation.Feature;
import trainableDeepSegmentation.examples.Example;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;

import static trainableDeepSegmentation.examples.ExamplesUtils.getNumClassesInExamples;

public class InstancesCreator {

    Logger logger = new IJLazySwingLogger();

    public InstancesCreator( Logger logger )
    {
        this.logger = logger;
    }

    /**
     * Create training instances out of the user markings
     *
     * @return set of instances (feature vectors in Weka format)
     */
    public static Instances createInstancesFromExamples( ArrayList< Example > examples,
                                                         String relationName,
                                                         ArrayList< String > featureNames,
                                                  ArrayList< String > classNames )
    {

        Instances instances = createInstancesHeader(
                relationName,
                featureNames,
                classNames  );

        for ( Example example : examples )
        {
            // loop over all pixels of the example
            // and add the feature values for each pixel to the trainingData
            // note: subsetting of active features happens in another function
            for ( double[] values : example.instanceValuesArray )
            {
                instances.add( new DenseInstance(1.0, values) );
            }
        }

        instances.setRelationName( relationName );

        return instances;

    }


    public static Instances createInstancesHeader(
            String instancesName,
            ArrayList< String > featureNames,
            ArrayList< String > classNames  )
    {

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for ( String feature : featureNames )
        {
            attributes.add( new Attribute(feature) );
        }
        attributes.add( new Attribute("class", classNames ) );

        // initialize set of instances
        Instances instances = new Instances(instancesName, attributes, 1);
        // Set the index of the class attribute
        instances.setClassIndex( featureNames.size() );

        return ( instances );

    }



    public static Instances removeAttributes( Instances instances,
                                              ArrayList< Integer > goners )
    {


        Instances attributeSubset = new Instances( instances );

        for( int j = goners.size() - 1; j >= 0; j-- )
        {
            int id = goners.get( j );
            attributeSubset.deleteAttributeAt( id );
        }

        return ( attributeSubset );
    }


}
