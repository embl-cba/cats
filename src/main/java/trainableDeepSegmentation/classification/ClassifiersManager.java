package trainableDeepSegmentation.classification;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.*;

public class ClassifiersManager {

    HashMap< String, ClassInst > classifiers;


    public ClassifiersManager( )
    {
        this.classifiers = new HashMap< String, ClassInst >();
    }

    public void setClassifier( Classifier classifier, Instances instances )
    {
        Instances instancesHeader = new Instances( instances, 0, 1 );
        ClassInst classInst = new ClassInst( classifier, instancesHeader );
        classifiers.put( instancesHeader.relationName(), classInst );
    }

    public Classifier getClassifier( String key )
    {
        return ( classifiers.get(key).classifier );
    }


    public List< Attribute > getClassifierAttributes( String key )
    {
        return ( Collections.list( classifiers.get(key).instances.enumerateAttributes() ) );
    }

    public Set< String > getNames()
    {
        return ( classifiers.keySet() );
    }



    private class ClassInst
    {
        public Classifier classifier;
        public Instances instances;

        public ClassInst( Classifier classifier, Instances instances )
        {
            this.classifier = classifier;
            this.instances = instances;
        }
    }



}
