package trainableDeepSegmentation.classification;

import hr.irb.fastRandomForest.FastRandomForest;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.*;

public class ClassifierManager {

    Map< String, ClassInst > classifiers;


    public ClassifierManager( )
    {
        this.classifiers = new LinkedHashMap<>();
    }

    public String setClassifier( Classifier classifier, Instances instances )
    {
        Instances instancesHeader = new Instances( instances, 0, 1 );
        ClassInst classInst = new ClassInst( classifier, instancesHeader );
        String key = instancesHeader.relationName();
        classifiers.put( key , classInst );
        return key;
    }

    public FastRandomForest getClassifier( String key )
    {
        return ( ( FastRandomForest) classifiers.get(key).classifier );
    }

    public String getMostRecentClassifierKey( )
    {
        if ( classifiers.keySet().size() > 0 )
        {
            String lastKey = ( String ) classifiers.keySet().toArray()[ classifiers.keySet().size() - 1 ];
            return ( lastKey );
        }
        else
        {
            return null;
        }
    }


    public ArrayList< Attribute > getClassifierAttributes( String key )
    {
        if ( ! classifiers.containsKey( key ) ) return null;

        return ( Collections.list( classifiers.get(key).instances.enumerateAttributes() ) );
    }

    public ArrayList< Attribute > getClassifierAttributesIncludingClass( String key )
    {

        if ( ! classifiers.containsKey( key ) ) return null;

        ArrayList< Attribute > attributes = Collections.list( classifiers.get(key).instances.enumerateAttributes() );
        attributes.add ( classifiers.get(key).instances.classAttribute() );
        return ( attributes );
    }

    public ArrayList< String > getClassifierAttributeNames( String key )
    {
        if ( ! classifiers.containsKey( key ) ) return null;

        List< Attribute > attributes = getClassifierAttributes( key );

        ArrayList< String > attributeNames = new ArrayList<>();

        for ( Attribute attribute : attributes )
        {
            attributeNames.add( attribute.name() );
        }

        return ( attributeNames );
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
