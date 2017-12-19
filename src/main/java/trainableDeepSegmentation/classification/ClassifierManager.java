package trainableDeepSegmentation.classification;

import hr.irb.fastRandomForest.FastRandomForest;
import trainableDeepSegmentation.WekaSegmentation;
import trainableDeepSegmentation.instances.InstancesMetadata;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.*;
import java.util.*;

public class ClassifierManager {

    Map< String, ClassifierInstancesMetadata > classifiers;

    public ClassifierManager( )
    {
        this.classifiers = new LinkedHashMap<>();
    }

    public String setClassifier( ClassifierInstancesMetadata classifierInstancesMetadata )
    {

        String key = classifierInstancesMetadata
                .instancesMetadata
                .getInstances()
                .relationName();

        classifiers.put( key , classifierInstancesMetadata );

        return key;
    }

    public String setClassifier( Classifier classifier,
                                 InstancesMetadata instancesMetadata )
    {
        ClassifierInstancesMetadata classifierInstancesMetadata =
                new ClassifierInstancesMetadata( classifier, instancesMetadata );

        return setClassifier( classifierInstancesMetadata );

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

        return ( Collections.list(
                classifiers
                .get(key)
                .instancesMetadata
                .getInstances()
                .enumerateAttributes() ) );

    }

    public Instances getInstancesHeader( String key )
    {
        Instances instancesHeader =
                new Instances( classifiers.get(key).instancesMetadata.getInstances() );

        return ( instancesHeader );
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

    public boolean saveClassifier( String key, String directory, String filename )
    {
        String filepath = directory + File.separator + filename;

        File sFile = null;
        boolean saveOK = true;

        WekaSegmentation.logger.info("\n# Saving classifier to " + filepath + " ..." );


        classifiers.get(key).instancesMetadata.putMetadataIntoInstances();

        try
        {
            sFile = new File( filepath );
            OutputStream os = new FileOutputStream( sFile );
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
            objectOutputStream.writeObject( classifiers.get(key).classifier );
            objectOutputStream.writeObject( classifiers.get(key).instancesMetadata.getInstances() );
            objectOutputStream.flush();
            objectOutputStream.close();
        }
        catch (Exception e)
        {
            WekaSegmentation.logger.error("Error when saving classifier to disk");
            WekaSegmentation.logger.info(e.toString());
            saveOK = false;
        }

        if (saveOK)
        {
            WekaSegmentation.logger.info("...done!");
        }

        classifiers.get(key).instancesMetadata.moveMetadataFromInstancesToMetadata();

        return saveOK;
    }


}
