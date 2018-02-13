package de.embl.cba.trainableDeepSegmentation.classification;

import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.instances.InstancesAndMetadata;
import de.embl.cba.trainableDeepSegmentation.weka.fastRandomForest.FastRandomForest;
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
                .instancesAndMetadata
                .getInstances()
                .relationName();

        classifiers.put( key , classifierInstancesMetadata );

        return key;
    }

    public String setClassifier( Classifier classifier,
                                 InstancesAndMetadata instancesAndMetadata)
    {
        ClassifierInstancesMetadata classifierInstancesMetadata =
                new ClassifierInstancesMetadata( classifier, instancesAndMetadata);

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
                .instancesAndMetadata
                .getInstances()
                .enumerateAttributes() ) );

    }

    public Instances getInstancesHeader( String key )
    {
        Instances instancesHeader =
                new Instances( classifiers.get(key).instancesAndMetadata.getInstances() );

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

        DeepSegmentation.logger.info("\n# Saving classifier to " + filepath + " ..." );


        classifiers.get(key).instancesAndMetadata.putMetadataIntoInstances();

        try
        {
            sFile = new File( filepath );
            OutputStream os = new FileOutputStream( sFile );
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
            objectOutputStream.writeObject( classifiers.get(key).classifier );
            objectOutputStream.writeObject( classifiers.get(key).instancesAndMetadata.getInstances() );
            objectOutputStream.flush();
            objectOutputStream.close();
        }
        catch (Exception e)
        {
            DeepSegmentation.logger.error("Error when saving classifier to disk");
            DeepSegmentation.logger.info(e.toString());
            saveOK = false;
        }

        if (saveOK)
        {
            DeepSegmentation.logger.info("...done!");
        }

        classifiers.get(key).instancesAndMetadata.moveMetadataFromInstancesToMetadata();

        return saveOK;
    }


}