package de.embl.cba.cats.classification;

import de.embl.cba.cats.CATS;
import de.embl.cba.cats.instances.InstancesAndMetadata;
import de.embl.cba.classifiers.weka.FastRandomForest;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.*;

public class ClassifierManager {

    Map< String, ClassifierInstancesMetadata > classifiers;

    String mostRecentlyAddedClassifierKey;

    public ClassifierManager( )
    {
        this.classifiers = new LinkedHashMap<>();
    }

    public String setClassifier( ClassifierInstancesMetadata classifierInstancesMetadata )
    {
        String key = classifierInstancesMetadata.instancesAndMetadata.getInstances().relationName();

        classifiers.put( key , classifierInstancesMetadata );

        mostRecentlyAddedClassifierKey = key;

        return key;
    }

    public String setClassifier( Classifier classifier, InstancesAndMetadata instancesAndMetadata)
    {
        ClassifierInstancesMetadata classifierInstancesMetadata = new ClassifierInstancesMetadata( classifier, instancesAndMetadata );

        return setClassifier( classifierInstancesMetadata );
    }

    public FastRandomForest getClassifier( String key )
    {
        return ( ( FastRandomForest) classifiers.get( key ).classifier );
    }

    public String getMostRecentClassifierKey( )
    {
        return mostRecentlyAddedClassifierKey;
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
        Instances instancesHeader = new Instances(
                classifiers.get( key ).instancesAndMetadata.getInstances(), 1 );

        return ( instancesHeader );
    }

    public ArrayList< String > getClassifierAttributeNames( String key )
    {
        if ( ! classifiers.containsKey( key ) ) return null;

        List< Attribute > attributes = getClassifierAttributes( key );

        ArrayList< String > attributeNames = new ArrayList<>();

        for ( Attribute attribute : attributes )
            attributeNames.add( attribute.name() );

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

        CATS.logger.info("\n# Saving classifier to " + filepath + " ..." );

        try
        {
            sFile = new File( filepath );
            OutputStream os = new FileOutputStream( sFile );
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
            objectOutputStream.writeObject( classifiers.get(key).classifier );
            objectOutputStream.writeObject( classifiers.get(key).instancesAndMetadata.getOneInstanceWithMetadata() );
            objectOutputStream.flush();
            objectOutputStream.close();
        }
        catch (Exception e)
        {
            CATS.logger.error("Error when saving classifier to disk");
            CATS.logger.info(e.toString());
            saveOK = false;
        }

        if (saveOK)
        {
            CATS.logger.info("...done!");
        }

        return saveOK;
    }


}
