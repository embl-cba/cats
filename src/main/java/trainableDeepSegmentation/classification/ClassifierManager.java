package trainableDeepSegmentation.classification;

import hr.irb.fastRandomForest.FastRandomForest;
import ij.IJ;
import trainableDeepSegmentation.WekaSegmentation;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.*;
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

    public Instances getInstancesHeader( String key )
    {
        Instances instancesHeader =
                new Instances( classifiers.get(key).instances, 0 );

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

        try
        {
            sFile = new File( filepath );
            OutputStream os = new FileOutputStream( sFile );
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
            objectOutputStream.writeObject( classifiers.get(key).classifier );
            objectOutputStream.writeObject( classifiers.get(key).instances );
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
            IJ.log("Saved classifier to " + filename);
        }

        return saveOK;
    }

    /**
     * Read header classifier from a .model file
     *
     * @param pathName complete path and file name
     * @return false if error
     */
    public String loadClassifier(String directory, String filename )
    {
        String filepath = directory + File.separator + filename;

        WekaSegmentation.logger.info("\n# Loading classifier from " + filepath + " ..." );

        FastRandomForest classifier = null;
        Instances instances = null;

        try
        {
            File selected = new File( filepath );

            InputStream is = new FileInputStream(selected);
            ObjectInputStream objectInputStream = new ObjectInputStream(is);
            classifier = (FastRandomForest) objectInputStream.readObject();
            instances = (Instances ) objectInputStream.readObject();
            objectInputStream.close();
        }
        catch (Exception e)
        {
            WekaSegmentation.logger.error("Error while loading classifier!");
            WekaSegmentation.logger.info(e.toString());
            return null;
        }


        String key = instances.relationName();

        classifiers.put( key, new ClassInst( classifier, instances ) );

        WekaSegmentation.logger.info( "Added classifier: " + filepath );
        WekaSegmentation.logger.info( "Classifier key: " + filepath );

        return key;
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
