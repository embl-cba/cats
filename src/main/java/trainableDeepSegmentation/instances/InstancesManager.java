package trainableDeepSegmentation.instances;

import bigDataTools.logging.IJLazySwingLogger;
import bigDataTools.logging.Logger;
import trainableDeepSegmentation.IntervalUtils;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.*;

public class InstancesManager {


    public enum Metadata {
        X("Metadata_X"),
        Y("Metadata_Y"),
        Z("Metadata_Z"),
        T("Metadata_T"),
        LabelID("Metadata_LabelID");
        private final String text;
        private Metadata(String s) {
            text = s;
        }
        @Override
        public String toString() {
            return text;
        }
    }


    Logger logger = new IJLazySwingLogger();

    SortedMap< String, InstancesAndMetadata > instancesMap = null;

    public InstancesManager()
    {
        instancesMap = new TreeMap<>();
    }

    public static Map< String, ArrayList< Double > > getEmptyMetadata()
    {
        Map< String, ArrayList< Double > > metadata = new HashMap<>(  );

        for ( Metadata item : Metadata.values() )
        {
            ArrayList< Double > list = new ArrayList<>();
            metadata.put( item.toString(), list );
        }

        return metadata;
    }

    public String putInstances( Instances instances )
    {
        String key = instances.relationName().split( "--" )[0];

        instancesMap.put( key, new InstancesAndMetadata( instances, null ) );

        return key;
    }

    public String putInstancesAndMetadata( Instances instances,
                                           Map< String, ArrayList< Double > >  metadata )
    {
        String key = instances.relationName().split( "--" )[0];

        instancesMap.put( key, new InstancesAndMetadata( instances, metadata ) );

        return key;
    }

    public String putInstancesAndMetadata( InstancesAndMetadata instancesAndMetadata )
    {
        String key = instancesAndMetadata.instances.relationName().split( "--" )[0];

        instancesMap.put( key, instancesAndMetadata );

        return key;
    }



    public String appendInstances( Instances newInstances )
    {
        String key = newInstances.relationName().split( "--" )[0];

        Instances instances = instancesMap.get( key );

        for( Instance instance : newInstances )
        {
            instances.add( instance );
        }

        return key;
    }

    public Instances getInstances( String key )
    {
        return ( instancesMap.get( key ).instances );
    }

    public Set< String > getKeys()
    {
        return ( instancesMap.keySet() );
    }

    /**
     * Write current instancesMap into an ARFF file
     * @param instances set of instancesMap
     * @param filename ARFF file name
     */
    private boolean saveInstancesToARFF( Instances instances,
                                         String directory,
                                         String filename)
    {

        BufferedWriter out = null;
        try{
            out = new BufferedWriter(
                    new OutputStreamWriter(
                            new FileOutputStream( directory
                                    + File.separator + filename ) ) );

            final Instances header = new Instances(instances, 0);
            out.write( header.toString() );

            for(int i = 0; i < instances.numInstances(); i++)
            {
                out.write(instances.get(i).toString()+"\n");
            }
        }
        catch(Exception e)
        {
            logger.error("Error: couldn't write instancesMap into .ARFF file.");
            e.printStackTrace();
            return false;
        }
        finally{
            try {
                out.close();
            } catch (IOException e) {
                e.printStackTrace();
                return false;
            }
        }

        return true;
    }

    public boolean saveInstancesToARFF( String key,
                                        String directory,
                                        String filename)
    {
        // TODO: add metadata here
        Instances instances = instancesMap.get( key ).instances;
        Map< String, ArrayList< Double > >  metadata = instancesMap.get( key ).metadata;

        //appendMetadata( instances, metadata );

        boolean status = saveInstancesToARFF( instancesMap.get( key ).instances,
                directory, filename );

        return status;
    }

    public Instances getCombinedInstances( List< String > keys )
    {

        Instances combined = new Instances( getInstances( keys.get( 0 ) ), 0 );;

        for ( int i = 0; i < keys.size(); ++i )
        {
            Instances nextInstances = getInstances( keys.get(i) );

            for( Instance instance : nextInstances )
            {
                combined.add( instance );
            }

        }

        return combined;

    }

    private void extractFeatureSettingsFromInstances()
    {
        // maybe simply put the settings into each feature name....

        // Check the features that were used in the loaded data
        /*
        Enumeration<Attribute> attributes = loadedTrainingData.enumerateAttributes();
        final int numFeatures = FeatureStack.availableFeatures.length;
        boolean[] usedFeatures = new boolean[numFeatures];
        while(attributes.hasMoreElements())
        {
            final Attribute a = attributes.nextElement();
            for(int i = 0 ; i < numFeatures; i++)
                if(a.name().startsWith(FeatureStack.availableFeatures[i]))
                    usedFeatures[i] = true;
        }*/
    }

    public String putInstancesFromARFF( String directory, String filename )
    {
        InstancesAndMetadata instancesAndMetadata
                = loadInstancesFromARFF( directory, filename );

        if ( instancesAndMetadata == null ) return null;

        String key = getName( instancesAndMetadata.instances );
        instancesMap.put( key, instancesAndMetadata );

        return key;
    }

    private String getName( Instances instances )
    {
        String name = instances.relationName().split( "--" )[0];

        return name;
    }

    /**
     * Read ARFF file
     * @param filename ARFF file name
     * @return set of instancesMap read from the file
     */
    private InstancesAndMetadata loadInstancesFromARFF( String directory, String filename )
    {
        String pathname = directory + File.separator + filename;

        logger.info("Loading data from " + pathname + "...");

        try{
            BufferedReader reader = new BufferedReader(
                    new FileReader( pathname ));
            try{
                Instances instances = new Instances( reader );
                reader.close();

                // TODO: separate metadata off
                ArrayList< double[] > metadata =
                        separateMetadata( instances );

                // set class attribute
                instances.setClassIndex( instances.numAttributes() - 1 );

                return ( new InstancesAndMetadata( instances, metadata ));
            }
            catch(IOException e)
            {
                logger.error("IOException");
            }
        }
        catch(FileNotFoundException e)
        {
            logger.error("File not found!");
        }

        return null;
    }

    private static ArrayList< double[] > separateMetadata( Instances instances )
    {
        ArrayList< String > metadata = new ArrayList<>(  );

        int n = instances.numAttributes();

        for ( Instance instance : instances )
        {

        }

        instances.deleteAttributeAt( 0 );

        int numAttributes = instances.numAttributes();
        int a = 1;

        return ( metadata );
    }

    public void appendMetadata( Instances instances, ArrayList< double[] > metadata )
    {
        Attribute attribute = new Attribute( "z-position" );
        instances.insertAttributeAt( attribute, IntervalUtils.X );

        for ( Instance instance : instances )
        {

        }
    }

    public static void logInstancesInformation( Instances instances, Logger logger )
    {
        logger.info( "\n# Instances information" );
        logger.info( "Number of instances: " + instances.size() );
        logger.info( "Number of attributes: " + instances.numAttributes() );

        // TODO: output per class a.s.o

    }



}
