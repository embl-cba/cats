package trainableDeepSegmentation.instances;

import trainableDeepSegmentation.Settings;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class InstancesAndMetadata {

    Instances instances;
    Map< Metadata, ArrayList< Double > > metadata;

    public void removeMetadataFromInstances( )
    {
        int numAttributes = instances.numAttributes();

        for ( int i = numAttributes - 1; i >= 0; --i )
        {
            Attribute attribute = instances.attribute( i );
            if ( Metadata.contains( attribute.name() ) )
            {
                instances.deleteAttributeAt( i );
            }
        }

        instances.setClassIndex( instances.numAttributes() - 1 );

    }

    public void putMetadataIntoInstances( )
    {
        int numInstances = instances.size();

        for ( Metadata metadata : Metadata.values() )
        {
            Attribute attribute = new Attribute( metadata.name() );
            instances.insertAttributeAt( attribute, 0 );
            for ( int i = 0; i < numInstances; ++i )
            {
                instances.get( i ).setValue( 0, getMetadata( metadata,  i )  );
            }
        }
    }

    public enum Metadata {
        Metadata_Position_X,
        Metadata_Position_Y,
        Metadata_Position_Z,
        Metadata_Position_T,
        Metadata_Label_Id,
        Metadata_Settings_ImageBackground,
        Metadata_Settings_Anisotropy,
        Metadata_Settings_MaxBinLevel,
        Metadata_Settings_BinFactor,
        Metadata_Settings_MaxDeepConvLevel;

        public static boolean contains( String test )
        {
            for ( Metadata c : Metadata.values() ) {
                if ( c.name().equals(test) ) {
                    return true;
                }
            }
            return false;
        }
    }


    public InstancesAndMetadata( Instances instances,
                                 Map< Metadata, ArrayList< Double > > metadata )
    {
        this.instances = instances;
        this.metadata = metadata;
    }

    public InstancesAndMetadata( Instances instances )
    {
        this.instances = instances;
        this.metadata = getEmptyMetadata();
    }

    public static Map< Metadata, ArrayList< Double > > getEmptyMetadata()
    {
        Map< Metadata, ArrayList< Double > > metadata = new HashMap<>(  );

        for ( Metadata aMetadata : Metadata.values() )
        {
            ArrayList< Double > list = new ArrayList<>();
            metadata.put( aMetadata, list );
        }

        return metadata;
    }

    public double getMetadata( Metadata aMetadata, int i )
    {
        return metadata.get( aMetadata ).get( i );
    }

    public void addMetadata( Metadata aMetadata, double value )
    {
        metadata.get( aMetadata ).add( value );
    }

    public void addInstance( Instance instance )
    {
        instances.add( instance );
    }

    public void moveMetadataOutOfInstances()
    {
        int numAttributes = instances.numAttributes();

        for ( int i = numAttributes - 1; i >= 0; --i )
        {
            Attribute attribute = instances.attribute( i );

            if ( Metadata.contains( attribute.name() ) )
            {
                for ( Instance instance : instances )
                {
                    metadata.get( Metadata.valueOf( attribute.name() ) ).add( instance.value( i ) );
                }
                instances.deleteAttributeAt( i );
            }

        }

        // set class attribute
        instances.setClassIndex( instances.numAttributes() - 1 );
    }

    public Instances getInstances()
    {
        return instances;
    }

    public Instance getInstance( int i )
    {
        return instances.get( i );
    }

    public void appendInstancesAndMetadata( InstancesAndMetadata instancesAndMetadata )
    {

        for( Instance instance : instancesAndMetadata.instances )
        {
            instances.add( instance );
        }

        for( Metadata aMetadata : instancesAndMetadata.metadata.keySet() )
        {
            for ( double value : instancesAndMetadata.metadata.get( aMetadata) )
            {
                metadata.get( aMetadata ).add( value );
            }
        }


    }

    public ArrayList< String > getAttributeNames()
    {
        ArrayList< String > names = new ArrayList<>(  );

        ArrayList< Attribute > attributes = Collections.list( instances.enumerateAttributes() );
        for ( Attribute attribute : attributes )
        {
            names.add( attribute.name() );
        }

        return names;

    }

}


