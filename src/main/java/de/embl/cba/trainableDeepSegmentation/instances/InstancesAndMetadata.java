package de.embl.cba.trainableDeepSegmentation.instances;

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
        Metadata_Settings_Binning_0,
        Metadata_Settings_Binning_1,
        Metadata_Settings_Binning_2,
        Metadata_Settings_Binning_3,
        Metadata_Settings_Binning_4,
        Metadata_Settings_Binning_5,
        Metadata_Settings_Binning_6,
        Metadata_Settings_Binning_7,
        Metadata_Settings_Binning_8,
        Metadata_Settings_Binning_9,
        Metadata_Settings_UseChannels,
        Metadata_Settings_MaxDeepConvLevel,
        Metadata_Settings_Log2;


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


    public Map< Metadata, ArrayList< Double > > getMetaData( )
    {
        return metadata;
    }

    public InstancesAndMetadata( Instances instances, Map< Metadata, ArrayList< Double > > metadata )
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
        if ( ! metadata.keySet().contains( aMetadata ) )
        {
            System.out.println( "ERROR: Metadata is not contained: " + aMetadata );
            return -1;
        }
        else
        {
            double value = metadata.get( aMetadata ).get( i );
            return value;
        }
    }

    public ArrayList< Double > getMetadata( Metadata aMetadata )
    {
        return metadata.get( aMetadata );
    }

    public void addMetadata( Metadata aMetadata, double value )
    {
        metadata.get( aMetadata ).add( value );
    }

    public void addInstance( Instance instance )
    {
        instances.add( instance );
    }

    public void moveMetadataFromInstancesToMetadata()
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

        int classIndex = instances.numAttributes() - 1;
        instances.setClassIndex( classIndex );
    }


    public Instances getOneInstanceWithMetadata()
    {
        putMetadataIntoInstances();
        Instances oneInstanceWithMetadata = new Instances( instances, 0 );
        oneInstanceWithMetadata.add( instances.get( 0 ) );
        moveMetadataFromInstancesToMetadata();

        return oneInstanceWithMetadata;
    }

    public Instances getInstances()
    {
        return instances;
    }

    public Instance getInstance( int i )
    {
        return instances.get( i );
    }

    public synchronized void append( InstancesAndMetadata instancesAndMetadata )
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

    public Map< Integer, ArrayList < Integer > >[] getLabelList()
    {

        Map< Integer, ArrayList < Integer > >[] labels =
                new LinkedHashMap[ instances.numClasses() ];

        for ( int c = 0; c < instances.numClasses(); ++c )
        {
            labels[c] = new LinkedHashMap<>(  );
        }

        ArrayList< Double > labelIds = metadata.get( Metadata.Metadata_Label_Id );
        for ( int i = 0; i < labelIds.size(); ++i )
        {
            int c = ( int ) instances.get( i ).classValue();
            int l = ( int ) ( double ) metadata.get( Metadata.Metadata_Label_Id ).get( i );

            if ( ! labels[ c ].containsKey( l ) )
            {
                labels[ c ].put( l, new ArrayList<>() );
            }

            labels[ c ].get( l ).add( i );
        }

        return labels;
    }

    public ArrayList< String > getClassNames()
    {
        ArrayList< String > names = new ArrayList<>(  );

        ArrayList< Object > classNames = Collections.list( instances.classAttribute().enumerateValues() );

        for ( Object className : classNames )
        {
            names.add( (String) className );
        }

        return names;

    }

}


