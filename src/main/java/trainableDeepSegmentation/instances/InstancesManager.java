package trainableDeepSegmentation.instances;

import bigDataTools.logging.IJLazySwingLogger;
import bigDataTools.logging.Logger;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class InstancesManager {

    Logger logger = new IJLazySwingLogger();

    SortedMap< String, InstancesAndMetadata > instancesMap = null;

    public InstancesManager()
    {
        instancesMap = new TreeMap<>();
    }

    public String putInstances( Instances instances )
    {
        String key = instances.relationName().split( "--" )[0];

        instancesMap.put( key, new InstancesAndMetadata( instances, null ) );

        return key;
    }

    public String putInstancesAndMetadata( InstancesAndMetadata instancesAndMetadata )
    {
        String key = getName( instancesAndMetadata.instances );

        instancesMap.put( key, instancesAndMetadata );

        return key;
    }

    public Instances getInstances( String key )
    {
        return ( instancesMap.get( key ).instances );
    }

    public InstancesAndMetadata getInstancesAndMetadata( String key )
    {
        return ( instancesMap.get( key ) );
    }

    public Set< String > getKeys()
    {
        return ( instancesMap.keySet() );
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

    private String getName( Instances instances )
    {
        String name = instances.relationName().split( "--" )[0];

        return name;
    }


}
