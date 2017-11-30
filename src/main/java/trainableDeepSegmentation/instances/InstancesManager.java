package trainableDeepSegmentation.instances;

import weka.core.Instances;

import java.util.*;

public class InstancesManager {

    SortedMap< String, InstancesMetadata > instancesMap = null;

    public InstancesManager()
    {
        instancesMap = new TreeMap<>();
    }

    public synchronized String putInstancesAndMetadata( InstancesMetadata instancesAndMetadata )
    {
        String key = getName( instancesAndMetadata.instances );

        instancesMap.put( key, instancesAndMetadata );

        return key;
    }

    public Instances getInstances( String key )
    {
        return ( instancesMap.get( key ).instances );
    }

    public InstancesMetadata getInstancesAndMetadata( String key )
    {
        return ( instancesMap.get( key ) );
    }

    public Set< String > getKeys()
    {
        return ( instancesMap.keySet() );
    }

    public InstancesMetadata getCombinedInstancesAndMetadata( List< String > keys )
    {
        // initialize empty IAM
        InstancesMetadata combinedIAM = new InstancesMetadata(
                new Instances( getInstances( keys.get( 0 ) ) , 0 ),
                InstancesMetadata.getEmptyMetadata() );

        for ( String key : keys )
        {
            combinedIAM.append( getInstancesAndMetadata( key ) );
        }

        return combinedIAM;

    }

    private String getName( Instances instances )
    {
        String name = instances.relationName().split( "--" )[0];

        return name;
    }


}
