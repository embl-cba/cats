package trainableDeepSegmentation.instances;

import weka.core.Instances;

import java.util.*;

public class InstancesManager {

    SortedMap< String, InstancesAndMetadata> instancesMap = null;

    public InstancesManager()
    {
        instancesMap = new TreeMap<>();
    }

    public synchronized String putInstancesAndMetadata( InstancesAndMetadata instancesAndMetadata )
    {
        String key = getName( instancesAndMetadata.instances );

        instancesMap.put( key, instancesAndMetadata );

        return key;
    }

    public synchronized void putInstancesAndMetadata( String key, InstancesAndMetadata instancesAndMetadata )
    {
        instancesMap.put( key, instancesAndMetadata );
    }

    public Instances getInstances( String key )
    {
        return ( instancesMap.get( key ).instances );
    }

    public InstancesAndMetadata getInstancesAndMetadata(String key )
    {
        return ( instancesMap.get( key ) );
    }

    public Set< String > getKeys()
    {
        return ( instancesMap.keySet() );
    }

    public InstancesAndMetadata getCombinedInstancesAndMetadata(List< String > keys )
    {

        InstancesAndMetadata combinedIAM = new InstancesAndMetadata( new Instances( getInstances( keys.get( 0 ) ) , 0 ), InstancesAndMetadata.getEmptyMetadata() );

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
