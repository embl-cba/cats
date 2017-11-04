package trainableDeepSegmentation.training;

import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

public class InstancesManager {

    Map< String, Instances > instancesMap = null;

    public InstancesManager()
    {
        instancesMap = new HashMap<>();
    }

    public void setInstances( Instances instances )
    {
        instancesMap.put( instances.relationName(), instances );
    }

    public Instances getInstances( String key )
    {
        return ( instancesMap.get( key ) );
    }




}
