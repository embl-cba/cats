package trainableDeepSegmentation.training;

import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class InstancesManager {

    Map< String, Instances > instances = null;

    public InstancesManager()
    {
        instances = new HashMap<>();
    }

    public void setInstances( Instances instances )
    {
        this.instances.put( instances.relationName(), instances );
    }

    public Instances getInstances( String key )
    {
        return ( instances.get( key ) );
    }

    public Set< String > getNames()
    {
        return ( instances.keySet() );
    }

}
