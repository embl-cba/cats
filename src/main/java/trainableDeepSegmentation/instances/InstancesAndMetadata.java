package trainableDeepSegmentation.instances;

import weka.core.Instances;

import java.util.ArrayList;
import java.util.Map;

public class InstancesAndMetadata {


    Instances instances;
    Map< String, ArrayList< Double > > metadata;

    public InstancesAndMetadata( Instances instances,
                                 Map< String, ArrayList< Double > > metadata )
    {
        this.instances = instances;
        this.metadata = metadata;
    }
}
