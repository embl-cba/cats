package trainableDeepSegmentation.classification;

import trainableDeepSegmentation.instances.InstancesMetadata;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class ClassifierInstancesMetadata
{
    public Classifier classifier;
    public InstancesMetadata instancesMetadata;

    public ClassifierInstancesMetadata( )
    {
        this.classifier = null;
        this.instancesMetadata = null;
    }

    public ClassifierInstancesMetadata( Classifier classifier,
                                        InstancesMetadata instancesMetadata )
    {
        this.classifier = classifier;
        this.instancesMetadata = instancesMetadata;
    }


}
