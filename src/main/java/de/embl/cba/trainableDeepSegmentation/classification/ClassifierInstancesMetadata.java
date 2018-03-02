package de.embl.cba.trainableDeepSegmentation.classification;

import de.embl.cba.trainableDeepSegmentation.instances.InstancesAndMetadata;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Map;

public class ClassifierInstancesMetadata
{
    public Classifier classifier;
    public InstancesAndMetadata instancesAndMetadata;

    public ClassifierInstancesMetadata( )
    {
        this.classifier = null;
        this.instancesAndMetadata = null;
    }

    public ClassifierInstancesMetadata( Classifier classifier, InstancesAndMetadata instancesAndMetadata)
    {
        this.classifier = classifier;

        // TODO: save only one instance and corresponding metadata
        //        final Instances oneInstance = new Instances( instancesAndMetadata.getInstances(), 1 );
        //     Map< InstancesAndMetadata.Metadata, ArrayList< Double > > metaData = instancesAndMetadata.getMetaData( );

        this.instancesAndMetadata = instancesAndMetadata;
    }


}
