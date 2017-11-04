package trainableDeepSegmentation.training;

import weka.core.Instances;

import java.util.Map;

public class TrainingData {

    public final Instances instances;
    public final Map< String, String > metaData;

    public TrainingData( Instances instances, Map< String, String > metaData )
    {
        this.instances = instances;
        this.metaData = metaData;
    }

}
