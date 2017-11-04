package trainableDeepSegmentation.training;

import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

public class TrainingDataManager {

    Map< String, TrainingData > trainingDataMap = null;

    public TrainingDataManager()
    {
        trainingDataMap = new HashMap<>();
    }

    public void setTrainingData( String key, TrainingData trainingData )
    {
        trainingDataMap.put( key, trainingData );
    }

    public Instances getInstances( String key )
    {
        return ( trainingDataMap.get( key ).instances );
    }




}
