package trainableDeepSegmentation;

import java.io.Serializable;

/**
 * Created by tischi on 19/08/17.
 */
public class Feature implements Serializable {
    String name = null;
    Integer usageInRF = null;
    boolean isActive = true;

    public Feature(String featureName, Integer featureUsageInRF, boolean isActive) {
        this.name = featureName;
        this.usageInRF = featureUsageInRF;
        this.isActive = isActive;
    }

    public int getUsageInRF()
    {
        return(usageInRF);
    }

}
