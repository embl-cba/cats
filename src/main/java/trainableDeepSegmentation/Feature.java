package trainableDeepSegmentation;

import java.io.Serializable;

/**
 * Created by tischi on 19/08/17.
 */
public class Feature implements Serializable {
    String featureName = null;
    Integer usageInRF = null;
    boolean isActive = true;

    public Feature(String featureName, Integer featureUsageInRF, boolean isActive) {
        this.featureName = featureName;
        this.usageInRF = featureUsageInRF;
        this.isActive = isActive;
    }

    public int getUsageInRF()
    {
        return(usageInRF);
    }

}
