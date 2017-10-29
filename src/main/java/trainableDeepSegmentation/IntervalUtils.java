package trainableDeepSegmentation;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;

public abstract class IntervalUtils {


    // TODO: move to Utils
    public static FinalInterval fixDimension( Interval interval, int d, long value)
    {

        int n = interval.numDimensions();
        long[] min = new long[n];
        long[] max = new long[n];
        interval.min(min);
        interval.max(max);

        min[d] = value;
        max[d] = value;

        return new FinalInterval(min, max);
    }

}
