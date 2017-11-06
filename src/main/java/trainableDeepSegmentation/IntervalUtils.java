package trainableDeepSegmentation;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;

import static trainableDeepSegmentation.ImageUtils.XYZT;
import static trainableDeepSegmentation.ImageUtils.dimNames;

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

    public static void logInterval( FinalInterval interval )
    {
        WekaSegmentation.logger.info("Interval: ");

        for ( int d : XYZT )
        {
            WekaSegmentation.logger.info( dimNames[d] + ": " + interval.min(d) + ", " + interval.max(d));
        }

    }
}
