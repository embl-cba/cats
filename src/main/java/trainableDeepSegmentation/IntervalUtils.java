package trainableDeepSegmentation;

import bigDataTools.Region5D;
import javafx.geometry.Point3D;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;

public abstract class IntervalUtils {


    public static String[] dimNames = new String[]{"x", "y", "c", "z", "t"};
    public static int X = 0;
    public static int Y = 1;
    public static int C = 2;
    public static int Z = 3;
    public static int[] XYZ = new int[]{X, Y, Z};
    public static int T = 4;
    public static int[] XYZT = new int[]{X, Y, Z, T};

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

    // TODO: move to Utils
    public static FinalInterval replaceValues( Interval interval, int d, long minValue, long maxValue)
    {

        int n = interval.numDimensions();
        long[] min = new long[n];
        long[] max = new long[n];
        interval.min(min);
        interval.max(max);

        min[d] = minValue;
        max[d] = maxValue;

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

    public static Region5D convertIntervalToRegion5D( FinalInterval interval )
    {
        Region5D region5D = new Region5D();

        region5D.offset = new Point3D(
                interval.min( X ),
                interval.min( Y ),
                interval.min( Z ) );
        region5D.size = new Point3D(
                interval.dimension( X ),
                interval.dimension( Y ),
                interval.dimension( Z ) );
        region5D.c = (int) interval.min( C );
        region5D.t = (int) interval.min( T );
        region5D.subSampling = new Point3D( 1, 1, 1);

        return ( region5D );
    }
}
