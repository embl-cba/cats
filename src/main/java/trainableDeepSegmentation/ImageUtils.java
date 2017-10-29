package trainableDeepSegmentation;

import bigDataTools.Region5D;
import javafx.geometry.Point3D;
import net.imglib2.FinalInterval;

public abstract class ImageUtils {


    public static String[] dimNames = new String[]{"x", "y", "c", "z", "t"};
    public static int X = 0;
    public static int Y = 1;
    public static int C = 2;
    public static int Z = 3;
    public static int[] XYZ = new int[]{X, Y, Z};
    public static int T = 4;
    public static int[] XYZT = new int[]{X, Y, Z, T};

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
