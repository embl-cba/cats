package trainableDeepSegmentation;

import de.embl.cba.bigDataTools.Region5D;
import ij.ImagePlus;
import ij.ImageStack;
import javafx.geometry.Point3D;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;

import java.util.ArrayList;

public abstract class IntervalUtils {


    public static String[] dimNames = new String[]{"x", "y", "c", "z", "t"};
    public static int X = 0;
    public static int Y = 1;
    public static int C = 2;
    public static int Z = 3;
    public static int[] XY = new int[]{X, Y};
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
    public static FinalInterval getIntervalByReplacingValues( Interval interval, int d, long minValue, long maxValue)
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

    public static ArrayList<FinalInterval> getXYTiles( FinalInterval interval,
                                                       int nxy,
                                                       long[] imgDims )
    {

        WekaSegmentation.logger.info("\n# Creating xy tiles");

        ArrayList<FinalInterval> tiles = new ArrayList<>();

        long[] tileSizes = new long[5];

        for ( int d : XY )
        {

            tileSizes[ d ] = (int) Math.ceil ( 1.0 * interval.dimension(d) / nxy );

            // make sure sizes fit into image
            tileSizes[d] = Math.min( tileSizes[d], imgDims[d] );

        }


        WekaSegmentation.logger.info("Tile sizes [x,y]: "
                + tileSizes[ X ]
                + ", " + tileSizes[ Y ] );


        for ( int y = (int) interval.min( Y ); y <= interval.max( Y ); y += tileSizes[ Y ])
        {
            for ( int x = (int) interval.min( X ); x <= interval.max( X ); x += tileSizes[ X ])
            {

                long[] min = new long[5];
                min[ X ] = x;
                min[ Y ] = y;
                min[ Z ] = interval.min( Z );
                min[ T ] = interval.min( T );

                long[] max = new long[5];
                max[ X ] = x + tileSizes[ X ] - 1;
                max[ Y ] = y + tileSizes[ Y ] - 1;
                max[ Z ] = interval.max( Z );
                max[ T ] = interval.max( T );

                // make sure to stay within image bounds
                for ( int d : XY )
                {
                    max[ d ] = Math.min( interval.max( d ), max[ d ] );
                }

                tiles.add( new FinalInterval(min, max) );

            }
        }

        WekaSegmentation.logger.info("Number of tiles: " + tiles.size());

        return (tiles);
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

    public static ArrayList<FinalInterval> getTiles( FinalInterval interval,
                                                     Integer numTiles,
                                                     WekaSegmentation ws)
    {

        logInterval( interval );

        ArrayList<FinalInterval> tiles = new ArrayList<>();

        long[] imgDims = ws.getInputImageDimensions();
        long[] tileSizes = new long[5];

        for ( int d : XYZ )
        {
            if ( numTiles > 0 )
            {
                tileSizes[ d ] = (int) Math.ceil ( 1.0 * interval.dimension(d) / Math.pow( numTiles, 1.0/3.0 ) );
            }
            else if ( interval.dimension(d) <= ws.getMaximalRegionSize() )
            {
                // everything can be computed at once
                tileSizes[d] = interval.dimension(d);
            }
            else
            {
                // we need to tile
                int n = (int) Math.ceil( (1.0 * interval.dimension(d)) / ws.getMaximalRegionSize());
                tileSizes[ d ] = (int) Math.ceil ( 1.0 * interval.dimension(d) / n );
            }

            // make sure sizes fit into image
            tileSizes[d] = Math.min( tileSizes[d], imgDims[d] );
        }

        tileSizes[ T ] = 1;

        WekaSegmentation.logger.info("Tile sizes [x,y,z]: "
                + tileSizes[ X]
                + ", " + tileSizes[ Y]
                + ", " + tileSizes[ Z]);


        for ( int t = (int) interval.min( T); t <= interval.max( T); t += 1)
        {
            for ( int z = (int) interval.min( Z); z <= interval.max( Z); z += tileSizes[ Z])
            {
                for ( int y = (int) interval.min( Y); y <= interval.max( Y); y += tileSizes[ Y])
                {
                    for ( int x = (int) interval.min( X); x <= interval.max( X); x += tileSizes[ X])
                    {
                        long[] min = new long[5];
                        min[ X ] = x;
                        min[ Y ] = y;
                        min[ Z ] = z;
                        min[ T ] = t;

                        long[] max = new long[5];
                        max[ X ] = x + tileSizes[ X ] - 1;
                        max[ Y ] = y + tileSizes[ Y ] - 1;
                        max[ Z ] = z + tileSizes[ Z ] - 1;
                        max[ T ] = t + tileSizes[ T ] - 1;

                        // make sure to stay within image bounds
                        for ( int d : XYZT )
                        {
                            max[ d ] = Math.min( interval.max( d ), max[ d ] );
                        }

                        tiles.add( new FinalInterval(min, max) );

                    }
                }
            }
        }

        WekaSegmentation.logger.info("Number of tiles: " + tiles.size());

        return (tiles);
    }

    public static FinalInterval getInterval( ImagePlus imp )
    {
        long[] min = new long[5];
        long[] max = new long[5];

        max[ X ] = imp.getWidth() - 1;
        max[ Y ] = imp.getHeight() - 1;
        max[ Z ] = imp.getNSlices() - 1;
        max[ C ] = imp.getNChannels() - 1;
        max[ T ] = imp.getNFrames() - 1;

        return new FinalInterval( min, max );
    }

    public static FinalInterval getEmptyInterval( )
    {
        long[] min = new long[5];
        long[] max = new long[5];
        return new FinalInterval( min, max );
    }


    public static ImagePlus createImagePlus( FinalInterval interval )
    {

        ImageStack stack = ImageStack.create(
                (int) interval.dimension( X ),
                (int) interval.dimension( Y ),
                (int) interval.dimension( Z ),
                8 );

        return new ImagePlus( "empty", stack );
    }

}
