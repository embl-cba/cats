package de.embl.cba.cats.utils;

import de.embl.cba.bigdataprocessor.utils.Region5D;
import ij.ImagePlus;
import ij.ImageStack;
import javafx.geometry.Point3D;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import de.embl.cba.cats.*;

import java.util.ArrayList;

public abstract class IntervalUtils {


    public static String[] dimNames = new String[]{"x", "y", "c", "z", "t"};
    public static int X = 0;
    public static int Y = 1;
    public static int C = 2;
    public static int Z = 3;
    public static int T = 4;

    public static int[] XY = new int[]{X, Y};
    public static int[] XYZ = new int[]{X, Y, Z};
    public static int[] XYZT = new int[]{X, Y, Z, T};

    // TODO: move to ResultUtils
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

    // TODO: move to ResultUtils
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
        CATS.logger.info("Interval: ");

        for ( int d : XYZT )
        {
            CATS.logger.info( dimNames[d] + ": " + interval.min(d) + ", " + interval.max(d));
        }

    }

    public static ArrayList<FinalInterval> getXYTiles( FinalInterval interval,
                                                       int nxy,
                                                       long[] imgDims )
    {

        CATS.logger.info("\n# Creating xy tiles");

        ArrayList<FinalInterval> tiles = new ArrayList<>();

        long[] tileSizes = new long[5];

        for ( int d : XY )
        {

            tileSizes[ d ] = (int) Math.ceil ( 1.0 * interval.dimension(d) / nxy );

            // make sure sizes fit into image
            tileSizes[d] = Math.min( tileSizes[d], imgDims[d] );

        }


        CATS.logger.info("Tile sizes [x,y]: "
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

        CATS.logger.info("Number of tiles: " + tiles.size());

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

    public static ArrayList<FinalInterval> createTiles( FinalInterval classificationInterval,
                                                        FinalInterval wholeImageInterval,
                                                        Integer numTiles,
                                                        int numFeatures,
                                                        boolean normalizeIntensities, // we do not
                                                        boolean doNotTileInXY,
                                                        CATS cats )
    {

        CATS.logger.info( "# Generating tiles for interval:" );
        logInterval( classificationInterval );

        ArrayList<FinalInterval> tiles = new ArrayList<>();

        long[] tileSizes = new long[5];

        double numTilesPerTimePoint = 1.0 * numTiles / classificationInterval.dimension( T );

        int volumeDimensionality = 3;
        if ( classificationInterval.dimension( Z ) == 1 ) volumeDimensionality = 2;

        if ( doNotTileInXY )
        {
            tileSizes[ X ] = classificationInterval.dimension( X );
            tileSizes[ Y ] = classificationInterval.dimension( Y );
            tileSizes[ Z ] = ( int ) Math.ceil( 1.0 * classificationInterval.dimension( Z ) / numTilesPerTimePoint );
        }
        else
        {
            for ( int d : XYZ )
            {
                if ( numTiles > 0 )
                {
                    tileSizes[ d ] = ( int ) Math.ceil( 1.0 * classificationInterval.dimension( d )
                            / Math.pow( numTilesPerTimePoint, 1.0 / volumeDimensionality ) );
                }
                else if ( classificationInterval.dimension( d ) <= cats.getMaximalRegionWidth( numFeatures ) )
                {
                    // everything can be computed at once
                    tileSizes[ d ] = classificationInterval.dimension( d );
                }
                else
                {
                    // we need to tile
                    int n = ( int ) Math.ceil( ( 1.0 * classificationInterval.dimension( d ) )
                            / cats.getMaximalRegionWidth( numFeatures ) );
                    tileSizes[ d ] = ( int ) Math.ceil( 1.0 * classificationInterval.dimension( d ) / n );
                }
            }
        }

        for ( int d : XYZ )
        {
            // make sure sizes fit into image
            tileSizes[ d ] = Math.min( tileSizes[ d ], wholeImageInterval.dimension( d ) );
        }


        tileSizes[ T ] = 1;

        CATS.logger.info("Tile sizes [x,y,z]: " + tileSizes[ X] + ", " + tileSizes[ Y]  + ", " + tileSizes[ Z ]);

        for ( int t = (int) classificationInterval.min( T ); t <= classificationInterval.max( T ); t += 1)
        {
            for ( int z = (int) classificationInterval.min( Z ); z <= classificationInterval.max( Z ); z += tileSizes[ Z ])
            {
                for ( int y = (int) classificationInterval.min( Y ); y <= classificationInterval.max( Y ); y += tileSizes[ Y ])
                {
                    for ( int x = (int) classificationInterval.min( X ); x <= classificationInterval.max( X ); x += tileSizes[ X ])
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
                            max[ d ] = Math.min( classificationInterval.max( d ), max[ d ] );
                        }

                        tiles.add( new FinalInterval(min, max) );

                    }
                }
            }
        }

        CATS.logger.info( "Number of tiles: " + tiles.size() );

        return ( tiles );
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

    public static FinalInterval getIntervalWithChannelsDimensionAsSingleton( ImagePlus imp )
    {
        long[] min = new long[5];
        long[] max = new long[5];

        max[ X ] = imp.getWidth() - 1;
        max[ Y ] = imp.getHeight() - 1;
        max[ Z ] = imp.getNSlices() - 1;
        min[ C ] = max[ C ] = 0; // Singleton, because classification resultImagePlus currently can only have one channel
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

    public static int[] getDimensions( ImagePlus imp )
    {
        int[] imgDims = new int[5];
        imgDims[ X ] = imp.getWidth();
        imgDims[ Y ] = imp.getHeight();
        imgDims[ C ] = imp.getNChannels();
        imgDims[ Z ] = imp.getNSlices();
        imgDims[ T ] = imp.getNFrames();
        return imgDims;
    }

    public static long getApproximatelyNeededBytesPerVoxel( double numFeatures )
    {
        long floatingPointBytes = 4; // 32-bit
        long neededBytesPerVoxel = (long) ( 2 * numFeatures * floatingPointBytes );
        return neededBytesPerVoxel;
    }

    public static long getApproximatelyNeededBytes( FinalInterval interval, double numFeatures )
    {
        long bytes = getApproximatelyNeededBytesPerVoxel( numFeatures );

        for ( int d = 0; d < interval.numDimensions(); ++d )
            bytes *= interval.dimension( d );

        return bytes;

    }

    public static String getIntervalAsCsvString( FinalInterval tile )
    {
        String intervalXYZT = "";
        for ( int d : XYZT )
        {
            intervalXYZT += tile.min( d ) + "," + tile.max( d );
            if (d != T) intervalXYZT += ",";
        }
        return intervalXYZT;
    }

    public static void ensureToStayWithinBounds( ImagePlus inputImage, long[] min, long[] max )
    {
        if ( min[ Z ] < 0 ) min[ Z ] = 0;
        if ( max[ Z ] > inputImage.getNSlices() - 1 ) max[ Z ] = inputImage.getNSlices() - 1;
    }
}
