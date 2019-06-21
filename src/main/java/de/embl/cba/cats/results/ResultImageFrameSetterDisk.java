package de.embl.cba.cats.results;

import net.imglib2.FinalInterval;
import de.embl.cba.cats.utils.IntervalUtils;

public class ResultImageFrameSetterDisk implements ResultImageFrameSetter {

    FinalInterval interval;
    byte[][][] resultChunk;
    ResultImageDisk resultImageDisk;

    public ResultImageFrameSetterDisk( ResultImage resultImageDisk, FinalInterval interval )
    {
        assert interval.min( IntervalUtils.T ) == interval.max( IntervalUtils.T );

        this.interval = interval;
        this.resultImageDisk = (ResultImageDisk) resultImageDisk;
        resultChunk = new byte[ (int) interval.dimension( IntervalUtils.Z ) ]
                [ (int) interval.dimension( IntervalUtils.Y ) ]
                [ (int) interval.dimension( IntervalUtils.X ) ];
    }

    @Override
    public void set( long x, long y, long z, int classId, double certainty )
    {
        int lutCertainty = (int) ( certainty * ( ResultImageSettings.CLASS_LUT_WIDTH - 1.0 ) );

        int classOffset = classId * ResultImageSettings.CLASS_LUT_WIDTH + 1;

        resultChunk[ (int) (z - interval.min( IntervalUtils.Z )) ]
                [ (int) (y - interval.min ( IntervalUtils.Y )) ]
                [ (int) (x - interval.min ( IntervalUtils.X )) ]
                = (byte) ( classOffset + lutCertainty );

    }

    @Override
    public void close()
    {
        resultImageDisk.write3dResultChunk( interval, resultChunk );
    }

}
