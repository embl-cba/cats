package trainableDeepSegmentation.results;

import net.imglib2.FinalInterval;
import trainableDeepSegmentation.utils.IntervalUtils;

public class ResultImageFrameSetterDisk implements ResultImageFrameSetter {

    FinalInterval interval;
    byte[][][] resultChunk;
    ResultImageDisk resultImage;

    public ResultImageFrameSetterDisk( ResultImage resultImage, FinalInterval interval )
    {
        assert interval.min( IntervalUtils.T ) == interval.max( IntervalUtils.T );

        this.interval = interval;
        this.resultImage = (ResultImageDisk) resultImage;
        resultChunk = new byte[ (int) interval.dimension( IntervalUtils.Z ) ]
                [ (int) interval.dimension( IntervalUtils.Y ) ]
                [ (int) interval.dimension( IntervalUtils.X ) ];
    }

    @Override
    public void set( long x, long y, long z, int classId, double certainty )
    {
        int lutCertainty = (int) ( certainty * ( resultImage.CLASS_LUT_WIDTH - 1.0 ) );

        int classOffset = classId * resultImage.CLASS_LUT_WIDTH + 1;

        resultChunk[ (int) (z - interval.min( IntervalUtils.Z )) ]
                [ (int) (y - interval.min ( IntervalUtils.Y )) ]
                [ (int) (x - interval.min ( IntervalUtils.X )) ]
                = (byte) ( classOffset + lutCertainty );

    }

    @Override
    public void close()
    {
        resultImage.write3dResultChunk( interval, resultChunk );
    }

}
