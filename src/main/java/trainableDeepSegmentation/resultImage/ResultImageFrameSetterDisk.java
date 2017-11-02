package trainableDeepSegmentation.resultImage;

import net.imglib2.FinalInterval;

import static trainableDeepSegmentation.ImageUtils.*;

public class ResultImageFrameSetterDisk implements ResultImageFrameSetter {

    FinalInterval interval;
    byte[][][] resultChunk;
    ResultImageDisk resultImage;

    public ResultImageFrameSetterDisk( ResultImage resultImage, FinalInterval interval )
    {
        assert interval.min( T ) == interval.max( T );

        this.interval = interval;
        this.resultImage = (ResultImageDisk) resultImage;
        resultChunk = new byte[ (int) interval.dimension( Z ) ]
                [ (int) interval.dimension( Y ) ]
                [ (int) interval.dimension( X ) ];
    }

    @Override
    public void set( long x, long y, long z, int classId, double certainty )
    {
        int lutCertainty = (int) ( certainty * ( resultImage.CLASS_LUT_WIDTH - 1.0 ) );

        int classOffset = classId * resultImage.CLASS_LUT_WIDTH + 1;

        resultChunk[ (int) (z - interval.min( Z )) ]
                [ (int) (y - interval.min ( Y )) ]
                [ (int) (x - interval.min ( X )) ]
                = (byte) ( classOffset + lutCertainty );

    }

    @Override
    public void close()
    {
        resultImage.write3dResultChunk( interval, resultChunk );
    }

}
