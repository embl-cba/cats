package trainableDeepSegmentation.resultImage;

import net.imglib2.FinalInterval;

import static trainableDeepSegmentation.ImageUtils.*;

public class ResultImageFrameSetterMemory implements ResultImageFrameSetter {

    FinalInterval interval;
    ResultImageMemory resultImage;

    public ResultImageFrameSetterMemory( ResultImage resultImage, FinalInterval interval )
    {
        assert interval.min( T ) == interval.max( T );
        
        this.interval = interval;
        this.resultImage = ( ResultImageMemory ) resultImage;
    }

    @Override
    public void set( long x, long y, long z, int classId, double certainty )
    {
        resultImage.set(  x,  y,  z, interval.min(T), classId, certainty );
    }

    @Override
    public void close()
    {
        // do nothing
    }

}
