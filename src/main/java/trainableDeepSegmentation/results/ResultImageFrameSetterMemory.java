package trainableDeepSegmentation.results;

import net.imglib2.FinalInterval;
import trainableDeepSegmentation.IntervalUtils;

public class ResultImageFrameSetterMemory implements ResultImageFrameSetter {

    FinalInterval interval;
    ResultImageMemory resultImage;

    public ResultImageFrameSetterMemory( ResultImage resultImage, FinalInterval interval )
    {
        assert interval.min( IntervalUtils.T ) == interval.max( IntervalUtils.T );
        
        this.interval = interval;
        this.resultImage = ( ResultImageMemory ) resultImage;
    }

    @Override
    public void set( long x, long y, long z, int classId, double certainty )
    {
        resultImage.set(  x,  y,  z, interval.min( IntervalUtils.T), classId, certainty );
    }

    @Override
    public void close()
    {
        // do nothing
    }

}
