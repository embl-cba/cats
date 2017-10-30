package trainableDeepSegmentation.resultImage;

import net.imglib2.FinalInterval;

import static trainableDeepSegmentation.ImageUtils.*;

public class ResultImageFrameSetterRAM implements ResultImageFrameSetter {

    FinalInterval interval;
    ResultImageRAM resultImage;

    public ResultImageFrameSetterRAM( ResultImage resultImage, FinalInterval interval )
    {
        this.interval = interval;
        this.resultImage = ( ResultImageRAM ) resultImage;
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
