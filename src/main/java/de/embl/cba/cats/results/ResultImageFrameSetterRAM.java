package de.embl.cba.cats.results;

import net.imglib2.FinalInterval;
import de.embl.cba.cats.utils.IntervalUtils;

public class ResultImageFrameSetterRAM implements ResultImageFrameSetter {

    FinalInterval interval;
    ResultImageRAM resultImage;

    public ResultImageFrameSetterRAM( ResultImage resultImage, FinalInterval interval )
    {
        assert interval.min( IntervalUtils.T ) == interval.max( IntervalUtils.T );
        
        this.interval = interval;
        this.resultImage = ( ResultImageRAM ) resultImage;
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
