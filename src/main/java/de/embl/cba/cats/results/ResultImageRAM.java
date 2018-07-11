package de.embl.cba.cats.results;

import de.embl.cba.bigDataTools.utils.Utils;
import de.embl.cba.cats.CATS;
import de.embl.cba.utils.logging.Logger;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;
import net.imglib2.util.Intervals;

import java.awt.image.ColorModel;

import static de.embl.cba.cats.utils.IntervalUtils.*;


public class ResultImageRAM implements ResultImage {

    public static final int CLASS_LUT_WIDTH = 10;

    ImagePlus result;
    CATS CATS;
    Logger logger;
    long[] dimensions;
    FinalInterval interval;

    public ResultImageRAM( CATS CATS, long[] dimensions )
    {
        this.CATS = CATS;
        this.logger = CATS.getLogger();
        this.result = de.embl.cba.cats.utils.Utils.create8bitImagePlus( dimensions );
        this.dimensions = dimensions;
        this.interval = null;
    }

    public ResultImageRAM( CATS CATS, FinalInterval interval )
    {
        this.CATS = CATS;
        this.logger = CATS.getLogger();
        this.interval = interval;
        this.dimensions = Intervals.dimensionsAsLongArray( interval );
        this.result = de.embl.cba.cats.utils.Utils.create8bitImagePlus( dimensions);
    }

    public long[] getDimensions()
    {
        return dimensions;
    }

    @Override
    public void exportResults( ResultExportSettings resultExportSettings )
    {
        resultExportSettings.classLutWidth = CLASS_LUT_WIDTH;
        resultExportSettings.logger = logger;
        resultExportSettings.resultImagePlus = result;
        resultExportSettings.resultImage = this;

        ResultUtils.exportResults( resultExportSettings );
    }

    @Override
    public ImageProcessor getSlice( int slice, int frame )
    {
        if ( interval != null )
        {
            slice -= interval.min( Z );
            frame -= interval.min( T );
        }

        int stackIndex = result.getStackIndex(  0, slice, frame );
        ImageProcessor ip = result.getStack().getProcessor( stackIndex );
        return ( ip );
    }

    public void setProcessor( ImageProcessor ip, int slice, int frame )
    {
        if ( interval != null )
        {
            slice -= interval.min( Z );
            frame -= interval.min( T );
        }

        int stackIndex = result.getStackIndex(  0, slice, frame );
        result.getStack().setProcessor( ip, stackIndex );
    }

    @Override
    public ResultImageFrameSetter getFrameSetter( FinalInterval interval )
    {
        return ( new ResultImageFrameSetterRAM( this, interval ) );
    }

    public void set( long x, long y, long z, long t, int classId, double certainty )
    {
        if ( interval != null )
        {
            x -= interval.min( X );
            y -= interval.min( Y );
            z -= interval.min( Z );
            t -= interval.min( T );
        }

        int lutCertainty = (int) ( certainty * ( CLASS_LUT_WIDTH - 1.0 ) );

        int classOffset = classId * CLASS_LUT_WIDTH + 1;

        int n = result.getStackIndex( 1, (int) z+1, (int) t+1 );
        result.getStack().getProcessor( n ).set( (int) x, (int) y, (byte) ( classOffset + lutCertainty ));
    }

    @Override
    public int getProbabilityRange()
    {
        return CLASS_LUT_WIDTH;
    }

    @Override
    public ImagePlus getDataCubeCopy( FinalInterval interval )
    {
        assert interval.min( C ) == interval.max( C );
        assert interval.min( T ) == interval.max( T );

        ImagePlus cube = Utils.getDataCube( result,
                convertIntervalToRegion5D( interval ) );

        return cube;
    }

    public ImagePlus getWholeImageCopy()
    {
        ImagePlus imp = result.duplicate();
        return imp;
    }

    @Override
    public FinalInterval getInterval()
    {
        return interval;
    }

    public ImagePlus getFrame( int frame )
    {
        ImagePlus imp = result;
        ImageStack stack = result.getStack();
        ImageStack stack2 = null;
        int c = 1;

        for(int slice = 1; slice <= imp.getNSlices(); ++slice) {
            int n1 = imp.getStackIndex(c, slice, frame);
            ImageProcessor ip = stack.getProcessor(n1);
            String label = stack.getSliceLabel(n1);
            if (stack2 == null) {
                stack2 = new ImageStack(ip.getWidth(), ip.getHeight(), (ColorModel )null);
            }
            stack2.addSlice(label, ip);
        }

        ImagePlus imp2 = imp.createImagePlus();
        imp2.setStack("Frame_" + frame + "_" + imp.getTitle(), stack2);
        imp2.setDimensions( 1, result.getNSlices(), 1);
        imp2.setOpenAsHyperStack(true);
        return imp2;
    }

}
