package de.embl.cba.cats.results;

import de.embl.cba.bigdataprocessor.BigDataProcessor;
import de.embl.cba.bigdataprocessor.utils.ImageDataInfo;
import de.embl.cba.bigdataprocessor.virtualstack2.VirtualStack2;
import de.embl.cba.cats.CATS;
import de.embl.cba.cats.utils.IntervalUtils;
import de.embl.cba.log.Logger;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;

import java.io.File;
import java.util.ArrayList;

public class ResultImageDisk implements ResultImage {

    ImagePlus result;
    CATS CATS;
    Logger logger;
    long[] dimensions;
    File directory;

    public ResultImageDisk( CATS CATS, String directory, long[] dimensions)
    {
        this.directory = new File( directory );
        this.CATS = CATS;
        this.logger = CATS.getLogger();
        this.result = createStream( directory, dimensions );
        this.dimensions = dimensions;
        this.result.setCalibration( CATS.getInputImage().getCalibration() );
    }

    public long[] getDimensions()
    {
        return dimensions;
    }

    @Override
    public ArrayList< ImagePlus > exportResults( ResultExportSettings resultExportSettings )
    {
        resultExportSettings.classLutWidth = ResultImageSettings.CLASS_LUT_WIDTH;
        resultExportSettings.logger = logger;
        resultExportSettings.resultImagePlus = result;
        resultExportSettings.resultImage = this;
        return ResultExport.exportResults( resultExportSettings );
    }

    @Override
    public ImageProcessor getSlice( int slice, int frame )
    {
        int stackIndex = result.getStackIndex(  0, slice, frame );
        ImageProcessor ip = result.getStack().getProcessor( stackIndex );
        return ( ip );
    }

    @Override
    public ResultImageFrameSetter getFrameSetter( FinalInterval interval )
    {
        return ( new ResultImageFrameSetterDisk( this, interval ) );
    }

    @Override
    public int getProbabilityRange()
    {
        return ResultImageSettings.CLASS_LUT_WIDTH;
    }

    @Override
    public ImagePlus getDataCubeCopy( FinalInterval interval )
    {
        VirtualStack2 vss = (VirtualStack2) result.getStack();
        ImagePlus dataCube = vss.getDataCube( IntervalUtils.convertIntervalToRegion5D( interval ), 1 );
        return dataCube;
    }

    private ImagePlus createStream( String directory, long[] dimensions )
    {
        // TODO: check for cancel!

        BigDataProcessor bdc = new BigDataProcessor();
        String tMax = String.format( "%05d", dimensions[ IntervalUtils.T ] );
        String zMax = String.format( "%05d", dimensions[ IntervalUtils.Z ] );

        String namingPattern = "classified--C<C01-01>--T<T00001-" + tMax + ">--Z<Z00001-"+zMax+">.tif";
        ImageDataInfo imageDataInfo = new ImageDataInfo();
        imageDataInfo.bitDepth = 8;
        int nIOthreads = 3;

        String[] list = new File(directory).list();
        if (list == null || list.length == 0)
        {
            // empty directory => create one empty image
            ImageStack stack = ImageStack.create(
                    ( int ) dimensions[ IntervalUtils.X ],
                    ( int ) dimensions[ IntervalUtils.Y ],
                    1, 8 );
            ImagePlus impC0T0Z0 = new ImagePlus( "", stack );
            FileSaver fileSaver = new FileSaver( impC0T0Z0 );
            fileSaver.saveAsTiff( directory + "/" + "classified--C01--T00001--Z00001.tif" );
        }

        ImagePlus result = bdc.openFromDirectory(
                directory,
                namingPattern,
                "None",
                "None",
                imageDataInfo,
                nIOthreads,
                false,
                true);

        result.setDimensions( 1, (int) dimensions[ IntervalUtils.Z ], (int) dimensions[ IntervalUtils.T ]);
        result.setOpenAsHyperStack( true );
        result.setTitle( "classification_result" );

        return ( result );
    }

    public void write3dResultChunk( FinalInterval interval, byte[][][] resultChunk )
    {
        assert interval.min( IntervalUtils.T ) == interval.max( IntervalUtils.T );

        VirtualStack2 stack = ( VirtualStack2 ) result.getStack();
        stack.saveByteCube( resultChunk, interval );

    }

    public ImagePlus getWholeImageCopy()
    {
        ImagePlus imp = result.duplicate();
        return imp;
    }

    @Override
    public FinalInterval getInterval()
    {
        return null;
    }

    public File getDirectory()
    {
        return directory;
    }

}
