package trainableDeepSegmentation.results;

import bigDataTools.VirtualStackOfStacks.VirtualStackOfStacks;
import bigDataTools.dataStreamingTools.DataStreamingTools;
import bigDataTools.logging.Logger;
import bigDataTools.utils.ImageDataInfo;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;
import trainableDeepSegmentation.IntervalUtils;
import trainableDeepSegmentation.WekaSegmentation;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import static trainableDeepSegmentation.results.Utils.saveImagePlusAsSeparateImarisChannels;

public class ResultImageDisk implements ResultImage {

    public static final int CLASS_LUT_WIDTH = 10;

    ImagePlus result;
    WekaSegmentation wekaSegmentation;
    Logger logger;
    long[] dimensions;

    public ResultImageDisk( WekaSegmentation wekaSegmentation,
                            String directory,
                            long[] dimensions)
    {
        this.wekaSegmentation = wekaSegmentation;
        this.logger = wekaSegmentation.getLogger();
        this.result = createStream( directory, dimensions );
        this.dimensions = dimensions;
        this.result.setCalibration( wekaSegmentation.getInputImage().getCalibration() );
    }


    @Override
    public void saveAsSeparateImarisChannels( String directory,
                                              ArrayList< Boolean > saveClass,
                                              int[] binning )
    {

        logger.info("Saving results as separate imaris channels.." );

        saveImagePlusAsSeparateImarisChannels(
                directory,
                saveClass,
                result,
                binning,
                logger,
                wekaSegmentation.getClassNames(),
                CLASS_LUT_WIDTH
        );

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
        return CLASS_LUT_WIDTH;
    }

    @Override
    public ImagePlus getDataCube( FinalInterval interval )
    {

        VirtualStackOfStacks vss = (VirtualStackOfStacks)result.getStack();
        ImagePlus dataCube = vss.getDataCube( IntervalUtils.convertIntervalToRegion5D( interval ),
                new int[] {-1,-1,}, 1 );
        return dataCube;
    }

    private ImagePlus createStream( String directory, long[] dimensions )
    {
        // TODO: check for cancel!

        DataStreamingTools dst = new DataStreamingTools();
        String tMax = String.format( "%05d", dimensions[ IntervalUtils.T ] );
        String zMax = String.format( "%05d", dimensions[ IntervalUtils.Z ] );

        String namingPattern = "classified--C<C01-01>--T<T00001-" +
                tMax + ">--Z<Z00001-"+zMax+">.tif";
        bigDataTools.utils.ImageDataInfo imageDataInfo = new ImageDataInfo();
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
            fileSaver.saveAsTiff( directory + "/" +
                    "classified--C01--T00001--Z00001.tif" );
        }

        ImagePlus result = dst.openFromDirectory(
                directory,
                namingPattern,
                "None",
                "None",
                imageDataInfo,
                nIOthreads,
                false,
                true);

        result.setDimensions(
                1,
                (int) dimensions[ IntervalUtils.Z ],
                (int) dimensions[ IntervalUtils.T ]);

        result.setOpenAsHyperStack(true);
        result.setTitle("classification_result");

        return ( result );
    }

    public void write3dResultChunk(
            FinalInterval interval,
            byte[][][] resultChunk )
    {
        assert interval.min( IntervalUtils.T ) == interval.max( IntervalUtils.T );

        VirtualStackOfStacks stack = ( VirtualStackOfStacks ) result.getStack();
        try
        {
            stack.saveByteCube( resultChunk, interval );
        }
        catch( IOException e)
        {
            wekaSegmentation.logger.warning(
                "ResultImage.write3dResultChunk: " + e.toString() );
        }
    }


    public ImagePlus getImagePlus()
    {
        logger.error( "Currently not implemented for disk resident result images." );
        return null;
    }
}
