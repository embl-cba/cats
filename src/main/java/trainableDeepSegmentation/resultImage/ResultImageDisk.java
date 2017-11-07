package trainableDeepSegmentation.resultImage;

import bigDataTools.*;
import bigDataTools.VirtualStackOfStacks.VirtualStackOfStacks;
import bigDataTools.dataStreamingTools.DataStreamingTools;
import bigDataTools.logging.Logger;
import bigDataTools.utils.ImageDataInfo;
import bigDataTools.utils.Utils;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.plugin.Duplicator;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;
import trainableDeepSegmentation.WekaSegmentation;

import java.io.IOException;
import java.util.ArrayList;

import static trainableDeepSegmentation.ImageUtils.*;
import static trainableDeepSegmentation.resultImage.Utils.saveImagePlusAsSeparateImarisChannels;

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
    }


    @Override
    public void saveAsSeparateImarisChannels( String directory,
                                              ArrayList< Boolean > saveClass )
    {

        logger.info("Saving results as separate imaris channels.." );

        saveImagePlusAsSeparateImarisChannels(
                directory,
                saveClass,
                result,
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

    private ImagePlus createStream( String directory, long[] dimensions )
    {
        // TODO: check for cancel!

        DataStreamingTools dst = new DataStreamingTools();
        String tMax = String.format( "%05d", dimensions[ T ] );
        String zMax = String.format( "%05d", dimensions[ Z ] );

        String namingPattern = "classified--C<C01-01>--T<T00001-" +
                tMax + ">--Z<Z00001-"+zMax+">.tif";
        bigDataTools.utils.ImageDataInfo imageDataInfo = new ImageDataInfo();
        imageDataInfo.bitDepth = 8;
        int nIOthreads = 3;

        // create one image
        ImageStack stack = ImageStack.create(
                (int) dimensions[ X ],
                (int) dimensions[ Y ],
                1, 8);
        ImagePlus impC0T0Z0 = new ImagePlus("", stack);
        FileSaver fileSaver = new FileSaver( impC0T0Z0 );
        fileSaver.saveAsTiff( directory + "/" +
                "classified--C01--T00001--Z00001.tif");

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
                (int) dimensions[ Z ],
                (int) dimensions[ T ]);

        result.setOpenAsHyperStack(true);
        result.setTitle("classification_result");

        return ( result );
    }

    public void write3dResultChunk(
            FinalInterval interval,
            byte[][][] resultChunk )
    {
        assert interval.min( T ) == interval.max( T );

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


}
