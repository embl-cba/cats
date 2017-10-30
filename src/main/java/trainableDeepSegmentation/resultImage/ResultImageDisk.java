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

    private void saveClassAsImaris( int classId, String directory )
    {

        String className = wekaSegmentation.getClassName( classId );

        ImarisDataSet imarisDataSet = new ImarisDataSet();

        imarisDataSet.setFromImagePlus( result,
                new int[]{1,1,1},
                directory,
                className,
                "/");

        ImarisWriter.writeHeader( imarisDataSet,
                directory,
                className + ".ims"
        );


        Hdf5DataCubeWriter writer = new Hdf5DataCubeWriter();

        for ( int t = 0; t < dimensions[ T ]; ++t )
        {

            ImagePlus impClass = getClassImage( classId, t );

            writer.writeImarisCompatibleResolutionPyramid(
                    impClass,
                    imarisDataSet,
                    0, t );

            logger.progress( "Wrote " + className+ " time-point:",
                     (t+1) + "/" + dimensions[ T ] );
        }
    }

    private ImagePlus getClassImage( int classId, int t )
    {
        ImagePlus impClass;

        Duplicator duplicator = new Duplicator();
        impClass = duplicator.run( result, 1, 1, 1, result.getNSlices(), t+1, t+1 );

        int[] intensityGate = new int[]
                { classId * CLASS_LUT_WIDTH + 1, (classId + 1 ) * CLASS_LUT_WIDTH };
        Utils.applyIntensityGate( impClass, intensityGate );

        return ( impClass );

    }


    @Override
    public void saveAsSeparateImarisChannels( String directory,
                                              ArrayList< Boolean > saveClass )
    {

        for ( int i = 0; i < saveClass.size(); ++i )
        {
            if ( saveClass.get( i ) )
            {
                saveClassAsImaris( i, directory );
            }
        }

        ImarisUtils.createImarisMetaFile( directory );
        logger.info( "Created Imaris Meta Header" );

    }

    @Override
    public ImageProcessor getSlice( int slice, int frame )
    {
        int stackIndex = result.getStackIndex(  0, slice, frame );
        ImageProcessor ip = result.getStack().getProcessor( stackIndex );
        return ( ip );
    }

    @Override
    public ResultImageFrameSetter getSetter( FinalInterval interval )
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
