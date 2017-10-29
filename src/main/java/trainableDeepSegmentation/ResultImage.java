package trainableDeepSegmentation;

import bigDataTools.Hdf5DataCubeWriter;
import bigDataTools.Region5D;
import bigDataTools.VirtualStackOfStacks.VirtualStackOfStacks;
import bigDataTools.dataStreamingTools.DataStreamingTools;
import bigDataTools.utils.ImageDataInfo;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;

import java.io.IOException;
import java.util.ArrayList;

import static trainableDeepSegmentation.ImageUtils.*;

public class ResultImage {

    public static final int CLASS_LUT_WIDTH = 10;

    ImagePlus result;
    WekaSegmentation wekaSegmentation;

    public ResultImage( WekaSegmentation wekaSegmentation,
                        String directory,
                        long[] dimensions)
    {
        this.wekaSegmentation = wekaSegmentation;
    }

    public void saveAsImarisSeparateChannels( String directory,
                                              ArrayList< Boolean > classesToSave )
    {
        Hdf5DataCubeWriter writer = new Hdf5DataCubeWriter();

        /*
        writer.writeImarisCompatibleResolutionPyramid(
                impBinned,
                imarisDataSetProperties,
                c, t );
                */
    }

    public ImageProcessor getSlice( int z, int t )
    {
        return null;
    }

    public Setter getSetter( FinalInterval interval )
    {
        return ( new Setter( interval ) );
    }

    private void createStream()
    {
        // TODO: check for cancel!

        DataStreamingTools dst = new DataStreamingTools();
        String tMax = String.format("%05d", trainingImage.getNFrames());
        String zMax = String.format("%05d", trainingImage.getNSlices());

        String namingPattern = "classified--C<C01-01>--T<T00001-" +
                tMax + ">--Z<Z00001-"+zMax+">.tif";
        bigDataTools.utils.ImageDataInfo imageDataInfo = new ImageDataInfo();
        imageDataInfo.bitDepth = 8;
        int nIOthreads = 3;

        // create one image
        ImageStack stack = ImageStack.create(trainingImage.getWidth(),
                trainingImage.getHeight(), 1, 8);
        ImagePlus impC0T0Z0 = new ImagePlus("", stack);
        FileSaver fileSaver = new FileSaver( impC0T0Z0 );
        fileSaver.saveAsTiff( directory + "/" +
                "classified--C01--T00001--Z00001.tif");

        classifiedImage = dst.openFromDirectory(
                directory,
                namingPattern,
                "None",
                "None",
                imageDataInfo,
                nIOthreads,
                false,
                true);

        int nZ = trainingImage.getNSlices();
        int nT = trainingImage.getNFrames();
        classifiedImage.setDimensions(
                1,
                nZ,
                nT);

        classifiedImage.setOpenAsHyperStack(true);
        classifiedImage.setTitle("classification_result");
    }

    private void write3dResultChunk(
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


    public class Setter
    {

        FinalInterval interval;
        byte[][][] resultChunk;

        public Setter ( FinalInterval interval )
        {
            this.interval = interval;
        }

        public void set( long x, long y, long z, int classId, double certainty)
        {
            int lutCertainty = (int) ( certainty * ( CLASS_LUT_WIDTH - 1.0 ) );

            int classOffset = classId * CLASS_LUT_WIDTH + 1;

            resultChunk[ (int) (z - interval.min( Z )) ]
                    [ (int) (y - interval.min ( Y )) ]
                    [ (int) (x - interval.min ( X )) ]
                    = (byte) ( classOffset + lutCertainty );

        }

        public void close( )
        {
            write3dResultChunk( interval, resultChunk );
        }

    }
}
