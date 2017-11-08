package trainableDeepSegmentation.results;

import bigDataTools.logging.Logger;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;
import trainableDeepSegmentation.WekaSegmentation;

import java.util.ArrayList;

import static trainableDeepSegmentation.ImageUtils.*;
import static trainableDeepSegmentation.results.Utils.saveImagePlusAsSeparateImarisChannels;

public class ResultImageMemory implements ResultImage {

    public static final int CLASS_LUT_WIDTH = 10;

    ImagePlus result;
    WekaSegmentation wekaSegmentation;
    Logger logger;
    long[] dimensions;

    public ResultImageMemory( WekaSegmentation wekaSegmentation,
                              long[] dimensions)
    {
        this.wekaSegmentation = wekaSegmentation;
        this.logger = wekaSegmentation.getLogger();
        this.result = createImagePlus( dimensions );
        this.dimensions = dimensions;
    }

    @Override
    public void saveAsSeparateImarisChannels( String directory,
                                              ArrayList< Boolean > saveClass,
                                              int[] binning)
    {
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
        return ( new ResultImageFrameSetterMemory( this, interval ) );
    }

    public void set( long x, long y, long z, long t, int classId, double certainty )
    {
        int lutCertainty = (int) ( certainty * ( CLASS_LUT_WIDTH - 1.0 ) );

        int classOffset = classId * CLASS_LUT_WIDTH + 1;

        int n = result.getStackIndex( 1, (int) z+1, (int) t+1 );
        result.getStack().getProcessor( n ).set(
                (int) x, (int) y,
                (byte) ( classOffset + lutCertainty ));
    }

    private ImagePlus createImagePlus( long[] dimensions )
    {
        ImageStack stack = ImageStack.create(
                (int) dimensions[ X ],
                (int) dimensions[ Y ],
                (int) (dimensions[ Z ] * dimensions[ T ]),
                8);

        result = new ImagePlus( "results", stack  );

        result.setDimensions(
                1,
                (int) dimensions[ Z ],
                (int) dimensions[ T ]);

        result.setOpenAsHyperStack(true);
        result.setTitle("results");

        return ( result );
    }

    @Override
    public int getProbabilityRange()
    {
        return CLASS_LUT_WIDTH;
    }

}
