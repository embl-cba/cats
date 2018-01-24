package trainableDeepSegmentation.results;

import de.embl.cba.bigDataTools.logging.Logger;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;
import trainableDeepSegmentation.IntervalUtils;
import trainableDeepSegmentation.WekaSegmentation;
import trainableDeepSegmentation.ij2plugins.IOUtils;

import java.awt.image.ColorModel;
import java.util.ArrayList;

import static de.embl.cba.bigDataTools.utils.Utils.getDataCubeFromImagePlus;

public class ResultImageRAM implements ResultImage {

    public static final int CLASS_LUT_WIDTH = 10;

    ImagePlus result;
    WekaSegmentation wekaSegmentation;
    Logger logger;
    long[] dimensions;

    public ResultImageRAM( WekaSegmentation wekaSegmentation,
                           long[] dimensions)
    {
        this.wekaSegmentation = wekaSegmentation;
        this.logger = wekaSegmentation.getLogger();
        this.result = createImagePlus( dimensions );
        this.dimensions = dimensions;
    }


    @Override
    public void saveClassesAsFiles( String directory, ArrayList< Boolean > classesToBeSaved, int[] binning, String fileType )
    {

        IOUtils.createDirectoryIfNotExists( directory );

        Utils.saveClassesAsFiles(
                directory,
                classesToBeSaved,
                fileType,
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

    public void setProcessor( ImageProcessor ip, int slice, int frame )
    {
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
                (int) dimensions[ IntervalUtils.X ],
                (int) dimensions[ IntervalUtils.Y ],
                (int) (dimensions[ IntervalUtils.Z ] * dimensions[ IntervalUtils.T ]),
                8);

        result = new ImagePlus( "results", stack  );

        result.setDimensions(
                1,
                (int) dimensions[ IntervalUtils.Z ],
                (int) dimensions[ IntervalUtils.T ]);

        result.setOpenAsHyperStack(true);
        result.setTitle("results");

        return ( result );
    }

    @Override
    public int getProbabilityRange()
    {
        return CLASS_LUT_WIDTH;
    }

    @Override
    public ImagePlus getDataCubeCopy( FinalInterval interval )
    {
        assert interval.min( IntervalUtils.C ) == interval.max( IntervalUtils.C );
        assert interval.min( IntervalUtils.T ) == interval.max( IntervalUtils.T );

        ImagePlus cube = getDataCubeFromImagePlus( result,
                IntervalUtils.convertIntervalToRegion5D( interval ));

        return cube;
    }

    public ImagePlus getWholeImageCopy()
    {
        ImagePlus imp = result.duplicate();
        return imp;
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
