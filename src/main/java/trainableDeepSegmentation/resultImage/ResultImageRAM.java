package trainableDeepSegmentation.resultImage;

import bigDataTools.Hdf5DataCubeWriter;
import bigDataTools.ImarisDataSet;
import bigDataTools.ImarisUtils;
import bigDataTools.ImarisWriter;
import bigDataTools.logging.Logger;
import bigDataTools.utils.Utils;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.Duplicator;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;
import trainableDeepSegmentation.WekaSegmentation;

import java.util.ArrayList;

import static trainableDeepSegmentation.ImageUtils.*;

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
        this.result = createImage( dimensions );
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
        return ( new ResultImageFrameSetterRAM( this, interval ) );
    }

    public void set( long x, long y, long z, long t, int classId, double certainty )
    {
        int lutCertainty = (int) ( certainty * ( CLASS_LUT_WIDTH - 1.0 ) );

        int classOffset = classId * CLASS_LUT_WIDTH + 1;

        int n = result.getStackIndex( 0, (int)z+1, (int) t+1 );
        result.getStack().getProcessor( n ).set(
                (int)x, (int)y,
                (byte) ( classOffset + lutCertainty ));
    }

    private ImagePlus createImage( long[] dimensions )
    {
        // TODO: check for cancel!

        ImageStack stack = ImageStack.create(
                (int) dimensions[ X ],
                (int) dimensions[ Y ],
                (int) (dimensions[ Z ] * dimensions[ T ]),
                8);

        result = new ImagePlus( "resultImage", stack  );

        result.setDimensions(
                1,
                (int) dimensions[ Z ],
                (int) dimensions[ T ]);

        result.setOpenAsHyperStack(true);
        result.setTitle("resultImage");

        return ( result );
    }


}
