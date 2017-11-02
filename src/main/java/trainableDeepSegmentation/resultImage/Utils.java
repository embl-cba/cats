package trainableDeepSegmentation.resultImage;

import bigDataTools.Hdf5DataCubeWriter;
import bigDataTools.ImarisDataSet;
import bigDataTools.ImarisUtils;
import bigDataTools.ImarisWriter;
import bigDataTools.logging.Logger;
import ij.ImagePlus;
import ij.plugin.Duplicator;

import java.util.ArrayList;

import static trainableDeepSegmentation.ImageUtils.T;

public abstract class Utils {


    private static void saveClassAsImaris( int classId,
                                           String directory,
                                           ImagePlus result,
                                           Logger logger,
                                           ArrayList< String > classNames,
                                           int CLASS_LUT_WIDTH)
    {

        String className = classNames.get( classId );

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

        for ( int t = 0; t < result.getNFrames(); ++t )
        {

            ImagePlus impClass = getClassImage( classId,
                    t, result, CLASS_LUT_WIDTH );

            writer.writeImarisCompatibleResolutionPyramid(
                    impClass,
                    imarisDataSet,
                    0, t );

            logger.progress( "Wrote " + className+ ", frame:",
                    (t+1) + "/" + result.getNFrames() );
        }
    }

    private static ImagePlus getClassImage( int classId,
                                            int t,
                                            ImagePlus result,
                                            int CLASS_LUT_WIDTH)
    {
        ImagePlus impClass;

        Duplicator duplicator = new Duplicator();
        impClass = duplicator.run( result, 1, 1, 1, result.getNSlices(), t+1, t+1 );

        int[] intensityGate = new int[]
                { classId * CLASS_LUT_WIDTH + 1, (classId + 1 ) * CLASS_LUT_WIDTH };
        bigDataTools.utils.Utils.applyIntensityGate( impClass, intensityGate );

        return ( impClass );

    }

    public static void saveImagePlusAsSeparateImarisChannels(
            String directory,
            ArrayList< Boolean > saveClass,
            ImagePlus result,
            Logger logger,
            ArrayList< String > classNames,
            int CLASS_LUT_WIDTH)
    {

        for ( int i = 0; i < saveClass.size(); ++i )
        {
            if ( saveClass.get( i ) )
            {
                saveClassAsImaris( i, directory, result, logger, classNames, CLASS_LUT_WIDTH );
            }
        }

        ImarisUtils.createImarisMetaFile( directory );
        logger.info("Created imaris meta file.");
    }

}
