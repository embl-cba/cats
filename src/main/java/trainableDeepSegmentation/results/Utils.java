package trainableDeepSegmentation.results;

import bigDataTools.Hdf5DataCubeWriter;
import bigDataTools.ImarisDataSet;
import bigDataTools.ImarisUtils;
import bigDataTools.ImarisWriter;
import bigDataTools.logging.Logger;
import ij.ImagePlus;
import ij.plugin.Binner;
import ij.plugin.Duplicator;

import java.util.ArrayList;

import static trainableDeepSegmentation.WekaSegmentation.logger;

public abstract class Utils {


    private static void saveClassAsImaris( int classId,
                                           String directory,
                                           ImagePlus result,
                                           int[] binning,
                                           Logger logger,
                                           ArrayList< String > classNames,
                                           int CLASS_LUT_WIDTH)
    {

        String className = classNames.get( classId );

        ImarisDataSet imarisDataSet = new ImarisDataSet();

        imarisDataSet.setFromImagePlus( result,
                binning,
                directory,
                className,
                "/");

        ArrayList< String > channelNames = new ArrayList<>();
        channelNames.add( className );

        imarisDataSet.setChannelNames( channelNames  );

        ImarisWriter.writeHeader( imarisDataSet,
                directory,
                className + ".ims"
        );


        Hdf5DataCubeWriter writer = new Hdf5DataCubeWriter();

        for ( int t = 0; t < result.getNFrames(); ++t )
        {

            ImagePlus impClass = getClassImage( classId, t, result, CLASS_LUT_WIDTH );

            if ( binning[0]*binning[1]*binning[2] > 1 )
            {
                Binner binner = new Binner();
                impClass = binner.shrink( impClass, binning[ 0 ],
                        binning[ 1 ], binning[ 2 ], Binner.AVERAGE );
            }

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

    public static void saveAsImarisChannel( ImagePlus rawData,
                                            String name,
                                            String directory,
                                            int[] binning )
    {
        // Set everything up
        ImarisDataSet imarisDataSet = new ImarisDataSet();
                imarisDataSet.setFromImagePlus( rawData,
                binning, directory, name, "/");

        // Channels
        ArrayList< String > channelNames = new ArrayList<>();
                channelNames.add( name );
                imarisDataSet.setChannelNames( channelNames  );

        // Header
                ImarisWriter.writeHeader( imarisDataSet,
                        directory,
                        name + ".ims"
                );

        Hdf5DataCubeWriter writer = new Hdf5DataCubeWriter();

                for ( int t = 0; t < rawData.getNFrames(); ++t )
        {

            Duplicator duplicator = new Duplicator();
            ImagePlus rawDataFrame = duplicator.run( rawData, 1, 1, 1, rawData.getNSlices(), t+1, t+1 );

            if ( binning[0]*binning[1]*binning[2] > 1 )
            {
                Binner binner = new Binner();
                rawDataFrame = binner.shrink( rawDataFrame, binning[ 0 ],
                        binning[ 1 ], binning[ 2 ], Binner.AVERAGE );
            }

            writer.writeImarisCompatibleResolutionPyramid(
                    rawDataFrame,
                    imarisDataSet,
                    0, t );

            logger.progress( "Wrote " + name + ", frame:",
                    (t+1) + "/" + rawData.getNFrames() );
        }
    }


    public static void saveImagePlusAsSeparateImarisChannels(
            String directory,
            ArrayList< Boolean > saveClass,
            ImagePlus result,
            int[] binning,
            Logger logger,
            ArrayList< String > classNames,
            int CLASS_LUT_WIDTH)
    {

        long volume = (long) 1.0 *
                result.getWidth() / binning[0] *
                result.getHeight() / binning[1] *
                result.getNSlices() / binning[2];

        if ( volume > Integer.MAX_VALUE - 10 )
        {
            logger.error( "Your image (after binning) is too large [voxels]: " + volume
                    + "\nDue to java indexing issues the maximum currently is around " + Integer.MAX_VALUE +
                    "\nPlease use more binning.");
            return;
        }

        for ( int i = 0; i < saveClass.size(); ++i )
        {
            if ( saveClass.get( i ) )
            {
                saveClassAsImaris( i, directory, result, binning, logger, classNames, CLASS_LUT_WIDTH );
            }
        }


    }

}
