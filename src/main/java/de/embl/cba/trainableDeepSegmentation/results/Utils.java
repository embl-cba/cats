package de.embl.cba.trainableDeepSegmentation.results;

import de.embl.cba.bigDataTools.Hdf5DataCubeWriter;
import de.embl.cba.bigDataTools.ImarisDataSet;
import de.embl.cba.bigDataTools.ImarisWriter;
import de.embl.cba.bigDataTools.logging.Logger;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.Binner;
import ij.plugin.Duplicator;

import java.io.File;
import java.util.ArrayList;

import static de.embl.cba.trainableDeepSegmentation.DeepSegmentation.logger;

public abstract class Utils {


    public static final String SEPARATE_IMARIS = "Separate Imaris Channels";

    public static final String SEPARATE_TIFF_FILES = "Separate Tiff Files";

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

        imarisDataSet.setFromImagePlus( result, binning, directory, className, "/");

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

            writer.writeImarisCompatibleResolutionPyramid( impClass, imarisDataSet, 0, t );

            logger.progress( "Wrote " + className+ ", frame:", (t+1) + "/" + result.getNFrames() );
        }
    }


    private static void saveClassAsTiff( int classId,
                                         String directory,
                                         ImagePlus result,
                                         int[] binning,
                                         Logger logger,
                                         ArrayList< String > classNames,
                                         int CLASS_LUT_WIDTH)
    {

        String className = classNames.get( classId );

        for ( int t = 0; t < result.getNFrames(); ++t )
        {
            ImagePlus impClass = getClassImage( classId, t, result, CLASS_LUT_WIDTH );

            if ( binning[0] * binning[1] * binning[2] > 1 )
            {
                Binner binner = new Binner();
                impClass = binner.shrink( impClass, binning[ 0 ], binning[ 1 ], binning[ 2 ], Binner.AVERAGE );
            }


            String path;

            if ( result.getNFrames() > 1 )
            {
                path = directory + File.separator + className + "--T" + String.format( "%05d", t ) + ".tif";
            }
            else
            {
                path = directory + File.separator + className + ".tif";
            }


            IJ.saveAsTiff( impClass, path );

            logger.progress( "Wrote " + className+
                                ", frame:", (t + 1) + "/" + result.getNFrames() +
                                ", path: " + path );
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
        de.embl.cba.bigDataTools.utils.Utils.applyIntensityGate( impClass, intensityGate );

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


    public static void saveClassesAsFiles(
            String directory,
            ArrayList< Boolean > classesToBeSaved,
            String fileType,
            ImagePlus result,
            int[] binning,
            Logger logger,
            ArrayList< String > classNames,
            int CLASS_LUT_WIDTH)
    {

        // if ( checkMaximalVolume( result, binning, logger ) ) return;

        if ( classesToBeSaved == null )
        {
            classesToBeSaved = selectAllClasses( classNames );
        }

        if ( binning == null )
        {
            binning = new int[] { 1, 1, 1 };
        }

        for ( int classIndex = 0; classIndex < classesToBeSaved.size(); ++classIndex )
        {
            if ( classesToBeSaved.get( classIndex ) )
            {
                if ( fileType.equals( Utils.SEPARATE_IMARIS ) )
                {
                    saveClassAsImaris( classIndex, directory, result, binning, logger, classNames, CLASS_LUT_WIDTH );
                }
                else if ( fileType.equals( Utils.SEPARATE_TIFF_FILES ) )
                {
                    saveClassAsTiff( classIndex, directory, result, binning, logger, classNames, CLASS_LUT_WIDTH );
                }
            }
        }


    }


    public static void saveResultImagePlusAsSeparateTiffFiles(
            String directory,
            ArrayList< Boolean > saveClass,
            ImagePlus result,
            int[] binning,
            Logger logger,
            ArrayList< String > classNames,
            int CLASS_LUT_WIDTH)
    {

        if ( checkMaximalVolume( result, binning, logger ) ) return;

        for ( int classIndex = 0; classIndex < saveClass.size(); ++classIndex )
        {
            if ( saveClass.get( classIndex ) )
            {
                saveClassAsTiff( classIndex, directory, result, binning, logger, classNames, CLASS_LUT_WIDTH );
            }
        }


    }

    private static boolean checkMaximalVolume( ImagePlus result, int[] binning, Logger logger )
    {
        long volume = (long) 1.0 * result.getWidth() / binning[0] * result.getHeight() / binning[1] * result.getNSlices() / binning[2];

        if ( volume > Integer.MAX_VALUE - 10 )
        {
            logger.error( "Your image (after binning) is too large [voxels]: " + volume
                    + "\nDue to java indexing issues the maximum currently is around " + Integer.MAX_VALUE +
                    "\nPlease use more binning.");
            return true;
        }
        return false;
    }

    public static final ArrayList< Boolean > selectAllClasses( ArrayList<String> classNames )
    {
        ArrayList< Boolean > classesToBeSaved = new ArrayList<>();
        for ( String className : classNames ) classesToBeSaved.add( true );
        return classesToBeSaved;
    }
}
