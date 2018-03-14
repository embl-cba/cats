package de.embl.cba.trainableDeepSegmentation.results;

import de.embl.cba.bigDataTools.Hdf5DataCubeWriter;
import de.embl.cba.bigDataTools.imaris.ImarisDataSet;
import de.embl.cba.bigDataTools.imaris.ImarisWriter;
import de.embl.cba.utils.logging.Logger;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.Binner;
import ij.plugin.Duplicator;

import java.io.File;
import java.util.ArrayList;

import static de.embl.cba.trainableDeepSegmentation.DeepSegmentation.logger;

public abstract class ResultUtils
{

    public static final String SEPARATE_IMARIS = "Save as Imaris";
    public static final String SEPARATE_TIFF_FILES = "Save as Tiff";
    public static final String SEPARATE_IMAGES = "Show images";

    private static void saveClassAsImaris( int classId,
                                           String directory,
                                           String fileNamePrefix,
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

        ImarisWriter.writeHeader( imarisDataSet, directory, fileNamePrefix + className + ".ims" );

        Hdf5DataCubeWriter writer = new Hdf5DataCubeWriter();

        for ( int t = 0; t < result.getNFrames(); ++t )
        {

            ImagePlus impClass = getClassImage( classId, t, result, CLASS_LUT_WIDTH );

            if ( binning[0]*binning[1]*binning[2] > 1 )
            {
                Binner binner = new Binner();
                impClass = binner.shrink( impClass, binning[ 0 ], binning[ 1 ], binning[ 2 ], Binner.AVERAGE );
            }

            writer.writeImarisCompatibleResolutionPyramid( impClass, imarisDataSet, 0, t );

            logger.progress( "Wrote " + className+ ", frame:", (t+1) + "/" + result.getNFrames() );
        }
    }


    private static void saveClassAsTiff( int classId,
                                         String directory,
                                         String fileNamePrefix,
                                         ImagePlus result,
                                         int[] binning,
                                         Logger logger,
                                         ArrayList< String > classNames,
                                         int CLASS_LUT_WIDTH )
    {

        String className = classNames.get( classId );

        for ( int t = 0; t < result.getNFrames(); ++t )
        {
            logger.progress( "Preparing " + className + ", frame:", (t + 1) + "/" + result.getNFrames() );

            ImagePlus impClass = getClassImage( classId, t, result, CLASS_LUT_WIDTH );

            if ( binning[0] * binning[1] * binning[2] > 1 )
            {
                Binner binner = new Binner();
                impClass = binner.shrink( impClass, binning[ 0 ], binning[ 1 ], binning[ 2 ], Binner.AVERAGE );
            }

            String path;

            if ( result.getNFrames() > 1 )
            {
                path = directory + File.separator + fileNamePrefix + className + "--T" + String.format( "%05d", t ) + ".tif";
            }
            else
            {
                path = directory + File.separator + fileNamePrefix + className + ".tif";
            }

            IJ.saveAsTiff( impClass, path );

            logger.progress( "Wrote " + className + ", frame:", (t + 1) + "/" + result.getNFrames() + ", path: " + path );
        }

    }


    private static void showClassAsImage( int classId,
                                          ImagePlus result,
                                          int[] binning,
                                          Logger logger,
                                          String imageNamePrefix,
                                          ArrayList< String > classNames,
                                          int CLASS_LUT_WIDTH )
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

            impClass.setTitle( imageNamePrefix + className + "--T" + ( t + 1 ) );
            impClass.show();

            logger.progress( "Displayed " + className + ", frame:", (t + 1) + "/" + result.getNFrames() );
        }

    }

    private static ImagePlus getClassImage( int classId,
                                            int t,
                                            ImagePlus result,
                                            int CLASS_LUT_WIDTH)
    {
        ImagePlus impClass;

        Duplicator duplicator = new Duplicator();
        impClass = duplicator.run( result, 1, 1, 1, result.getNSlices(), t + 1, t + 1 );

        int[] intensityGate = new int[]{ classId * CLASS_LUT_WIDTH + 1, (classId + 1 ) * CLASS_LUT_WIDTH };

        de.embl.cba.bigDataTools.utils.Utils.applyIntensityGate( impClass, intensityGate );

        return ( impClass );

    }

    public static void saveAsImarisChannels( ImagePlus rawData,
                                             String name,
                                             String directory,
                                             int[] binning )
    {
        // Set everything up
        ImarisDataSet imarisDataSet = new ImarisDataSet();
        imarisDataSet.setFromImagePlus( rawData, binning, directory, name, "/");

        // Channels
        //ArrayList< String > channelNames = new ArrayList<>();
        //channelNames.add( name );
        //imarisDataSet.setChannelNames( channelNames  );

        // Header
        ImarisWriter.writeHeader( imarisDataSet, directory, name + ".ims" );

        Hdf5DataCubeWriter writer = new Hdf5DataCubeWriter();

        for ( int t = 0; t < rawData.getNFrames(); ++t )
        {

            for ( int c = 0; c < rawData.getNChannels(); ++c )
            {
                Duplicator duplicator = new Duplicator();
                ImagePlus rawDataFrame = duplicator.run( rawData, c + 1, c + 1, 1, rawData.getNSlices(), t + 1, t + 1 );

                if ( binning[ 0 ] * binning[ 1 ] * binning[ 2 ] > 1 )
                {
                    Binner binner = new Binner();
                    rawDataFrame = binner.shrink( rawDataFrame, binning[ 0 ], binning[ 1 ], binning[ 2 ], Binner.AVERAGE );
                }

                writer.writeImarisCompatibleResolutionPyramid( rawDataFrame, imarisDataSet, c, t );

                logger.progress( "Wrote " + name + ", channel:" + ( c + 1 ) + "/" + rawData.getNChannels() + ", frame:", ( t + 1 ) + "/" + rawData.getNFrames() );

            }

        }
    }


    public static void saveClassesAsFiles(
            String directory,
            String fileNamePrefix,
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
                if ( fileType.equals( ResultUtils.SEPARATE_IMARIS ) )
                {
                    saveClassAsImaris( classIndex, directory, fileNamePrefix, result, binning, logger, classNames, CLASS_LUT_WIDTH );
                }
                else if ( fileType.equals( ResultUtils.SEPARATE_TIFF_FILES ) )
                {
                    saveClassAsTiff( classIndex, directory, fileNamePrefix, result, binning, logger, classNames, CLASS_LUT_WIDTH );
                }
            }
        }
    }


    public static void showClassesAsImages(
            String imageNamePrefix,
            ArrayList< Boolean > classesToBeShown,
            ImagePlus resultImage,
            int[] binning,
            Logger logger,
            ArrayList< String > classNames,
            int CLASS_LUT_WIDTH)
    {

        if ( classesToBeShown == null )
        {
            classesToBeShown = selectAllClasses( classNames );
        }

        if ( binning == null )
        {
            binning = new int[] { 1, 1, 1 };
        }

        for ( int classIndex = 0; classIndex < classesToBeShown.size(); ++classIndex )
        {
            if ( classesToBeShown.get( classIndex ) )
            {
                showClassAsImage( classIndex, resultImage, binning, logger, imageNamePrefix, classNames, CLASS_LUT_WIDTH );
            }
        }

    }

    public static final ArrayList< Boolean > selectAllClasses( ArrayList<String> classNames )
    {
        ArrayList< Boolean > classesToBeSaved = new ArrayList<>();
        for ( String className : classNames ) classesToBeSaved.add( true );
        return classesToBeSaved;
    }
}
