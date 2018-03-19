package de.embl.cba.trainableDeepSegmentation.results;

import de.embl.cba.bigDataTools.Hdf5DataCubeWriter;
import de.embl.cba.bigDataTools.imaris.ImarisDataSet;
import de.embl.cba.bigDataTools.imaris.ImarisUtils;
import de.embl.cba.bigDataTools.imaris.ImarisWriter;
import de.embl.cba.trainableDeepSegmentation.postprocessing.ProximityFilter3D;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
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
    public static final String SHOW_AS_SEPARATE_IMAGES = "Show images";

    private static void saveClassAsImaris( int classId, ResultExportSettings resultExportSettings )
    {

        String className = resultExportSettings.classNames.get( classId );

        ImarisDataSet imarisDataSet = new ImarisDataSet();

        imarisDataSet.setFromImagePlus(
                resultExportSettings.result,
                resultExportSettings.binning,
                resultExportSettings.directory,
                className,
                "/");

        ArrayList< String > channelNames = new ArrayList<>();

        channelNames.add( className );

        imarisDataSet.setChannelNames( channelNames  );

        ImarisWriter.writeHeader( imarisDataSet, resultExportSettings.directory, resultExportSettings.exportNamesPrefix + className + ".ims" );

        Hdf5DataCubeWriter writer = new Hdf5DataCubeWriter();

        for ( int t = 0; t < resultExportSettings.result.getNFrames(); ++t )
        {
            ImagePlus impClass = getBinnedAndProximityFilteredClassImage( classId, resultExportSettings, t );

            logger.progress( "Writing " + className+ ", frame:", (t+1) + "/" + resultExportSettings.result.getNFrames() + "..." );

            writer.writeImarisCompatibleResolutionPyramid( impClass, imarisDataSet, 0, t );

        }
    }

    private static void logDone( ResultExportSettings resultExportSettings, String className, int t, String s )
    {
        logger.progress( s + className + ", frame:", ( t + 1 ) + "/" + resultExportSettings.result.getNFrames() );
    }

    private static ImagePlus getBinnedClassImage( int classId, ResultExportSettings resultExportSettings, int t )
    {

        String className = resultExportSettings.classNames.get( classId );

        logger.progress( "Getting " + className + ", frame:", ( t + 1 ) + "/" + resultExportSettings.result.getNFrames() );

        ImagePlus impClass = getClassImage( classId, t, resultExportSettings.result, resultExportSettings.classLutWidth );

        if ( resultExportSettings.binning[0] * resultExportSettings.binning[1] * resultExportSettings.binning[2] > 1 )
        {
            logger.progress( "Binning " + className + ", frame:", (t+1) + "/" + resultExportSettings.result.getNFrames() + "..." );
            Binner binner = new Binner();
            impClass = binner.shrink( impClass, resultExportSettings.binning[ 0 ], resultExportSettings.binning[ 1 ], resultExportSettings.binning[ 2 ], Binner.AVERAGE );
        }

        return impClass;
    }


    private static ImagePlus getBinnedAndProximityFilteredClassImage( int classId, ResultExportSettings resultExportSettings, int t )
    {

        ImagePlus impClass = getBinnedClassImage( classId, resultExportSettings, t );

        if ( resultExportSettings.proximityFilterSettings.doSpatialProximityFiltering )
        {
            logger.info( "Applying proximity filter..." );
            impClass = ProximityFilter3D.multiply( impClass, resultExportSettings.proximityFilterSettings.dilatedBinaryReferenceMask );
        }

        return impClass;
    }



    private static void saveClassAsTiff( int classId, ResultExportSettings resultExportSettings )
    {

        String className = resultExportSettings.classNames.get( classId );

        for ( int t = 0; t < resultExportSettings.result.getNFrames(); ++t )
        {

            ImagePlus impClass = getBinnedAndProximityFilteredClassImage( classId, resultExportSettings, t );

            String path;

            if ( resultExportSettings.result.getNFrames() > 1 )
            {
                path = resultExportSettings.directory + File.separator + resultExportSettings.exportNamesPrefix + className + "--T" + String.format( "%05d", t ) + ".tif";
            }
            else
            {
                path = resultExportSettings.directory + File.separator + resultExportSettings.exportNamesPrefix + className + ".tif";
            }

            IJ.saveAsTiff( impClass, path );

            logDone( resultExportSettings, className, t, "Done with export of " );
        }

    }

    private static void showClassAsImage( int classId, ResultExportSettings resultExportSettings )
    {

        String className = resultExportSettings.classNames.get( classId );

        for ( int t = 0; t < resultExportSettings.result.getNFrames(); ++t )
        {
            ImagePlus impClass = getBinnedAndProximityFilteredClassImage( classId, resultExportSettings, t );
            impClass.setTitle( resultExportSettings.exportNamesPrefix + className + "--T" + ( t + 1 ) );
            impClass.show();
            logDone( resultExportSettings, className, t, "Displayed " );
        }

    }


    public static void saveAsImaris( ImagePlus rawData, String name, String directory, int[] binning )
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

    public static void exportResults( ResultExportSettings resultExportSettings )
    {

        if ( ! resultExportSettings.exportType.equals( ResultUtils.SHOW_AS_SEPARATE_IMAGES ) )
        {
            resultExportSettings.directory = IJ.getDirectory("Select a directory");
            if ( resultExportSettings.directory == null ) return;
            IOUtils.createDirectoryIfNotExists( resultExportSettings.directory );
        }

        if ( resultExportSettings.classesToBeExported == null )
        {
            resultExportSettings.classesToBeExported = selectAllClasses( resultExportSettings.classNames );
        }

        if ( resultExportSettings.binning == null )
        {
            resultExportSettings.binning = new int[] { 1, 1, 1 };
        }

        exportClasses( resultExportSettings );

        exportRawDataAndCreateImarisHeader( resultExportSettings );

        logger.info( "Export of results finished!" );

    }

    private static void exportRawDataAndCreateImarisHeader( ResultExportSettings resultExportSettings )
    {
        if ( resultExportSettings.exportType.equals( ResultUtils.SEPARATE_IMARIS ) )
        {
            if ( resultExportSettings.saveRawData )
            {
                saveAsImaris(
                        resultExportSettings.rawData,
                        resultExportSettings.exportNamesPrefix + "raw-data",
                        resultExportSettings.directory,
                        resultExportSettings.binning
                );
            }

            ImarisUtils.createImarisMetaFile( resultExportSettings.directory );
        }
    }

    private static void exportClasses( ResultExportSettings resultExportSettings )
    {
        prepareProximityFilter( resultExportSettings );

        for ( int classIndex = 0; classIndex < resultExportSettings.classesToBeExported.size(); ++classIndex )
        {
            if ( resultExportSettings.classesToBeExported.get( classIndex ) )
            {
                if ( resultExportSettings.exportType.equals( ResultUtils.SEPARATE_IMARIS ) )
                {
                    saveClassAsImaris( classIndex, resultExportSettings );
                }
                else if ( resultExportSettings.exportType.equals( ResultUtils.SEPARATE_TIFF_FILES ) )
                {
                    saveClassAsTiff( classIndex, resultExportSettings );
                }
                else if ( resultExportSettings.exportType.equals( ResultUtils.SHOW_AS_SEPARATE_IMAGES ) )
                {
                    showClassAsImage( classIndex, resultExportSettings );
                }
            }
        }
    }

    private static void prepareProximityFilter( ResultExportSettings resultExportSettings )
    {
        ProximityFilterSettings settings = resultExportSettings.proximityFilterSettings;

        if ( settings.doSpatialProximityFiltering )
        {
            logger.info( "Computing proximity mask..." );
            ImagePlus impReferenceClass = getBinnedClassImage( settings.referenceClassId, resultExportSettings, 0  );
            settings.dilatedBinaryReferenceMask = ProximityFilter3D.getDilatedBinaryUsingEDT( impReferenceClass, settings.distanceInPixelsAfterBinning  );
        }
    }


    public static void saveClassesAsFiles( ResultExportSettings resultExportSettings )
    {
        // if ( checkMaximalVolume( result, binning, logger ) ) return;

        if ( resultExportSettings.classesToBeExported == null )
        {
            resultExportSettings.classesToBeExported = selectAllClasses( resultExportSettings.classNames );
        }

        if ( resultExportSettings.binning == null )
        {
            resultExportSettings.binning = new int[] { 1, 1, 1 };
        }

        for ( int classIndex = 0; classIndex < resultExportSettings.classesToBeExported.size(); ++classIndex )
        {
            if ( resultExportSettings.classesToBeExported.get( classIndex ) )
            {
                if ( resultExportSettings.exportType.equals( ResultUtils.SEPARATE_IMARIS ) )
                {
                    saveClassAsImaris( classIndex, resultExportSettings );
                }
                else if ( resultExportSettings.exportType.equals( ResultUtils.SEPARATE_TIFF_FILES ) )
                {
                    saveClassAsTiff( classIndex, resultExportSettings );
                }
            }
        }
    }


    public static void showClassesAsImages( ResultExportSettings resultExportSettings )
    {

        if ( resultExportSettings.classesToBeExported == null )
        {
            resultExportSettings.classesToBeExported = selectAllClasses( resultExportSettings.classNames );
        }

        if ( resultExportSettings.binning == null )
        {
            resultExportSettings.binning = new int[] { 1, 1, 1 };
        }

        for ( int classIndex = 0; classIndex < resultExportSettings.classesToBeExported.size(); ++classIndex )
        {
            if ( resultExportSettings.classesToBeExported.get( classIndex ) )
            {
                showClassAsImage( classIndex, resultExportSettings );
            }
        }

    }

    public static final ArrayList< Boolean > selectAllClasses( ArrayList<String> classNames )
    {
        ArrayList< Boolean > classesToBeSaved = new ArrayList<>();
        for ( String className : classNames ) classesToBeSaved.add( true );
        return classesToBeSaved;
    }

    private static ImagePlus getClassImage( int classId, int t, ImagePlus result, int CLASS_LUT_WIDTH)
    {
        ImagePlus impClass;

        Duplicator duplicator = new Duplicator();
        impClass = duplicator.run( result, 1, 1, 1, result.getNSlices(), t + 1, t + 1 );

        int[] intensityGate = new int[]{ classId * CLASS_LUT_WIDTH + 1, (classId + 1 ) * CLASS_LUT_WIDTH };

        de.embl.cba.bigDataTools.utils.Utils.applyIntensityGate( impClass, intensityGate );

        return ( impClass );

    }
}
