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
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;

import java.io.File;
import java.util.ArrayList;

import static de.embl.cba.trainableDeepSegmentation.DeepSegmentation.logger;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.*;

public abstract class ResultUtils
{

    private static void saveClassAsImaris( int classId, ResultExportSettings resultExportSettings )
    {
        String className = resultExportSettings.classNames.get( classId );

        ImarisDataSet imarisDataSet = new ImarisDataSet();

        imarisDataSet.setFromImagePlus(
                resultExportSettings.resultImagePlus,
                resultExportSettings.binning,
                resultExportSettings.directory,
                className,
                "/");

        ArrayList< String > channelNames = new ArrayList<>();

        channelNames.add( className );

        imarisDataSet.setChannelNames( channelNames  );

        ImarisWriter.writeHeader( imarisDataSet, resultExportSettings.directory, resultExportSettings.exportNamesPrefix + className + ".ims" );

        Hdf5DataCubeWriter writer = new Hdf5DataCubeWriter();

        for ( int t = resultExportSettings.timePointsFirstLast[ 0 ]; t <= resultExportSettings.timePointsFirstLast[ 1 ]; ++t )
        {
            ImagePlus impClass = getBinnedAndProximityFilteredClassImage( classId, resultExportSettings, t );

            logger.progress( "Writing " + className+ ", frame:", (t+1) + "/" + resultExportSettings.resultImagePlus.getNFrames() + "..." );

            writer.writeImarisCompatibleResolutionPyramid( impClass, imarisDataSet, 0, t );

        }
    }

    private static void logDone( ResultExportSettings resultExportSettings, String className, int t, String s )
    {
        logger.progress( s + className + ", frame:", ( t + 1 ) + "/" + resultExportSettings.resultImagePlus.getNFrames() );
    }

    private static ImagePlus getBinnedClassImage( int classId, ResultExportSettings resultExportSettings, int t )
    {

        String className = resultExportSettings.classNames.get( classId );

        logger.progress( "Getting " + className + ", frame:", ( t + 1 ) + "/" + resultExportSettings.resultImagePlus.getNFrames() );

        ImagePlus impClass = getClassImage( classId, t, resultExportSettings.resultImagePlus, resultExportSettings.classLutWidth );

        if ( resultExportSettings.binning[0] * resultExportSettings.binning[1] * resultExportSettings.binning[2] > 1 )
        {
            logger.progress( "Binning " + className + ", frame:", (t+1) + "/" + resultExportSettings.resultImagePlus.getNFrames() + "..." );
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

        for ( int t = resultExportSettings.timePointsFirstLast[ 0 ]; t <= resultExportSettings.timePointsFirstLast[ 1 ]; ++t )
        {

            ImagePlus impClass = getBinnedAndProximityFilteredClassImage( classId, resultExportSettings, t );

            String path;

            if ( resultExportSettings.resultImagePlus.getNFrames() > 1 )
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

        for ( int t = resultExportSettings.timePointsFirstLast[ 0 ]; t <= resultExportSettings.timePointsFirstLast[ 1 ]; ++t )
        {
            ImagePlus impClass = getBinnedAndProximityFilteredClassImage( classId, resultExportSettings, t );
            impClass.setTitle( resultExportSettings.exportNamesPrefix + className + "--T" + ( t + 1 ) );
            impClass.show();
            logDone( resultExportSettings, className, t, "Displayed " );
        }

    }


    public static void saveAsImaris(  ResultExportSettings resultExportSettings  )
    {

        ImagePlus rawData = resultExportSettings.rawData;
        String name =  resultExportSettings.exportNamesPrefix + "raw-data";
        String directory =  resultExportSettings.directory;
        int[] binning =  resultExportSettings.binning;

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

        for ( int t = resultExportSettings.timePointsFirstLast[ 0 ]; t <= resultExportSettings.timePointsFirstLast[ 1 ]; ++t )
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

        logger.info( "Exporting results, using modality: " + resultExportSettings.exportType );
        logger.info( "Exporting results to: " + resultExportSettings.directory );

        if ( ! resultExportSettings.exportType.equals( ResultExportSettings.SHOW_AS_SEPARATE_IMAGES ) )
        {
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

        logger.info( "Export of results finished." );

    }

    private static void exportRawDataAndCreateImarisHeader( ResultExportSettings resultExportSettings )
    {
        if ( resultExportSettings.exportType.equals( ResultExportSettings.SEPARATE_IMARIS ) )
        {
            if ( resultExportSettings.saveRawData )
            {
                saveAsImaris( resultExportSettings );
            }

            ImarisUtils.createImarisMetaFile( resultExportSettings.directory );
        }
    }

    private static void exportClasses( ResultExportSettings resultExportSettings )
    {

        if ( resultExportSettings.exportType.equals( ResultExportSettings.SEPARATE_MULTI_CLASS_TIFF_SLICES ) )
        {
            saveAsSeparateMultiClassTiffSlices( resultExportSettings );
        }
        else
        {
            prepareProximityFilter( resultExportSettings );

            for ( int classIndex = 0; classIndex < resultExportSettings.classesToBeExported.size(); ++classIndex )
            {
                if ( resultExportSettings.classesToBeExported.get( classIndex ) )
                {
                    if ( resultExportSettings.exportType.equals( ResultExportSettings.SEPARATE_IMARIS ) )
                    {
                        saveClassAsImaris( classIndex, resultExportSettings );
                    }
                    else if ( resultExportSettings.exportType.equals( ResultExportSettings.SEPARATE_TIFF_FILES ) )
                    {
                        saveClassAsTiff( classIndex, resultExportSettings );
                    }
                    else if ( resultExportSettings.exportType.equals( ResultExportSettings.SHOW_AS_SEPARATE_IMAGES ) )
                    {
                        showClassAsImage( classIndex, resultExportSettings );
                    }
                }
            }
        }

    }

    private static void saveAsSeparateMultiClassTiffSlices( ResultExportSettings resultExportSettings )
    {
        FinalInterval interval = resultExportSettings.resultImage.getInterval();

        String directory = resultExportSettings.directory;

        for ( long t = interval.min( T ); t <= interval.max( T ); ++t )
        {
            for ( long z = interval.min( Z ); z <= interval.max( Z ); ++z )
            {
                int slice = (int) z + 1;
                int frame = (int) t + 1;
                ImageProcessor ip = resultExportSettings.resultImage.getSlice( slice, frame);
                String filename = "classified--C01--T" + String.format( "%05d", frame ) + "--Z" + String.format( "%05d", slice ) + ".tif";
                String path = directory + File.separator + filename;
                IJ.saveAsTiff( new ImagePlus( filename, ip ), path );
            }
        }

    }

    private static void prepareProximityFilter( ResultExportSettings resultExportSettings )
    {
        ProximityFilterSettings settings = resultExportSettings.proximityFilterSettings;

        if (  settings.doSpatialProximityFiltering )
        {
            logger.info( "Computing proximity mask..." );
            ImagePlus impReferenceClass = getBinnedClassImage( settings.referenceClassId, resultExportSettings, 0  );
            settings.dilatedBinaryReferenceMask = ProximityFilter3D.getDilatedBinaryUsingEDT( impReferenceClass, settings.distanceInPixelsAfterBinning  );
        }
    }


    public static void saveClassesAsFiles( ResultExportSettings resultExportSettings )
    {
        // if ( checkMaximalVolume( resultImagePlus, binning, logger ) ) return;

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
                if ( resultExportSettings.exportType.equals( ResultExportSettings.SEPARATE_IMARIS ) )
                {
                    saveClassAsImaris( classIndex, resultExportSettings );
                }
                else if ( resultExportSettings.exportType.equals( ResultExportSettings.SEPARATE_TIFF_FILES ) )
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
