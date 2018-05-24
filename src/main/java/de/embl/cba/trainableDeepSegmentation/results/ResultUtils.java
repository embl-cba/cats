package de.embl.cba.trainableDeepSegmentation.results;

import de.embl.cba.bigDataTools.Hdf5DataCubeWriter;
import de.embl.cba.bigDataTools.imaris.ImarisDataSet;
import de.embl.cba.bigDataTools.imaris.ImarisUtils;
import de.embl.cba.bigDataTools.imaris.ImarisWriter;
import de.embl.cba.bigDataTools.logging.IJLazySwingLogger;
import de.embl.cba.bigDataTools.logging.Logger;
import de.embl.cba.trainableDeepSegmentation.postprocessing.ProximityFilter3D;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.Binner;
import ij.plugin.Duplicator;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;

import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static de.embl.cba.trainableDeepSegmentation.DeepSegmentation.logger;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.*;

public abstract class ResultUtils
{

    private static void saveClassAsImaris( int classId, ResultExportSettings resultExportSettings )
    {
        String fileName = resultExportSettings.classNames.get( classId );

        ImarisDataSet imarisDataSet = new ImarisDataSet();

        imarisDataSet.setFromImagePlus(
                resultExportSettings.resultImagePlus,
                resultExportSettings.binning,
                resultExportSettings.directory,
                fileName,
                "/");

        ArrayList< String > channelNames = new ArrayList<>();

        channelNames.add( fileName );

        imarisDataSet.setChannelNames( channelNames  );

        ImarisWriter.writeHeader( imarisDataSet, resultExportSettings.directory, resultExportSettings.exportNamesPrefix + fileName + ".ims" );

        Hdf5DataCubeWriter writer = new Hdf5DataCubeWriter();

        for ( int t = resultExportSettings.timePointsFirstLast[ 0 ]; t <= resultExportSettings.timePointsFirstLast[ 1 ]; ++t )
        {
            ImagePlus impClass = getBinnedAndProximityFilteredClassImage( classId, resultExportSettings, t );

            logger.progress( "Writing " + fileName+ ", frame:", ( t + 1 ) + "/" + resultExportSettings.resultImagePlus.getNFrames() + "..." );

            writer.writeImarisCompatibleResolutionPyramid( impClass, imarisDataSet, 0, t );
        }
    }


    public static void saveRawDataAsImaris( ResultExportSettings resultExportSettings  )
    {
        if ( resultExportSettings.rawData.getNChannels() > 1 )
        {
            logger.error( "Multi-channel raw data export for imaris is currently not implemented." );
            return;
        }

        String fileName = "raw-data";

        ImarisDataSet imarisDataSet = new ImarisDataSet();

        imarisDataSet.setFromImagePlus(
                resultExportSettings.rawData,
                resultExportSettings.binning,
                resultExportSettings.directory,
                fileName,
                "/");

        ArrayList< String > channelNames = new ArrayList<>();

        channelNames.add( fileName );

        imarisDataSet.setChannelNames( channelNames  );

        // Header
        ImarisWriter.writeHeader( imarisDataSet, resultExportSettings.directory, resultExportSettings.exportNamesPrefix + fileName + ".ims" );

        Hdf5DataCubeWriter writer = new Hdf5DataCubeWriter();

        for ( int t = resultExportSettings.timePointsFirstLast[ 0 ]; t <= resultExportSettings.timePointsFirstLast[ 1 ]; ++t )
        {
            ImagePlus rawDataFrame = getBinnedRawDataFrame( resultExportSettings, t );

            logger.progress( "Writing " + fileName+ ", frame:", ( t + 1 ) + "/" + resultExportSettings.resultImagePlus.getNFrames() + "..." );

            writer.writeImarisCompatibleResolutionPyramid( rawDataFrame, imarisDataSet, 0, t );

        }
    }


    private static void logDone( ResultExportSettings resultExportSettings, String className, int t, String s )
    {
        logger.progress( s + className + ", frame:", ( t + 1 ) + "/" + resultExportSettings.resultImagePlus.getNFrames() );
    }

    public static ImagePlus getBinnedClassImage( int classId, ResultExportSettings resultExportSettings, int t )
    {

        ImagePlus impClass = getClassImage( classId, t, resultExportSettings.resultImagePlus, resultExportSettings.classLutWidth );

        if ( resultExportSettings.binning[0] * resultExportSettings.binning[1] * resultExportSettings.binning[2] > 1 )
        {
            Binner binner = new Binner();
            impClass = binner.shrink( impClass, resultExportSettings.binning[ 0 ], resultExportSettings.binning[ 1 ], resultExportSettings.binning[ 2 ], Binner.AVERAGE );
        }

        return impClass;
    }


    public static ImagePlus getBinnedClassImageMemoryEfficient(
            int classId, ResultExportSettings resultExportSettings, int t,
            de.embl.cba.utils.logging.Logger logger, int numThreads )
    {

        int nz = (int) resultExportSettings.resultImage.getDimensions()[ Z ];
        int nx = (int) resultExportSettings.resultImage.getDimensions()[ X ];
        int ny = (int) resultExportSettings.resultImage.getDimensions()[ Y ];

        int dx = resultExportSettings.binning[0];
        int dy = resultExportSettings.binning[1];
        int dz = resultExportSettings.binning[2];

        int classLutWidth = resultExportSettings.classLutWidth;
        int[] intensityGate = new int[]{ classId * classLutWidth + 1, (classId + 1 ) * classLutWidth };

        Binner binner = new Binner();

        ImageStack binnedStack = new ImageStack( nx / dx, ny / dy,  (int) Math.ceil( nz / dz ) );

        long startTime = System.currentTimeMillis();

        ExecutorService exe = Executors.newFixedThreadPool( numThreads );
        ArrayList< Future > futures = new ArrayList<>(  );

        for ( int iz = 0; iz < nz; iz += dz )
        {
            //futures.add(
            //        exe.submit(

            // TODO: put into callable function, returing the binned processors and put them later in a stack
            ImageStack tmpStack = new ImageStack ( nx , ny );

            for ( int iz2 = iz; iz2 < iz + dz; ++iz2 )
            {
                tmpStack.addSlice( resultExportSettings.resultImage.getSlice( iz2 + 1, t + 1 )  );
            }

            ImagePlus tmpImage = new ImagePlus( "", tmpStack );
            de.embl.cba.bigDataTools.utils.Utils.applyIntensityGate( tmpImage, intensityGate );
            ImagePlus tmpBinned = binner.shrink( tmpImage, dx, dy, dz, Binner.AVERAGE );

            binnedStack.setProcessor( tmpBinned.getProcessor(), (iz / dz) + 1 );

            logger.progress( "Creating binned class image", null, startTime, iz, nz   );
        }

        ImagePlus binnedClassImage = new ImagePlus( "binnedClassImage", binnedStack );

        return binnedClassImage;
    }


    public static ImagePlus getBinnedAndProximityFilteredClassImage( int classId, ResultExportSettings resultExportSettings, int t )
    {

        ImagePlus impClass = getBinnedClassImage( classId, resultExportSettings, t );

        if ( resultExportSettings.proximityFilterSettings.doSpatialProximityFiltering )
        {
            logger.info( "Applying proximity filter..." );
            impClass = ProximityFilter3D.multiply( impClass, resultExportSettings.proximityFilterSettings.dilatedBinaryReferenceMask );
        }

        return impClass;
    }

    public static void saveClassAsTiff( int classId, ResultExportSettings resultExportSettings )
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



    private static ImagePlus getBinnedRawDataFrame( ResultExportSettings resultExportSettings, int t )
    {
        Duplicator duplicator = new Duplicator();

        ImagePlus rawDataFrame = duplicator.run( resultExportSettings.rawData, 1, 1, 1, resultExportSettings.rawData.getNSlices(), t + 1, t + 1 );

        if ( resultExportSettings.binning[ 0 ] * resultExportSettings.binning[ 1 ] * resultExportSettings.binning[ 2 ] > 1 )
        {
            Binner binner = new Binner();
            rawDataFrame = binner.shrink( rawDataFrame, resultExportSettings.binning[ 0 ], resultExportSettings.binning[ 1 ], resultExportSettings.binning[ 2 ], Binner.AVERAGE );
        }
        return rawDataFrame;
    }

    public static void exportResults( ResultExportSettings resultExportSettings )
    {

        logger.info( "Exporting results, using modality: " + resultExportSettings.exportType );
        logger.info( "Exporting results to: " + resultExportSettings.directory );

        setExportTimePoints( resultExportSettings );

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

        exportRawData( resultExportSettings );

        createImarisHeader( resultExportSettings );

        logger.info( "Export of results finished." );

    }

    private static void setExportTimePoints( ResultExportSettings resultExportSettings )
    {
        if ( resultExportSettings.timePointsFirstLast == null )
        {
            resultExportSettings.timePointsFirstLast = new int[2];
            resultExportSettings.timePointsFirstLast[ 0 ] = 0;
            resultExportSettings.timePointsFirstLast[ 1 ] = resultExportSettings.resultImagePlus.getNFrames() - 1;
        }
    }

    private static void createImarisHeader( ResultExportSettings resultExportSettings )
    {
        if ( resultExportSettings.exportType.equals( ResultExportSettings.SEPARATE_IMARIS ) )
        {
            ImarisUtils.createImarisMetaFile( resultExportSettings.directory );
        }
    }

    private static void exportRawData( ResultExportSettings resultExportSettings )
    {
        if ( resultExportSettings.exportType.equals( ResultExportSettings.SEPARATE_IMARIS ) )
        {
            if ( resultExportSettings.saveRawData )
            {
                saveRawDataAsImaris( resultExportSettings );
            }
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

    private static ImagePlus getClassImage( int classId, int t, ImagePlus result, int CLASS_LUT_WIDTH )
    {
        ImagePlus impClass;

        Duplicator duplicator = new Duplicator();

        impClass = duplicator.run( result, 1, 1, 1, result.getNSlices(), t + 1, t + 1 );

        int[] intensityGate = new int[]{ classId * CLASS_LUT_WIDTH + 1, (classId + 1 ) * CLASS_LUT_WIDTH };

        de.embl.cba.bigDataTools.utils.Utils.applyIntensityGate( impClass, intensityGate );

        return ( impClass );

    }
}
