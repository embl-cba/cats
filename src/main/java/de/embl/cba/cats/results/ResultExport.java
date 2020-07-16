package de.embl.cba.cats.results;

import de.embl.cba.cats.postprocessing.ProximityFilter3D;
import de.embl.cba.cats.utils.IOUtils;
import de.embl.cba.imaris.H5DataCubeWriter;
import de.embl.cba.imaris.ImarisDataSet;
import de.embl.cba.imaris.ImarisUtils;
import de.embl.cba.imaris.ImarisWriter;
import de.embl.cba.log.Logger;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Prefs;
import ij.plugin.Binner;
import ij.plugin.Duplicator;
import ij.process.ImageProcessor;
import ij.process.LUT;
import net.imglib2.FinalInterval;

import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static de.embl.cba.bigdataprocessor.utils.Utils.applyIntensityGate;
import static de.embl.cba.cats.CATS.logger;
import static de.embl.cba.cats.utils.IntervalUtils.*;

public abstract class ResultExport
{
    private static void saveClassAsImaris( int classId, ResultExportSettings resultExportSettings )
    {
        String fileName = resultExportSettings.classNames.get( classId );

        ImarisDataSet imarisDataSet = new ImarisDataSet(
                resultExportSettings.resultImagePlus,
                resultExportSettings.binning,
                resultExportSettings.directory,
                fileName );

        setChannelName( fileName, imarisDataSet );

        ImarisWriter.writeHeaderFile(
                imarisDataSet, resultExportSettings.directory,
                resultExportSettings.exportNamesPrefix + fileName + ".ims" );

        H5DataCubeWriter writer = new H5DataCubeWriter();

        for ( int t = resultExportSettings.timePointsFirstLast[ 0 ]; t <= resultExportSettings.timePointsFirstLast[ 1 ]; ++t )
        {
            ImagePlus impClass = getBinnedAndProximityFilteredClassImage( classId, resultExportSettings, t );

            logger.progress( "Writing " + fileName+ ", frame:", ( t + 1 )
                    + "/" + resultExportSettings.resultImagePlus.getNFrames() + "..." );

            writer.writeImarisCompatibleResolutionPyramid( impClass, imarisDataSet, 0, t );
        }
    }

    public static void saveRawDataAsImaris( ResultExportSettings resultExportSettings  )
    {
        String fileName = "raw-data";

        ImarisDataSet imarisDataSet = new ImarisDataSet(
                resultExportSettings.inputImagePlus,
                resultExportSettings.binning,
                resultExportSettings.directory,
                fileName );

        // Header
        ImarisWriter.writeHeaderFile( imarisDataSet, resultExportSettings.directory, resultExportSettings.exportNamesPrefix + fileName + ".ims" );

        H5DataCubeWriter writer = new H5DataCubeWriter();

        for ( int c = 0; c < imarisDataSet.getNumChannels(); ++c )
        {
            for ( int t = resultExportSettings.timePointsFirstLast[ 0 ];
                  t <= resultExportSettings.timePointsFirstLast[ 1 ];
                  ++t )
            {

                logger.progress( "Writing " + fileName,
                        ", frame: " + ( t + 1 ) + "/" + resultExportSettings.resultImagePlus.getNFrames()
                                + ", channel: "+ ( c + 1 ) + "/" + imarisDataSet.getNumChannels() + "..."
                );

                //logger.info( "Copying into RAM..." );
                ImagePlus rawDataFrame = getBinnedRawDataFrame( resultExportSettings, c, t );

                //logger.info( "Writing as Imaris..." );
                writer.writeImarisCompatibleResolutionPyramid( rawDataFrame, imarisDataSet, c, t );
            }
        }
    }


    private ImagePlus getChannelView( final ImagePlus imp, final int channel )
    {
//        final int imagePlusChannelDimension = 2;
//        RandomAccessibleInterval rai = ImageJFunctions.wrap( imp );
//        final IntervalView singleChannelView = Views.hyperSlice( rai, imagePlusChannelDimension, channel );
        return null;
    }

    private static void setChannelName( String fileName, ImarisDataSet imarisDataSet )
    {
        ArrayList< String > channelNames = new ArrayList<>();

        channelNames.add( fileName );

        imarisDataSet.setChannelNames( channelNames );
    }

    private static void logDone( ResultExportSettings resultExportSettings,
                                 String className,
                                 int t,
                                 String s )
    {
        logger.progress( s + className + ", frame:",
                ( t + 1 ) + "/" + resultExportSettings.resultImagePlus.getNFrames() );
    }

    public static ImagePlus getBinnedClassImage(
            int classId,
            ResultExportSettings resultExportSettings,
            int t )
    {

        ImagePlus impClass = getClassImage( classId, t, resultExportSettings );

        if ( resultExportSettings.binning[0] * resultExportSettings.binning[1] * resultExportSettings.binning[2] > 1 )
        {
            Binner binner = new Binner();
            impClass = binner.shrink( impClass,
                    resultExportSettings.binning[ 0 ],
                    resultExportSettings.binning[ 1 ],
                    resultExportSettings.binning[ 2 ], Binner.AVERAGE );
        }

        return impClass;
    }


    public static ImagePlus getBinnedClassImageMemoryEfficient(
            int classId, ResultExportSettings settings, int t,
            Logger logger, int numThreads )
    {
        logger.info( "Computing probability image for " + settings.classNames.get( classId ) + ", using " + numThreads + " threads." );

        int nz = (int) settings.resultImage.getDimensions()[ Z ];
        int nx = (int) settings.resultImage.getDimensions()[ X ];
        int ny = (int) settings.resultImage.getDimensions()[ Y ];

        int dx = settings.binning[0];
        int dy = settings.binning[1];
        int dz = settings.binning[2];

        ImageStack binnedStack =
                new ImageStack(
                        nx / dx,
                        ny / dy,
                        (int) Math.ceil( 1.0 * nz / dz ) );

        long startTime = System.currentTimeMillis();

        ExecutorService exe = Executors.newFixedThreadPool( numThreads );
        ArrayList< Future< ImagePlus > > futures = new ArrayList<>(  );

        for ( int iz = 0; iz < nz; iz += dz )
        {
            futures.add(
                    exe.submit(
                            CallableResultImageBinner.getBinned(
                                    settings,
                                    classId,
                                    iz, iz + dz - 1, t,
                                    logger,
                                    startTime,
                                    nz )
                    )
            );
        }


        int i = 0;
        for ( Future<ImagePlus> future : futures )
        {
            // getInstancesAndMetadata feature images
            try
            {
                ImagePlus binnedSlice = future.get();
                binnedStack.setProcessor( binnedSlice.getProcessor(), ++i );
                System.gc();
            }
            catch ( InterruptedException e )
            {
                e.printStackTrace();
            }
            catch ( ExecutionException e )
            {
                e.printStackTrace();
            }
        }

        futures = null;
        exe.shutdown();
        System.gc();

        ImagePlus binnedClassImage = new ImagePlus( "binned_class_" + classId, binnedStack );

        return binnedClassImage;
    }


    public static ImagePlus getBinnedAndProximityFilteredClassImage(
            int classId, ResultExportSettings resultExportSettings, int t )
    {
        ImagePlus impClass =
                getBinnedClassImageMemoryEfficient(
                        classId, resultExportSettings, t, logger, Prefs.getThreads() );

        if ( resultExportSettings.proximityFilterSettings.doSpatialProximityFiltering )
        {
            logger.info( "Applying proximity filter..." );
            impClass = ProximityFilter3D.multiply(
                    impClass,
                    resultExportSettings.proximityFilterSettings.dilatedBinaryReferenceMask );
        }

        return impClass;
    }

    public static void saveClassAsTiff( int classId, ResultExportSettings resultExportSettings )
    {
        String className = resultExportSettings.classNames.get( classId );

        for ( int t = resultExportSettings.timePointsFirstLast[ 0 ];
              t <= resultExportSettings.timePointsFirstLast[ 1 ]; ++t )
        {

            ImagePlus impClass =
                    getBinnedAndProximityFilteredClassImage( classId, resultExportSettings, t );

            String path;

            if ( resultExportSettings.resultImagePlus.getNFrames() > 1 )
            {
                path = resultExportSettings.directory +
                        File.separator + resultExportSettings.exportNamesPrefix
                        + className + "--T" + String.format( "%05d", t ) + ".tif";
            }
            else
            {
                path = resultExportSettings.directory +
                        File.separator + resultExportSettings.exportNamesPrefix
                        + className + ".tif";
            }

            IJ.saveAsTiff( impClass, path );

            logDone( resultExportSettings, className, t, "Done with export of " );
        }
    }

    private static ImagePlus createProbabilitiyImagePlusForClass(
            int classId,
            ResultExportSettings resultExportSettings )
    {

        String className = resultExportSettings.classNames.get( classId );

        final ImageStack stackOfAllTimepoints =
                new ImageStack(
                        resultExportSettings.inputImagePlus.getWidth(),
                        resultExportSettings.inputImagePlus.getHeight() );

        int numSlices = resultExportSettings.inputImagePlus.getNSlices();

        int numTimepoints =
                resultExportSettings.timePointsFirstLast[ 1 ]
                        - resultExportSettings.timePointsFirstLast[ 0 ] + 1;

        for ( int t = resultExportSettings.timePointsFirstLast[ 0 ];
              t <= resultExportSettings.timePointsFirstLast[ 1 ]; ++t )
        {
            ImagePlus impClass =
                    getBinnedAndProximityFilteredClassImage( classId, resultExportSettings, t );

            final ImageStack stackOfThisTimepoint = impClass.getStack();

            for ( int slice = 0; slice < numSlices; ++slice )
                stackOfAllTimepoints.addSlice( stackOfThisTimepoint.getProcessor( slice + 1 ) );
        }

        setSliceLabels(
                resultExportSettings.inputImagePlus.getStack(),
                stackOfAllTimepoints,
                className );

        final ImagePlus imp =
                new ImagePlus(
                        resultExportSettings.exportNamesPrefix + className,
                        stackOfAllTimepoints );

        imp.setDimensions( 1, numSlices, numTimepoints );
        imp.setOpenAsHyperStack( true );

        logDone( resultExportSettings, className, numTimepoints - 1, "Created " );

        return imp;

    }



    private static ImagePlus getBinnedRawDataFrame( ResultExportSettings resultExportSettings, int c, int t )
    {
        Duplicator duplicator = new Duplicator();

        ImagePlus rawDataFrame = duplicator.run( resultExportSettings.inputImagePlus, c + 1, c + 1, 1, resultExportSettings.inputImagePlus.getNSlices(), t + 1, t + 1 );

        if ( resultExportSettings.binning[ 0 ] * resultExportSettings.binning[ 1 ] * resultExportSettings.binning[ 2 ] > 1 )
        {
            Binner binner = new Binner();
            rawDataFrame = binner.shrink( rawDataFrame, resultExportSettings.binning[ 0 ], resultExportSettings.binning[ 1 ], resultExportSettings.binning[ 2 ], Binner.AVERAGE );
        }

        return rawDataFrame;
    }

    public static ArrayList< ImagePlus > exportResults(
            ResultExportSettings resultExportSettings )
    {
        logger.info( "# Exporting results, using modality: " + resultExportSettings.exportType );

        configureTimePointsExport( resultExportSettings );

        if ( isSaveImage( resultExportSettings ) )
            IOUtils.createDirectoryIfNotExists( resultExportSettings.directory );

        configureClassExport( resultExportSettings );

        configureExportBinning( resultExportSettings );

        exportRawData( resultExportSettings );

        final ArrayList< ImagePlus > classImps = exportClasses( resultExportSettings );

        createImarisHeader( resultExportSettings );

        logger.info( "Export of results finished." );

        return classImps;
    }

    private static void setSliceLabels(
            ImageStack source,
            ImageStack target,
            String className )
    {

        if ( source.getSize() != target.getSize() )
        {
            logger.info( "Results slice naming not yet " +
                    "implemented for multi-channel images." );
            return;
        }

        final int numImagePlanes = source.getSize();

        for ( int planeId = 0; planeId < numImagePlanes; ++planeId )
        {
            String sliceLabel = source.getSliceLabel( planeId + 1 );

            if ( sliceLabel != null )
            {
                if ( sliceLabel.contains( "." ) )
                {
                    final String[] split = sliceLabel.split( "\\." );
                    sliceLabel = split[ 0 ];
                }

                target.setSliceLabel( sliceLabel + "-" + className, planeId + 1 );
            }
            else
            {
                target.setSliceLabel( className, planeId + 1 );
            }
        }
    }

    public static boolean isSaveImage( ResultExportSettings resultExportSettings )
    {
        switch ( resultExportSettings.exportType )
        {
            case ResultExportSettings.SAVE_AS_CLASS_PROBABILITY_TIFF_STACKS:
                return true;
            case ResultExportSettings.SAVE_AS_IMARIS_STACKS:
                return true;
            case ResultExportSettings.SAVE_AS_CLASS_LABEL_MASK_TIFF_STACKS:
                return true;
            case ResultExportSettings.SAVE_AS_CLASS_PROBABILITIES_TIFF_SLICES:
                return true;
            default:
                return false;
        }
    }

    private static void configureClassExport( ResultExportSettings resultExportSettings )
    {
        if ( resultExportSettings.classesToBeExported == null )
        {
            resultExportSettings.classesToBeExported
                    = selectAllClasses( resultExportSettings.classNames );
        }
    }

    private static void configureTimePointsExport( ResultExportSettings resultExportSettings )
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
        if ( resultExportSettings.exportType.equals( ResultExportSettings.SAVE_AS_IMARIS_STACKS ) )
        {
            ImarisUtils.createImarisMetaFile( resultExportSettings.directory );
        }
    }

    private static void exportRawData( ResultExportSettings resultExportSettings )
    {
        if ( resultExportSettings.exportType.equals( ResultExportSettings.SAVE_AS_IMARIS_STACKS ) )
        {
            if ( resultExportSettings.saveRawData )
            {
                if ( resultExportSettings.exportType.equals(
                        ResultExportSettings.SAVE_AS_IMARIS_STACKS ) )
                {
                    saveRawDataAsImaris( resultExportSettings );
                }
                else if ( resultExportSettings.exportType.equals(
                        ResultExportSettings.SAVE_AS_CLASS_PROBABILITY_TIFF_STACKS ) )
                {
                    // TODO
                }
                else if ( resultExportSettings.exportType.equals(
                        ResultExportSettings.SHOW_AS_PROBABILITIES ) )
                {
                    // TODO
                }
            }
        }
    }

    private static ArrayList< ImagePlus > exportClasses(
            ResultExportSettings settings )
    {
        final ArrayList< ImagePlus > classImps = new ArrayList<>();

        if ( settings.exportType.equals(
                ResultExportSettings.SAVE_AS_CLASS_PROBABILITIES_TIFF_SLICES ) )
        {
            saveAsMultiClassTiffSlices( settings );
        }
        else if ( settings.exportType.equals(
                ResultExportSettings.SAVE_AS_CLASS_LABEL_MASK_TIFF_STACKS ) )
        {
            saveAsClassLabelMaskTiffStacks( settings );
        }
        else if ( settings.exportType.equals(
                ResultExportSettings.SHOW_AS_LABEL_MASKS ) )
        {
            showAsClassLabelMask( settings );
        }
        else
        {
            prepareProximityFilter( settings );

            for ( int classIndex = 0;
                  classIndex < settings.classesToBeExported.size(); ++classIndex )
            {
                if ( settings.classesToBeExported.get( classIndex ) )
                {
                    if ( settings.exportType.equals(
                            ResultExportSettings.SAVE_AS_IMARIS_STACKS ) )
                    {
                        saveClassAsImaris( classIndex, settings );
                    }
                    else if ( settings.exportType.equals(
                            ResultExportSettings.SAVE_AS_CLASS_PROBABILITY_TIFF_STACKS ) )
                    {
                        saveClassAsTiff( classIndex, settings );
                    }
                    else if ( settings.exportType.equals(
                            ResultExportSettings.SHOW_AS_PROBABILITIES ) )
                    {
                        final ImagePlus imp = createProbabilitiyImagePlusForClass(
                                classIndex, settings );
                        imp.show();
                    }
                    else if ( settings.exportType.equals(
                            ResultExportSettings.GET_AS_IMAGEPLUS_ARRAYLIST ) )
                    {
                        classImps.add(
                                createProbabilitiyImagePlusForClass(
                                        classIndex, settings ) );
                    }
                }
            }
        }

        return classImps;

    }

    private static void saveAsMultiClassTiffSlices( ResultExportSettings resultExportSettings )
    {
        // TODO: Binning is ignored here

        FinalInterval interval = resultExportSettings.resultImage.getInterval();

        String directory = resultExportSettings.directory;

        for ( long t = interval.min( T ); t <= interval.max( T ); ++t )
        {
            for ( long z = interval.min( Z ); z <= interval.max( Z ); ++z )
            {
                int slice = (int) z + 1;
                int frame = (int) t + 1;
                ImageProcessor ip = resultExportSettings.resultImage.getSlice( slice, frame);
                String filename = "classified--C01--T" + String.format( "%05d", frame )
                        + "--Z" + String.format( "%05d", slice ) + ".tif";
                String path = directory + File.separator + filename;
                IJ.saveAsTiff( new ImagePlus( filename, ip ), path );
            }
        }
    }

    private static void saveAsClassLabelMaskTiffStacks( ResultExportSettings settings )
    {
        // TODO: Binning is currently ignored here

        String directory = settings.directory;
        final ImagePlus result = settings.resultImagePlus;

        for ( int t = settings.timePointsFirstLast[ 0 ];
              t <= settings.timePointsFirstLast[ 1 ]; ++t )
        {
            final int frame =  t + 1;

            ImagePlus resultFrame = getLabelMask( settings, result, frame );

            IJ.saveAsTiff( resultFrame, getLabelMaskPath( settings, directory, frame ) );
        }

    }

    private static void showAsClassLabelMask( ResultExportSettings settings )
    {
        // TODO: Binning is currently ignored here

        final ImagePlus result = settings.resultImagePlus;

        for ( int t = settings.timePointsFirstLast[ 0 ];
              t <= settings.timePointsFirstLast[ 1 ]; ++t )
        {
            final int frame =  t + 1;

            ImagePlus resultFrame = getLabelMask( settings, result, frame );

            resultFrame.show();
        }

    }

    private static ImagePlus getLabelMask(
            ResultExportSettings settings, ImagePlus result, int frame )
    {
        ImagePlus resultFrame = new Duplicator().run(
                result,
                1, 1,
                1, result.getNSlices(),
                frame, frame );

        final LUT classLabelLUT = createClassLabelLUT( settings );

        for ( int z = 1; z <= resultFrame.getNSlices(); z++ )
        {
            final ImageProcessor processor = resultFrame.getStack().getProcessor( z );
            byte[] pixels = (byte[]) processor.getPixels();

            /**
             * 0 = Not classified at all
             * 1 - classLutWidth -> 1 = class 1
             * classLutWidth + 1 - 2 * classLutWidth -> 2 = class 2
             * ...
             */
            for ( int i = 0; i < pixels.length; i++ )
                pixels[ i ] = ( byte ) Math.ceil( 1.0 * pixels[ i ] / settings.classLutWidth );

        }


        ImageProcessor ip = resultFrame.getChannelProcessor();
        ip.setColorModel( classLabelLUT );
        if (resultFrame.getStackSize()>1)
            resultFrame.getStack().setColorModel( classLabelLUT );
        resultFrame.updateAndRepaintWindow();

        //IJ.run( resultFrame, "Spectrum", "");
//        IJ.run( resultFrame, "", "" );
//        resultFrame.setC( 0 );
        //resultFrame.getProcessor().setLut( classLabelLUT );
        resultFrame.setDisplayRange( 0, settings.classNames.size() );
        resultFrame.setTitle( "class label mask" );
        resultFrame.show();
        return resultFrame;
    }

    private static LUT createClassLabelLUT( ResultExportSettings settings )
    {
        final byte[] red = new byte[ 256 ];
        final byte[] green = new byte[ 256 ];
        final byte[] blue = new byte[ 256 ];

        for ( int iClass = 0; iClass < settings.classColors.length; iClass++ )
        {
            red[ iClass + 1 ] = ( byte ) ( settings.classColors[ iClass ].getRed() );
            green[ iClass + 1 ] = ( byte ) ( settings.classColors[ iClass ].getGreen() );
            blue[ iClass + 1 ] = ( byte ) ( settings.classColors[ iClass ].getBlue() );
        }

        return new LUT( red, green, blue );

    }

    private static String getLabelMaskPath(
            ResultExportSettings settings,
            String directory,
            int frame )
    {
        String filename;
        if ( settings.resultImagePlus.getNFrames() > 1 )
            filename = settings.exportNamesPrefix
                    + "labelMask" + "--T" + String.format( "%05d", frame ) + ".tif";
        else
            filename = settings.exportNamesPrefix
                    + "labelMask.tif";

        return directory + File.separator + filename;
    }

    private static void prepareProximityFilter( ResultExportSettings resultExportSettings )
    {
        ProximityFilterSettings settings = resultExportSettings.proximityFilterSettings;

        if (  settings.doSpatialProximityFiltering )
        {
            logger.info( "Computing proximity mask..." );
            ImagePlus impReferenceClass =
                    getBinnedClassImage( settings.referenceClassId, resultExportSettings, 0  );
            settings.dilatedBinaryReferenceMask =
                    ProximityFilter3D.getDilatedBinaryUsingEDT(
                            impReferenceClass,
                            settings.distanceInPixelsAfterBinning  );
        }
    }

    private static void configureExportBinning( ResultExportSettings resultExportSettings )
    {
        if ( resultExportSettings.binning == null )
        {
            resultExportSettings.binning = new int[] { 1, 1, 1 };
        }
    }

    public static final ArrayList< Boolean > selectAllClasses( ArrayList<String> classNames )
    {
        ArrayList< Boolean > classesToBeSaved = new ArrayList<>();
        for ( String className : classNames ) classesToBeSaved.add( true );
        return classesToBeSaved;
    }

    private static ImagePlus getClassImage( int classId, int t, ResultExportSettings settings)
    {
        Duplicator duplicator = new Duplicator();

        ImagePlus impClass = duplicator.run( settings.resultImagePlus,
                1, 1,
                1, settings.resultImagePlus.getNSlices(),
                t + 1, t + 1 );

        applyClassIntensityGate( classId, settings, impClass );

        convertToProperBitDepth( impClass, settings );

        return ( impClass );

    }

    private static void applyClassIntensityGate(
            int classId,
            ResultExportSettings settings,
            ImagePlus impClass )
    {
        int[] intensityGate = new int[]{
                classId * settings.classLutWidth + 1,
                (classId + 1 ) * settings.classLutWidth };

        applyIntensityGate( impClass, intensityGate );
    }

    public static void convertToProperBitDepth( ImagePlus impClass,
                                                ResultExportSettings settings )
    {
        int factorToFillBitDepth = (int) ( 255.0  / settings.classLutWidth );

        if ( settings.inputImagePlus.getBitDepth() == 16 )
        {
            IJ.run( impClass, "16-bit", "" );
            factorToFillBitDepth = (int) ( 65535.0  / settings.classLutWidth );
        }

        if ( settings.inputImagePlus.getBitDepth() == 32 )
        {
            IJ.run( impClass, "32-bit", "" );
            factorToFillBitDepth = (int) ( 255.0  / settings.classLutWidth );
        }

        IJ.run( impClass, "Multiply...", "value=" +
                factorToFillBitDepth + " stack");
    }
}
