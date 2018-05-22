package de.embl.cba.trainableDeepSegmentation.postprocessing;

import de.embl.cba.bigDataTools.utils.Utils;
import de.embl.cba.trainableDeepSegmentation.DeepSegmentation;
import de.embl.cba.trainableDeepSegmentation.results.ResultExportSettings;
import de.embl.cba.trainableDeepSegmentation.results.ResultUtils;
import fiji.util.gui.GenericDialogPlus;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.Duplicator;
import inra.ijpb.binary.BinaryImages;
import inra.ijpb.morphology.AttributeFiltering;
import inra.ijpb.segment.Threshold;
import mcib3d.geom.Objects3DPopulation;
import mcib3d.image3d.ImageShort;
import mcib3d.image3d.Segment3DImage;
import mcib_plugins.tools.RoiManager3D_2;

public class ObjectSegmentation
{

    ObjectSegmentationSettings objectSegmentationSettings;
    DeepSegmentation deepSegmentation;

    class ObjectSegmentationSettings
    {
        public int minVolumeInPixels;
        public int[] binning;
        public int t;
        public int classId;
        public double probabilityThreshold;

    }

    public ObjectSegmentation( DeepSegmentation deepSegmentation )
    {
        this.deepSegmentation = deepSegmentation;
    }

    public SegmentedObjects runUI( )
    {

        GenericDialog gd = openGenericDialog();
        if ( gd == null ) return null;
        setSettingsFromUI( gd );

        ImagePlus labelMask = segmentUsingMorphoLibJ( objectSegmentationSettings );
        labelMask.show();

        Objects3DPopulation objects3DPopulation = new Objects3DPopulation(  new ImageShort( labelMask ) );

        SegmentedObjects segmentedObjects = new SegmentedObjects();
        segmentedObjects.objects3DPopulation = objects3DPopulation;
        segmentedObjects.name = deepSegmentation.getClassName( objectSegmentationSettings.classId );

        return segmentedObjects;

    }

    private GenericDialog openGenericDialog()
    {
        GenericDialog gd = new GenericDialogPlus("Object Segmentation");

        gd.addChoice( "Class", deepSegmentation.getClassNames().toArray( new String[0] ), deepSegmentation.getClassNames().get( 0 ) );

        gd.addStringField( "Binning during segmentation", "1,1,1", 10  );

        gd.addNumericField( "Time frame ", 1, 0 );

        gd.addNumericField( "Certainty threshold [0-1] ", 0.20, 2);

        gd.showDialog();

        if ( gd.wasCanceled() ) return null;
        return gd;
    }

    private void setSettingsFromUI( GenericDialog gd )
    {
        objectSegmentationSettings = new ObjectSegmentationSettings();
        objectSegmentationSettings.classId = gd.getNextChoiceIndex();
        objectSegmentationSettings.binning =  Utils.delimitedStringToIntegerArray( gd.getNextString().trim(), ",");
        objectSegmentationSettings.t = (int) gd.getNextNumber() - 1;
        objectSegmentationSettings.probabilityThreshold = gd.getNextNumber();
    }

    private ImagePlus segmentUsingMorphoLibJ( ObjectSegmentationSettings settings )
    {

        ResultExportSettings resultExportSettings = new ResultExportSettings();
        resultExportSettings.resultImagePlus = deepSegmentation.getResultImage().getWholeImageCopy();
        resultExportSettings.binning = settings.binning;
        resultExportSettings.classLutWidth = deepSegmentation.getResultImage().getProbabilityRange();

        ImagePlus probabilities = ResultUtils.getBinnedClassImage( settings.classId, resultExportSettings, settings.t );

        double threshold = deepSegmentation.getResultImage().getProbabilityRange() * settings.probabilityThreshold;

        int connectivity = 6;

        deepSegmentation.logger.info( "\nComputing label mask..." );

        deepSegmentation.logger.info( "Threshold: " + threshold  );
        ImagePlus binary = Threshold.threshold( probabilities, threshold, 255.0 );

        //deepSegmentation.logger.info( "MinNumVoxels: " + settings.minVolumeInPixels );
        //binary = new ImagePlus( "", AttributeFiltering.volumeOpening( binary.getStack(), settings.minVolumeInPixels) );

        deepSegmentation.logger.info( "\nComputing connected components..." );
        ImagePlus labelMask = BinaryImages.componentsLabeling( binary, connectivity, 16 );
        int numObjects = (int) labelMask.getDisplayRangeMax();
        deepSegmentation.logger.info( "...found objects: " + numObjects );

        return labelMask;

        //deepSegmentation.logger.info( "\nVolume measurements..." );
        //ResultsTable volumes = GeometricMeasures3D.volume( labelMask.getStack(), new double[]{ 1, 1, 1 } );
        //volumes.show( "Volumes" );

        //double[][] centroids = GeometricMeasures3D.centroids( labelMask.getStack(), getAllLabels( numObjects ) );

    }


    private void segmentUsing3dImageSuite( ObjectSegmentationSettings settings )
    {

        ResultExportSettings resultExportSettings = new ResultExportSettings();
        resultExportSettings.resultImagePlus = deepSegmentation.getResultImage().getWholeImageCopy();
        resultExportSettings.binning = settings.binning;
        resultExportSettings.classLutWidth = deepSegmentation.getResultImage().getProbabilityRange();

        ImagePlus probabilities = ResultUtils.getBinnedClassImage( settings.classId, resultExportSettings, settings.t );


        float threshold = (float) deepSegmentation.getResultImage().getProbabilityRange() * (float) settings.probabilityThreshold;

        int connectivity = 6;

        deepSegmentation.logger.info( "\nComputing label mask..." );

        Segment3DImage nucleiSegmentor = new Segment3DImage( probabilities, threshold, Float.MAX_VALUE );
        nucleiSegmentor.setMinSizeObject( settings.minVolumeInPixels );
        nucleiSegmentor.setMaxSizeObject( Integer.MAX_VALUE );
        nucleiSegmentor.segment();

        nucleiSegmentor.getLabelledObjectsImage3D();


        /*
        deepSegmentation.logger.info( "Threshold: " + threshold  );
        ImagePlus binary = Threshold.threshold( probabilities, threshold, 255.0 );

        deepSegmentation.logger.info( "MinNumVoxels: " + minNumVoxels );
        ImagePlus binaryFiltered = new ImagePlus( "", AttributeFiltering.volumeOpening( th.getStack(), minNumVoxels) );

        deepSegmentation.logger.info( "\nComputing connected components..." );
        ImagePlus labelMask = BinaryImages.componentsLabeling( binaryFiltered, connectivity, 16 );
        int numObjects = (int) labelMask.getDisplayRangeMax();

        deepSegmentation.logger.info( "\nVolume measurements..." );
        ResultsTable volumes = GeometricMeasures3D.volume( labelMask.getStack(), new double[]{ 1, 1, 1 } );
        volumes.show( "Volumes" );

        double[][] centroids = GeometricMeasures3D.centroids( labelMask.getStack(), getAllLabels( numObjects ) );
        */

    }


    private int[] getAllLabels( int numObjects )
    {
        int[] labels = new int[numObjects];
        for ( int i = 0; i < numObjects; ++i )
        {
            labels[ i ] = i + 1;
        }
        return labels;
    }

    public static ImagePlus createLabelMaskForChannelAndFrame(
            ImagePlus image,
            int frame,
            int channel,
            int minNumVoxels,
            int lowerThreshold,
            int upperThreshold
    )
    {

        int connectivity = 6;

        long start = System.currentTimeMillis();

//        logger.info( "\n# Computing label mask..." );

        Duplicator duplicator = new Duplicator();
        ImagePlus imp = duplicator.run( image, channel, channel, 1, image.getNSlices(), frame, frame );

//        logger.info( "Threshold: " + lower + ", " + upper );
        ImagePlus th = Threshold.threshold( imp, lowerThreshold, upperThreshold );

//        logger.info( "MinNumVoxels: " + minNumVoxels );
        ImagePlus th_sf = new ImagePlus( "", AttributeFiltering.volumeOpening( th.getStack(), minNumVoxels) );

//        logger.info( "Connectivity: " + conn );
        ImagePlus labelMask = BinaryImages.componentsLabeling( th_sf, connectivity, 16);

//        logger.info( "...done! It took [min]:" + (System.currentTimeMillis() - start ) / ( 1000 * 60) );

        return labelMask;
    }




}
