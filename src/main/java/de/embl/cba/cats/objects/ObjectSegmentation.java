package de.embl.cba.cats.objects;

import de.embl.cba.bigDataTools.utils.Utils;
import de.embl.cba.cats.CATS;
import de.embl.cba.cats.results.ResultExportSettings;
import de.embl.cba.cats.results.ResultExport;
import fiji.util.gui.GenericDialogPlus;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.Duplicator;
import inra.ijpb.binary.BinaryImages;
import inra.ijpb.morphology.AttributeFiltering;
import inra.ijpb.segment.Threshold;
import mcib3d.geom.Objects3DPopulation;
import mcib3d.image3d.*;

public class ObjectSegmentation
{

    ObjectSegmentationSettings settings;
    CATS CATS;

    class ObjectSegmentationSettings
    {
        public int minVolumeInPixels = 0;
        public int[] binning;
        public int t;
        public int classId;
        public double probabilityThreshold;
        public float threshold;
        public String method;
        public boolean showSegmentationImages;
    }

    public ObjectSegmentation( CATS CATS )
    {
        this.CATS = CATS;
    }

    public SegmentedObjects runFromUI( )
    {

        GenericDialog gd = showSegmentationDialog();

        if ( gd == null ) return null;

        setSettingsFromInitialisationDialog( gd );

        ImagePlus probabilities = getBinnedProbabilityImage( settings );

        probabilities.show();

        NonBlockingGenericDialog thresholdDialog = showThresholdDialog();

        setThreshold( (float) thresholdDialog.getNextNumber() );

        SegmentedObjects segmentedObjects = getSegmentedObjects( probabilities );

        return segmentedObjects;

    }

    private SegmentedObjects getSegmentedObjects( ImagePlus probabilities )
    {
        long start = System.currentTimeMillis();

        SegmentedObjects segmentedObjects;

        switch ( settings.method )
        {
            case MORPHOLIBJ:
                segmentedObjects = segmentUsingMorphoLibJ( settings, probabilities );
                break;
            case IMAGE_SUITE_3D:
                segmentedObjects = segmentUsing3dImageSuite( settings, probabilities );
                break;
            default:
                segmentedObjects = null;
        }


        CATS.logger.info( "\nSegmentation done in [s]: " + ( System.currentTimeMillis() - start ) / 1000 );
        return segmentedObjects;
    }

    private void setThreshold( float threshold )
    {
        settings.threshold = threshold; // (float) cats.getResultImage().getProbabilityRange() * (float) settings.probabilityThreshold;
//        settings.threshold  = ensureThresholdWithinRange( settings.threshold  );
    }

    private SegmentedObjects getSegmentedObjects( ImageInt labelMask )
    {
        Objects3DPopulation objects3DPopulation = new Objects3DPopulation(  labelMask );

        SegmentedObjects segmentedObjects = new SegmentedObjects();
        segmentedObjects.objects3DPopulation = objects3DPopulation;
        segmentedObjects.name = CATS.getClassName( settings.classId );
        return segmentedObjects;

    }

    private final static String MORPHOLIBJ = "MorphoLibJ";
    private final static String IMAGE_SUITE_3D = "3D Image Suite";


    private GenericDialog showSegmentationDialog()
    {
        GenericDialog gd = new GenericDialogPlus("Object Segmentation");

        gd.addChoice( "Class", CATS.getClassNames().toArray( new String[0] ), CATS.getClassNames().get( 1 ) );

        gd.addNumericField( "Time frame ", 1, 0 );

//        gd.addNumericField( "Certainty threshold [0-1] ", 0.20, 2);

//        gd.addChoice( "Do segmentation using ", new String[]{MORPHOLIBJ,IMAGE_SUITE_3D}, IMAGE_SUITE_3D);

        gd.addStringField( "Binning during segmentation", "1,1,1", 10  );

//        gd.addCheckbox( "Show (binned) segmentation images", true );

        gd.showDialog();

        if ( gd.wasCanceled() ) return null;
        return gd;
    }

    private NonBlockingGenericDialog showThresholdDialog()
    {
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Probability threshold");

        gd.addMessage( "Please inspect the probability image to decide on a threshold." );

        int maxThreshold = CATS.getResultImage().getProbabilityRange();

        gd.addNumericField( "Threshold [ 1 - " + maxThreshold + " ]", 3, 0 );

        gd.showDialog();

        if ( gd.wasCanceled() ) return null;

        return gd;
    }

    private void setSettingsFromInitialisationDialog( GenericDialog gd )
    {
        settings = new ObjectSegmentationSettings();
        settings.classId = gd.getNextChoiceIndex();
        settings.t = (int) gd.getNextNumber() - 1;
//        settings.probabilityThreshold = gd.getNextNumber();
        settings.method = IMAGE_SUITE_3D; //      settings.method = gd.getNextChoice();
        settings.binning =  Utils.delimitedStringToIntegerArray( gd.getNextString().trim(), ",");
        settings.showSegmentationImages = true; // gd.getNextBoolean();
    }

    private SegmentedObjects segmentUsingMorphoLibJ( ObjectSegmentationSettings settings, ImagePlus probabilities )
    {

        double threshold = CATS.getResultImage().getProbabilityRange() * settings.probabilityThreshold;

        int connectivity = 6;

        CATS.logger.info( "\nComputing label mask..." );

        CATS.logger.info( "\nApplying threshold: " + threshold + " ..." );
        ImagePlus binary = Threshold.threshold( probabilities, threshold, 255.0 );

        CATS.logger.info( "\nApplying object size filter: " + settings.minVolumeInPixels + " ..." );
        binary = new ImagePlus( "", AttributeFiltering.volumeOpening( binary.getStack(), settings.minVolumeInPixels) );

        CATS.logger.info( "\nComputing connected components..." );
        ImagePlus labelMask = BinaryImages.componentsLabeling( binary, connectivity, 16 );
        int numObjects = (int) labelMask.getDisplayRangeMax();
        CATS.logger.info( "...found objects: " + numObjects );

        CATS.logger.info( "\nConverting label mask to objects..." );
        SegmentedObjects segmentedObjects = getSegmentedObjects( ImageInt.wrap( labelMask ) );

        return segmentedObjects;

        //cats.logger.info( "\nVolume measurements..." );
        //ResultsTable volumes = GeometricMeasures3D.volume( labelMask.getStack(), new double[]{ 1, 1, 1 } );
        //volumes.show( "Volumes" );

        //double[][] centroids = GeometricMeasures3D.centroids( labelMask.getStack(), getAllLabels( numObjects ) );

    }

    private ImagePlus getBinnedProbabilityImage( ObjectSegmentationSettings settings )
    {
        ResultExportSettings resultExportSettings = new ResultExportSettings();
        resultExportSettings.resultImage = CATS.getResultImage();
        resultExportSettings.binning = settings.binning;
        resultExportSettings.classLutWidth = CATS.getResultImage().getProbabilityRange();

        ImagePlus probabilities =  ResultExport.getBinnedClassImageMemoryEfficient(
                settings.classId, resultExportSettings, settings.t,
                CATS.getLogger(), CATS.numThreads );

        probabilities.setTitle( "probabilities" );

        if ( settings.showSegmentationImages )
        {
            probabilities.show();
            probabilities.setDisplayRange( 0, CATS.getResultImage().getProbabilityRange() );
        }

        return probabilities;
    }


    private SegmentedObjects segmentUsing3dImageSuite( ObjectSegmentationSettings settings, ImagePlus probabilities )
    {

        CATS.logger.info( "\nSegmenting image..." );

        ImageLabeller labeler = new ImageLabeller();

        if ( settings.minVolumeInPixels > 0)
        {
            labeler.setMinSize( settings.minVolumeInPixels );
        }
        //if (max > 0) {
        //    labeler.setMaxsize(max);
        //}

        ImageHandler img = ImageHandler.wrap( probabilities );
        ImageInt bin = img.thresholdAboveInclusive( settings.threshold );
        ImageInt lbl = labeler.getLabels( bin );

        CATS.logger.info( "...done. " );

        CATS.logger.info( "\nCreating objects..." );
        SegmentedObjects segmentedObjects = getSegmentedObjects( lbl );
        segmentedObjects.objects3DPopulation.setScaleXY( settings.binning[ 0 ] );
        segmentedObjects.objects3DPopulation.setScaleZ( settings.binning[ 2 ] );
        CATS.logger.info( "...found objects: " + segmentedObjects.objects3DPopulation.getNbObjects()  );

        return segmentedObjects;

    }

    private float ensureThresholdWithinRange( float threshold )
    {
        threshold = (float) Math.max( threshold, 1.0 );
        threshold = (float) Math.min( threshold, CATS.getResultImage().getProbabilityRange() );
        return threshold;
    }


    private SegmentedObjects getSegmentedObjects( Segment3DImage segment3DImage )
    {

        ImageInt labelMask = segment3DImage.getLabelledObjectsImage3D();
        Objects3DPopulation objects3DPopulation = new Objects3DPopulation(  labelMask );

        if ( settings.showSegmentationImages )
        {
            labelMask.getImagePlus().show();
        }

        objects3DPopulation.setScaleXY( settings.binning[ 0 ] );
        objects3DPopulation.setScaleZ( settings.binning[ 2 ] );

        SegmentedObjects segmentedObjects = new SegmentedObjects();
        segmentedObjects.objects3DPopulation = objects3DPopulation;
        segmentedObjects.name = CATS.getClassName( settings.classId );
        return segmentedObjects;
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
