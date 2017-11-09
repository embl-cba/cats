package trainableDeepSegmentation.training;

import bigDataTools.logging.IJLazySwingLogger;
import bigDataTools.logging.Logger;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import javafx.geometry.Point3D;
import net.imglib2.FinalInterval;
import trainableDeepSegmentation.*;
import trainableDeepSegmentation.examples.Example;
import trainableDeepSegmentation.results.ResultImage;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

import static trainableDeepSegmentation.ImageUtils.*;

public class InstancesCreator {

    public InstancesCreator( )
    {
    }

    /**
     * Create training instancesMap out of the user markings
     *
     * @return set of instancesMap (feature vectors in Weka format)
     */
    public static Instances createInstancesFromLabels( ArrayList< Example > examples,
                                                       String inputImageTitle,
                                                       Settings settings,
                                                       ArrayList< String > featureNames,
                                                       ArrayList< String > classNames )
    {

        String instancesInfo = getInfoString( inputImageTitle, settings );

        Instances instances = createInstancesHeader(
                instancesInfo,
                featureNames,
                classNames  );

        for ( Example example : examples )
        {
            // loop over all pixels of the example
            // and add the feature values for each pixel to the trainingData
            // note: sub-setting of active features happens in another function
            for ( double[] values : example.instanceValuesArray )
            {
                instances.add( new DenseInstance(1.0, values) );
            }
        }

        return instances;

    }



    private static String getInfoString( String inputImageTitle, Settings settings )
    {

        String info = inputImageTitle;

        info += "--" + Settings.ANISOTROPY + ":" + settings.anisotropy;
        info += "--" + Settings.MAX_BIN_LEVEL + ":" + settings.maxBinLevel;
        info += "--" + Settings.BIN_FACTOR + ":" + settings.binFactor;
        info += "--" + Settings.MAX_DEEP_CONV_LEVEL + ":" + settings.maxDeepConvLevel;

        return ( info );

    }


    // TODO: improve separation from WekaSegmentation
    public static Instances createInstancesFromLabelImageRegion(
            WekaSegmentation wekaSegmentation,
            ImagePlus inputImage,
            ImagePlus labelImage,
            ResultImage result,
            String instancesName,
            FinalInterval interval,
            int numInstancesPerClassAndPlane,
            int numThreads,
            Logger logger )
    {
        final int numClasses = wekaSegmentation.getNumClasses();
        if ( logger == null ) logger = new IJLazySwingLogger();

		/*
		Img img = ImageJFunctions.wrap( labelImage );
		RectangleNeighborhoodGPL neighborhood = new RectangleNeighborhoodGPL<>( img );
		neighborhood.setPosition( 1,1 );
		neighborhood.setSpan( span );
		*/


        logger.info( "Computing features for label image region...");
        IntervalUtils.logInterval( interval );
        logger.info( "Instances per class and plane: " + numInstancesPerClassAndPlane);

        long startTime = System.currentTimeMillis();

        // Compute features
        FeatureProvider featureProvider = new FeatureProvider();
        featureProvider.setLogger( logger );
        featureProvider.isLogging( true );
        featureProvider.setInputImage( inputImage );
        featureProvider.setWekaSegmentation( wekaSegmentation );
        featureProvider.setInterval( interval );
        featureProvider.setActiveChannels( wekaSegmentation.settings.activeChannels );
        featureProvider.computeFeatures( numThreads,
                wekaSegmentation.maximumMultithreadedLevel );

        logger.info ( "...computed features  in [ms]: " +
                ( System.currentTimeMillis() - startTime ) );

        logger.info( "Computing instance values...");
        startTime = System.currentTimeMillis();

        // TODO: determine numClasses from labelImage!
        wekaSegmentation.settings.classNames = new ArrayList<>();
        wekaSegmentation.settings.classNames.add("label_im_class_0");
        wekaSegmentation.settings.classNames.add("label_im_class_1");

        int nClasses = wekaSegmentation.getNumClasses();
        int nf = featureProvider.getNumFeatures();

        int[] pixelsPerClass = new int[nClasses];

        double[][][] featureSlice = featureProvider.getReusableFeatureSlice();

        Instances instances = InstancesCreator.createInstancesHeader(
                instancesName,
                featureProvider.getFeatureNames(),
                wekaSegmentation.getClassNames());

        // Collect instancesMap per plane
        for ( int z = (int) interval.min( Z ); z <= interval.max( Z ); ++z)
        {

            logLabelImageTrainingProgress( logger, z, interval, "...");

            // Create lists of coordinates of pixels of each class
            //
            ArrayList<Point3D >[] classCoordinates = new ArrayList[numClasses];
            for (int i = 0; i < wekaSegmentation.getNumClasses(); i++)
            {
                classCoordinates[i] = new ArrayList<>();
            }

            ImageProcessor ip = labelImage.getStack().getProcessor(z + 1);

            for ( int y = (int) interval.min( Y ); y <= interval.max( Y ); ++y)
            {
                for ( int x = (int) interval.min( X ); x <= interval.max( X ); ++x)
                {
                    int classIndex = ip.get(x, y);
                    classCoordinates[classIndex].add(new Point3D(x, y, z));
                }
            }

            // Select random samples from each class
            Random rand = new Random();

            featureProvider.setFeatureSlicesValues( z, featureSlice, numThreads );

            for (int iClass = 0; iClass < nClasses; ++iClass)
            {
                if ( ! classCoordinates[iClass].isEmpty() )
                {
                    for (int i = 0; i < numInstancesPerClassAndPlane; ++i)
                    {
                        int randomSample = rand.nextInt(classCoordinates[iClass].size());

                        // We have to put the featureSlice for this z-plane into
                        // an ArrayList, because there could be multiple channels,
                        // and this is what 'setFeatureValuesAndClassIndex' expects as input
                        double[] featureValuesWithClassNum = new double[nf + 1];

                        featureProvider.setFeatureValuesAndClassIndex(
                                featureValuesWithClassNum,
                                (int) classCoordinates[iClass].get(randomSample).getX(),
                                (int) classCoordinates[iClass].get(randomSample).getY(),
                                featureSlice,
                                iClass);

                        DenseInstance denseInstance = new DenseInstance(
                                1.0,
                                featureValuesWithClassNum);

                        instances.add( denseInstance );

                        pixelsPerClass[iClass]++;

                    }
                }
            }

        }


        logger.info ( "...computed instance values in [min]: " +
                wekaSegmentation.getMinutes( System.currentTimeMillis(), startTime ) );

        //for( int j = 0; j < numOfClasses ; j ++ )
        //	IJ.log("Added " + numSamples + " instancesComboBox of '" + loadedClassNames.get( j ) +"'.");

        logger.info("Label image training data added " +
                "(" + instances.numInstances() +
                " instancesComboBox, " + instances.numAttributes() +
                " attributes, " + instances.numClasses() + " classes).");

        for ( int iClass = 0; iClass < nClasses; ++iClass )
        {
            logger.info( "Class " + iClass + " [pixels]: " + pixelsPerClass[ iClass ]);
            if( pixelsPerClass[iClass] == 0 )
            {
                logger.error("No labels of class found: " + iClass);
            }
        }

        return ( instances );

    }



    // TODO: improve separation from WekaSegmentation
    public static Instances getLocallyPairedInstancesFromLabelImage(
            WekaSegmentation wekaSegmentation,
            ImagePlus inputImage,
            ImagePlus labelImage,
            ResultImage result,
            String instancesName,
            FinalInterval interval,
            int numInstancesPerClassAndPlane,
            int numThreads,
            Logger logger )
    {

        final int numClasses = wekaSegmentation.getNumClasses();
        int radius = 5;
        Random rand = new Random();
        if ( logger == null ) logger = new IJLazySwingLogger();


        logger.info( "Computing features for label image region...");
        IntervalUtils.logInterval( interval );
        logger.info( "Instances per class and plane: " + numInstancesPerClassAndPlane);

        long startTime = System.currentTimeMillis();

        // Compute features
        FeatureProvider featureProvider = new FeatureProvider();
        featureProvider.setLogger( logger );
        featureProvider.isLogging( true );
        featureProvider.setInputImage( inputImage );
        featureProvider.setWekaSegmentation( wekaSegmentation );
        featureProvider.setInterval( interval );
        featureProvider.setActiveChannels( wekaSegmentation.settings.activeChannels );
        featureProvider.computeFeatures( numThreads,
                wekaSegmentation.maximumMultithreadedLevel,
                true );

        int nf = featureProvider.getNumFeatures();

        logger.info ( "...computed features  in [ms]: " +
                ( System.currentTimeMillis() - startTime ) );

        logger.info( "Computing instance values...");
        startTime = System.currentTimeMillis();

        // TODO: determine numClasses from labelImage!
        wekaSegmentation.settings.classNames = new ArrayList<>();
        wekaSegmentation.settings.classNames.add("label_im_class_0");
        wekaSegmentation.settings.classNames.add("label_im_class_1");

        int[] pixelsPerClass = new int[ numClasses ];

        double[][][] featureSlice = featureProvider.getReusableFeatureSlice();

        Instances instances = InstancesCreator.createInstancesHeader(
                instancesName,
                featureProvider.getFeatureNames(),
                wekaSegmentation.getClassNames());


        // Collect instancesMap per plane
        for ( int z = (int) interval.min( Z ); z <= interval.max( Z ); ++z)
        {

            logLabelImageTrainingProgress( logger, z, interval, "...");

            // Create lists of coordinates of pixels of each class
            //
            ArrayList< int[] >[] classCoordinates = new ArrayList[ numClasses ];

            for ( int i = 0; i < numClasses; i++)
            {
                classCoordinates[i] = new ArrayList<>();
            }

            ImageProcessor labelImageSlice = labelImage.getStack().getProcessor(z + 1);

            for ( int y = (int) interval.min( Y ); y <= interval.max( Y ); ++y)
            {
                for ( int x = (int) interval.min( X ); x <= interval.max( X ); ++x)
                {
                    int classIndex = labelImageSlice.get(x, y);
                    classCoordinates[classIndex].add( new int[]{ x, y });
                }
            }

            featureProvider.setFeatureSlicesValues( z, featureSlice, numThreads );

            for (int i = 0; i < numInstancesPerClassAndPlane; ++i)
            {
                for (int iClass = 0; iClass < numClasses; ++iClass)
                {
                    if ( ! classCoordinates[iClass].isEmpty() )
                    {
                        int[] center = getRandomCoordinate( iClass, classCoordinates, rand );

                        ArrayList< int[] >[] localClassCoordinates =
                                getLocalClassCoordinates( numClasses,
                                        labelImageSlice,
                                        interval,
                                        center,
                                        radius );

                        for (int localClass = 0; localClass < numClasses; ++localClass)
                        {

                            for (int k = 0; k < 3; k++ )
                            {

                                int[] xy = getRandomCoordinate( localClass, localClassCoordinates, rand );

                                if ( xy == null )
                                {
                                    // no local coordinate has been found
                                    // thus we take a global one
                                    xy = getRandomCoordinate( localClass, classCoordinates, rand );
                                }

                                addInstance( instances, featureProvider, xy, featureSlice, localClass );

                                pixelsPerClass[ localClass ]++;

                            }

                        }


                    }
                }
            }

        }


        logger.info ( "...computed instance values in [min]: " +
                wekaSegmentation.getMinutes( System.currentTimeMillis(), startTime ) );

        //for( int j = 0; j < numOfClasses ; j ++ )
        //	IJ.log("Added " + numSamples + " instancesComboBox of '" + loadedClassNames.get( j ) +"'.");

        logger.info("Label image training data added " +
                "(" + instances.numInstances() +
                " instancesComboBox, " + instances.numAttributes() +
                " attributes, " + instances.numClasses() + " classes).");

        for ( int iClass = 0; iClass < numClasses; ++iClass )
        {
            logger.info( "Class " + iClass + " [pixels]: " + pixelsPerClass[ iClass ]);
            if( pixelsPerClass[iClass] == 0 )
            {
                logger.error("No labels of class found: " + iClass);
            }
        }

        return ( instances );

    }


    // TODO: improve separation from WekaSegmentation
    public static Instances getUsefulInstancesFromLabelImage(
            WekaSegmentation wekaSegmentation,
            ImagePlus inputImage,
            ImagePlus labelImage,
            ResultImage resultImage,
            FeatureProvider featureProvider,
            String instancesName,
            FinalInterval interval,
            int numInstancesPerClassAndPlane,
            int numThreads,
            Logger logger )
    {

        final int numClasses = wekaSegmentation.getNumClasses();
        int radius = 5;
        int t = (int) interval.min( T );

        Random rand = new Random();

        if ( logger == null ) logger = new IJLazySwingLogger();
        logger.info( "Computing features for label image region...");
        IntervalUtils.logInterval( interval );
        logger.info( "Instances per class and plane: " + numInstancesPerClassAndPlane);


        // Compute features only if they do not exist yet
        if ( featureProvider == null)
        {
            featureProvider = new FeatureProvider();
            featureProvider.setLogger( logger );
            featureProvider.isLogging( true );
            featureProvider.setInputImage( inputImage );
            featureProvider.setWekaSegmentation( wekaSegmentation );
            featureProvider.setInterval( interval );
            featureProvider.setActiveChannels( wekaSegmentation.settings.activeChannels );
            featureProvider.computeFeatures( numThreads );
        }


        logger.info( "Computing instance values...");
        long startTime = System.currentTimeMillis();

        // TODO: determine numClasses from labelImage!
        wekaSegmentation.settings.classNames = new ArrayList<>();
        wekaSegmentation.settings.classNames.add("label_im_class_0");
        wekaSegmentation.settings.classNames.add("label_im_class_1");

        int[] pixelsPerClass = new int[ numClasses ];

        double[][][] featureSlice = featureProvider.getReusableFeatureSlice();

        Instances instances = InstancesCreator.createInstancesHeader(
                instancesName,
                featureProvider.getFeatureNames(),
                wekaSegmentation.getClassNames());


        int numAccuracies = resultImage.getProbabilityRange();

        // Collect instances per plane
        for ( int z = (int) interval.min( Z ); z <= interval.max( Z ); ++z)
        {

            logLabelImageTrainingProgress( logger, z, interval, "...");

            ArrayList < int[] > [][] classCoordinates = getEmptyClassCoordinates(
                    numClasses, numAccuracies );

            int[][] accuracies = new int[numClasses][4]; // TOTAL, CORRECT, FP, FN

            setClassCoordinatesAndAccuracies(
                    classCoordinates,
                    accuracies,
                    z, t,
                    labelImage,
                    resultImage,
                    interval );

            featureProvider.setFeatureSlicesValues( z, featureSlice, numThreads );

            for (int iClass = 0; iClass < numClasses; ++iClass)
            {
                for (int i = 0; i < numInstancesPerClassAndPlane; ++i)
                {
                    int[] xy = getUsefulRandomCoordinate( iClass, classCoordinates, rand );

                    addInstance( instances, featureProvider, xy, featureSlice, iClass );

                    pixelsPerClass[ iClass ]++;

                }
            }

        }


        logger.info ( "...computed instance values in [min]: " +
                wekaSegmentation.getMinutes( System.currentTimeMillis(), startTime ) );

        //for( int j = 0; j < numOfClasses ; j ++ )
        //	IJ.log("Added " + numSamples + " instancesComboBox of '" + loadedClassNames.get( j ) +"'.");

        logger.info("Label image training data added " +
                "(" + instances.numInstances() +
                " instancesComboBox, " + instances.numAttributes() +
                " attributes, " + instances.numClasses() + " classes).");

        for ( int iClass = 0; iClass < numClasses; ++iClass )
        {
            logger.info( "Class " + iClass + " [pixels]: " + pixelsPerClass[ iClass ]);
            if( pixelsPerClass[iClass] == 0 )
            {
                logger.error("No labels of class found: " + iClass);
            }
        }

        return ( instances );

    }


    private static ArrayList < int[] > [][] getEmptyClassCoordinates(
            int numClasses,
            int numAccuracies )

    {

        ArrayList < int[] >[][] classCoordinates;
        classCoordinates = new ArrayList[numClasses][numAccuracies];

        for ( int iClass = 0; iClass < numClasses; iClass++)
        {
            for ( int a = 0; a < numAccuracies; ++a )
            {
                classCoordinates[iClass][a] = new ArrayList<>();
            }
        }

        return classCoordinates;
    }

    final static int TOTAL = 0, CORRECT = 1, FP = 2, FN = 3;

    private static void setClassCoordinatesAndAccuracies(
                                    ArrayList < int[] >[][] classCoordinates,
                                    int[][] accuracies,
                                    int z, int t,
                                    ImagePlus labelImage,
                                    ResultImage resultImage,
                                    FinalInterval interval)
    {

        int maxProbability = resultImage.getProbabilityRange();

        ImageProcessor labelImageSlice = labelImage.getStack().getProcessor(z + 1);

        for ( int y = (int) interval.min( Y ); y <= interval.max( Y ); ++y)
        {
            for ( int x = ( int ) interval.min( X ); x <= interval.max( X ); ++x )
            {
                int[] classifiedClassAndProbability =
                        resultImage.getClassAndProbability( x, y, z, t );

                int realClass = labelImageSlice.get( x, y );
                int classifiedClass = classifiedClassAndProbability[ 0 ];
                int correctness = classifiedClassAndProbability[ 1 ];

                if ( realClass == classifiedClass )
                {
                    accuracies[ realClass ][ TOTAL ]++;
                    accuracies[ realClass ][ CORRECT ]++;

                }
                else
                {
                    correctness *= -1;
                    accuracies[ realClass ][ TOTAL ]++;
                    accuracies[ realClass ][ FN ]++;
                    accuracies[ classifiedClass ][ FP ]++;
                }

                // shift such that it becomes a positive index
                correctness += maxProbability;

                classCoordinates[realClass][correctness].add( new int[]{ x, y } );

            }
        }

    }


    private static ArrayList< int[] >[] getLocalClassCoordinates(
            int numClasses,
            ImageProcessor labelImageSlice,
            FinalInterval interval,
            int[] center,
            int radius )
    {
        int nd = 2;
        int[] dims = new int[]{0,1};
        int[] min = new int[nd];
        int[] max = new int[nd];

        for ( int d : dims )
        {
            min[d] = Math.max( center[d] - radius, (int) interval.min(d));
            max[d] = Math.min( center[d] + radius, (int) interval.max(d) );
        }

        ArrayList< int[] >[] localClassCoordinates =
                new ArrayList[ numClasses ];

        for ( int iClass = 0; iClass < numClasses; ++iClass )
        {
            localClassCoordinates[iClass] = new ArrayList<>();
        }

        for ( int x = min[0]; x <= max[0]; ++x )
        {
            for ( int y = min[1]; y <= max[1]; ++y )
            {
                int iClass = labelImageSlice.get(x , y);
                localClassCoordinates[iClass].add( new int[]{x, y} );
            }
        }

        return ( localClassCoordinates );

    }

    private static void addInstance( Instances instances,
                                     FeatureProvider featureProvider,
                                     int[] xy,
                                     double[][][] featureSlice,
                                     int iClass)
    {
        double[] featureValuesWithClassNum = new double[featureProvider.getNumFeatures() + 1];

        featureProvider.setFeatureValuesAndClassIndex(
                featureValuesWithClassNum,
                xy[0], xy[1],
                featureSlice,
                iClass);

        DenseInstance denseInstance = new DenseInstance(
                1.0,
                featureValuesWithClassNum);

        instances.add( denseInstance );
    }


    private static int[] getRandomCoordinate( int iClass,
                                              ArrayList< int[] >[] classCoordinates,
                                              Random random )
    {

        if ( classCoordinates[ iClass ].size() == 0 ) return null;

        int randomSample = random.nextInt( classCoordinates[ iClass ].size() );

        int[] xy = new int[2];
        for ( int d = 0; d < 2; ++d )
        {
            xy[ d ] = classCoordinates[ iClass ].get( randomSample )[ d ];
        }
        return ( xy );
    }


    private static int[] getUsefulRandomCoordinate( int iClass,
                                                    ArrayList< int[] >[][] classCoordinates,
                                                    Random random )
    {

        // - find one that is not too close to the old one? (bad if they all cluster....)

        int[] xy = null;

        for ( int accuracy = 0; accuracy < classCoordinates[iClass].length; ++accuracy )
        {
            int numInstances = classCoordinates[iClass][accuracy].size();

            if ( numInstances > 0 )
            {

                int randomSample = random.nextInt( numInstances );

                xy = classCoordinates[iClass][accuracy].get( randomSample );

                // remove it not to draw it again
                classCoordinates[iClass][accuracy].remove( randomSample );

                break;
            }

        }

        return xy;

    }


    private static final void logLabelImageTrainingProgress(
            Logger logger,
            int z,
            FinalInterval interval,
            String currentTask )
    {
        logger.progress("z (current, min, max): ",
                "" + z
                        + ", " + interval.min( Z )
                        + ", " + interval.max( Z )
                        + "; " + currentTask);

    }



    public static Instances createInstancesHeader(
            String instancesName,
            ArrayList< String > featureNames,
            ArrayList< String > classNames  )
    {

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for ( String feature : featureNames )
        {
            attributes.add( new Attribute(feature) );
        }
        attributes.add( new Attribute("class", classNames ) );

        // initialize set of instancesMap
        Instances instances = new Instances(instancesName, attributes, 1);
        // Set the index of the class attribute
        instances.setClassIndex( featureNames.size() );

        return ( instances );

    }



    public static Instances removeAttributes( Instances instances,
                                              ArrayList< Integer > goners )
    {


        Instances attributeSubset = new Instances( instances );

        for( int j = goners.size() - 1; j >= 0; j-- )
        {
            int id = goners.get( j );
            attributeSubset.deleteAttributeAt( id );
        }

        return ( attributeSubset );
    }



    /*
    private ImageProcessor createWrongnessImage()
    {
        ImageProcessor ip = null;

        return ( ip );
    }


    private ArrayList< int[] > getCoordinatesOfLocalMaxima(
            ImageProcessor ip,
            int tolerance )
    {
        MaximumFinder mf = new MaximumFinder();
        Polygon maxima = mf.getMaxima(ip, tolerance, false);
        //print("count="+maxima.npoints);
    }*/



}
