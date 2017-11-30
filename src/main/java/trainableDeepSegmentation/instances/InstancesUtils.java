package trainableDeepSegmentation.instances;

import bigDataTools.logging.Logger;
import ij.ImagePlus;
import ij.Prefs;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;
import trainableDeepSegmentation.*;
import trainableDeepSegmentation.examples.Example;
import trainableDeepSegmentation.results.ResultImage;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static trainableDeepSegmentation.IntervalUtils.*;
import static trainableDeepSegmentation.WekaSegmentation.*;
import static trainableDeepSegmentation.instances.InstancesMetadata.Metadata.*;

public class InstancesUtils {

    public InstancesUtils( )
    {
    }

    /**
     * Create instances instancesMap out of the user markings
     *
     * @return set of instancesMap (feature vectors in Weka format)
     */
    public static InstancesMetadata getInstancesAndMetadataFromLabels( ArrayList< Example > examples,
                                                                       String inputImageTitle,
                                                                       Settings settings,
                                                                       ArrayList< String > featureNames,
                                                                       ArrayList< String > classNames )
    {

        Instances instances = getInstancesHeader(
                inputImageTitle,
                featureNames,
                classNames  );

        InstancesMetadata instancesAndMetadata
                = new InstancesMetadata( instances );

        for ( int e = 0; e < examples.size(); ++e )
        {
            Example example = examples.get( e );
            for ( int p = 0; p < example.points.length; ++p )
            {
                Instance instance = new DenseInstance(1.0, example.instanceValuesArray.get( p ) );
                instancesAndMetadata.addInstance( instance );
                instancesAndMetadata.addMetadata( Metadata_Position_X , example.points[p].x );
                instancesAndMetadata.addMetadata( Metadata_Position_Y , example.points[p].y );
                instancesAndMetadata.addMetadata( Metadata_Position_Z , example.z );
                instancesAndMetadata.addMetadata( Metadata_Position_T , example.t );
                instancesAndMetadata.addMetadata( Metadata_Label_Id, e );
                // TODO: use the Metadata enum also for the settings to be able to loop
                instancesAndMetadata.addMetadata( Metadata_Settings_ImageBackground, settings.imageBackground );
                instancesAndMetadata.addMetadata( Metadata_Settings_Anisotropy, settings.anisotropy );
                instancesAndMetadata.addMetadata( Metadata_Settings_MaxBinLevel, settings.maxBinLevel );
                instancesAndMetadata.addMetadata( Metadata_Settings_MaxDeepConvLevel, settings.maxDeepConvLevel );
                instancesAndMetadata.addMetadata( Metadata_Settings_BinFactor, settings.binFactor );
            }

        }

        return ( instancesAndMetadata );

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

    public static Callable<InstancesMetadata > getUsefulInstancesFromLabelImage(
            WekaSegmentation wekaSegmentation,
            ImagePlus labelImage,
            ResultImage resultImage,
            ImageProcessor ipInstancesDistribution,
            FeatureProvider featureProvider,
            String instancesName,
            FinalInterval interval,
            int maxInstancesPerClassAndPlane,
            int numThreads,
            int localRadius,
            boolean isFirstTime)
    {
        return new Callable< InstancesMetadata >() {

            public InstancesMetadata call()
            {

                int maxAccuracy = resultImage.getProbabilityRange() - 2;

                final int numClasses = wekaSegmentation.getNumClasses(); // TODO: getInstancesAndMetadata from label image

                int t = ( int ) interval.min( T );

                Random rand = new Random();

                int[] pixelsPerClass = new int[ numClasses ];

                Instances instances = InstancesUtils.getInstancesHeader(
                        instancesName,
                        featureProvider.getAllFeatureNames(),
                        wekaSegmentation.getClassNames() );

                InstancesMetadata instancesAndMetadata = new InstancesMetadata( instances );

                //int[][] classificationAccuracies = new int[ numClasses ][ 4 ]; // TOTAL, CORRECT, FP, FN
                double[][][] featureSlice;

                // Collect instances per plane
                for ( int z = ( int ) interval.min( Z ); z <= interval.max( Z ); ++z )
                {
                    logLabelImageTrainingProgress( logger, z, interval, "getting class coordinates..." );

                    ArrayList< int[] >[][] classCoordinates = getEmptyClassCoordinates(
                            numClasses, resultImage.getProbabilityRange() );

                    setClassCoordinatesAndAccuracies(
                            classCoordinates,
                            z, t,
                            labelImage,
                            resultImage,
                            interval,
                            isFirstTime );

                    featureSlice = featureProvider.getCachedFeatureSlice( z );
                    if ( featureSlice == null )
                    {
                        logLabelImageTrainingProgress( logger, z, interval,
                                "getting feature slice... (using " +
                                numThreads + " threads)." );
                        featureSlice = featureProvider.getReusableFeatureSlice();
                        featureProvider.setFeatureSlicesValues( z, featureSlice, numThreads );
                    }

                    logLabelImageTrainingProgress( logger, z,
                            interval, "getting instances..." );

                    collectInstancesLoop:
                    for ( int i = 0; i < maxInstancesPerClassAndPlane; ++i )
                    {

                        if ( wekaSegmentation.stopCurrentTasks )
                        {
                            return null;
                        }

                        for ( int iClass = 0; iClass < numClasses; ++iClass )
                        {

                            int[] xy = getUsefulRandomCoordinate(
                                    iClass,
                                    classCoordinates,
                                    rand,
                                    maxAccuracy);

                            if ( xy == null )
                            {
                                // stop, because no more example of this class are available
                                // and we do not want to add (unbalanced)
                                // examples of the other classes
                                break collectInstancesLoop;
                            }

                            ArrayList< int[] >[] localClassCoordinates =
                                    getLocalClassCoordinates( numClasses,
                                            labelImage, z, t,
                                            interval,
                                            xy,
                                            localRadius );

                            for ( int localClass = 0; localClass < numClasses; ++localClass )
                            {

                                int[] xyLocal;

                                if ( localClass == iClass )
                                {
                                    xyLocal = xy;
                                }
                                else
                                {
                                    xyLocal = getRandomCoordinate( localClass, localClassCoordinates, rand );

                                    if ( xyLocal == null )
                                    {
                                        // no local coordinate found
                                        // thus we take a useful global one
                                        xyLocal = getUsefulRandomCoordinate(
                                                localClass,
                                                classCoordinates,
                                                rand,
                                                maxAccuracy );
                                    }

                                }

                                if ( xyLocal == null )
                                {
                                    // this class really has no more labels => stop
                                    break collectInstancesLoop;
                                }


                                addToInstancesDistribution( ipInstancesDistribution, xyLocal, interval );

                                removeNeighbors( localClass, classCoordinates, xyLocal, localRadius );

                                Instance instance = getInstance( featureProvider, xyLocal, featureSlice, localClass );

                                instancesAndMetadata.addInstance( instance );
                                instancesAndMetadata.addMetadata( Metadata_Position_X, xyLocal[ 0 ] );
                                instancesAndMetadata.addMetadata( Metadata_Position_Y, xyLocal[ 1 ] );
                                instancesAndMetadata.addMetadata( Metadata_Position_Z, z );
                                instancesAndMetadata.addMetadata( Metadata_Position_T, t );
                                instancesAndMetadata.addMetadata( Metadata_Label_Id, -1 );

                                // TODO: make this a loop!!!
                                instancesAndMetadata.addMetadata( Metadata_Settings_ImageBackground,
                                        wekaSegmentation.settings.imageBackground );
                                instancesAndMetadata.addMetadata( Metadata_Settings_Anisotropy,
                                        wekaSegmentation.settings.anisotropy );
                                instancesAndMetadata.addMetadata( Metadata_Settings_MaxBinLevel,
                                        wekaSegmentation.settings.maxBinLevel );
                                instancesAndMetadata.addMetadata( Metadata_Settings_BinFactor,
                                        wekaSegmentation.settings.binFactor );
                                instancesAndMetadata.addMetadata( Metadata_Settings_MaxDeepConvLevel,
                                        wekaSegmentation.settings.maxDeepConvLevel );


                                pixelsPerClass[ localClass ]++;

                            }
                        }
                    }
                }

                return ( instancesAndMetadata );
            }
        };
    }


    public ArrayList< Integer >[] getLabelIdsPerClass( InstancesMetadata iam )
    {
        int numClasses = iam.instances.numClasses();
        ArrayList< Integer >[]  labelIdsPerClass = new ArrayList[numClasses];

        for ( int iClass = 0; iClass < numClasses; ++iClass )
        {
            labelIdsPerClass[iClass] = new ArrayList<>();
        }

        for ( int i = 0; i < iam.instances.size(); ++i )
        {
            labelIdsPerClass [ (int) iam.getInstance( i ).classValue() ]
                    .add( (int) iam.getMetadata( Metadata_Label_Id, i ) );
        }

        return labelIdsPerClass;
    }

    private static void addToInstancesDistribution( ImageProcessor instancesDistribution,
                                             int[] xy,
                                             FinalInterval interval )
    {
        if ( instancesDistribution != null )
        {
            int xx = xy[ 0 ] - ( int ) interval.min( X );
            int yy = xy[ 1 ] - ( int ) interval.min( Y );
            int v = instancesDistribution.get( xx, yy ) + 1;
            instancesDistribution.set( xx, yy, v );
        }
    }

    public static void reportClassificationAccuracies( long[][] classificationAccuracies,
                                                        Logger logger)
    {
        logger.info( "\n# Classification accuracies");

        for ( int iClass = 0; iClass < classificationAccuracies.length; ++iClass )
        {
            long total = classificationAccuracies[iClass][TOTAL];
            if ( total == 0 ) total = -1; // to avoid division by zero

            logger.info("Class " + iClass
                    + "; " + "Total ground truth: " + total
                    + "; " + "Correct: " + classificationAccuracies[iClass][CORRECT]
                    + "; " + "Percent correct: " + ( 100 * classificationAccuracies[iClass][CORRECT] ) / total
                    + "; " + "False positive: " + classificationAccuracies[iClass][FP]
                    + "; " + "False negative: " + classificationAccuracies[iClass][FN]
            );
        }
    }

    private static ArrayList < int[] > [][] getEmptyClassCoordinates(
            int numClasses,
            int numProbabilities )
    {

        int numAccuracies = 2 * numProbabilities + 1;
        ArrayList < int[] >[][] classCoordinates;
        classCoordinates = new ArrayList[numClasses][ numAccuracies ];

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
                                    int z, int t,
                                    ImagePlus labelImage,
                                    ResultImage resultImage,
                                    FinalInterval interval,
                                    boolean onlySetCoordinates)
    {

        int maxProbability = resultImage.getProbabilityRange();

        ImageProcessor labelImageSlice = labelImage.getStack().getProcessor(z + 1);

        ImageProcessor ipResultImage = resultImage.getSlice( z + 1, t + 1  );

        for ( int y = (int) interval.min( Y ); y <= interval.max( Y ); ++y)
        {
            for ( int x = ( int ) interval.min( X ); x <= interval.max( X ); ++x )
            {

                int realClass = labelImageSlice.get( x, y );

                if ( onlySetCoordinates )
                {
                    // put all metadata into the zero
                    // accuracy slot, this makes sense for the
                    // first iteration where we do not
                    // want to use the result image yet
                    classCoordinates[realClass][0].add( new int[]{ x, y } );
                }
                else
                {

                    byte result = (byte) ipResultImage.get( x, y );

                    int classifiedClass = ( result - 1 ) / maxProbability;
                    int correctness = result - classifiedClass * maxProbability;

                    if ( realClass != classifiedClass )
                    {
                        correctness *= -1;
                    }

                    // shift such that it becomes a positive index
                    correctness += maxProbability;

                    classCoordinates[realClass][correctness].add( new int[]{ x, y } );
                }

            }
        }

    }

    public static long[][] setAccuracies(
            ImagePlus labelImage,
            ResultImage resultImage,
            ImagePlus accuraciesImage,
            FinalInterval interval,
            WekaSegmentation ws)
    {

        logger.info( "\n# Label image training accuracies");

        int numClasses = 2; // TODO

        long[][] accuracies = new long[numClasses][5];

        ExecutorService exe = Executors.newFixedThreadPool( Prefs.getThreads() );
        ArrayList< Future > futures = new ArrayList< >(  );

        for ( int z = (int) interval.min( Z ); z <= interval.max( Z ); ++z )
        {
            futures.add(
                    exe.submit(
                        setAccuracies(
                            accuracies,
                            labelImage,
                            resultImage,
                            accuraciesImage,
                            z,
                            interval,
                            ws
                        )
                    )
            );
        }

        ThreadUtils.joinThreads( futures, ws.getLogger() );

        return ( accuracies );

    }


    public static Runnable setAccuracies(
            long[][] accuracies,
            ImagePlus labelImage,
            ResultImage resultImage,
            ImagePlus accuraciesImage,
            int z,
            FinalInterval interval,
            WekaSegmentation ws )
    {

        final Object lock = new Object();

        return new Runnable() {
            @Override
            public void run()
            {

                int maxProbability = resultImage.getProbabilityRange();
                int t = (int) interval.min( T );

                if ( ws.stopCurrentTasks ) return;

                logger.progress( "Measuring accuracies",
                        "in z-slice " + ( z + 1 ) + "/" + ( interval.max( Z ) + 1 ) );

                ImageProcessor labelImageSlice = labelImage.getStack().getProcessor( z + 1 );
                ImageProcessor ipAccuracies = null;
                ImageProcessor ipResult = resultImage.getSlice( z + 1, t + 1  );
                int probabilityRange = resultImage.getProbabilityRange();

                if ( accuraciesImage != null )
                {
                    ipAccuracies = accuraciesImage.getImageStack().getProcessor( z + 1 - ( int ) interval.min( Z ) );
                }

                for ( int y = ( int ) interval.min( Y ); y <= interval.max( Y ); ++y )
                {
                    for ( int x = ( int ) interval.min( X ); x <= interval.max( X ); ++x )
                    {
                        int realClass = labelImageSlice.get( x, y );

                        byte result = (byte) ipResult.get( x, y );

                        int[] classAndProbability = new int[2];

                        classAndProbability[0] = ( result - 1 ) / probabilityRange;
                        classAndProbability[1] = result - classAndProbability[0] * probabilityRange;

                        int classifiedClass = classAndProbability[ 0 ];
                        int correctness = classAndProbability[ 1 ];

                        synchronized ( lock )
                        {
                            accuracies[ realClass ][ TOTAL ]++;
                            if ( realClass == classifiedClass )
                            {
                                accuracies[ realClass ][ CORRECT ]++;
                            }
                            else
                            {
                                correctness *= -1;
                                accuracies[ realClass ][ FN ]++;
                                accuracies[ classifiedClass ][ FP ]++;
                            }
                        }

                        correctness += maxProbability;

                        if ( ipAccuracies != null )
                        {
                            ipAccuracies.set(
                                    x - ( int ) interval.min( X ),
                                    y - ( int ) interval.min( Y ),
                                    correctness );
                        }

                    }

                }

                accuraciesImage.updateImage();
            }
        };
    }

    private static ArrayList< int[] >[] getLocalClassCoordinates(
            int numClasses,
            ImagePlus labelImage,
            int z, int t,
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

        ImageProcessor labelImageSlice = labelImage.getStack().getProcessor(z + 1);

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
        double[] featureValuesWithClassNum = new double[featureProvider.getNumAllFeatures() + 1];

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

    public static Instance getInstance(
            FeatureProvider featureProvider,
            int[] xy,
            double[][][] featureSlice,
            int iClass )
    {
        double[] featureValuesWithClassNum
                = new double[ featureProvider.getNumAllFeatures() + 1 ];

        featureProvider.setFeatureValuesAndClassIndex(
                featureValuesWithClassNum,
                xy[0], xy[1],
                featureSlice,
                iClass);

        DenseInstance denseInstance = new DenseInstance(
                1.0,
                featureValuesWithClassNum);

        return denseInstance;
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

    private static void removeNeighbors( int iClass,
                                         ArrayList< int[] >[][] classCoordinates,
                                         int[] xyRef,
                                         int radius)
    {

        for ( int accuracy = 0; accuracy < classCoordinates[iClass].length; ++accuracy )
        {
            int numInstances = classCoordinates[ iClass ][ accuracy ].size();

            for ( int i = numInstances - 1; i >= 0; --i )
            {
                int[] xy = classCoordinates[ iClass ][ accuracy ].get( i );

                if ( ( Math.abs( xy[ 0 ] - xyRef[ 0 ] ) <= radius ) ||
                        ( Math.abs( xy[ 1 ] - xyRef[ 1 ] ) <= radius ) )
                {
                    classCoordinates[ iClass ][ accuracy ].remove( i );
                }
            }
        }
    }

    private static int[] getUsefulRandomCoordinate( int iClass,
                                                    ArrayList< int[] >[][] classCoordinates,
                                                    Random random,
                                                    int maxAccuracy )
    {

        int[] xy = null;

        for ( int accuracy = 0; accuracy < maxAccuracy; ++accuracy )
        {
            int numInstances = classCoordinates[iClass][accuracy].size();

            if ( numInstances > 0 )
            {
                int randomSample = random.nextInt( numInstances );

                xy = classCoordinates[iClass][accuracy].get( randomSample );

                return xy;
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
        logger.progress("Getting instances: z (current, min, max): ",
                "" + z
                        + ", " + interval.min( Z )
                        + ", " + interval.max( Z )
                        + "; " + currentTask);

    }

    public static Instances getInstancesHeader(
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
        Instances instances = new Instances( instancesName, attributes, 1);
        // Set the index of the class attribute
        instances.setClassIndex( instances.numAttributes() - 1 );

        return ( instances );

    }

    public static InstancesMetadata removeAttributes( InstancesMetadata instancesAndMetadata,
                                                      ArrayList< Integer > goners )
    {

        Instances attributeSubset = new Instances( instancesAndMetadata.instances );

        for( int j = goners.size() - 1; j >= 0; j-- )
        {
            int id = goners.get( j );
            attributeSubset.deleteAttributeAt( id );
        }

        return ( new InstancesMetadata( attributeSubset, instancesAndMetadata.metadata ) );
    }

    public static void logInstancesInformation( Instances instances )
    {
        logger.info( "\n# Instances information" );
        logger.info( "Number of instances: " + instances.size() );
        logger.info( "Number of attributes: " + instances.numAttributes() );

        long[] instancesPerClass = new long[ instances.numClasses() ];

        for ( Instance instance : instances )
        {
            instancesPerClass[ (int) instance.classValue() ] ++;
        }

        for ( int i = 0; i < instancesPerClass.length; ++i )
        {
            logger.info( "Instances in class " + i + ": " + instancesPerClass[ i ]);
        }

    }

    public static long[] getClassDistribution( Instances instances )
    {

        long[] instancesPerClass = new long[ instances.numClasses() ];

        for ( Instance instance : instances )
        {
            instancesPerClass[ (int) instance.classValue() ] ++;
        }

        return instancesPerClass;

    }

    /**
     * Read ARFF file
     * @param filename ARFF file name
     * @return set of instancesMap read from the file
     */
    static Instances loadInstancesFromARFF( String directory, String filename )
    {
        String pathname = directory + File.separator + filename;

        logger.info("Loading instances from " + pathname + " ...");

        try{
            BufferedReader reader = new BufferedReader( new FileReader( pathname ) );

            try
            {
                Instances instances = new Instances( reader );
                reader.close();
                return ( instances );
            }
            catch(IOException e)
            {
                logger.error("IOException");
            }
        }
        catch(FileNotFoundException e)
        {
            logger.error("File not found!");
        }

        return null;
    }

    public static InstancesMetadata loadInstancesAndMetadataFromARFF(
            String directory, String filename )
    {
        Instances instances = loadInstancesFromARFF( directory, filename );

        if ( instances == null ) return null;

        InstancesMetadata instancesAndMetadata
                = new InstancesMetadata( instances );

        instancesAndMetadata.moveMetadataFromInstancesToMetadata();

        return instancesAndMetadata;
    }


    public static boolean saveInstancesAsARFF( Instances instances,
                                        String directory,
                                        String filename )
    {

        BufferedWriter out = null;
        try{
            out = new BufferedWriter(
                    new OutputStreamWriter(
                            new FileOutputStream( directory
                                    + File.separator + filename ) ) );

            final Instances header = new Instances(instances, 0);
            out.write( header.toString() );

            for(int i = 0; i < instances.numInstances(); i++)
            {
                out.write(instances.get(i).toString()+"\n");
            }
        }
        catch(Exception e)
        {
            logger.error("Error: couldn't write instancesMap into .ARFF file.");
            e.printStackTrace();
            return false;
        }
        finally{
            try {
                out.close();
            } catch (IOException e) {
                e.printStackTrace();
                return false;
            }
        }

        return true;
    }

    public static boolean saveInstancesAndMetadataAsARFF( InstancesMetadata instancesAndMetadata,
                                                          String directory,
                                                          String filename)
    {

        instancesAndMetadata.putMetadataIntoInstances();

        boolean success = saveInstancesAsARFF( instancesAndMetadata.instances, directory, filename );

        instancesAndMetadata.removeMetadataFromInstances();

        return success;

    }

    public static int getNumLabelIds( InstancesMetadata instancesAndMetadata )
    {
        Set<Double> uniqueLabelIds = new HashSet<>(instancesAndMetadata.getMetadata( Metadata_Label_Id ));
        return uniqueLabelIds.size();
    }

    public static void setSettingsFromInstancesAndMetadata(
            Settings settings,
            InstancesMetadata instancesAndMetadata)
    {
        settings.imageBackground = (int) instancesAndMetadata.getMetadata( Metadata_Settings_ImageBackground, 0 );
        settings.maxDeepConvLevel = (int) instancesAndMetadata.getMetadata( Metadata_Settings_MaxDeepConvLevel, 0 );
        settings.binFactor = (int) instancesAndMetadata.getMetadata( Metadata_Settings_BinFactor, 0 );
        settings.maxBinLevel = (int) instancesAndMetadata.getMetadata( Metadata_Settings_MaxBinLevel, 0 );
        settings.anisotropy = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Anisotropy, 0 );
        settings.classNames = instancesAndMetadata.getClassNames();
    }

}
