package de.embl.cba.trainableDeepSegmentation.instances;

import de.embl.cba.trainableDeepSegmentation.labelimagetraining.AccuracyEvaluation;
import de.embl.cba.trainableDeepSegmentation.settings.FeatureSettings;
import de.embl.cba.utils.logging.Logger;
import ij.ImagePlus;
import ij.Prefs;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;
import de.embl.cba.trainableDeepSegmentation.*;
import de.embl.cba.trainableDeepSegmentation.labels.examples.Example;
import de.embl.cba.trainableDeepSegmentation.features.FeatureProvider;
import de.embl.cba.trainableDeepSegmentation.results.ResultImage;
import de.embl.cba.trainableDeepSegmentation.settings.SettingsUtils;
import de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils;
import de.embl.cba.trainableDeepSegmentation.utils.ThreadUtils;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.*;
import static de.embl.cba.trainableDeepSegmentation.DeepSegmentation.*;
import static de.embl.cba.trainableDeepSegmentation.instances.InstancesAndMetadata.Metadata.*;

public class InstancesUtils {

    public InstancesUtils( )
    {
    }

    /**
     * Create instances instancesMap out of the user markings
     *
     * @return set of instancesMap (feature vectors in Weka format)
     */
    public static InstancesAndMetadata createInstancesAndMetadataFromExamples(
            ArrayList< Example > examples,
            String instancesName,
            FeatureSettings featureSettings,
            ArrayList< String > featureNames,
            ArrayList< String > classNames )
    {

        Instances instances = getInstancesHeader( instancesName, featureNames, classNames  );

        InstancesAndMetadata instancesAndMetadata = new InstancesAndMetadata( instances );

        for ( int e = 0; e < examples.size(); ++e )
        {
            Example example = examples.get( e );

            if ( example.instanceValuesArrays !=  null && ! example.instanceValuesAreCurrentlyBeingComputed )
            {
                for ( ArrayList< double[] > instanceValuesArray : example.instanceValuesArrays )
                {
                    if ( instanceValuesArray.size() != example.points.length )
                    {
                        int a = 1;
                        //logger.warning( "Example point list size does not equal instances array size" );
                        //continue;
                    }

                    Set< Point > addedPoints = new LinkedHashSet<>();

                    for ( int p = 0; p < instanceValuesArray.size(); ++p )
                    {
                        if ( ! addedPoints.contains( example.points[ p ]  ) )
                        {
                            Instance instance = new DenseInstance( 1.0, instanceValuesArray.get( p ) );
                            instancesAndMetadata.addInstance( instance );
                            instancesAndMetadata.addMetadata( Metadata_Position_X, example.points[ p ].x );
                            instancesAndMetadata.addMetadata( Metadata_Position_Y, example.points[ p ].y );
                            instancesAndMetadata.addMetadata( Metadata_Position_Z, example.z );
                            instancesAndMetadata.addMetadata( Metadata_Position_T, example.t );
                            instancesAndMetadata.addMetadata( Metadata_Label_Id, e );
                            SettingsUtils.addSettingsToMetadata( featureSettings, instancesAndMetadata );
                            addedPoints.add( example.points[ p ] );
                        }
                        else
                        {
                            int a = 1; // point was added already (this can happen as the IJ Rois can contain points multiple times.)
                        }
                    }

                }
            }
        }

        return ( instancesAndMetadata );

    }


    public static Callable<InstancesAndMetadata> getUsefulInstancesFromLabelImage(
            DeepSegmentation deepSegmentation,
            ImagePlus inputImageWithLabels,
            int labelChannel,
            ResultImage resultImage,
            ImageProcessor ipInstancesDistribution,
            FeatureProvider featureProvider,
            String instancesName,
            FinalInterval interval,
            int maxInstancesPerClassAndPlane,
            int numThreads,
            int localRadius,
            ArrayList< Double > classWeights )
    {
        return new Callable<InstancesAndMetadata>() {

            public InstancesAndMetadata call()
            {
                double accuracySmoothingSigma = 2.0;
                int probabilityRange = resultImage.getProbabilityRange();

                int maxAccuracy = probabilityRange;

                final int numClasses = deepSegmentation.getNumClasses(); // TODO: getInstancesAndMetadata from label image

                int t = ( int ) interval.min( T );

                Random rand = new Random();

                int[] pixelsPerClass = new int[ numClasses ];
                ArrayList< int[] >[][] globalClassCoordinates;
                ArrayList< int[] >[][] localClassCoordinates;

                FinalInterval localInterval;

                Instances instances = InstancesUtils.getInstancesHeader(
                        instancesName,
                        featureProvider.getAllFeatureNames(),
                        deepSegmentation.getClassNames() );

                InstancesAndMetadata instancesAndMetadata = new InstancesAndMetadata( instances );

                double[][][] featureSlice;

                // Collect instances per plane
                for ( int z = ( int ) interval.min( Z ); z <= interval.max( Z ); ++z )
                {
                    logLabelImageTrainingProgress( logger, z, interval, "getting class coordinates..." );

                    int stackIndex = inputImageWithLabels.getStackIndex( labelChannel + 1, z + 1, t + 1  );
                    ImageProcessor labelSlice = inputImageWithLabels.getStack().getProcessor(stackIndex );
                    ImageProcessor resultSlice = resultImage.getSlice( z + 1, t + 1  );

                    ImageProcessor accuracySliceRegion = getAccuracySlice(
                            probabilityRange,
                            labelSlice, resultSlice, interval
                    );

                    accuracySliceRegion.blurGaussian( accuracySmoothingSigma );

                    // DEBUG
                    // ImagePlus imp2 = new  ImagePlus( "" + z, correctnessSliceRegion); imp2.show();

                    globalClassCoordinates = getClassCoordinatesAndAccuracies(
                            probabilityRange,
                            numClasses,
                            labelSlice,
                            accuracySliceRegion,
                            interval );

                    featureSlice = featureProvider.getCachedFeatureSlice( z );

                    if ( featureSlice == null )
                    {
                        logLabelImageTrainingProgress( logger, z, interval,
                                "getting feature slice... (using " +
                                numThreads + " numWorkers)." );
                        featureSlice = featureProvider.getReusableFeatureSlice();
                        featureProvider.setFeatureSlicesValues( z, featureSlice, numThreads );
                    }

                    logLabelImageTrainingProgress( logger, z,
                            interval, "getting instances..." );

                    collectInstancesLoop:
                    for ( int i = 0; i < maxInstancesPerClassAndPlane; ++i )
                    {
                        if ( deepSegmentation.stopCurrentTasks )
                        {
                            return null;
                        }

                        for ( int iClass = 0; iClass < numClasses; ++iClass )
                        {

                            int[] xy = getUsefulRandomCoordinate(
                                    iClass,
                                    globalClassCoordinates,
                                    rand,
                                    maxAccuracy );

                            if ( xy == null )
                            {
                                // stop, because no more example of this class are available
                                // and we do not want to add (unbalanced)
                                // examples of the other classes
                                break collectInstancesLoop;
                            }

                            // Compute local interval within bounds of global interval.
                            //
                            localInterval = IntervalUtils.getIntervalByReplacingValues( interval, X,
                                    Math.max( xy[0] - localRadius, interval.min( X ) ),
                                    Math.min( xy[0] + localRadius, interval.max( X ) )
                            );

                            localInterval = IntervalUtils.getIntervalByReplacingValues( localInterval, Y,
                                    Math.max( xy[1] - localRadius, interval.min( Y ) ),
                                    Math.min( xy[1] + localRadius, interval.max( Y ) )
                            );

                            localClassCoordinates = getClassCoordinatesAndAccuracies(
                                    probabilityRange,
                                    numClasses,
                                    labelSlice,
                                    accuracySliceRegion,
                                    localInterval );

                            for ( int localClass = 0; localClass < numClasses; ++localClass )
                            {

                                int[] xyLocal;

                                if ( localClass == iClass )
                                {
                                    xyLocal = xy;
                                }
                                else
                                {
                                    xyLocal = getUsefulRandomCoordinate(
                                            localClass,
                                            localClassCoordinates,
                                            rand,
                                            maxAccuracy );

                                    if ( xyLocal == null )
                                    {
                                        // No useful local coordinate was found.
                                        // Thus we take a useful global one.
                                        xyLocal = getUsefulRandomCoordinate(
                                                localClass,
                                                globalClassCoordinates,
                                                rand,
                                                probabilityRange );
                                    }

                                }

                                if ( xyLocal == null )
                                {
                                    // this class really has no more labels => stop
                                    break collectInstancesLoop;
                                }


                                addToInstancesDistribution( ipInstancesDistribution, xyLocal, interval );

                                removeNeighbors( localClass, globalClassCoordinates, xyLocal, localRadius );

                                Instance instance = getInstance( featureProvider, xyLocal, featureSlice, localClass );

                                instance.setWeight( classWeights.get( localClass ) );

                                instancesAndMetadata.addInstance( instance );
                                instancesAndMetadata.addMetadata( Metadata_Position_X, xyLocal[ 0 ] );
                                instancesAndMetadata.addMetadata( Metadata_Position_Y, xyLocal[ 1 ] );
                                instancesAndMetadata.addMetadata( Metadata_Position_Z, z );
                                instancesAndMetadata.addMetadata( Metadata_Position_T, t );
                                instancesAndMetadata.addMetadata( Metadata_Label_Id, -1 );

                                SettingsUtils.addSettingsToMetadata( deepSegmentation.featureSettings, instancesAndMetadata );

                                pixelsPerClass[ localClass ]++;

                            }
                        }
                    }
                }



                return ( instancesAndMetadata );
            }
        };
    }


    public ArrayList< Integer >[] getLabelIdsPerClass( InstancesAndMetadata iam )
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

    private static void addToInstancesDistribution( ImageProcessor instancesDistribution, int[] xy, FinalInterval interval )
    {
        if ( instancesDistribution != null )
        {
            int xx = xy[ 0 ] - ( int ) interval.min( X );
            int yy = xy[ 1 ] - ( int ) interval.min( Y );
            int v = instancesDistribution.get( xx, yy ) + 1;
            instancesDistribution.set( xx, yy, v );
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

    private static ArrayList< int[] >[][] getClassCoordinatesAndAccuracies(
                                    int maxProbability,
                                    int numClasses,
                                    ImageProcessor labelImageSlice,
                                    ImageProcessor correctnessSliceRegion,
                                    FinalInterval interval )
    {


        ArrayList< int[] >[][] classCoordinates = getEmptyClassCoordinates(
                numClasses, maxProbability );

        for ( int y = (int) interval.min( Y ); y <= interval.max( Y ); ++y)
        {
            for ( int x = ( int ) interval.min( X ); x <= interval.max( X ); ++x )
            {
                int realClass = labelImageSlice.get( x, y );
                int correctness = correctnessSliceRegion
                        .get( x -(int) interval.min( X ),
                                y -(int) interval.min( Y ) );

                classCoordinates[realClass][correctness].add( new int[]{ x, y } );
            }
        }

        return classCoordinates;
    }


    private static ImageProcessor getAccuracySlice(
            int maxProbability,
            ImageProcessor labelSlice,
            ImageProcessor resultSlice,
            FinalInterval interval )
    {


        labelSlice.setRoi( new Rectangle(
                (int) interval.min( X ),
                (int) interval.min( Y ),
                (int) interval.dimension( X ),
                (int) interval.dimension( Y ) ) );

        ImageProcessor accuracySlice = labelSlice.crop();

        int realClass;
        byte result;
        int classifiedClass;
        int accuracy;

        for ( int y = (int) interval.min( Y ); y <= interval.max( Y ); ++y)
        {
            for ( int x = ( int ) interval.min( X ); x <= interval.max( X ); ++x )
            {
                realClass = labelSlice.get( x, y );
                result = (byte) resultSlice.get( x, y );

                if ( result == 0 )
                {
                    accuracy = 0; // no classification available yet
                }
                else
                {
                    classifiedClass = ( result - 1 ) / maxProbability;

                    accuracy = result - classifiedClass * maxProbability;
                    if ( realClass != classifiedClass ) accuracy *= -1;
                    accuracy += maxProbability;
                }

                accuracySlice.set(
                        x - (int) interval.min( X ),
                        y - (int) interval.min( Y ),
                        accuracy);

            }
        }

        return accuracySlice;

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

    private static int[] getMostWrongCoordinate( int iClass,
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

    public static Instances getInstancesHeader( String instancesName, ArrayList< String > featureNames, ArrayList< String > classNames  )
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

    public static InstancesAndMetadata removeAttributes(InstancesAndMetadata instancesAndMetadata, ArrayList< Integer > goners )
    {

        logger.info( "Copying instances..." );

        Instances instancesWithAttributeSubset = new Instances( instancesAndMetadata.instances );

        logger.info( "Removing non useful attributes..." );

        int n = goners.size();

        for( int j = n - 1, i = 0; j >= 0; j--, i++ )
        {
            // logger.progress( "Removed attributes:", "" + i + "/" +  n  );
            int id = goners.get( j );
            instancesWithAttributeSubset.deleteAttributeAt( id );
        }

        instancesWithAttributeSubset.setClassIndex( instancesWithAttributeSubset.numAttributes() - 1 );

        return ( new InstancesAndMetadata( instancesWithAttributeSubset, instancesAndMetadata.metadata ) );
    }

    public static int[] getAttIndicesWindowByRemovingAttributes( InstancesAndMetadata instancesAndMetadata, ArrayList< Integer > goners )
    {

        int nGoners = goners.size();
        int nTotal = instancesAndMetadata.instances.numAttributes() - 1;
        int nTotalAfterRemoval = nTotal - nGoners;

        int[] attIndicesWindow = new int[nTotalAfterRemoval];

        for( int i = 0, j = 0; i < nTotal; ++i )
        {
            if ( ! goners.contains( i ) )
            {
                attIndicesWindow[ j++ ] = i;
            }
        }

        return ( attIndicesWindow );
    }

    public static InstancesAndMetadata onlyKeepAttributes(InstancesAndMetadata instancesAndMetadata,
                                                          ArrayList< Integer > keepers)
    {
        return onlyKeepAttributes( instancesAndMetadata, keepers, null );
    }

    public static InstancesAndMetadata onlyKeepAttributes(InstancesAndMetadata instancesAndMetadata,
                                                          ArrayList< Integer > keepers,
                                                          Integer numInstances )
    {
        //logger.info( "Removing non useful attributes from instances..." );

        Instances instancesWithAttributeSubset;

        if ( numInstances != null )
        {
            instancesWithAttributeSubset = new Instances( instancesAndMetadata.instances, 0, numInstances );
        }
        else
        {
            instancesWithAttributeSubset = new Instances( instancesAndMetadata.instances );
        }

        int numAttributes = instancesAndMetadata.instances.numAttributes();

        for( int i = numAttributes - 1; i >= 0; --i )
        {
            if ( ! keepers.contains( i )  && i != instancesWithAttributeSubset.classIndex() )
            {
                instancesWithAttributeSubset.deleteAttributeAt( i );
            }
        }

        instancesWithAttributeSubset.setClassIndex( instancesWithAttributeSubset.numAttributes() - 1 );

        //logger.info( "...done." );

        return ( new InstancesAndMetadata( instancesWithAttributeSubset, instancesAndMetadata.metadata ) );
    }

    public static int[] getAttIndicesWindowByKeepingAttributes(ArrayList< Integer > keepers )
    {
        int[] attIndicesWindow = new int[ keepers.size() ];

        for( int i = 0; i < keepers.size(); ++i )
        {
            attIndicesWindow[ i ] = keepers.get( i );
        }

        return attIndicesWindow;
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

    public static InstancesAndMetadata loadInstancesAndMetadataFromARFF( String directory, String filename )
    {
        Instances instances = loadInstancesFromARFF( directory, filename );

        if ( instances == null ) return null;

        InstancesAndMetadata instancesAndMetadata = new InstancesAndMetadata( instances );

        instancesAndMetadata.moveMetadataFromInstancesToMetadata();

        return instancesAndMetadata;
    }


    public static boolean saveInstancesAsARFF( Instances instances, String directory, String filename )
    {

        BufferedWriter out = null;
        try{
            out = new BufferedWriter(
                    new OutputStreamWriter(
                            new FileOutputStream( directory + File.separator + filename ) ) );

            final Instances header = new Instances( instances, 0 );
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

    public static boolean saveInstancesAndMetadataAsARFF( InstancesAndMetadata instancesAndMetadata, String directory, String filename)
    {

        instancesAndMetadata.putMetadataIntoInstances();

        boolean success = saveInstancesAsARFF( instancesAndMetadata.instances, directory, filename );

        instancesAndMetadata.removeMetadataFromInstances();

        return success;

    }

    public static int getNumLabelIds( InstancesAndMetadata instancesAndMetadata )
    {
        Set<Double> uniqueLabelIds = new HashSet<>(instancesAndMetadata.getMetadata( Metadata_Label_Id ));
        return uniqueLabelIds.size();
    }


}
