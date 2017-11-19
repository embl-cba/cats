package trainableDeepSegmentation.instances;

import bigDataTools.logging.IJLazySwingLogger;
import bigDataTools.logging.Logger;
import ij.ImagePlus;
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
import java.util.ArrayList;
import java.util.Random;

import static trainableDeepSegmentation.IntervalUtils.*;
import static trainableDeepSegmentation.WekaSegmentation.*;
import static trainableDeepSegmentation.instances.InstancesAndMetadata.Metadata.*;

public class InstancesUtils {

    public InstancesUtils( )
    {
    }

    /**
     * Create instances instancesMap out of the user markings
     *
     * @return set of instancesMap (feature vectors in Weka format)
     */
    public static InstancesAndMetadata getInstancesAndMetadataFromLabels( ArrayList< Example > examples,
                                                                          String inputImageTitle,
                                                                          Settings settings,
                                                                          ArrayList< String > featureNames,
                                                                          ArrayList< String > classNames )
    {

        String instancesInfo = getInfoString( inputImageTitle, settings );

        Instances instances = getInstancesHeader(
                instancesInfo,
                featureNames,
                classNames  );

        InstancesAndMetadata instancesAndMetadata
                = new InstancesAndMetadata( instances );

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
                instancesAndMetadata.addMetadata( Metadata_ImageBackground, settings.imageBackground );
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



    // TODO: improve separation from WekaSegmentation
    public static InstancesAndMetadata createUsefulInstancesFromLabelImage(
            WekaSegmentation wekaSegmentation,
            ImagePlus labelImage,
            ResultImage resultImage,
            ImageProcessor instancesDistribution,
            FeatureProvider featureProvider,
            String instancesName,
            FinalInterval interval,
            int numInstancesPerClassAndPlane,
            int numThreads,
            boolean isFirstTime)
    {
        int radius = 5;

        logger.info("\n# Getting instances from Label image");

        final int numClasses = wekaSegmentation.getNumClasses();
        int t = (int) interval.min( T );

        Random rand = new Random();

        logger.info( "Instances per class and plane: " + numInstancesPerClassAndPlane);

        long startTime = System.currentTimeMillis();

        // TODO: determine numClasses from labelImage!
        wekaSegmentation.settings.classNames = new ArrayList<>();
        wekaSegmentation.settings.classNames.add("label_im_class_0");
        wekaSegmentation.settings.classNames.add("label_im_class_1");

        int[] pixelsPerClass = new int[ numClasses ];

        double[][][] reusableFeatureSlice = featureProvider.getReusableFeatureSlice();

        Instances instances = InstancesUtils.getInstancesHeader(
                instancesName,
                featureProvider.getAllFeatureNames(),
                wekaSegmentation.getClassNames());

        InstancesAndMetadata instancesAndMetadata = new InstancesAndMetadata( instances );

        int[][] classificationAccuracies = new int[numClasses][4]; // TOTAL, CORRECT, FP, FN

        // Collect instances per plane
        for ( int z = (int) interval.min( Z ); z <= interval.max( Z ); ++z)
        {
            logLabelImageTrainingProgress( logger, z, interval, "...");

            ArrayList < int[] > [][] classCoordinates = getEmptyClassCoordinates(
                    numClasses, resultImage.getProbabilityRange() );

            addClassCoordinatesAndAccuracies(
                    classCoordinates,
                    classificationAccuracies,
                    z, t,
                    labelImage,
                    resultImage,
                    interval,
                    isFirstTime);

            double[][][] featureSlice = featureProvider.getCachedFeatureSlice( z );

            if ( featureSlice == null )
            {
                featureProvider.setFeatureSlicesValues( z, reusableFeatureSlice, numThreads );
                featureSlice = reusableFeatureSlice;
            }


            for ( int i = 0; i < numInstancesPerClassAndPlane; ++i )
            {

                if ( wekaSegmentation.stopCurrentTasks )
                {
                    return null;
                }

                for ( int iClass = 0; iClass < numClasses; ++iClass )
                {

                    int[] xy = getUsefulRandomCoordinate( iClass, classCoordinates, rand );

                    if ( xy == null )
                    {
                        // stop, because no more example of this class are available
                        // and we do not want to add (unbalanced)
                        // examples of the other classes
                        i = numInstancesPerClassAndPlane;
                        break;
                    }

                    ArrayList< int[] >[] localClassCoordinates =
                            getLocalClassCoordinates( numClasses,
                                    labelImage, z, t,
                                    interval,
                                    xy,
                                    radius );

                    for (int localClass = 0; localClass < numClasses; ++localClass)
                    {

                        int[] xyLocal = null;

                        if ( localClass == iClass )
                        {
                            xyLocal = xy;
                        }
                        else
                        {
                            xyLocal = getRandomCoordinate( localClass, localClassCoordinates, rand );

                            if ( xyLocal == null )
                            {
                                // no local coordinate has been found
                                // thus we take a useful global one
                                xyLocal = getUsefulRandomCoordinate( localClass, classCoordinates, rand );
                            }

                        }

                        if ( xyLocal == null )
                        {
                            // this class really has no more labels => stop
                            i = numInstancesPerClassAndPlane;
                            break;
                        }

                        addToInstancesDistribution( instancesDistribution, xyLocal, interval );

                        removeNeighbors( localClass, classCoordinates, xyLocal, radius );

                        Instance instance = getInstance( featureProvider, xyLocal, featureSlice, localClass );

                        instancesAndMetadata.addInstance( instance );
                        instancesAndMetadata.addMetadata( Metadata_Position_X, xyLocal[ 0 ] );
                        instancesAndMetadata.addMetadata( Metadata_Position_Y, xyLocal[ 1 ] );
                        instancesAndMetadata.addMetadata( Metadata_Position_Z, z );
                        instancesAndMetadata.addMetadata( Metadata_Position_T, t );

                        pixelsPerClass[ localClass ]++;

                        }

                    }

                }
            }

        logger.info ( "...computed instance values in [min]: " +
                wekaSegmentation.getMinutes( System.currentTimeMillis(), startTime ) );

        //for( int j = 0; j < numOfClasses ; j ++ )
        //	IJ.log("Added " + numSamples + " instancesComboBox of '" + loadedClassNames.get( j ) +"'.");

        for ( int iClass = 0; iClass < numClasses; ++iClass )
        {
            logger.info( "Class " + iClass + " [pixels]: " + pixelsPerClass[ iClass ]);
            if( pixelsPerClass[iClass] == 0 )
            {
                logger.error("No labels of class found: " + iClass);
            }
        }

        if ( ! isFirstTime )
        {
            reportClassificationAccuracies( classificationAccuracies, logger );
        }

        return ( instancesAndMetadata );

    }


    private static void addToInstancesDistribution( ImageProcessor instancesDistribution,
                                             int[] xy,
                                             FinalInterval interval )
    {
        int xx = xy[ 0 ] - ( int ) interval.min( X );
        int yy = xy[ 1 ] - ( int ) interval.min( Y );
        int v = instancesDistribution.get( xx, yy ) + 1;
        instancesDistribution.set( xx, yy, v );
    }

    public static void reportClassificationAccuracies( int[][] classificationAccuracies,
                                                        Logger logger)
    {
        logger.info( "\n# Classification accuracies");

        for ( int iClass = 0; iClass < classificationAccuracies.length; ++iClass )
        {
            int total = classificationAccuracies[iClass][TOTAL];
            if ( total == 0 ) total = -1; // to avoid division by zero

            logger.info("Class " + iClass
                    + "; " + "Percent correct: " + ( 100 * classificationAccuracies[iClass][CORRECT] ) / total
                    + "; " + "True: " + total
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

    private static void addClassCoordinatesAndAccuracies(
                                    ArrayList < int[] >[][] classCoordinates,
                                    int[][] accuracies,
                                    int z, int t,
                                    ImagePlus labelImage,
                                    ResultImage resultImage,
                                    FinalInterval interval,
                                    boolean onlySetCoordinates)
    {

        int maxProbability = resultImage.getProbabilityRange();

        ImageProcessor labelImageSlice = labelImage.getStack().getProcessor(z + 1);

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

                    int[] classifiedClassAndProbability =
                            resultImage.getClassAndProbability( x, y, z, t );

                    int classifiedClass = classifiedClassAndProbability[ 0 ];
                    int correctness = classifiedClassAndProbability[ 1 ];

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

                    // shift such that it becomes a positive index
                    correctness += maxProbability;

                    classCoordinates[realClass][correctness].add( new int[]{ x, y } );
                }

            }
        }

    }

    public static int[][] getAccuracies(
            ImagePlus labelImage,
            ResultImage resultImage,
            ImageProcessor ipAccuracy,
            FinalInterval interval )
    {
        int maxProbability = resultImage.getProbabilityRange();

        int numClasses = 2; // TODO: get from ResultImage
        int t = (int) interval.min( T );

        int[][] accuracies = new int[numClasses][5];

        for ( int z = (int) interval.min( Z ); z <= interval.max( Z ); ++z )
        {
            ImageProcessor labelImageSlice = labelImage.getStack().getProcessor(z + 1);

            for ( int y = (int) interval.min( Y ); y <= interval.max( Y ); ++y)
            {
                for ( int x = ( int ) interval.min( X ); x <= interval.max( X ); ++x )
                {
                    int realClass = labelImageSlice.get( x, y );
                    int[] classifiedClassAndProbability =
                            resultImage.getClassAndProbability( x, y, z, t );

                    int classifiedClass = classifiedClassAndProbability[ 0 ];
                    int correctness = classifiedClassAndProbability[ 1 ];

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
                    correctness += maxProbability;

                    if ( ipAccuracy != null )
                    {
                        ipAccuracy.set( x - ( int ) interval.min( X ),
                                y - ( int ) interval.min( Y ),
                                correctness );
                    }

                }
            }
        }

        return (accuracies);

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
                                                    Random random )
    {

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
        Instances instances = new Instances(instancesName, attributes, 1);
        // Set the index of the class attribute
        instances.setClassIndex( instances.numAttributes() - 1 );

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

    public static void logInstancesInformation( Instances instances )
    {
        logger.info( "\n# Instances information" );
        logger.info( "Number of instances: " + instances.size() );
        logger.info( "Number of attributes: " + instances.numAttributes() );

        // TODO: output per class a.s.o.

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

    public static InstancesAndMetadata loadInstancesAndMetadataFromARFF(
            String directory, String filename )
    {
        Instances instances = loadInstancesFromARFF( directory, filename );

        if ( instances == null ) return null;

        InstancesAndMetadata instancesAndMetadata
                = new InstancesAndMetadata( instances );

        instancesAndMetadata.moveMetadataOutOfInstances();

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

    public static boolean saveInstancesAndMetadataAsARFF( InstancesAndMetadata instancesAndMetadata,
                                                          String directory,
                                                          String filename)
    {

        instancesAndMetadata.putMetadataIntoInstances();

        boolean success = saveInstancesAsARFF( instancesAndMetadata.instances, directory, filename );

        instancesAndMetadata.removeMetadataFromInstances();

        return success;

    }



}
