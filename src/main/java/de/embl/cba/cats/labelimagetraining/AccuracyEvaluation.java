package de.embl.cba.cats.labelimagetraining;

import de.embl.cba.cats.CATS;
import de.embl.cba.cats.results.ResultImage;
import de.embl.cba.cats.utils.ThreadUtils;
import de.embl.cba.utils.logging.Logger;
import ij.ImagePlus;
import ij.Prefs;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;

import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static de.embl.cba.cats.CATS.logger;
import static de.embl.cba.cats.utils.IntervalUtils.X;
import static de.embl.cba.cats.utils.IntervalUtils.Y;
import static de.embl.cba.cats.utils.IntervalUtils.Z;

public class AccuracyEvaluation
{
    public final static int TOTAL = 0;
    public final static int TP = 1;
    public final static int FP = 2;
    public final static int FN = 3;

    public static void reportClassificationAccuracies( long[][][] classificationAccuracies,
                                                       int t,
                                                       Logger logger)
    {
        logger.info( "\n# Classification accuracies for time-point: " + t );


        int numSlices = classificationAccuracies.length;
        int numClasses = classificationAccuracies[ 0 ].length;


        for ( int iClass = 0; iClass < numClasses; ++iClass )
        {
            long gt = 0;
            long tp = 0;
            long fp = 0;
            long fn = 0;

            for ( int z = 0; z < numSlices; ++z )
            {
                gt += classificationAccuracies[ z ][ iClass ][ TOTAL ];
                tp += classificationAccuracies[ z ][ iClass ][ TP ];
                fp += classificationAccuracies[ z ][ iClass ][ FP ];
                fn += classificationAccuracies[ z ][ iClass ][ FN ];
            }

            if ( gt == 0 ) gt = -1; // to avoid division by zero

            double jaccardIndex = 100.0 * tp / ( tp + fp + fn );

            logger.info("Class " + iClass
                    + "; " + "Percent correct: " + ( 100.0 * tp ) / gt
                    + "; " + "Jaccard index: " + jaccardIndex
                    + "; " + "Ground truth: " + gt
                    + "; " + "True positive: " + tp
                    + "; " + "False positive: " + fp
                    + "; " + "False negative: " + fn
            );
        }
    }

    public static long[][][] computeLabelImageBasedAccuracies(
            ImagePlus inputImageWithLabels,
            ResultImage resultImage,
            int labelChannel,
            ImagePlus accuraciesImage,
            FinalInterval interval,
            int t,
            CATS CATS )
    {

        logger.info( "\n# Label image training accuracies");

        int numClasses = 2; // TODO

        long[][][] accuracies = new long[ (int) interval.dimension( Z ) ][ numClasses ][ 5 ];

        ExecutorService exe = Executors.newFixedThreadPool( Prefs.getThreads() );
        ArrayList< Future > futures = new ArrayList< >(  );

        for ( int z = (int) interval.min( Z ); z <= interval.max( Z ); ++z )
        {
            futures.add(
                    exe.submit(
                            computeSliceAccuracies(
                                accuracies,
                                inputImageWithLabels,
                                resultImage,
                                accuraciesImage,
                                labelChannel,
                                z,
                                t,
                                interval,
									CATS
                            )
                    )
            );
        }

        ThreadUtils.joinThreads( futures, CATS.getLogger(), exe );

        return ( accuracies );

    }

    public static Runnable computeSliceAccuracies(
            long[][][] accuracies,
            ImagePlus inputImageWithLabels,
            ResultImage resultImage,
            ImagePlus accuraciesImage,
            int labelChannel,
            int z,
            int t,
            FinalInterval interval,
            CATS CATS )
    {

        return new Runnable() {

            @Override
            public void run()
            {

                if ( CATS.stopCurrentTasks ) return;

                logger.progress( "Measuring accuracies",
                        "t: " + t + "; z: " + ( z + 1 ) + " / " + ( interval.max( Z ) + 1 ) );

                int maxProbability = resultImage.getProbabilityRange();

                int stackIndex =  inputImageWithLabels.getStackIndex( labelChannel + 1, z + 1, t + 1 );
                ImageProcessor labelImageSlice = inputImageWithLabels.getStack().getProcessor( stackIndex );
                ImageProcessor ipAccuracies = null;
                ImageProcessor ipResult = resultImage.getSlice( z + 1, t + 1  );

                int probabilityRange = resultImage.getProbabilityRange();

                if ( accuraciesImage != null )
                {
                    ipAccuracies = accuraciesImage.getImageStack().getProcessor( z + 1 - ( int ) interval.min( Z ) );
                }

                int zIndexInAccuraciesArray = (int) ( z - interval.min( Z ) );

                for ( int y = ( int ) interval.min( Y ); y <= interval.max( Y ); ++y )
                {
                    for ( int x = ( int ) interval.min( X ); x <= interval.max( X ); ++x )
                    {
                        int realClass = labelImageSlice.get( x, y );

                        byte result = (byte) ipResult.get( x, y );

                        int[] classAndProbability = new int[ 2 ];

                        classAndProbability[0] = ( result - 1 ) / probabilityRange;
                        classAndProbability[1] = result - classAndProbability[0] * probabilityRange;

                        int classifiedClass = classAndProbability[ 0 ];
                        int correctness = classAndProbability[ 1 ];

                        accuracies[ zIndexInAccuraciesArray ][ realClass ][ TOTAL ]++;

                        if ( realClass == classifiedClass )
                        {
                            accuracies[ zIndexInAccuraciesArray ][ realClass ][ TP ]++;
                        }
                        else
                        {
                            correctness *= -1;
                            accuracies[ zIndexInAccuraciesArray ][ realClass ][ FN ]++;
                            accuracies[ zIndexInAccuraciesArray ][ classifiedClass ][ FP ]++;
                        }

                        correctness += maxProbability;

                        if ( ipAccuracies != null )
                        {
                            ipAccuracies.set( x - ( int ) interval.min( X ), y - ( int ) interval.min( Y ), correctness );
                        }

                    }

                }

                accuraciesImage.updateImage();
            }
        };
    }
}
