package de.embl.cba.trainableDeepSegmentation.utils;

import de.embl.cba.cluster.JobFuture;
import de.embl.cba.utils.logging.Logger;

import java.util.ArrayList;
import java.util.HashMap;

public class SlurmJobMonitor
{

    class SlurmErrors
    {
        int numResubmittedJobs = 0;
        int numXvfbErrors = 0;
        int numSlurmStepErrors = 0;
        int numUnkownErrors = 0;
        int numFailedMoreThanFiveTimes = 0;
    }

    public void monitorJobProgress( ArrayList< JobFuture > jobFutures, Logger logger )
    {
        ArrayList< JobFuture > doneJobs = new ArrayList<>();

        HashMap< JobFuture, Integer > resubmissionAttempts = new HashMap<>();

        SlurmErrors slurmErrors = new SlurmErrors();

        while ( doneJobs.size() < jobFutures.size() )
        {
            for ( JobFuture jobFuture : jobFutures )
            {

                try { Thread.sleep( 5000 ); } catch ( InterruptedException e ) { e.printStackTrace(); }

                logJobStati( jobFutures, logger, doneJobs, slurmErrors );

                if ( jobFuture.isStarted() )
                {

                    String currentOutput = jobFuture.getOutput();

                    if  ( ! doneJobs.contains( jobFuture ) )
                    {
                        String[] currentOutputLines = currentOutput.split( "\n" );
                        String lastLine = currentOutputLines[ currentOutputLines.length - 1 ];
                        logger.info( "Current last line of job output: " + lastLine );
                    }

                    String resubmissionNeeded = jobFuture.needsResubmission();

                    if ( ! resubmissionNeeded.equals( JobFuture.EVERYTHING_FINE ) )
                    {
                        if ( resubmissionAttempts.containsKey( jobFuture ) )
                        {
                            int numResubmissions = resubmissionAttempts.get( jobFuture );
                            resubmissionAttempts.put( jobFuture, ++numResubmissions );
                        }
                        else
                        {
                            resubmissionAttempts.put( jobFuture, 1 );
                        }

                        if ( resubmissionAttempts.containsKey( jobFuture ) && resubmissionAttempts.get( jobFuture ) > 5 )
                        {
                            logger.info( "# Job failed more than 5 times. Will not resubmit." );
                            slurmErrors.numFailedMoreThanFiveTimes++;
                        }
                        else
                        {
                            logger.info( "RESUBMITTING: " + jobFuture.getJobID() );
                            jobFuture.resubmit();

                            if ( resubmissionNeeded.equals( JobFuture.XVFB_ERROR ) )
                            {
                                slurmErrors.numXvfbErrors++;
                            }
                            else if ( resubmissionNeeded.equals( JobFuture.SLURM_STEP_ERROR ) )
                            {
                                slurmErrors.numSlurmStepErrors++;
                            }
                            else
                            {
                                logger.info( "UNKOWN ERROR: " + resubmissionNeeded );

                                slurmErrors.numUnkownErrors++;
                            }

                            slurmErrors.numResubmittedJobs++;
                        }

                    }
                    else if ( jobFuture.isDone() )
                    {
                        logger.info("Final and full job output:" );
                        logger.info( currentOutput );
                        doneJobs.add( jobFuture );
                        if ( doneJobs.size() == jobFutures.size() )
                        {
                            break;
                        }
                    }
                }
                else
                {
                    logger.info( "Job " + jobFuture.getJobID() + " has not yet started." );
                }


            }
        }

        logJobStati( jobFutures, logger, doneJobs, slurmErrors );

        logger.info( "All jobs finished." );

    }

    private static void logJobStati( ArrayList< JobFuture > jobFutures, Logger logger, ArrayList< JobFuture > doneJobs, SlurmErrors slurmErrors)
    {
        logger.info( " " );
        logger.info( "# JOB STATI" );
        logger.info( " " );
        logger.info( "Total number of jobs: " + jobFutures.size() );
        logger.info( "Finished jobs: " + doneJobs.size() );
        logger.info( "Job resubmissions: " + slurmErrors.numResubmittedJobs );
        logger.info( "Xvfb errors: " + slurmErrors.numXvfbErrors );
        logger.info( "Slurm step errors: " + slurmErrors.numSlurmStepErrors );
        logger.info( "Unknown errors: " + slurmErrors.numUnkownErrors );
        logger.info( "Failed jobs (Resubmissions failed more than 5 times): " + slurmErrors.numFailedMoreThanFiveTimes );
        logger.info( " " );
    }
}
