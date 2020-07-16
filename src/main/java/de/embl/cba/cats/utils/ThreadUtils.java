package de.embl.cba.cats.utils;

import de.embl.cba.log.Logger;

import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class ThreadUtils {


    public static boolean stopThreads( Logger logger, boolean stopCurrentThreads, int i, int n )
    {
        if ( stopCurrentThreads || Thread.currentThread().isInterrupted() )
        {
            logger.progress("Thread stopped:", "" + i + "/" + n);
            return true;
        }
        else
        {
            return false;
        }
    }


    public static boolean stopThreads( Logger logger, ExecutorService exe, boolean stopCurrentThreads, int i, int n )
    {
        if ( stopCurrentThreads || Thread.currentThread().isInterrupted() )
        {
            logger.progress("Thread stopped:", "" + i + "/" + n);
            exe.shutdownNow();
            return true;
        }
        else
        {
            return false;
        }
    }

    public static void joinThreads( ArrayList<Future> futures, Logger logger, ExecutorService exe )
    {
        if ( futures.size() > 0)
        {
            try
            {
                for (Future future : futures)
                    future.get();
            }
            catch (InterruptedException e)
            {
                e.printStackTrace();
                return;
            }
            catch (ExecutionException e)
            {
                e.printStackTrace();
                return;
            }
            catch (OutOfMemoryError err)
            {
                logger.error("Out of memoryMB. Please, "
                        + "provide more memoryMB and/or use less numWorkers " +
                        "[ImageJ > Edit > Options > Memory & Threads].");
                err.printStackTrace();
                return;
            }
        }

        futures = null;
        exe.shutdown();

        return;
    }
}
