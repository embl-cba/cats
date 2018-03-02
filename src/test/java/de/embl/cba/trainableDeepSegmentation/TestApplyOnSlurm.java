package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierAsSlurmJobsCommand;
import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierCommand;
import de.embl.cba.trainableDeepSegmentation.utils.IOUtils;
import de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.ImageJ;
import net.imglib2.FinalInterval;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static de.embl.cba.trainableDeepSegmentation.FIBSEMCell.TEST_RESOURCES;

public class TestApplyOnSlurm
{

    public static void main ( String... args )
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();


        ImagePlus imp = IJ.openImage(TEST_RESOURCES + "fib-sem--cell--8x8x8nm.zip" );
        FinalInterval interval = IntervalUtils.getInterval( imp );

        /*
        FileInfo fileInfo = imp.getOriginalFileInfo();
        String inputImagePath = fileInfo.directory + File.separator + fileInfo.fileName;
        */

        String inputImagePath = "/Volumes/cba/tischer/projects/em-automated-segmentation--data/fib-sem--cell--8x8x8nm.zip";
        String outputDirectory = "/Volumes/cba/tischer/projects/em-automated-segmentation--data/fib-sem--cell--classification/";
        File classifierFile = new File( "/Volumes/cba/tischer/projects/em-automated-segmentation--data/fib-sem--cell.classifier" );


        Map< String, Object > parameters = new HashMap<>(  );
        parameters.put( IOUtils.OUTPUT_DIRECTORY, outputDirectory );

        parameters.put( ApplyClassifierAsSlurmJobsCommand.INTERVAL, interval );

        parameters.put( ApplyClassifierCommand.CLASSIFIER_FILE, classifierFile );

        parameters.put( IOUtils.INPUT_IMAGE_PATH, inputImagePath );
        parameters.put( IOUtils.INPUT_MODALITY, IOUtils.OPEN_USING_IMAGEJ1 );

        parameters.put( ApplyClassifierAsSlurmJobsCommand.WORKERS, 16 );

        ij.command().run( ApplyClassifierAsSlurmJobsCommand.class, true, parameters );

    }
}
