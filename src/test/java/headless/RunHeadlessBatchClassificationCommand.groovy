/**
 * Batch classification of a folder of images.
 * 
 * 
 * Important notes: 
 * - If you have the 3DImageSuite installed, there may be an old version of the weka classfier: weka-3.7.6.jar
 * in your /jars folder. You need to delete this file in case it is there.
 */

import de.embl.cba.cats.ui.BatchClassificationCommand;
import java.io.File;
import ij.IJ;
import ij.Prefs;

command = new BatchClassificationCommand();

// Specify folder with input images
command.inputDirectory = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/blobs/input");

// Specify classifier
command.classifierFile = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/blobs/classifier/blobs_00.classifier");

// Regular expression to only classify files in the inputDirectory matching this pattern (".*" matches everything)
command.filenameRegExp = ".*"; 

// Directory for the output
command.outputDirectory = new File("/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/blobs/output");

// specify resources, you may try to over-commit to see whether it runs faster
command.numThreads = Prefs.getThreads();;
command.memoryMB = IJ.maxMemory() / ( 1024 * 1024 ); // IJ.maxMemory() is in Bytes, thus we need to divide by Mega ( = 1024 * 1024 )
		
command.run();