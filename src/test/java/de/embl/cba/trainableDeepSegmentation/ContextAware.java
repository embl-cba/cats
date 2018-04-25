package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.bigDataTools.dataStreamingTools.DataStreamingTools;
import de.embl.cba.trainableDeepSegmentation.settings.Settings;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.imglib2.FinalInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;

import java.util.ArrayList;
import java.util.TreeSet;

import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.*;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.C;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.T;

public class ContextAware
{

    public static void main( final String[] args )
    {

        new ImageJ();

        // Open Image
        //
        ImagePlus imp = IJ.openImage( "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/context-aware-v6-scale1.5-noise.tif" );

        DeepSegmentation deepSegmentation = new DeepSegmentation( );
        deepSegmentation.setInputImage( imp );
        deepSegmentation.setResultImageRAM( );
        deepSegmentation.loadInstancesAndMetadata( "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/context-aware-v6-scale1.5-noise.ARFF"  );

        deepSegmentation.settings.smoothingScales = new TreeSet<>();
        deepSegmentation.settings.smoothingScales.add( 1 );
        deepSegmentation.settings.smoothingScales.add( 2 );
        deepSegmentation.settings.binFactors.set( 1,  -1 );
        deepSegmentation.settings.binFactors.set( 2,  -1 );
        deepSegmentation.settings.binFactors.set( 3,  -1 );

        deepSegmentation.recomputeLabelInstances = true;
        deepSegmentation.updateExamplesInstancesAndMetadata();

        deepSegmentation.trainClassifier( );

        deepSegmentation.applyClassifierWithTiling();
    }


}
