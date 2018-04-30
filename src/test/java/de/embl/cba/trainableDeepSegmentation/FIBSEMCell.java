package de.embl.cba.trainableDeepSegmentation;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.imglib2.FinalInterval;

import java.util.TreeSet;

import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.*;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.C;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.T;

public class FIBSEMCell
{

    public static void main( final String[] args )
    {

        // Annotations: Slice 53

        new ImageJ();

        // Open Image
        //
        ImagePlus imp = IJ.openImage( "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/fib-sem--cell--8x8x8nm.zip" );

        DeepSegmentation deepSegmentation = new DeepSegmentation( );
        deepSegmentation.setInputImage( imp );
        deepSegmentation.setResultImageRAM( );
        deepSegmentation.loadInstancesAndMetadata( "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/fib-sem--cell--8x8x8nm.ARFF" );

        //deepSegmentation.recomputeLabelInstances = true;
        //deepSegmentation.updateExamplesInstancesAndMetadata();

        //configureIlastikSettings( deepSegmentation );

        deepSegmentation.classifierNumTrees = 10;
        deepSegmentation.trainClassifier( );

        long[] min = new long[ 5 ];
        long[] max = new long[ 5 ];
        min[ X ] = 50; max[ X ] = 400;
        min[ Y ] = 50; max[ Y ] = 400;
        min[ Z ] = 20; max[ Z ] = 40;
        min[ C ] = 0; max[ C ] = 0;
        min[ T ] = 0; max[ T ] = 0;
        FinalInterval interval = new FinalInterval( min, max );
        deepSegmentation.applyClassifierWithTiling( interval );

        deepSegmentation.getInputImage().show();
        deepSegmentation.getResultImage().getWholeImageCopy().show();
    }

    private static void configureIlastikSettings( DeepSegmentation deepSegmentation )
    {
        deepSegmentation.featureSettings.smoothingScales = new TreeSet<>();
        deepSegmentation.featureSettings.smoothingScales.add( 1 );
        deepSegmentation.featureSettings.smoothingScales.add( 2 );
        deepSegmentation.featureSettings.smoothingScales.add( 4 );
        deepSegmentation.featureSettings.smoothingScales.add( 8 );
        deepSegmentation.featureSettings.smoothingScales.add( 16 );
        deepSegmentation.featureSettings.binFactors.set( 1,  -1 );
        deepSegmentation.featureSettings.binFactors.set( 2,  -1 );
        deepSegmentation.featureSettings.binFactors.set( 3,  -1 );
        deepSegmentation.featureSettings.computeGaussian = true;

        deepSegmentation.recomputeLabelInstances = true;
        deepSegmentation.updateExamplesInstancesAndMetadata();
    }


}
