package de.embl.cba.cats;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.imglib2.FinalInterval;

import java.util.TreeSet;

import static de.embl.cba.cats.utils.IntervalUtils.*;
import static de.embl.cba.cats.utils.IntervalUtils.C;
import static de.embl.cba.cats.utils.IntervalUtils.T;

public class FIBSEMCell
{

    public static void main( final String[] args )
    {

        // Annotations: Slice 53

        new ImageJ();

        // Open Image
        //
        ImagePlus imp = IJ.openImage( "/Users/tischer/Documents/fiji-plugin-CATS/src/test/resources/fib-sem--cell--8x8x8nm.zip" );

        CATS CATS = new CATS( );
        CATS.setInputImage( imp );
        CATS.setResultImageRAM( );
        CATS.loadInstancesAndMetadata( "/Users/tischer/Documents/fiji-plugin-CATS/src/test/resources/fib-sem--cell--8x8x8nm.ARFF" );

        //CATS.recomputeLabelInstances = true;
        //CATS.updateLabelInstancesAndMetadata();

        //configureIlastikSettings( CATS );

        CATS.classifierNumTrees = 10;
        CATS.trainClassifier( );

        long[] min = new long[ 5 ];
        long[] max = new long[ 5 ];
        min[ X ] = 50; max[ X ] = 400;
        min[ Y ] = 50; max[ Y ] = 400;
        min[ Z ] = 20; max[ Z ] = 40;
        min[ C ] = 0; max[ C ] = 0;
        min[ T ] = 0; max[ T ] = 0;
        FinalInterval interval = new FinalInterval( min, max );
        CATS.applyClassifierWithTiling( interval );

        CATS.getInputImage().show();
        CATS.getResultImage().getWholeImageCopy().show();
    }

    private static void configureIlastikSettings( CATS CATS )
    {
        CATS.featureSettings.smoothingScales = new TreeSet<>();
        CATS.featureSettings.smoothingScales.add( 1 );
        CATS.featureSettings.smoothingScales.add( 2 );
        CATS.featureSettings.smoothingScales.add( 4 );
        CATS.featureSettings.smoothingScales.add( 8 );
        CATS.featureSettings.smoothingScales.add( 16 );
        CATS.featureSettings.binFactors.set( 1,  -1 );
        CATS.featureSettings.binFactors.set( 2,  -1 );
        CATS.featureSettings.binFactors.set( 3,  -1 );
        CATS.featureSettings.computeGaussian = true;

        CATS.recomputeLabelInstances = true;
        CATS.updateLabelInstancesAndMetadata();
    }


}
