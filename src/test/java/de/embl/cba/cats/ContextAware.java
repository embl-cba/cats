package de.embl.cba.cats;

import de.embl.cba.cats.features.DownSampler;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

import java.util.TreeSet;

public class ContextAware
{



    public static void main( final String[] args )
    {

        new ImageJ();

        // Open Image
        //
        ImagePlus imp = IJ.openImage( "/Users/tischer/Documents/fiji-plugin-CATS/src/test/resources/context-aware-v6-scale1.5-noise.tif" );

        CATS CATS = new CATS( );
        CATS.setInputImage( imp );
        CATS.setResultImageRAM( );
        CATS.loadInstancesAndMetadata( "/Users/tischer/Documents/fiji-plugin-CATS/src/test/resources/context-aware-v6-scale1.5-noise.ARFF"  );

        CATS.featureSettings.downSamplingMethod = DownSampler.getID( DownSampler.TRANSFORMJ_SCALE_LINEAR );
        CATS.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels = new TreeSet<>(  );
        CATS.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels.add( 0 );
        CATS.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels.add( 1 );
        CATS.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels.add( 3 );
        CATS.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels.add( 7 );

        //CATS.recomputeLabelInstances = true;
        //CATS.updateLabelInstancesAndMetadata();

        configureIlastikSettings( CATS );

        CATS.classifierNumTrees = 10;
        CATS.trainClassifier( );

        CATS.applyClassifierWithTiling();

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

        CATS.recomputeLabelInstances = true;
        CATS.updateLabelInstancesAndMetadata();
    }


}
