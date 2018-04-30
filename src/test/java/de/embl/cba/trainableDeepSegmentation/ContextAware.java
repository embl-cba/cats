package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.trainableDeepSegmentation.features.DownSampler;
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
        ImagePlus imp = IJ.openImage( "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/context-aware-v6-scale1.5-noise.tif" );

        DeepSegmentation deepSegmentation = new DeepSegmentation( );
        deepSegmentation.setInputImage( imp );
        deepSegmentation.setResultImageRAM( );
        deepSegmentation.loadInstancesAndMetadata( "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/context-aware-v6-scale1.5-noise.ARFF"  );

        deepSegmentation.featureSettings.downSamplingMethod = DownSampler.getID( DownSampler.TRANSFORMJ_SCALE_LINEAR );
        deepSegmentation.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels = new TreeSet<>(  );
        deepSegmentation.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels.add( 0 );
        deepSegmentation.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels.add( 1 );
        deepSegmentation.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels.add( 3 );
        deepSegmentation.featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels.add( 7 );

        //deepSegmentation.recomputeLabelInstances = true;
        //deepSegmentation.updateExamplesInstancesAndMetadata();

        configureIlastikSettings( deepSegmentation );

        deepSegmentation.classifierNumTrees = 10;
        deepSegmentation.trainClassifier( );

        deepSegmentation.applyClassifierWithTiling();

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

        deepSegmentation.recomputeLabelInstances = true;
        deepSegmentation.updateExamplesInstancesAndMetadata();
    }


}
