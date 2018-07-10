package de.embl.cba.cats.postprocessing;

import ij.IJ;
import ij.ImagePlus;

import inra.ijpb.segment.Threshold;
import inra.ijpb.morphology.Morphology;
import inra.ijpb.morphology.Strel3D;
import inra.ijpb.math.ImageCalculator;
import sc.fiji.localThickness.EDT_S1D;


public class ProximityFilter3D
{
    public static ImagePlus filter( ImagePlus reference, ImagePlus toBeProximityFiltered, int radiusInPixels )
    {

        ImagePlus binaryDilated = getDilatedBinaryUsingEDT( reference, radiusInPixels );

        // Apply dilated reference stack mask on image toBeProximityFiltered
        //
        ImageCalculator.Operation multiply = ImageCalculator.Operation.TIMES;
        ImagePlus proximityFiltered = ImageCalculator.combineImages( binaryDilated, toBeProximityFiltered, multiply );

        return proximityFiltered;
    }

    public static ImagePlus multiply( ImagePlus imp1, ImagePlus imp2 )
    {
        // Apply dilated reference stack mask on image toBeProximityFiltered
        //
        ImageCalculator.Operation multiply = ImageCalculator.Operation.TIMES;
        ImagePlus product = ImageCalculator.combineImages( imp1, imp2, multiply );
        return product;
    }


    public static ImagePlus getDilatedBinaryUsingEDT( ImagePlus reference, int radiusInPixels )
    {
        // Create binary mask of reference stack
        //
        //ImagePlus imp = Threshold.threshold( reference, 1, 255 );

        // Create dilated reference stack
        //
        ImagePlus edt = getEDT( reference );
        ImagePlus edtMask = Threshold.threshold( edt, 0, radiusInPixels );
        edt.close();
        IJ.run( edtMask, "Divide...", "value=255 stack");

        return edtMask;
    }

    private static ImagePlus getEDT( ImagePlus imp )
    {

        EDT_S1D edt = new EDT_S1D();
        edt.runSilent = true;
        edt.inverse = true;
        edt.showOptions = false;
        edt.thresh = 1;
        edt.setup( "", imp );
        edt.run( imp.getProcessor() );
        imp = edt.getResultImage();
        return imp;
    }


    public static ImagePlus getDilatedBinaryUsingDilation( ImagePlus reference, int radiusInPixels )
    {
        // Create binary mask of reference stack
        //
        ImagePlus th = Threshold.threshold( reference, 1, 255 );
        IJ.run( th, "Divide...", "value=255 stack");

        // Create dilated reference stack
        //
        Strel3D ball3D = Strel3D.Shape.BALL.fromRadius( radiusInPixels );
        ImagePlus thDilated = new ImagePlus( "dilated", Morphology.dilation( th.getImageStack(), ball3D ) );


        // IJ.run(imp, "Distance Transform 3D", "");

        return thDilated;
    }

}
