package de.embl.cba.cats.results;

import de.embl.cba.cats.results.ResultImage;
import de.embl.cba.utils.logging.Logger;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.Binner;
import ij.process.StackProcessor;

import java.util.concurrent.Callable;

import static de.embl.cba.cats.utils.IntervalUtils.X;
import static de.embl.cba.cats.utils.IntervalUtils.Y;
import static de.embl.cba.cats.utils.IntervalUtils.Z;

public class CallableResultImageBinner
{


    public static Callable<ImagePlus> getBinned( ResultExportSettings settings,
                                                 int classId,
                                                 int z0, int z1,
                                                 int t,
                                                 Logger logger,
                                                 long startTime,
                                                 int nz )
    {
        return () -> {

            int nx = (int) settings.resultImage.getDimensions()[ X ];
            int ny = (int) settings.resultImage.getDimensions()[ Y ];

            int dx = settings.binning[0];
            int dy = settings.binning[1];
            int dz = settings.binning[2];

            int classLutWidth = settings.resultImage.getProbabilityRange();
            int[] intensityGate = new int[]{ classId * classLutWidth + 1, (classId + 1 ) * classLutWidth };

            ImageStack tmpStack = new ImageStack ( nx , ny );

            for ( int z = z0; z <= z1; ++z )
            {
                tmpStack.addSlice( settings.resultImage.getSlice( z + 1, t + 1 )  );
            }

            ImagePlus tmpImage = new ImagePlus( "", tmpStack );
            de.embl.cba.bigDataTools.utils.Utils.applyIntensityGate( tmpImage, intensityGate );

            Binner binner = new Binner();
            ImagePlus binned = binner.shrink( tmpImage, dx, dy, dz, Binner.AVERAGE );

            ResultExport.convertToProperBitDepth( binned, settings, classId );

            logger.progress( "Creating binned class image", null, startTime, z0, nz );

            return binned;

        };
    }

}
