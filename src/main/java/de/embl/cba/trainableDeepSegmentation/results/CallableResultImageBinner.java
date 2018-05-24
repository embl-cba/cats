package de.embl.cba.trainableDeepSegmentation.results;

import de.embl.cba.trainableDeepSegmentation.results.ResultImage;
import de.embl.cba.utils.logging.Logger;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.Binner;
import ij.process.StackProcessor;

import java.util.concurrent.Callable;

import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.X;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.Y;
import static de.embl.cba.trainableDeepSegmentation.utils.IntervalUtils.Z;

public class CallableResultImageBinner
{


    public static Callable<ImagePlus> getBinned( ResultImage resultImage,
                                                 int classId,
                                                 int[] binning,
                                                 int z0, int z1,
                                                 int t,
                                                 Logger logger,
                                                 long startTime,
                                                 int nz )
    {
        return () -> {

            int nx = (int) resultImage.getDimensions()[ X ];
            int ny = (int) resultImage.getDimensions()[ Y ];

            int dx = binning[0];
            int dy = binning[1];
            int dz = binning[2];

            int classLutWidth = resultImage.getProbabilityRange();
            int[] intensityGate = new int[]{ classId * classLutWidth + 1, (classId + 1 ) * classLutWidth };

            ImageStack tmpStack = new ImageStack ( nx , ny );

            for ( int z = z0; z <= z1; ++z )
            {
                tmpStack.addSlice( resultImage.getSlice( z + 1, t + 1 )  );
            }

            ImagePlus tmpImage = new ImagePlus( "", tmpStack );
            de.embl.cba.bigDataTools.utils.Utils.applyIntensityGate( tmpImage, intensityGate );

            Binner binner = new Binner();
            ImagePlus binned = binner.shrink( tmpImage, dx, dy, dz, Binner.AVERAGE );


            logger.progress( "Creating binned class image", null, startTime, z0, nz );

            return binned;

        };
    }

}
