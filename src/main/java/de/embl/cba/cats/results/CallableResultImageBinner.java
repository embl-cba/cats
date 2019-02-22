package de.embl.cba.cats.results;

import de.embl.cba.bigdataconverter.utils.Utils;
import de.embl.cba.utils.logging.Logger;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.Binner;

import java.util.concurrent.Callable;

import static de.embl.cba.cats.utils.IntervalUtils.X;
import static de.embl.cba.cats.utils.IntervalUtils.Y;

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
                tmpStack.addSlice( settings.resultImage.getSlice( z + 1, t + 1 ).duplicate()  );
            }

            ImagePlus gated = new ImagePlus( "gated-" + classId, tmpStack );
            Utils.applyIntensityGate( gated, intensityGate );
//            tmpImage.show();

            ImagePlus binned = gated;

            if ( dx != 1 || dy !=1 || dz != 1 )
            {
                Binner binner = new Binner();
                binned = binner.shrink( gated, dx, dy, dz, Binner.AVERAGE );
            }

//            binned.setTitle( "binned-" + classId );
//            binned.show();

            ResultExport.convertToProperBitDepth( binned, settings, classId );
//            binned.setTitle( "binned-bitdepth-" + classId );
//            binned.show();

            logger.progress( "Creating binned class image", null, startTime, z0, nz );

            return binned;

        };
    }

}
