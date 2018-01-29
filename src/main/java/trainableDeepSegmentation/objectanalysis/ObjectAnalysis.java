package trainableDeepSegmentation.objectanalysis;

import ij.ImagePlus;
import ij.plugin.Duplicator;
import inra.ijpb.binary.BinaryImages;
import inra.ijpb.morphology.AttributeFiltering;
import inra.ijpb.segment.Threshold;

public class ObjectAnalysis
{
    public static ImagePlus createLabelMaskForChannelAndFrame(
            ImagePlus image,
            int frame,
            int channel,
            int minNumVoxels,
            int lowerThreshold,
            int upperThreshold)
    {

        int connectivity = 6;

        long start = System.currentTimeMillis();

//        logger.info( "\n# Computing label mask..." );

        Duplicator duplicator = new Duplicator();
        ImagePlus imp = duplicator.run( image, channel, channel, 1, image.getNSlices(), frame, frame );

//        logger.info( "Threshold: " + lower + ", " + upper );
        ImagePlus th = Threshold.threshold( imp, lowerThreshold, upperThreshold );

//        logger.info( "MinNumVoxels: " + minNumVoxels );
        ImagePlus th_sf = new ImagePlus( "", AttributeFiltering.volumeOpening( th.getStack(), minNumVoxels) );

//        logger.info( "Connectivity: " + conn );
        ImagePlus labelMask = BinaryImages.componentsLabeling( th_sf, connectivity, 16);

//        logger.info( "...done! It took [min]:" + (System.currentTimeMillis() - start ) / ( 1000 * 60) );

        return labelMask;
    }
}
