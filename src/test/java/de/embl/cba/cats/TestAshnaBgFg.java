package de.embl.cba.cats;

import de.embl.cba.bigDataTools.dataStreamingTools.DataStreamingTools;
import ij.IJ;
import ij.ImageJ;
import net.imglib2.FinalInterval;

import static de.embl.cba.cats.utils.IntervalUtils.*;
import static de.embl.cba.cats.utils.IntervalUtils.C;
import static de.embl.cba.cats.utils.IntervalUtils.T;

public class TestAshnaBgFg {

    final static String INSTANCES_PATH = "/Users/de.embl.cba.cats.weka/Documents/ashna/instances/bin_1_2_3_4--log2--001.ARFF";


    public static void main( final String[] args )
    {

        boolean GUI = true;

        new ImageJ();

        // Open Image
        //
        DataStreamingTools dst = new DataStreamingTools();
        dst.openFromDirectory(
                "/Users/de.embl.cba.cats.weka/Documents/ashna/reg-3x3",
                "None",
                ".*--C.*",
                "Resolution 0/Data",
                null,
                3,
                true,
                false);
        IJ.wait( 1000 );

        IJ.run("Properties...", "unit=nm pixel_width=309 pixel_height=309 voxel_depth=1000");


        if ( GUI )
        {
            de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin();
            weka_segmentation.run( "" );
        }
        else
        {
            long[] min = new long[ 5 ];
            long[] max = new long[ 5 ];
            min[ X ] = 111; max[ X ] = 211;
            min[ Y ] = 111; max[ Y ] = 211;
            min[ Z ] = 80; max[ Z ] = 90;
            min[ C ] = 0; max[ C ] = 0;
            min[ T ] = 0; max[ T ] = 0;

            FinalInterval interval = new FinalInterval( min, max );

            // FinalInterval interval = IntervalUtils.getInterval( IJ.getImage() );


            // Pixel classification
            //
            CATS ws = new CATS( );
            ws.setInputImage( IJ.getImage() );
            ws.setResultImageRAM( );
            String instancesKey = ws.loadInstancesAndMetadata( INSTANCES_PATH );
            //String classifierKey = ws.trainClassifier( instancesKey );
            //ws.applyClassifierWithTiling( classifierKey, interval );
            //ws.getResultImage().getWholeImageCopy().show();
            ws.applyBgFgClassification( interval, instancesKey );

            /*
            ws.recomputeLabelInstances = true;
            ws.featureSettings.log2 = true;
            ws.updateLabelInstancesAndMetadata( "updated" );
            ws.saveInstances( "updated",
                    "/Users/de.embl.cba.cats.weka/Documents/ashna/instances/",
                    "bin_1_2_3_4--log2--001.ARFF");

             */

        }


    }

}
