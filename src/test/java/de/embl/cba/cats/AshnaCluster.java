package de.embl.cba.cats;

import de.embl.cba.bigDataTools.dataStreamingTools.DataStreamingTools;
import de.embl.cba.cats.utils.IOUtils;
import ij.IJ;
import ij.ImagePlus;

public class AshnaCluster
{
    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        ImagePlus imp = IOUtils.openImageWithLazyLoadingTools( "/Volumes/cba/tischer/projects/ashna-spim/reg2-3x3", "None", ".*--C.*", "Resolution 0/Data" );
        imp.show();

        de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin weka_segmentation = new de.embl.cba.cats.ui.DeepSegmentationIJ1Plugin();
        weka_segmentation.run("");
    }
}
