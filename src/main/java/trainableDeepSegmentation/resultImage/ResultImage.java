package trainableDeepSegmentation.resultImage;

import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;

import java.util.ArrayList;

public interface ResultImage {

    void saveAsSeparateImarisChannels( String directory,
                                       ArrayList< Boolean > saveClass,
                                       int[] binning );

    ImageProcessor getSlice( int slice, int frame );

    ResultImageFrameSetter getFrameSetter( FinalInterval interval );


}
