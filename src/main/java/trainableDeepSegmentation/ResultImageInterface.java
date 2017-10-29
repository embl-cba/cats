package trainableDeepSegmentation;

import ij.ImagePlus;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;

import java.util.ArrayList;

public interface ResultImageInterface {

    void saveAsImarisSeparateChannels( String directory,
                                       ArrayList< Boolean > classesToSave );

    void setSliceInterval( FinalInterval sliceInterval,
                           byte[] classifiedSlice );

    ImageProcessor getSlice( int z, int t);

    class Setter{};

}
