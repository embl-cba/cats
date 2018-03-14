package de.embl.cba.trainableDeepSegmentation.results;

import ij.ImagePlus;
import ij.process.ImageProcessor;
import net.imglib2.FinalInterval;

import java.util.ArrayList;

public interface ResultImage {

    void saveClassesAsFiles( String directory, String fileNamePrefix, ArrayList< Boolean > classesToBeSaved, int[] binning, String fileType );

    void showClassesAsImages( String imageNamePrefix, ArrayList< Boolean > classesToBeShown, int[] binning );

    ImageProcessor getSlice( int slice, int frame );

    ResultImageFrameSetter getFrameSetter( FinalInterval interval );

    default int[] getClassAndProbability( int x, int y, int z, int t )
    {
        ImageProcessor ip = getSlice( z + 1, t + 1  );
        byte result = (byte) ip.get( x, y );

        int[] classAndProbability = new int[2];

        classAndProbability[0] = ( result - 1 ) / getProbabilityRange();
        classAndProbability[1] = result - classAndProbability[0] * getProbabilityRange();

        return classAndProbability;
    }

    int getProbabilityRange();

    ImagePlus getDataCubeCopy( FinalInterval interval );

    ImagePlus getWholeImageCopy();
}
