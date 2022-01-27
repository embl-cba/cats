package run;

import ij.IJ;
import net.imagej.ImageJ;

public class RunImageJ
{
    public static void main( final String... args )
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        IJ.openImage("https://imagej.net/images/boats.gif").show();
    }
}
