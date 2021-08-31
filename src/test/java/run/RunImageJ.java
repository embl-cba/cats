package run;

import net.imagej.ImageJ;

public class RunImageJ
{
    public static void main( final String... args )
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();
    }
}
