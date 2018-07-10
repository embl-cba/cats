package de.embl.cba.cats;

import de.embl.cba.cats.ui.CATSPlugin;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.ImageJ;

import static de.embl.cba.cats.TestUtils.TEST_RESOURCES;

public class CatsUI
{

    public static void main(final String... args) throws Exception
    {

        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ImagePlus inputImagePlus = IJ.openImage(TEST_RESOURCES + "mri-stack.zip" );
        inputImagePlus.show();

        ij.command().run( CATSPlugin.class, true );

    }


}
