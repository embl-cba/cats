package trainableDeepSegmentation;

import ij.IJ;
import ij.ImagePlus;
import trainableDeepSegmentation.ui.Weka_Deep_Segmentation;

public class MriStack
{

	public final static String TEST_RESOURCES = "/Users/tischer/Documents/fiji-plugin-deepSegmentation/src/test/resources/";

	public static void main( final String[] args )
	{
		final net.imagej.ImageJ ij = new net.imagej.ImageJ();
		ij.ui().showUI();

		ImagePlus inputImagePlus = IJ.openImage(TEST_RESOURCES + "mri-stack.zip" );
		inputImagePlus.show();

		IJ.wait(100);

		Weka_Deep_Segmentation weka_segmentation = new Weka_Deep_Segmentation();
		weka_segmentation.run("");

	}

}