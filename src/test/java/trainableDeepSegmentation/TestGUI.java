package trainableDeepSegmentation;

import bigDataTools.dataStreamingTools.DataStreamingTools;
import bigDataTools.utils.ImageDataInfo;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

/**
 * Simple test to launch the plugin
 * 
 * @author Ignacio Arganda-Carreras
 *
 */
public class TestGUI {
	
	/**
	 * Main method to test and debug the Trainable Weka
	 * Segmentation GUI
	 *  
	 * @param args
	 */
	public static void main( final String[] args )
	{
		new ImageJ();

		// EM
		/*
		DataStreamingTools dst = new DataStreamingTools();
		String namingPattern = "classified--C<C00-00>--T<T00000-00000>--Z<Z00001-01162>.tif";
		bigDataTools.utils.ImageDataInfo imageDataInfo = new ImageDataInfo();
		imageDataInfo.channelFolders = new String[] {""};
		imageDataInfo.bitDepth = 8;
		int nIOthreads = 3;

		String directory = "/Users/tischi/Desktop/example-data/EM/result/";
		dst.openFromDirectory(
				directory,
				namingPattern,
				"None",
				"data",
				imageDataInfo,
				nIOthreads,
				true);
		IJ.wait(1000);
		IJ.getImage().setTitle("classification_result");

		dst.openFromDirectory(
				"/Users/tischi/Desktop/example-data/EM/",
				"None",
				".*--inverted.tif",
				"data",
				null,
				nIOthreads,
				true);
		*/


		/*
		DataStreamingTools dst = new DataStreamingTools();

		dst.openFromDirectory(
				"/Volumes/almf/tischer/Nils/bothChannels/ch0/",
				"None",
				".*.tif",
				"data",
				null,
				3,
				true,
				false);
				*/

		// FLY EYE
		/*
		IJ.open("/Users/tischi/Documents/imagej-courses/data/supervised_segmentation/scanningEM_flyEye.tif");
		ImagePlus imp = IJ.getImage();
		ImagePlus impClass = IJ.createImage("classification_result",imp.getWidth(),imp.getHeight(),1,8);
		impClass.show();
		imp.hide(); imp.show();

		IJ.wait(100);
		Weka_Deep_Segmentation weka_segmentation = new Weka_Deep_Segmentation();
		weka_segmentation.run("");
		*/

		/*
		DataStreamingTools dst = new DataStreamingTools();
		dst.openFromDirectory(
				"/Volumes/almfspim/tischi/tobias-primordium/stack8-right/",
				".*--C<c>--T<t>.tif",
				".*.tif",
				"data",
				null,
				3,
				true,
				false);*/
		//IJ.open("/Users/tischi/Desktop/mitosis.tif");
		//IJ.open( "/Users/tischi/Desktop/mri-stack-big-2d-movie.tif" );

		IJ.open( "/Users/tischi/Desktop/segmentation-challenges/brainiac2-mit-edu-SNEMI3D/train-input/train-input-clahe.tif" );
		//IJ.open( "/Users/tischi/Desktop/Nils.tif" );

		//IJ.open("/Users/tischi/Documents/imagej-courses/data/supervised_segmentation/scanningEM_flyEye.tif");
		IJ.wait(100);

		Weka_Deep_Segmentation weka_segmentation = new Weka_Deep_Segmentation();
		weka_segmentation.run("");


		/*
		Thread t1 = new Thread(new Runnable() {
			public void run() {
				byte nIOthreads = 3;
				dataStreamingTools.openFromDirectory("/Users/tischi/Documents/fiji-plugin-deep-segmentation/data",
						"None", ".*--invert.tif", "data", nIOthreads);
			}
		}); t1.start();
		*/


		/*
		ImagePlus imp = IJ.getImage();
		VirtualStackOfStacks vss = (VirtualStackOfStacks) imp.getStack();
		Region5D region5D = new Region5D();
		region5D.size = new Point3D(40.0,40.0,40.0);
		region5D.offset = new Point3D(0,0,0);
		region5D.t = 0;
		region5D.c = 0;
		region5D.subSampling = new Point3D(1,1,1);
		ImagePlus cube = vss.getDataCube(region5D, 0, 3);
		cube.show();
		*/

		//Debug.run("Trainable Deep Weka Segmentation", "");

		//Debug.run("Trainable Deep Weka Segmentation",
		//		"open=/Users/tischi/Desktop/EM--crop--registered--iso10nm--invert.tif");

	}

}
