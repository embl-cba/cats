package trainableSegmentation;

import bigDataTools.Region5D;
import bigDataTools.VirtualStackOfStacks.FastTiffDecoder;
import bigDataTools.VirtualStackOfStacks.FileInfoSer;
import bigDataTools.VirtualStackOfStacks.VirtualStackOfStacks;
import bigDataTools.dataStreamingTools.DataStreamingTools;
import fiji.Debug;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import javafx.geometry.Point3D;
import net.imglib2.exception.IncompatibleTypeException;
import trainableSegmentation.filters.HessianImgLib2;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.concurrent.ExecutionException;

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
		// Call the plugin with empty arguments (this
		// will pop up an Open dialog)		
		//Debug.run("Trainable Weka Segmentation 3D", "");

		//Debug.run("Trainable Deep Weka Segmentation", "open=/Users/tischi/Desktop/EM--crop2.tif");
		//Debug.run("Trainable Deep Weka Segmentation", "open=/Users/tischi/Desktop/example-data/weka-test-3d.tif");

		/*
		double a = 0.134445;
		byte b = (byte)(100*a);
		double f = 150;
		byte c = (byte)f;
		int g = 1;
		*/


		new ImageJ();


		/*
		FastTiffDecoder ftd = new FastTiffDecoder("/Users/tischi/Desktop/tmp3-out", "blobs.tif");
		FileInfoSer[] info = null;
		try
		{
			info = ftd.getTiffInfo();
		} catch (IOException e)
		{
			e.printStackTrace();
		}


		RandomAccessFile raf = null;
		try
		{
			raf = new RandomAccessFile("/Users/tischi/Desktop/tmp3-out/blobs.tif", "rw");
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}

		try {
			raf.seek(info[0].offset); // Go to byte at offset position 5.
			int a1 = raf.read() & 0xFF; // Write byte 70 (overwrites original byte at this offset).
			int a2 = raf.read() & 0xFF; // Write byte 70 (overwrites original byte at this offset).
			raf.writeByte(11);
			raf.writeByte(11);
			raf.close();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
		*/

		/*
		ImagePlus imp = IJ.openImage("http://imagej.nih.gov/ij/images/mri-stack.zip");
		imp.show();

		HessianImgLib2 hessianImgLib2 = new HessianImgLib2();
		try
		{
			hessianImgLib2.run( imp, 1);
		} catch (IncompatibleTypeException e)
		{
			e.printStackTrace();
		} catch (InterruptedException e)
		{
			e.printStackTrace();
		} catch (ExecutionException e)
		{
			e.printStackTrace();
		}*/




		final DataStreamingTools dataStreamingTools = new DataStreamingTools();
		byte nIOthreads = 3;
		dataStreamingTools.openFromDirectory("/Users/tischi/Desktop/example-data",
				"None", ".*--inverted.tif", "data", nIOthreads);
		IJ.wait(500);
		Weka_Segmentation weka_segmentation = new Weka_Segmentation();
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
