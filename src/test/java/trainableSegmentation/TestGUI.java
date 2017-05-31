package trainableSegmentation;

import fiji.Debug;

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

		Debug.run("Trainable Deep Weka Segmentation", "open=/Users/tischi/Desktop/EM--crop2.tif");
		//Debug.run("Trainable Deep Weka Segmentation", "open=/Users/tischi/Desktop/example-data/weka-test-3d.tif");

	}

}
