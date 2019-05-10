package playground;

import ij.IJ;

public class FreeMemory
{
	public static void main( String[] args )
	{
		final double maxMemory = IJ.maxMemory() / ( 1024 * 1024 * 1024 );
		final long freeMemory = Runtime.getRuntime().freeMemory();
	}
}
