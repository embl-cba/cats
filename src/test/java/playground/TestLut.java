package playground;

import ij.IJ;
import ij.ImagePlus;
import ij.process.LUT;
import net.imagej.ImageJ;

import java.awt.*;

public class TestLut
{

	public static void main( String[] args )
	{

		new ImageJ().ui().showUI();

		final ImagePlus imagePlus =
				IJ.openImage("http://wsr.imagej.net/images/blobs.gif");

		//imagePlus.show();
		imagePlus.setLut( createClassLabelLUT( ) );
		imagePlus.show();

	}


	public static LUT createClassLabelLUT( )
	{
		final byte[] red = new byte[ 256 ];
		final byte[] green = new byte[ 256 ];
		final byte[] blue = new byte[ 256 ];

		for ( int iClass = 0; iClass < classColors.length; iClass++ )
		{
			red[ iClass + 1 ] = ( byte ) ( classColors[ iClass ].getRed() );
			green[ iClass + 1 ] = ( byte ) ( classColors[ iClass ].getGreen() );
			blue[ iClass + 1 ] = ( byte ) ( classColors[ iClass ].getBlue() );
		}

		return new LUT( red, green, blue );

	}


	public static Color[] classColors = new Color[]{
			Color.gray,
			Color.green,
			Color.red,
			Color.blue,
			Color.cyan,
			Color.pink,
			Color.white,
			Color.magenta,
			Color.orange,
			Color.black,
			Color.yellow,
			Color.gray,
			Color.green,
			Color.red,
			Color.blue,
			Color.cyan,
			Color.pink,
			Color.white,
			Color.magenta,
			Color.orange,
			Color.black
	};
}
