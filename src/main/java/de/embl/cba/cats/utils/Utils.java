package de.embl.cba.cats.utils;

/**
 *
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * Authors: Ignacio Arganda-Carreras (iarganda@mit.edu)
 */

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.filter.GaussianBlur;
import ij.process.*;
import org.scijava.vecmath.Point3f;
import util.FindConnectedRegions;
import util.FindConnectedRegions.Results;

import java.util.ArrayList;

import static de.embl.cba.cats.utils.IntervalUtils.*;

/**
 * This class implements useful methods for the Weka Segmentation library.
 */
public final class Utils {

    private Utils() throws InstantiationException {
        throw new InstantiationException("This class is not created for instantiation");
    }

	public static String removeExtension( String s ) {

		String separator = System.getProperty("file.separator");
		String filename;

		// Remove the path upto the filename.
		int lastSeparatorIndex = s.lastIndexOf(separator);
		if (lastSeparatorIndex == -1) {
			filename = s;
		} else {
			filename = s.substring(lastSeparatorIndex + 1);
		}

		// Remove the extension.
		int extensionIndex = filename.lastIndexOf(".");
		if (extensionIndex == -1)
			return filename;

		return filename.substring(0, extensionIndex);
	}


	/**
	 * Connected components based on Find Connected Regions (from Mark Longair)
	 * @param im input image
	 * @param adjacency number of neighbors to check (4, 8...)
	 * @return list of images per region, all-regions image and regions info
	 */
	public static Results connectedComponents(final ImagePlus im, final int adjacency)
	{
		if( adjacency != 4 && adjacency != 8 )
			return null;

		final boolean diagonal = adjacency == 8 ? true : false;

		FindConnectedRegions fcr = new FindConnectedRegions();
		try {
			final Results r = fcr.run( im,
				 diagonal,
				 false,
				 true,
				 false,
				 false,
				 false,
				 false,
				 0,
				 1,
				 -1,
				 true /* noUI */ );
			return r;

		} catch( IllegalArgumentException iae ) {
			IJ.error(""+iae);
			return null;
		}

	}


	/*
	public static void filterSmallObjects3D(
			ImagePlus imp,
			int[] sizesMinMax)
	{
		Segment3DImageModified segment3DImage = new Segment3DImageModified(imp, (float)0.9, (float)255.0 );
		segment3DImage.setMinSizeObject(sizesMinMax[0]);
		segment3DImage.segment();
		ImageStack filteredStack = segment3DImage.getBinaryObjectsStack(1);
		imp.setStack(filteredStack);
	}*/



	/**
	 * Connected components based on Find Connected Regions (from Mark Longair)
	 * @param im input image
	 * @param adjacency number of neighbors to check (4, 8...)
	 * @param minSize minimum size (in pixels) of the components
	 * @return list of images per region, all-regions image and regions info
	 */
	public static Results connectedComponents(
			final ImagePlus im, 
			final int adjacency,
			final int minSize)
	{
		if( adjacency != 4 && adjacency != 8 )
			return null;

		final boolean diagonal = adjacency == 8 ? true : false;

		FindConnectedRegions fcr = new FindConnectedRegions();
		try {
			final Results r = fcr.run( im,
				 diagonal,
				 false,
				 true,
				 false,
				 false,
				 false,
				 false,
				 0,
				 minSize,
				 -1,
				 true /* noUI */ );
			return r;

		} catch( IllegalArgumentException iae ) {
			IJ.error(""+iae);
			return null;
		}

	}
	


	/**
	 * Applies the morphological erode operator.
	 * Reused from the SIOX package. 
	 *
	 * @param fp probability image to be eroded.
	 */
	public static void erode(FloatProcessor fp)
	{
		final float[] cm = (float[]) fp.getPixels();
		final int xres = fp.getWidth();
		final int yres = fp.getHeight();
		
		
		for (int y=0; y<yres; y++) {
			for (int x=0; x<xres-1; x++) {
				final int idx=(y*xres)+x;
				cm[idx]=Math.min(cm[idx], cm[idx+1]);
			}
		}
		for (int y=0; y<yres; y++) {
			for (int x=xres-1; x>=1; x--) {
				final int idx=(y*xres)+x;
				cm[idx]=Math.min(cm[idx-1], cm[idx]);
			}
		}
		for (int y=0; y<yres-1; y++) {
			for (int x=0; x<xres; x++) {
				final int idx=(y*xres)+x;
				cm[idx]=Math.min(cm[idx], cm[((y+1)*xres)+x]);
			}
		}
		for (int y=yres-1; y>=1; y--) {
			for (int x=0; x<xres; x++) {
				final int idx=(y*xres)+x;
				cm[idx]=Math.min(cm[((y-1)*xres)+x], cm[idx]);
			}
		}
	}
	
	/**
	 * Applies the morphological dilate operator.
	 * Reused from the SIOX package. 
	 * Can be used to close small holes in the probability image.
	 *
	 * @param fp probability image to be dilated
	 */
	public static void dilate(FloatProcessor fp)	
	{
		final float[] cm = (float[]) fp.getPixels();
		final int xres = fp.getWidth();
		final int yres = fp.getHeight();

		for (int y=0; y<yres; y++) {
			for (int x=0; x<xres-1; x++) {
				final int idx=(y*xres)+x;
				cm[idx]=Math.max(cm[idx], cm[idx+1]);
			}
		}
		for (int y=0; y<yres; y++) {
			for (int x=xres-1; x>=1; x--) {
				final int idx=(y*xres)+x;
				cm[idx]=Math.max(cm[idx-1], cm[idx]);
			}
		}
		for (int y=0; y<yres-1; y++) {
			for (int x=0; x<xres; x++) {
				final int idx=(y*xres)+x;
				cm[idx]=Math.max(cm[idx], cm[((y+1)*xres)+x]);
			}
		}
		for (int y=yres-1; y>=1; y--) {
			for (int x=0; x<xres; x++) {
				final int idx=(y*xres)+x;
				cm[idx]=Math.max(cm[((y-1)*xres)+x], cm[idx]);
			}
		}
	}

	/**
	 * Apply binary threshold to input image
	 * @param ip input image
	 * @param thresholdValue threshold value (all pixel above that value will be set to 255, the rest to 0)
	 * @return binary resultImagePlus
	 */
	public static ByteProcessor threshold (ImageProcessor ip, double thresholdValue)
	{
		final ByteProcessor result = new ByteProcessor(ip.getWidth(), ip.getHeight());
		
		for(int x=0; x<ip.getWidth(); x++)
			for(int y=0; y<ip.getHeight(); y++)
			{
				if( ip.getPixelValue(x, y) > thresholdValue)
					result.putPixelValue(x, y, 255);
				else
					result.putPixelValue(x, y, 0);
			}
						
		return result;
	}
	
	/**
	 * Post-process probability image to getInstancesAndMetadata more reasonable objects3DPopulation
	 * at a certain threshold
	 * 
	 * @param probabilityMap probability image
	 * @param smoothIterations number of smoothing iterations
	 * @param threshold threshold to use
	 * @param minSize minimum object size (in pixels)
	 * @param binarize flag to binarize results
	 */
	public static void postProcess( 
			FloatProcessor probabilityMap, 
			int smoothIterations,
			double threshold,
			int minSize,
			boolean binarize)
	{
		//smooth( probabilityMap, 0.33f, 0.33f, 0.33f );
		GaussianBlur gb = new GaussianBlur();
		gb.blur(probabilityMap, 2);
		
		normalize01( probabilityMap );
		erode( probabilityMap );
		
		filterSmallObjectsAndHoles(probabilityMap, threshold, minSize);
		
		for (int i=0; i<smoothIterations; i++) 
			//smooth( probabilityMap, 0.33f, 0.33f, 0.33f );
			gb.blur(probabilityMap, 2);
		normalize01( probabilityMap );
		
		filterSmallObjectsAndHoles(probabilityMap, threshold, minSize);
		
		if( binarize )
		{
			float[] pixels = (float[]) probabilityMap.getPixels();
			for(int i=0; i<pixels.length; i++)
				if( pixels[ i ] > threshold )
					pixels[ i ] = 1.0f;
				else
					pixels[ i ] = 0.0f;
			
		}	
		
		dilate( probabilityMap );
		normalize01(probabilityMap);
	}

	/**
	 * Normalize float image so the pixel are between 0 and 1
	 * @param fp input image
	 */
	public static void normalize01( FloatProcessor fp )
	{
		fp.resetMinAndMax();
		double max = fp.getMax();
		double min = fp.getMin();				
		
		double scale = max>min?1.0/(max-min):1.0;
		int size = fp.getWidth()*fp.getHeight();
		float[] pixels = (float[])fp.getPixels();
		double v;
		for (int i=0; i<size; i++) {
			v = pixels[i] - min;
			if (v<0.0) v = 0.0;
			v *= scale;
			if (v>1.0) v = 1.0;
			pixels[i] = (float)v;
		}
	}
	
	/**
	 * Filter small objects3DPopulation and holes at a specific threshold value
	 * @param probabilityMap probability image
	 * @param thresholdValue threshold to use
	 * @param minSize minimum size of the objects3DPopulation (in pixels)
	 */
	public static void filterSmallObjectsAndHoles(
			FloatProcessor probabilityMap,
			double thresholdValue, 
			int minSize) 
	{
		// apply threshold 
		ByteProcessor thresholded = threshold(probabilityMap, thresholdValue);
		
		// Calculate connected components above the minimum size
		Results res = connectedComponents( new ImagePlus("thresholded", thresholded), 4, minSize);
		
		//res.allRegions.show();
		
		// Binarize components image (after removing small objects3DPopulation)
		ByteProcessor th = threshold( res.allRegions.getProcessor(), 0.5 );
		
		// Localize small objects3DPopulation by the difference with the original thresholded image
		ByteProcessor th2 = (ByteProcessor) th.duplicate();
		th2.copyBits(thresholded, 0, 0, Blitter.DIFFERENCE);				
		
		byte[] th2pixels = (byte[])th2.getPixels();
		final float[] probPixels = (float[])probabilityMap.getPixels();
				
		// Set those pixels to background in the probability image
		for(int i=0; i<th2pixels.length; i++)
		{
			if( th2pixels[ i ] != 0)
				probPixels[ i ] = 0;
		}
				
		// Localize holes by the removing them first from the image
		// without small objects3DPopulation and then looking at the difference
		th2 = (ByteProcessor) th.duplicate();
		
		// Fill holes in the thresholded components image
		fill( th2, 255, 0 );
						
		th2.copyBits(th, 0, 0, Blitter.DIFFERENCE);
		th2pixels = (byte[])th2.getPixels();
										
		// Set those pixels to foreground in the probability image
		for(int i=0; i<th2pixels.length; i++)
		{
			if( th2pixels[ i ] != 0)
				probPixels[ i ] = 1;
		}
		
	}
	
	
	// Binary fill by Gabriel Landini, G.Landini at bham.ac.uk
	// 21/May/2008
	public static void fill(ImageProcessor ip, int foreground, int background) 
	{
		int width = ip.getWidth();
		int height = ip.getHeight();
		FloodFiller ff = new FloodFiller(ip);
		ip.setColor(127);
		for (int y=0; y<height; y++) {
			if (ip.getPixel(0,y)==background) ff.fill(0, y);
			if (ip.getPixel(width-1,y)==background) ff.fill(width-1, y);
		}
		for (int x=0; x<width; x++){
			if (ip.getPixel(x,0)==background) ff.fill(x, 0);
			if (ip.getPixel(x,height-1)==background) ff.fill(x, height-1);
		}
		byte[] pixels = (byte[])ip.getPixels();
		int n = width*height;
		for (int i=0; i<n; i++) {
			if (pixels[i]==127)
				pixels[i] = (byte)background;
			else
				pixels[i] = (byte)foreground;
		}
	}
	
	/**
	 * Get the binary class coordinates from a label image (2D image or stack)
	 * 
	 * @param labelImage labels (they can be in any format, black = 0)
	 * @param mask binary mask to select the pixels to be extracted
	 * @return array with the two lists (black and white) of sample coordinates
	 */
	public static ArrayList< Point3f >[] getClassCoordinates( 
			ImagePlus labelImage,
			ImagePlus mask)
	{
		final ArrayList< Point3f >[] classPoints = new ArrayList[2];
		classPoints[ 0 ] = new ArrayList< Point3f >();
		classPoints[ 1 ] = new ArrayList< Point3f >();
		
		final int width = labelImage.getWidth();
		final int height = labelImage.getHeight();
		final int size = labelImage.getImageStackSize();
		
		if( null != mask )
		{					
			for(int slice = 1; slice <= size; slice ++)
			{
				final float[] labelsPix = (float[]) labelImage.getImageStack().getProcessor( slice ).convertToFloat().getPixels();
				final float[] maskPix = (float[]) mask.getImageStack().getProcessor( slice ).convertToFloat().getPixels();
				
				for(int x = 0; x < width; x++)
					for( int y = 0; y < height; y++ )
						if( maskPix[ x + y * width] > 0 )
						{
							if( labelsPix[ x + y * width] != 0)				
								classPoints[ 1 ].add( new Point3f( new float[]{ x, y, slice-1}) );					
							else				
								classPoints[ 0 ].add( new Point3f( new float[]{ x, y, slice-1}) );
						}
			}
		}
		else
		{
			for(int slice = 1; slice <= size; slice ++)
			{
				final float[] labelsPix = (float[]) labelImage.getImageStack().getProcessor( slice ).convertToFloat().getPixels();
				
				for(int x = 0; x < width; x++)
					for( int y = 0; y < height; y++ )					
							if( labelsPix[ x + y * width] != 0)				
								classPoints[ 1 ].add( new Point3f( new float[]{ x, y, slice-1}) );					
							else				
								classPoints[ 0 ].add( new Point3f( new float[]{ x, y, slice-1}) );
					
			}
		}
		return classPoints;
	}

	public static ImagePlus[] maxPool( 
			ImagePlus input,
			ImagePlus label,
			int sizeX,
			int sizeY)
	{
		final int maxPoolWidth = input.getWidth() / sizeX;
		final int maxPoolHeight = input.getHeight() / sizeY;
		
		final int inputWidth = input.getWidth();
		final int inputHeight = input.getHeight();
		
		ImageStack isMaxPoolInput = new ImageStack(maxPoolWidth, maxPoolHeight);
		ImageStack isMaxPoolLabel = new ImageStack(maxPoolWidth, maxPoolHeight);
		
		ImagePlus[] maxPool = new ImagePlus[ 2 ];
		
		for(int slice = 1; slice <= input.getImageStackSize(); slice ++)
		{
			IJ.log("Processing slice " + slice + "...");
			
			double[] inputPix = new double [ maxPoolWidth * maxPoolHeight ];
			byte[] labelPix = new byte [ maxPoolWidth * maxPoolHeight ];
				
			for(int y=0, pos2 = 0; y<inputHeight; y += sizeY)
				for(int x=0; x<inputWidth; x += sizeX)				
				{
					double max = 0;										
					
					for(int x2=0; x2<sizeX; x2++)
						for(int y2=0; y2<sizeY; y2++)
						{
							final int pos = (y2 + y) * inputWidth + x2 + x;
							
							double val = ((float[]) input.getImageStack().getProcessor( slice ).getPixels())[pos];
							
							if (val > max)
							{								
								inputPix[ pos2 ] = val;
								labelPix[ pos2 ] = ((byte[]) label.getImageStack().getProcessor( slice ).getPixels())[ pos ];
							}							
						}
					pos2++;
				}
			
			isMaxPoolInput.addSlice( new FloatProcessor( maxPoolWidth, maxPoolHeight, inputPix));
			isMaxPoolLabel.addSlice( new ByteProcessor( maxPoolWidth, maxPoolHeight, labelPix, null ));
			
		}
		
		maxPool[ 0 ] = new ImagePlus("Input", isMaxPoolInput );
		maxPool[ 1 ] = new ImagePlus("Labels", isMaxPoolLabel );
		
		return maxPool;
	}
	
	public static ImagePlus[] maxPoolNoReduction( 
			ImagePlus input,
			ImagePlus label,
			int sizeX,
			int sizeY)
	{		
		final int width = input.getWidth();
		final int height = input.getHeight();
		
		ImageStack isMaxPoolInput = new ImageStack(width, height);
		ImageStack isMaxPoolLabel = new ImageStack(width, height);
		
		ImagePlus[] maxPool = new ImagePlus[ 2 ];
		
		for(int slice = 1; slice <= input.getImageStackSize(); slice ++)
		{
			IJ.log("Processing slice " + slice + "...");
			
			double[] inputPix = new double [ width * height ];
			byte[] labelPix = new byte [ width * height ];
				
			for(int y=0; y<height; y += sizeY)
				for(int x=0; x<width; x += sizeX)				
				{
					double max = 0;										
					double maxVal = 0;
					byte maxLabel = 0;
					
					for(int x2=0; x2<sizeX; x2++)
						for(int y2=0; y2<sizeY; y2++)
						{
							final int pos = (y2 + y) * width + x2 + x;
							
							double val = ((float[]) input.getImageStack().getProcessor( slice ).getPixels())[pos];
							
							if (val > max)
							{								
								maxVal = val;
								maxLabel = ((byte[]) label.getImageStack().getProcessor( slice ).getPixels())[ pos ];
							}							
						}
					
					for(int x2=0; x2<sizeX; x2++)
						for(int y2=0; y2<sizeY; y2++)
						{
							final int pos = (y2 + y) * width + x2 + x;
							inputPix [ pos ] = maxVal;
							labelPix [ pos ] = maxLabel;
						}
			
				}
			
			isMaxPoolInput.addSlice( new FloatProcessor( width, height, inputPix));
			isMaxPoolLabel.addSlice( new ByteProcessor( width, height, labelPix, null ));
			
		}
		
		maxPool[ 0 ] = new ImagePlus("Input", isMaxPoolInput );
		maxPool[ 1 ] = new ImagePlus("Labels", isMaxPoolLabel );
		
		return maxPool;
	}

    public static String getSimpleString( String inputString )
    {
        String dataSetName = inputString.toString().replace( ".*", "" ).trim();
        String trimmedDataSetName = dataSetName.replaceAll("\\(.*?\\)" , "" );
        trimmedDataSetName = trimmedDataSetName.replace("\\" , "" );
        trimmedDataSetName = trimmedDataSetName.replace(".tif" , "" );
		trimmedDataSetName = trimmedDataSetName.replaceAll( "--$", "" );
		trimmedDataSetName = trimmedDataSetName.replaceAll( "^--", "" );

		return trimmedDataSetName;
    }

	public static ImagePlus create8bitImagePlus( long[] dimensions )
	{
		ImageStack stack = ImageStack.create(
				(int) dimensions[ X ],
				(int) dimensions[ Y ],
				(int) (dimensions[ Z ] * dimensions[ T ]),
				8);

		ImagePlus imp = new ImagePlus( "results", stack  );

		imp.setDimensions(
				1,
				(int) dimensions[ Z ],
				(int) dimensions[ T ]);

		imp.setOpenAsHyperStack(true);
		imp.setTitle("Objects");

		return imp;
	}
}
