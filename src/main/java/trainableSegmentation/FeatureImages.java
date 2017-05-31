package trainableSegmentation;

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
 * Authors: Ignacio Arganda-Carreras (iarganda@mit.edu), Verena Kaynig (verena.kaynig@inf.ethz.ch),
 *          Albert Cardona (acardona@ini.phys.ethz.ch)
 */

import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
import weka.core.Attribute;
import weka.core.DenseInstance;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This class stores the feature stacks of a set of input slices.
 * It can be used so for 2D stacks or as the container of 3D features (by
 * using a feature stack per section). 
 * 
 * @author Ignacio Arganda-Carreras (iarganda@mit.edu)
 *
 */
public interface FeatureImages
{


	/**
	 * Update all feature stacks in the list (multi-thread fashion)
	 */
	public boolean updateFeaturesMT();

	/**
	 * Get the number of feature stacks
	 * 
	 * @return number of feature stacks stored in the array
	 */
	public int getNumFeatures();

	/**
	 * Create instance (feature vector) of a specific coordinate
	 *
	 * @param x x- axis coordinate
	 * @param y y- axis coordinate
	 * @param classValue class value to be assigned
	 * @return corresponding instance
	 */
	DenseInstance createInstance(
			int x,
			int y,
			int z,
			int t,
			int classValue );

	void setInstance(
			int x,
			int y,
			int z,
			int t,
			int classValue,
			final ReusableDenseInstance ins,
			final double[] auxArray );


	void setFeatureSlice(int slice, int frame, double[][][] featureSlice);

	void setMinimumSigma( double sigma );

	void setMaximumSigma( double sigma );

	boolean saveStackAsTiff( String filePath );

	/**
	 * Reset the reference index (used when the are 
	 * changes in the features)
	 */
	void resetReference();
	
	/**
	 * Set the reference index (used when the are 
	 * changes in the features)
	 */
	public void setReference( int index );
	
	/**
	 * Shut down the executor service
	 */
	public void shutDownNow();

	public ArrayList<Attribute> getFeatureNamesAsAttributes();

	/**
	 * Check if the array has not been yet initialized
	 * 
	 * @return true if the array has been initialized
	 */
	public boolean isEmpty();

	/**
	 * Get a specific label of the reference stack
	 * @param index slice index (&gt;=1)
	 * @return label name
	 */
	public String getLabel(int index);
	
	/**
	 * Get the features enabled for the reference stack
	 * @return features to be calculated on each stack
	 */
	public boolean[] getEnabledFeatures();

	public int getReferenceSliceIndex();

	public double getFeatureValue( int x, int y, int z, int t, int i );

	public int getWidth();

	public int getHeight();

	public int getDepth();

	public int getSize();

}

	
