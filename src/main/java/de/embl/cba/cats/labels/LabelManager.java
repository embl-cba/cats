package de.embl.cba.cats.labels;

import ij.gui.PolygonRoi;
import ij.gui.Roi;

import java.awt.*;
import java.util.ArrayList;

import static de.embl.cba.cats.labels.LabelUtils.getNumClassesInLabels;

public class LabelManager
{
	private ArrayList< Label > labels;

	public LabelManager( )
	{
		this.labels = new ArrayList< > () ;
	}

	public int getNumLabels()
	{
		if ( labels == null )
		{
			return 0;
		}
		else
		{
			return labels.size();
		}
	}


	public void setLabels( ArrayList< Label > labels )
	{
		this.labels = labels;
	}

	public ArrayList< Label > getLabels()
	{
		return labels;
	}


	public void addLabel( Label label )
	{
		labels.add( label );
	}

	public void removeLabel( int classNum, int z, int t, int index )
	{
		int i = 0;
		for ( int iLabel = 0; iLabel < labels.size(); iLabel++)
		{
			Label label = labels.get(iLabel);
			if (( label.z == z)
					&& ( label.t == t)
					&& ( label.classNum == classNum))
			{
				if ( index == i++ ) // i'th label for this z,t,class
				{
					labels.remove( iLabel );
					return;
				}
			}
		}
	}

	public ArrayList< Roi > getLabelsAsRois( int classNum, int z, int t )
	{
		ArrayList< Roi > rois = new ArrayList<>();

		for ( Label label : labels )
		{
			if (( label.z == z)
					&& ( label.t == t)
					&& ( label.classNum == classNum))
			{
				float[] x = new float[ label.points.length];
				float[] y = new float[ label.points.length];
				for ( int iPoint = 0; iPoint < label.points.length; iPoint++)
				{
					x[iPoint] = (float) label.points[iPoint].getX();
					y[iPoint] = (float) label.points[iPoint].getY();
				}
				Roi roi = new PolygonRoi(x, y, PolygonRoi.FREELINE);
				roi.setStrokeWidth((double) label.strokeWidth);
				rois.add(roi);
			}

		}

		return rois;
	}

	public double getAverageNumberOfPointsPerLabel( )
	{
		double totalLength = 0;

		for ( Label label : labels )
		{
			totalLength += label.points.length;
		}

		return totalLength / getNumLabels();
	}

	public int[] getNumLabelsPerClass( )
	{
		int[] numLabelsPerClass = new int[ getNumClassesInLabels( labels ) ];

		for ( de.embl.cba.cats.labels.Label label : labels )
		{
			numLabelsPerClass[ label.classNum ]++;
		}

		return ( numLabelsPerClass );
	}


	public boolean areAnyLabelInstanceValuesCurrentlyBeingComputed()
	{
		for ( Label label : labels )
		{
			if ( label.instanceValuesAreCurrentlyBeingComputed ) return true;
		}

		return false;
	}
	

}
