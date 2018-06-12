package de.embl.cba.trainableDeepSegmentation.postprocessing;

import ij.ImagePlus;
import mcib3d.geom.Objects3DPopulation;

public class SegmentedObjects
{
    public Objects3DPopulation objects3DPopulation;
    public ImagePlus labelMask;
    public String name;
    public int t = 0;
}
