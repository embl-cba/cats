package de.embl.cba.trainableDeepSegmentation.results;

import ij.ImagePlus;

public class ProximityFilterSettings
{
    boolean doSpatialProximityFiltering;
    int distanceInPixelsAfterBinning;
    int referenceClassId;
    ImagePlus dilatedBinaryReferenceMask;
}
