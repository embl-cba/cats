package de.embl.cba.cats.results;

import ij.ImagePlus;

public class ProximityFilterSettings
{
    boolean doSpatialProximityFiltering = false;
    int distanceInPixelsAfterBinning;
    int referenceClassId;
    ImagePlus dilatedBinaryReferenceMask;
}
