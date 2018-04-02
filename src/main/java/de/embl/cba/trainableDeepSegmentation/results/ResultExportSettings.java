package de.embl.cba.trainableDeepSegmentation.results;

import de.embl.cba.utils.logging.Logger;
import ij.ImagePlus;

import java.util.ArrayList;

public class ResultExportSettings
{
    public String directory;
    public String exportNamesPrefix;
    public ArrayList< Boolean > classesToBeExported;
    public int[] binning;
    public String exportType;
    public ProximityFilterSettings proximityFilterSettings = new ProximityFilterSettings();
    public ImagePlus rawData;
    public ImagePlus result;
    public ResultImage resultImage;
    public boolean saveRawData;
    public ArrayList< String > classNames;
    public Logger logger;
    public int classLutWidth;

}
