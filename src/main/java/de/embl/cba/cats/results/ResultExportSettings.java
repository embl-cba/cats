package de.embl.cba.cats.results;

import de.embl.cba.utils.logging.Logger;
import ij.ImagePlus;

import java.util.ArrayList;

public class ResultExportSettings
{
    public static final String IMARIS_STACKS = "Save as probabilities - Imaris";
    public static final String SAVE_AS_CLASS_PROBABILITY_TIFF_STACKS = "Save as probabilities - Tiff";
    public static final String SAVE_AS_CLASS_LABEL_MASK_TIFF_STACKS = "Save as label mask - Tiff";
    public static final String CLASS_PROBABILITIES_TIFF_SLICES = "Save as probabilities - Tiff slices";
    public static final String SHOW_AS_PROBABILITIES = "Show as probabilities";
    public static final String SHOW_AS_LABEL_MASKS = "Show as label masks";

    public static final String GET_AS_IMAGEPLUS_ARRAYLIST = "Get as ImagePlus Array";
	public String directory;
    public String exportNamesPrefix = "";
    public ArrayList< Boolean > classesToBeExported;
    public int[] binning;
    public String exportType;
    public ProximityFilterSettings proximityFilterSettings = new ProximityFilterSettings();
    public ImagePlus inputImagePlus;
    public ImagePlus resultImagePlus;
    public ResultImage resultImage;
    public boolean saveRawData;
    public ArrayList< String > classNames;
    public Logger logger;
    public int classLutWidth;
    public int[] timePointsFirstLast;

}
