package de.embl.cba.cats.classification;

import de.embl.cba.cats.CATS;
import de.embl.cba.cats.utils.IntervalUtils;
import de.embl.cba.utils.logging.Logger;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import net.imglib2.FinalInterval;

import java.awt.*;

import static de.embl.cba.cats.utils.IntervalUtils.*;

public class ClassificationRangeUtils
{
    public static final String SELECTION_PM10Z = "Selected roi +- 10 slices";
    public static final String WHOLE_DATA_SET = "Whole data set";
    static Logger logger = CATS.logger;


    public static FinalInterval getIntervalFromRoi( ImagePlus inputImage, String rangeString )
    {
        Roi roi = inputImage.getRoi();

        if ( roi == null || !( roi.getType() == Roi.RECTANGLE || roi.getType() == Roi.FREELINE) )
        {
            IJ.showMessage( "Please use ImageJ's rectangle or freeline tool to select a region." );
            return null;
        }

        long[] min = new long[ 5 ];
        long[] max = new long[ 5 ];

        ClassificationRangeUtils.setMinMax( inputImage, rangeString, roi, min, max );

        IntervalUtils.ensureToStayWithinBounds( inputImage, min, max );

        inputImage.killRoi();

        return new FinalInterval( min, max );
    }


    public static void setMinMax( ImagePlus inputImage, String rangeString, Roi roi, long[] min, long[] max )
    {
        Rectangle rectangle = roi.getBounds();

        min[ X ] = ( int ) rectangle.getX();
        max[ X ] = min[ 0 ] + ( int ) rectangle.getWidth() - 1;

        min[ Y ] = ( int ) rectangle.getY();
        max[ Y ] = min[ 1 ] + ( int ) rectangle.getHeight() - 1;

        min[ Z ] = max[ Z ] = inputImage.getZ() - 1;

        if ( rangeString.equals( SELECTION_PM10Z ))
        {
            min[ Z ] = inputImage.getZ() - 1 - 10;
            max[ Z ] = inputImage.getZ() - 1 + 10;
        }

        min[ T ] = max[ T ] = inputImage.getT() - 1;
        min[ C ] = max[ C ] = inputImage.getC() - 1;


        ClassificationRangeUtils.adaptZTtoUsersInput( inputImage, rangeString, min, max );
    }

    public static void adaptZTtoUsersInput( ImagePlus inputImage, String rangeString, long[] min, long[] max )
    {
        try
        {
            int[] range = de.embl.cba.bigDataTools.utils.Utils.delimitedStringToIntegerArray( rangeString, "," );

            if ( inputImage.getNFrames() == 1 )
            {
                min[ Z ] = range[ 0 ] - 1;
                max[ Z ] = range[ 1 ] - 1;
            }
            else if ( inputImage.getNSlices() == 1 )
            {
                min[ T ] = range[ 0 ] - 1;
                max[ T ] = range[ 1 ] - 1;
            }
            else
            {
                min[ Z ] = range[ 0 ] - 1;
                max[ Z ] = range[ 1 ] - 1;

                if ( range.length == 4 )
                {
                    min[ T ] = range[ 2 ] - 1;
                    max[ T ] = range[ 3 ] - 1;
                }
            }

            logger.info( "Using selected z and t range..." );

        }
        catch ( NumberFormatException e )
        {
            logger.info( "No (or invalid) z and t range selected." );
        }
    }
}
