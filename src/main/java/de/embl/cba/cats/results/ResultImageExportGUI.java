package de.embl.cba.cats.results;

import de.embl.cba.bigDataTools.utils.Utils;
import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;

import java.util.ArrayList;

public abstract class ResultImageExportGUI
{

    public static void showExportGUI( String inputImageTitle,
                                      ResultImage resultImage,
                                      ImagePlus rawData,
                                      ArrayList< String > classNames )
    {

        String[] exportChoices = new String[]{ ResultExportSettings.SHOW_AS_SEPARATE_IMAGES, ResultExportSettings.SEPARATE_IMARIS, ResultExportSettings.SEPARATE_TIFF_FILES };

        GenericDialog gd = new GenericDialogPlus("Export Segmentation Results");

        // gd.addStringField( "Class names prefix: ", inputImageTitle + "--", 10  );

        gd.addStringField( "Binning: ", "1,1,1", 10  );

        gd.addMessage( "--- Export ---" );

        gd.addCheckbox( "raw data", true );

        for ( String className : classNames )
        {
            gd.addCheckbox( className, true );
        }


//        gd.addMessage( "--- Spatial proximity filtering (currently not working for movies) ---" );

//        gd.addCheckbox( "Do spatial proximity filtering", false );
//        gd.addNumericField( "Distance (after binning) [pixels]:", 10, 0 );

//        gd.addChoice( "Reference class:", classNames.toArray( new String[classNames.size()] ), classNames.get( 0 ) );

        gd.addMessage( "--- Export modality ---" );

        gd.addChoice( "Export as:", exportChoices, ResultExportSettings.SEPARATE_IMARIS );

        // gd.addStringField( "TimePoints [from, to] ", "1," + inputImagePlus.getNFrames());

        gd.showDialog();

        if ( gd.wasCanceled() ) return;

        ResultExportSettings resultExportSettings = new ResultExportSettings();
        resultExportSettings.classNames = classNames;
        resultExportSettings.resultImage = resultImage;
        resultExportSettings.inputImagePlus = rawData;
        resultExportSettings.timePointsFirstLast = new int[] { 0, rawData.getNFrames() - 1 };

        setSettingsFromGUI( classNames, gd, resultExportSettings );

        if ( getOutputDirectory( resultExportSettings ) ) return;

        resultImage.exportResults( resultExportSettings );

    }

    private static boolean getOutputDirectory( ResultExportSettings resultExportSettings )
    {
        if ( ! resultExportSettings.exportType.equals( ResultExportSettings.SHOW_AS_SEPARATE_IMAGES ) )
        {
            resultExportSettings.directory = IJ.getDirectory("Select output directory");
            if ( resultExportSettings.directory == null ) return true;
        }
        return false;
    }

    private static void setSettingsFromGUI( ArrayList< String > classNames, GenericDialog gd, ResultExportSettings resultExportSettings )
    {
        // resultExportSettings.exportNamesPrefix = gd.getNextString();

        resultExportSettings.binning = Utils.delimitedStringToIntegerArray( gd.getNextString().trim(), ",");

        setExport( classNames, gd, resultExportSettings );

//        setSpatialProximityFiltering( gd, resultExportSettings );

        resultExportSettings.exportType = gd.getNextChoice();


        // resultExportSettings.timePointsFirstLast = Utils.delimitedStringToIntegerArray( gd.getNextString().trim(), ",");

        // for ( int i = 0; i < resultExportSettings.timePointsFirstLast.length; ++i )
        // {
        //    resultExportSettings.timePointsFirstLast[ i ] -= 1; // convert to zero-based ids
        // }

    }

    private static void setExport( ArrayList< String > classNames, GenericDialog gd, ResultExportSettings resultExportSettings )
    {
        resultExportSettings.saveRawData = gd.getNextBoolean();

        resultExportSettings.classesToBeExported = new ArrayList<>();

        for ( String className : classNames ) resultExportSettings.classesToBeExported.add( gd.getNextBoolean() );
    }

    private static void setSpatialProximityFiltering( GenericDialog gd, ResultExportSettings resultExportSettings )
    {
        resultExportSettings.proximityFilterSettings = new ProximityFilterSettings();
        resultExportSettings.proximityFilterSettings.doSpatialProximityFiltering = gd.getNextBoolean();
        resultExportSettings.proximityFilterSettings.distanceInPixelsAfterBinning = (int) gd.getNextNumber();
        resultExportSettings.proximityFilterSettings.referenceClassId = (int) gd.getNextChoiceIndex();
    }


}
