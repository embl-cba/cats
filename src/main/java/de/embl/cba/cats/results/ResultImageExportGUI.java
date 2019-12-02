package de.embl.cba.cats.results;

import de.embl.cba.bigdataprocessor.utils.Utils;
import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;

import java.awt.*;
import java.util.ArrayList;

public abstract class ResultImageExportGUI
{

    public static void showExportGUI( ResultImage resultImage,
                                      ImagePlus rawData,
                                      ArrayList< String > classNames, Color[] classColors )
    {

        String[] exportChoices = new String[]{
                ResultExportSettings.SHOW_AS_LABEL_MASKS,
                ResultExportSettings.SHOW_AS_PROBABILITIES,
                ResultExportSettings.SAVE_AS_IMARIS_STACKS,
                ResultExportSettings.SAVE_AS_CLASS_PROBABILITY_TIFF_STACKS };

        GenericDialog gd = new GenericDialogPlus("Export Segmentation Results");

        // gd.addStringField( "Class names prefix: ", inputImageTitle + "--", 10  );

        gd.addStringField( "Binning: ", "1,1,1", 10  );

        gd.addMessage( "--- Export ---" );

        gd.addCheckbox( "raw data", true );

        for ( String className : classNames )
            gd.addCheckbox( className, true );

        gd.addMessage( "--- Export modality ---" );

        gd.addChoice( "Export as:",
                exportChoices,
                ResultExportSettings.SHOW_AS_LABEL_MASKS );

        // gd.addStringField( "TimePoints [from, to] ", "1," + inputImagePlus.getNFrames());

        gd.showDialog();

        if ( gd.wasCanceled() ) return;

        ResultExportSettings settings = new ResultExportSettings();
        settings.classNames = classNames;
        settings.resultImage = resultImage;
        settings.inputImagePlus = rawData;
        settings.timePointsFirstLast = new int[] { 0, rawData.getNFrames() - 1 };
        settings.classColors = classColors;

        setSettingsFromGUI( classNames, gd, settings );

        if ( ResultExport.isSaveImage( settings ) )
            if ( ! getOutputDirectory( settings ) ) return;

        resultImage.exportResults( settings );
    }

    private static boolean getOutputDirectory( ResultExportSettings settings )
    {
        settings.directory = IJ.getDirectory("Select output directory");
        if ( settings.directory == null ) return false;
        return true;
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
