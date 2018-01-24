package trainableDeepSegmentation.results;

import de.embl.cba.bigDataTools.ImarisUtils;
import de.embl.cba.bigDataTools.utils.Utils;
import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;

import java.util.ArrayList;

import static trainableDeepSegmentation.WekaSegmentation.logger;
import static trainableDeepSegmentation.results.Utils.SEPARATE_IMARIS;
import static trainableDeepSegmentation.results.Utils.SEPARATE_TIFF_FILES;
import static trainableDeepSegmentation.results.Utils.saveAsImarisChannel;

public abstract class ResultImageGUI {

    public static void showExportGUI( ResultImage resultImage,
                                      ImagePlus rawData,
                                      ArrayList< String > classNames )
    {
        String[] exportChoices = new String[]{
                trainableDeepSegmentation.results.Utils.SEPARATE_IMARIS,
                trainableDeepSegmentation.results.Utils.SEPARATE_TIFF_FILES
        };

        ArrayList < Boolean > classesToBeSaved = new ArrayList<>();

        GenericDialog gd = new GenericDialogPlus("Export Segmentation Results");

        gd.addStringField( "Binning: ", "1,1,1", 10  );

        gd.addMessage( "Export raw data:" );
        gd.addCheckbox( rawData.getTitle(), true );

        gd.addMessage( "Export class:" );
        for ( String className : classNames ) gd.addCheckbox( className, true );
        gd.addChoice( "Export as:", exportChoices, trainableDeepSegmentation.results.Utils.SEPARATE_IMARIS );

        gd.showDialog();

        if ( gd.wasCanceled() ) return;

        int[] binning = Utils.delimitedStringToIntegerArray( gd.getNextString().trim(), ",");

        boolean saveRawData = gd.getNextBoolean();
        for ( String className : classNames ) classesToBeSaved.add( gd.getNextBoolean() );

        String exportModality = gd.getNextChoice();

        String directory = IJ.getDirectory("Select a directory");

        if ( exportModality.equals( SEPARATE_IMARIS ) )
        {
            saveAsSeparateImaris( resultImage, rawData, classesToBeSaved, binning, saveRawData, exportModality, directory );
        }
        else if ( exportModality.equals( SEPARATE_TIFF_FILES ) )
        {
            resultImage.saveClassesAsFiles( directory, classesToBeSaved, binning, exportModality );
        }

        logger.info("Created imaris meta file.");

    }

    private static void saveAsSeparateImaris( ResultImage resultImage, ImagePlus rawData, ArrayList< Boolean > classesToBeSaved, int[] binning, boolean saveRawData, String exportModality, String directory )
    {
        resultImage.saveClassesAsFiles( directory, classesToBeSaved, binning, exportModality );

        if ( saveRawData )
        {
            saveAsImarisChannel( rawData, "raw-data", directory, binning );
        }

        ImarisUtils.createImarisMetaFile( directory );
    }

}
