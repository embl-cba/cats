package de.embl.cba.trainableDeepSegmentation.results;

import de.embl.cba.bigDataTools.imaris.ImarisUtils;
import de.embl.cba.bigDataTools.utils.Utils;
import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;

import java.util.ArrayList;

import static de.embl.cba.trainableDeepSegmentation.results.ResultUtils.*;

public abstract class ResultImageGUI {

    public static void showExportGUI( String inputImageTitle,
                                      ResultImage resultImage,
                                      ImagePlus rawData,
                                      ArrayList< String > classNames )
    {

        String[] exportChoices = new String[]{ SEPARATE_IMAGES, SEPARATE_IMARIS, SEPARATE_TIFF_FILES };

        ArrayList < Boolean > classesToBeSaved = new ArrayList<>();

        GenericDialog gd = new GenericDialogPlus("Export Segmentation Results");

        gd.addStringField( "Class names prefix: ", inputImageTitle + "--", 10  );

        gd.addStringField( "Binning: ", "1,1,1", 10  );

        gd.addMessage( "Export raw data:" );

        gd.addCheckbox( rawData.getTitle(), true );

        gd.addMessage( "Export class:" );

        for ( String className : classNames ) gd.addCheckbox( className, true );

        gd.addChoice( "Export as:", exportChoices, ResultUtils.SEPARATE_IMARIS );

        gd.showDialog();

        if ( gd.wasCanceled() ) return;

        String classNamePrefix = gd.getNextString();

        int[] binning = Utils.delimitedStringToIntegerArray( gd.getNextString().trim(), ",");

        boolean saveRawData = gd.getNextBoolean();

        for ( String className : classNames ) classesToBeSaved.add( gd.getNextBoolean() );

        String exportModality = gd.getNextChoice();


        if ( exportModality.equals( SEPARATE_IMARIS ) )
        {
            String directory = IJ.getDirectory("Select a directory");
            if ( directory == null ) return;
            saveAsSeparateImaris( directory, classNamePrefix, resultImage, rawData, classesToBeSaved, binning, saveRawData, exportModality );
        }
        else if ( exportModality.equals( SEPARATE_TIFF_FILES ) )
        {
            String directory = IJ.getDirectory("Select a directory");
            if ( directory == null ) return;
            resultImage.saveClassesAsFiles( directory, classNamePrefix, classesToBeSaved, binning, exportModality );
        }
        else if ( exportModality.equals( SEPARATE_IMAGES ) )
        {
            resultImage.showClassesAsImages( classNamePrefix, classesToBeSaved, binning );
        }

    }

    private static void saveAsSeparateImaris( String directory, String fileNamePrefix, ResultImage resultImage, ImagePlus rawData, ArrayList< Boolean > classesToBeSaved, int[] binning, boolean saveRawData, String exportModality )
    {
        resultImage.saveClassesAsFiles( directory, fileNamePrefix, classesToBeSaved, binning, exportModality );

        if ( saveRawData )
        {
            saveAsImarisChannels( rawData, fileNamePrefix + "raw-data", directory, binning );
        }

        ImarisUtils.createImarisMetaFile( directory );
    }

}
