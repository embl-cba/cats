package trainableDeepSegmentation.resultImage;

import bigDataTools.Hdf5DataCubeWriter;
import bigDataTools.ImarisDataSet;
import bigDataTools.ImarisUtils;
import bigDataTools.ImarisWriter;
import bigDataTools.utils.Utils;
import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.Binner;
import ij.plugin.Duplicator;
import net.imglib2.FinalInterval;

import java.util.ArrayList;

import static trainableDeepSegmentation.WekaSegmentation.logger;
import static trainableDeepSegmentation.resultImage.Utils.saveAsImarisChannel;

public abstract class ResultImageGUI {

    private static final String SEPARATE_IMARIS = "Separate Imaris Channels";

    public static void showExportGUI( ResultImage resultImage,
                                      ImagePlus rawData,
                                      ArrayList< String > classNames )
    {
        String[] exportChoices = new String[]
                {
                        SEPARATE_IMARIS
                };

        ArrayList < Boolean > saveClass = new ArrayList<>();


        GenericDialog gd = new GenericDialogPlus("Export Segmentation Results");

        gd.addStringField( "Binning: ", "1,1,1", 10  );

        gd.addMessage( "Export raw data:" );
        gd.addCheckbox( rawData.getTitle(), true );

        gd.addMessage( "Export class:" );
        for ( String className : classNames ) gd.addCheckbox( className, true );
        gd.addChoice( "Export as:", exportChoices, SEPARATE_IMARIS );

        gd.showDialog();

        if ( gd.wasCanceled() )
            return;

        int[] binning = Utils.delimitedStringToIntegerArray(
                gd.getNextString().trim(), ",");


        boolean saveRawData = gd.getNextBoolean();
        for ( String className : classNames ) saveClass.add( gd.getNextBoolean() );

        String exportModality = gd.getNextChoice();

        String directory = IJ.getDirectory("Select a directory");

        switch ( exportModality )
        {
            case SEPARATE_IMARIS:
                resultImage.saveAsSeparateImarisChannels( directory, saveClass, binning );
                break;
        }

        if ( saveRawData )
        {
             saveAsImarisChannel( rawData, "raw-data", directory, binning );
        }

        ImarisUtils.createImarisMetaFile( directory );
        logger.info("Created imaris meta file.");

    }

}
