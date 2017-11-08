package trainableDeepSegmentation.resultImage;

import bigDataTools.utils.Utils;
import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.gui.GenericDialog;

import java.util.ArrayList;

public abstract class ResultImageGUI {

    private static final String SEPARATE_IMARIS = "Separate Imaris Channels";

    public static void showExportGUI( ResultImage resultImage,
                                      ArrayList< String > classNames )
    {
        String[] exportChoices = new String[]
                {
                        SEPARATE_IMARIS
                };

        ArrayList < Boolean > saveClass = new ArrayList<>();


        GenericDialog gd = new GenericDialogPlus("Export Segmentation Results");

        gd.addStringField( "Binning: ", "1,1,1", 10  );

        gd.addMessage( "Export class:" );
        for ( String className : classNames ) gd.addCheckbox( className, true );
        gd.addChoice( "Export as:", exportChoices, SEPARATE_IMARIS );

        gd.showDialog();

        if ( gd.wasCanceled() )
            return;

        int[] binning = Utils.delimitedStringToIntegerArray(
                gd.getNextString().trim(), ",");



        for ( String className : classNames ) saveClass.add( gd.getNextBoolean() );

        String exportModality = gd.getNextChoice();

        switch ( exportModality )
        {
            case SEPARATE_IMARIS:
                String directory = IJ.getDirectory("Select a directory");
                resultImage.saveAsSeparateImarisChannels( directory, saveClass, binning );
                break;
        }

    }

}
