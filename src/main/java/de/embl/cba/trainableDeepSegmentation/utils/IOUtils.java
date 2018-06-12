package de.embl.cba.trainableDeepSegmentation.utils;

import de.embl.cba.bigDataTools.dataStreamingTools.DataStreamingTools;
import de.embl.cba.utils.fileutils.FileRegMatcher;
import ij.IJ;
import ij.ImagePlus;
import ij.io.OpenDialog;
import ij.io.SaveDialog;

import javax.swing.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class IOUtils
{

    public static final String SAVE = "Save";
    public static final String SHOW = "Show";

    public static final String SAVE_RESULTS_TABLE = "Save results table";
    public static final String SAVE_RESULTS_TABLE_AND_LABEL_MASK = "Save results table and label mask";
    public static final String SHOW_RESULTS_TABLE = "Show results table";
    public static final String INPUT_MODALITY = "inputModality";
    public static final String INPUT_IMAGE_VSS_DIRECTORY = "inputImageVSSDirectory";
    public static final String INPUT_IMAGE_VSS_SCHEME = "inputImageVSSScheme";
    public static final String INPUT_IMAGE_VSS_PATTERN = "inputImageVSSPattern";
    public static final String INPUT_IMAGE_VSS_HDF5_DATA_SET_NAME = "inputImageVSSHdf5DataSetName";

    public static final String OPEN_USING_IMAGEJ1 = "Open using ImageJ1";
    public static final String OPEN_USING_IMAGE_J1_VIRTUAL = "Open using ImageJ1 virtual";
    public static final String OPEN_USING_IMAGEJ1_IMAGE_SEQUENCE = "Open using ImageJ1 ImageSequence";
    public static final String OPEN_USING_LAZY_LOADING_TOOLS = "Open using Lazy Loading Tools";
    public static final String OUTPUT_MODALITY = "outputModality";
    public static final String SAVE_AS_IMARIS = "Save class probabilities as imaris files";
    public static final String SAVE_AS_TIFF_STACKS = "Save class probabilities as Tiff stacks";
    public static final String SHOW_AS_ONE_IMAGE = "Show all probabilities in one image";
    public static final String STREAM_TO_RESULT_IMAGE_DISK = "Stream class probabilities to disk resident resultImagePlus image";
    public static final String SAVE_AS_MULTI_CLASS_TIFF_SLICES = "Save class probabilities as Tiff slices";

    public static final String OUTPUT_DIRECTORY = "outputDirectory";
    public static final String INPUT_IMAGE_FILE = "inputImageFile";


    public static ImagePlus openImageWithIJOpenImage( File imageFile )
    {
        ImagePlus input = IJ.openImage( imageFile.getAbsolutePath() );
        return input;
    }

    public static ImagePlus openImageWithIJOpenVirtualImage( File imageFile )
    {
        ImagePlus input = IJ.openVirtual( imageFile.getAbsolutePath() );
        return input;
    }

    public static ImagePlus openImageWithIJImportImageSequence( File imageFile  )
    {
        String directory = imageFile.getParent();
        String regExp = imageFile.getName();
        IJ.run("Image Sequence...", "open=["+ directory +"]" +"file=(" + regExp + ") sort");
        ImagePlus image = IJ.getImage();
        image.setTitle( regExp );
        return image;
    }

    public static ImagePlus openImageWithLazyLoadingTools( String directory, String namingScheme, String filePattern, String hdf5DataSetName  )
    {

        DataStreamingTools dst = new DataStreamingTools();
        ImagePlus image = dst.openFromDirectory(
                directory,
                namingScheme,
                filePattern,
                hdf5DataSetName,
                null,
                3,
                false,
                false);

        image.setTitle( namingScheme );
        IJ.wait( 1000 );

        return image;
    }



    public static String createDataSetNameFromPattern( String dataSetPattern )
    {
        String dataSetName = dataSetPattern; //.replace( ".*", "" );

        return dataSetName;
    }

    public static Path createDirectoryIfNotExists( String directory )
    {

        Path path = Paths.get( directory );

        if ( ! Files.exists( path ) )
        {
            try
            {
                Files.createDirectories( path );
            } catch ( IOException e )
            {
                e.printStackTrace();
            }
        }

        return path;
    }


    public static List< Path > getDataSetPatterns( String directory, String regExpMaster, String[] regExpGroups )
    {

        FileRegMatcher regMatcher = new FileRegMatcher();

        regMatcher.setParameters( regExpMaster, regExpGroups );

        regMatcher.matchFiles( directory );

        List< File > filePatterns = regMatcher.getMatchedFilesList();

        List< Path > filePatternPaths = new ArrayList<>();

        for ( File f : filePatterns )
        {
            String pattern = f.getAbsolutePath();
            filePatternPaths.add( Paths.get( pattern) );
        }

        return filePatternPaths;
    }

    public static String[] getOpenDirFile( String title )
    {
        OpenDialog od = new OpenDialog(title, OpenDialog.getLastDirectory(), "");
        if ( od.getFileName() == null ) return null;

        String[] dirFile = new String[]{ od.getDirectory(), od.getFileName() };

        return dirFile;
    }

    public static String classNameDialog()
    {
        String inputName = JOptionPane.showInputDialog("Please input a new class name");

        if( null == inputName )
            return null;


        if ( null == inputName || 0 == inputName.length() )
        {
            IJ.error("Invalid name for class");
            return null;
        }
        inputName = inputName.trim();

        if (0 == inputName.toLowerCase().indexOf("add to "))
            inputName = inputName.substring(7);
        return inputName;
    }

    public static String[] getSaveDirFile( String title, String defaultName , String extension )
    {
        SaveDialog sd = new SaveDialog("Save as...", defaultName, extension);

        if ( sd.getFileName() == null ) return null;

        String[] dirFile = new String[] { sd.getDirectory(), sd.getFileName() };

        return dirFile;
    }

    public static String[] getSaveDirFile( String title, String extension )
    {
        return getSaveDirFile( title, "", extension );
    }
}
