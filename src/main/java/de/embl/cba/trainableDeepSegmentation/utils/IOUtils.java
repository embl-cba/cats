package de.embl.cba.trainableDeepSegmentation.utils;

import ij.IJ;
import ij.ImagePlus;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class IOUtils
{

    public static final String SAVE_RESULTS_TABLE = "Save results table";
    public static final String SHOW_RESULTS_TABLE = "Show results table";
    public static final String INPUT_MODALITY = "inputModality";
    public static final String OPEN_USING_IMAGE_J1 = "Open using ImageJ1";
    public static final String OPEN_USING_IMAGEJ1_IMAGE_SEQUENCE = "Open using ImageJ1 ImageSequence";
    public static final String OUTPUT_MODALITY = "outputModality";
    public static final String SAVE_AS_IMARIS = "Save class probabilities as imaris files";
    public static final String SAVE_AS_TIFF_STACKS = "Save class probabilities as Tiff stacks";
    public static final String SHOW_AS_ONE_IMAGE = "Show all probabilities in one image";
    public static final String SAVE_AS_TIFF_SLICES = "Save class probabilities as Tiff slices";
    public static final String OUTPUT_DIRECTORY = "outputDirectory";
    public static final String INPUT_IMAGE_PATH = "inputImagePath";

    public static ImagePlus loadImage( File imageFile )
    {

        ImagePlus image;

        if ( imageFile.getName().contains( ".*" ) || imageFile.getName().contains( "?" ) )
        {
            image = loadImageWithImportImageSequence( imageFile );
        }
        else
        {
            image = loadImageWithIJOpenImage( imageFile);
        }

        return image;
    }

    private static ImagePlus loadImageWithIJOpenImage( File imageFile )
    {
        ImagePlus input = IJ.openImage( imageFile.getAbsolutePath() );
        return input;
    }

    private static ImagePlus loadImageWithImportImageSequence( File imageFile  )
    {
        String directory = imageFile.getParent();
        String regExp = imageFile.getName();
        IJ.run("Image Sequence...", "open=["+ directory +"]" +"file=(" + regExp + ") sort");
        ImagePlus image = IJ.getImage();
        image.setTitle( regExp );
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

    public static List< Path > clusterMounted( List< Path > paths )
    {
        ArrayList< Path > newPaths = new ArrayList<>();

        for ( Path path : paths )
        {
            newPaths.add( clusterMounted( path ) );
        }

        return newPaths;
    }

    public static Path clusterMounted( Path path )
    {

        String pathString = path.toString();
        String newPathString = null;

        if ( isMac() )
        {
            newPathString = pathString.replace( "/Volumes/", "/g/" );
        }
        else if ( isWindows() )
        {
            try
            {
                Runtime runTime = Runtime.getRuntime();
                Process process = null;
                process = runTime.exec( "net use" );

                InputStream inStream = process.getInputStream();
                InputStreamReader inputStreamReader = new InputStreamReader( inStream );
                BufferedReader bufferedReader = new BufferedReader( inputStreamReader );
                String line = null;
                String[] components = null;

                while ( null != ( line = bufferedReader.readLine() ) )
                {
                    components = line.split( "\\s+" );
                    if ( ( components.length > 2 ) && ( components[ 1 ].equals( pathString.substring( 0, 2 ) ) ) )
                    {
                        newPathString = pathString.replace( components[ 1 ], components[ 2 ] );
                    }
                }

            }
            catch ( IOException e )
            {
                e.printStackTrace();
            }
        }
        else
        {
            newPathString = pathString;
        }


        return Paths.get( newPathString );

    }

    public static String getOsName()
    {
        String OS = System.getProperty("os.name");
        return OS;
    }

    public static boolean isWindows()
    {
        return getOsName().startsWith("Windows");
    }


    public static boolean isMac()
    {
        String OS = getOsName();

        if ( ( OS.toLowerCase().indexOf( "mac" ) >= 0 ) || ( OS.toLowerCase().indexOf( "darwin" ) >= 0 ) )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

}
