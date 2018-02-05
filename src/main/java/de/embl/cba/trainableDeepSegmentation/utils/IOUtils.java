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

    public static List< Path > asEMBLClusterMounted( List< Path > paths )
    {
        ArrayList< Path > newPaths = new ArrayList<>();

        for ( Path path : paths )
        {
            newPaths.add( asEMBLClusterMounted( path ) );
        }

        return newPaths;
    }

    public static Path asEMBLClusterMounted( Path path )
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
