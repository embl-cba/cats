package trainableDeepSegmentation.commands;

import ij.IJ;
import ij.ImagePlus;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class IOUtils
{

    static ImagePlus loadImage( File imageFile )
    {

        ImagePlus image;

        if ( imageFile.getName().contains( ".*" ) )
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
}
