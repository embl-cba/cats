import de.embl.cba.cats.ui.CATSCommand;
import ij.IJ;
import ij.ImagePlus;
import net.imagej.ImageJ;

public class RunCatsCommand
{

    public static void main( final String... args )
    {

        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ImagePlus imp = IJ.openImage(
                RunCatsCommand.class.getResource(
                        "fib-sem-cell/fib-sem--cell--8x8x8nm.zip" ).getFile() );

		imp.show();

        ij.command().run( CATSCommand.class, true );
    }


}
