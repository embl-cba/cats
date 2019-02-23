import de.embl.cba.cats.ui.AnalyzeObjectsCommand;
import net.imagej.ImageJ;

public class AnalyzeObjectCommandUI
{

    public static void main(final String... args) throws Exception
    {


        Runtime.getRuntime();
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();
        ij.command().run( AnalyzeObjectsCommand.class, true );

    }

}
