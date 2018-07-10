import de.embl.cba.cats.commands.AnalyzeObjectsCommand;
import de.embl.cba.cats.utils.IOUtils;
import net.imagej.ImageJ;

import java.util.HashMap;
import java.util.Map;

public class AnalyzeObjectCommandUI
{

    public static void main(final String... args) throws Exception
    {

        final ImageJ ij = new ImageJ();
        ij.ui().showUI();
        ij.command().run( AnalyzeObjectsCommand.class, true );

    }

}
