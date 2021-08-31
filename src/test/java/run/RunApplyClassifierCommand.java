package run;

import de.embl.cba.cats.ui.ApplyClassifierAdvancedCommand;
import net.imagej.ImageJ;

public class RunApplyClassifierCommand
{
    public static void main( final String... args )
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ij.command().run( ApplyClassifierAdvancedCommand.class, true );
    }
}
