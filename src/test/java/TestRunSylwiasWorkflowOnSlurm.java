import de.embl.cba.cats.ui.BatchClassificationOnSlurmCommand;
import net.imagej.ImageJ;

public class TestRunSylwiasWorkflowOnSlurm
{

    public static void main(final String... args) throws Exception
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ij.command().run( BatchClassificationOnSlurmCommand.class, true );
    }

}
