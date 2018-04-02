package de.embl.cba.trainableDeepSegmentation;

import de.embl.cba.trainableDeepSegmentation.commands.BatchClassificationOnSlurm;
import net.imagej.ImageJ;

public class TestRunSylwiasWorkflowOnSlurm
{

    public static void main(final String... args) throws Exception
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        ij.command().run( BatchClassificationOnSlurm.class, true );
    }

}
