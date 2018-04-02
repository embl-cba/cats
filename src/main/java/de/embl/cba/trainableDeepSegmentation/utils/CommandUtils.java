package de.embl.cba.trainableDeepSegmentation.utils;

import de.embl.cba.cluster.ImageJCommandsSubmitter;
import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierOnSlurmCommand;
import de.embl.cba.utils.fileutils.PathMapper;
import ij.IJ;
import org.scijava.Context;
import org.scijava.command.CommandService;

import java.io.File;
import java.util.Map;

public class CommandUtils
{
    public static void runSlurmCommand( Map< String, Object > parameters )
    {
        Context ctx = (Context ) IJ.runPlugIn("org.scijava.Context", "");
        CommandService commandService = ctx.service( CommandService.class );
        commandService.run( ApplyClassifierOnSlurmCommand.class, true, parameters );
    }

    public static String getImageJExecutionString( File imageJFile )
    {
        if ( imageJFile == null )
        {
            return ImageJCommandsSubmitter.IMAGEJ_EXECTUABLE_ALMF_CLUSTER_XVFB;
        }
        else
        {
            String clusterMountedImageJ = PathMapper.asEMBLClusterMounted( imageJFile.getAbsolutePath() );
            return "xvfb-run -a -e XVFB_ERR_PATH " + clusterMountedImageJ + " --mem=MEMORY_MB --run";
        }
    }
}
