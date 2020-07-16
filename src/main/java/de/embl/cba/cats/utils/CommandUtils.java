package de.embl.cba.cats.utils;

import de.embl.cba.cluster.ImageJCommandsSubmitter;
import de.embl.cba.cats.ui.ApplyClassifierOnSlurmCommand;
import de.embl.cba.util.PathMapper;
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

    public static String getXvfbImageJExecutionString( File imageJFile )
    {
        if ( imageJFile == null )
        {
            return ImageJCommandsSubmitter.IMAGEJ_EXECTUABLE_ALMF_CLUSTER_XVFB;
        }
        else
        {
            String clusterMountedImageJ = PathMapper.asEMBLClusterMounted( imageJFile.getAbsolutePath() );
            return "xvfb-run -a -e XVFB_ERR_PATH " + clusterMountedImageJ + " --mem=MEMORY_MB --headless --allow-multiple --run";
        }
    }

    public static String getHeadlessImageJExecutionString( File imageJFile )
    {
        if ( imageJFile == null )
        {
            return ImageJCommandsSubmitter.IMAGEJ_EXECTUABLE_ALMF_CLUSTER_HEADLESS;
        }
        else
        {
            String clusterMountedImageJ = PathMapper.asEMBLClusterMounted( imageJFile.getAbsolutePath() );
            return clusterMountedImageJ + " --headless --mem=MEMORY_MB --allow-multiple --run";
        }
    }
}
