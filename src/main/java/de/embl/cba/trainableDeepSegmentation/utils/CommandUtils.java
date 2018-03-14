package de.embl.cba.trainableDeepSegmentation.utils;

import de.embl.cba.trainableDeepSegmentation.commands.ApplyClassifierOnSlurmCommand;
import ij.IJ;
import org.scijava.Context;
import org.scijava.command.CommandService;

import java.util.Map;

public class CommandUtils
{
    public static void runSlurmCommand( Map< String, Object > parameters )
    {
        Context ctx = (Context ) IJ.runPlugIn("org.scijava.Context", "");
        CommandService commandService = ctx.service( CommandService.class );
        commandService.run( ApplyClassifierOnSlurmCommand.class, true, parameters );
    }

}
