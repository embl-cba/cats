package de.embl.cba.cats.utils;

import ij.IJ;
import net.imagej.ImageJ;
import org.scijava.Context;
import org.scijava.command.CommandService;
import org.scijava.module.Module;
import org.scijava.plugin.Parameter;
import org.scijava.script.ScriptService;

import java.util.concurrent.ExecutionException;

public class GetPasswordFromUIJan
{

    public String run()
    {

        Context ctx = (Context) IJ.runPlugIn("org.scijava.Context", "");
        ScriptService scriptService = ctx.service( ScriptService.class );

        String script = "#@BOTH String (label = \"Please enter password\", style = password) pwd";

        Module module = null;
        try
        {
            module = scriptService.run( "GetPassword.groovy", script, true ).get();
        }
        catch ( InterruptedException e )
        {
            e.printStackTrace();
        }
        catch ( ExecutionException e )
        {
            e.printStackTrace();
        }

        String password = (String) module.getOutput( "pwd" );

        return password;
    }

}
