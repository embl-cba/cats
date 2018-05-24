package de.embl.cba.trainableDeepSegmentation.utils;

import de.embl.cba.trainableDeepSegmentation.commands.GetPasswordCommand;
import org.scijava.module.Module;
import org.scijava.module.ModuleInfo;
import org.scijava.module.ModuleService;
import org.scijava.plugin.Parameter;
import org.scijava.script.ScriptService;

import java.util.concurrent.ExecutionException;

public class GetPasswordFromUI
{
    @Parameter
    private ScriptService scriptService;

    @Parameter
    private ModuleService moduleService;

    public String getPassword()
    {

        // Module module = moduleService.run( GetPasswordCommand.class, "", true ).getBinned();

        // String password = (String) module.getOutput( "password" );

        return "";
    }

}
