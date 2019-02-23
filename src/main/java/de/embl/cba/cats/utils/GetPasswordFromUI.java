package de.embl.cba.cats.utils;

import org.scijava.module.ModuleService;
import org.scijava.plugin.Parameter;
import org.scijava.script.ScriptService;

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
