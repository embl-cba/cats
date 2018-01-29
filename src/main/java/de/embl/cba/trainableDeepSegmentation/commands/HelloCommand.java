package de.embl.cba.trainableDeepSegmentation.commands;

import ij.IJ;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

@Plugin(type = Command.class, menuPath = "Plugins>HelloCommand" )
public class HelloCommand implements Command
{

    @Parameter(label = "Please enter your name", required = true )
    public String name;


    public void run()
    {
        IJ.log( "HelloCommand " + name + "!" );
    }

}
