package de.embl.cba.trainableDeepSegmentation.commands;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.widget.TextWidget;

public class GetPasswordCommand implements Command
{
    @Parameter( label = "Password", style = TextWidget.PASSWORD_STYLE, persist = false, type = ItemIO.BOTH )
    private String password;
    public static String PASSWORD = "password";

    @Override
    public void run()
    {

    }
}
