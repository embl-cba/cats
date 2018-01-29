package de.embl.cba.trainableDeepSegmentation.commands;

import java.util.Iterator;
import java.util.Map;

public abstract class Commands
{

    public static String createImageJPluginCommandLineCall( String ImageJcmd, String pluginName, Map<String, Object> parameters )
    {
        String command = addImageJExecutable( ImageJcmd );

        command = addPluginName( pluginName, command );

        command = addParameters( parameters, command );

        return command;
    }

    private static String addPluginName( String pluginName, String command )
    {
        command += " \"" + pluginName + "\"";
        return command;
    }

    private static String addImageJExecutable( String ImageJcmd )
    {
        return ImageJcmd;
    }

    private static String addParameters( Map< String, Object > parameters, String command )
    {
        command += " \"";
        Iterator<String> keys = parameters.keySet().iterator();
        while( keys.hasNext() )
        {
            String key = addImageJExecutable( keys.next() );
            command += key + "=" + "'" + parameters.get( key ) + "'";
            if ( keys.hasNext() ) command += ",";
        }
        command += "\" ";
        return command;
    }

}
