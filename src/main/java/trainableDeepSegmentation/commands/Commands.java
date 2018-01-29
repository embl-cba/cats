package trainableDeepSegmentation.commands;

import ij.IJ;

import java.util.Iterator;
import java.util.Map;

public abstract class Commands
{

    public static String createCommand( String pluginName, Map<String, Object> parameters )
    {
        IJ.log( "\n# Command line call:" );

        String command = "";
        command += "\"" + pluginName + "\"";
        command += " \"";

        Iterator<String> keys = parameters.keySet().iterator();
        while( keys.hasNext() )
        {
            String key = keys.next();
            command += key + "=" + "'" + parameters.get( key ) + "'";
            if ( keys.hasNext() ) command += ",";
        }
        command += "\" ";
        return command;
    }

}
