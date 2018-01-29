package trainableDeepSegmentation;

import net.imagej.ImageJ;
import trainableDeepSegmentation.commands.HelloCommand;

import java.util.HashMap;
import java.util.Map;

public class TestHelloIJ2Plugin
{

    // Main
    public static void main(final String... args) throws Exception {

        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        Map< String, Object > parameters = new HashMap<>(  );
        parameters.put( "name", "Frida" );

        ij.command().run( HelloCommand.class, false, parameters );

    }
}
