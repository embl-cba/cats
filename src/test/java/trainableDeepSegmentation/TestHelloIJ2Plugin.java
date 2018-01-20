package trainableDeepSegmentation;

import net.imagej.ImageJ;
import trainableDeepSegmentation.ij2plugins.Hello;

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

        ij.command().run( Hello.class, false, parameters );

    }
}
