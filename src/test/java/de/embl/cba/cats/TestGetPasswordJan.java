package de.embl.cba.cats;

import de.embl.cba.cats.utils.GetPasswordFromUIJan;
import net.imagej.ImageJ;

import java.util.concurrent.ExecutionException;

public class TestGetPasswordJan
{
    public static void main( final String[] args )
    {
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();

        GetPasswordFromUIJan getPasswordFromUIJan = new GetPasswordFromUIJan();
        String pwd = getPasswordFromUIJan.run();
    }
}
