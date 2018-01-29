package de.embl.cba.trainableDeepSegmentation.settings;

import java.util.ArrayList;
import java.util.Set;
import java.util.TreeSet;

public class Settings {

    public final static String ANISOTROPY = "anisotropy";
    public final static String BIN_FACTOR = "binFactor";
    public final static String MAX_BIN_LEVEL = "maxBinLevel";
    public final static String MAX_DEEP_CONV_LEVEL = "maxDeepConvLevel";

    public double anisotropy = 1.0;

    public int[] binFactors = new int[]{1,2,3,4,-1,-1,-1,-1,-1,-1,-1};

    public int maxDeepConvLevel = 3; // 3

    public int imageBackground = 0; // gray-values

    public boolean log2 = false;

    public Set< Integer > activeChannels = new TreeSet<>();

    public ArrayList < String > classNames = new ArrayList<>();

    public boolean equals( Settings settings )
    {
        if ( anisotropy != settings.anisotropy ) return false;
        if ( maxDeepConvLevel != settings.maxDeepConvLevel ) return false;
        if ( imageBackground != settings.imageBackground ) return false;
        if ( ! activeChannels.equals( settings.activeChannels ) ) return false;
        if ( ! binFactors.equals( settings.binFactors ) ) return false;
        if ( ! classNames.equals( settings.classNames ) ) return false;

        return true;
    }

    public Settings()
    {
    }

    public Settings copy()
    {
        Settings settings = new Settings();
        settings.classNames = new ArrayList<>( classNames );
        settings.binFactors = binFactors.clone();
        settings.activeChannels = new TreeSet<>( activeChannels );
        settings.anisotropy = anisotropy;
        settings.log2 = log2;
        settings.maxDeepConvLevel = maxDeepConvLevel;
        settings.imageBackground = imageBackground;
        return settings;
    }

    public void setActiveChannels( String activeChannels )
    {
        String[] ss = activeChannels.split(",");

        this.activeChannels = new TreeSet<>();
        for ( String s : ss)
        {
            this.activeChannels.add(Integer.parseInt(s.trim()) - 1); // zero-based
        }
    }
}
