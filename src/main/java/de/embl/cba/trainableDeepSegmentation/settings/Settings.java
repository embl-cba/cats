package de.embl.cba.trainableDeepSegmentation.settings;

import de.embl.cba.trainableDeepSegmentation.features.DownSampler;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Set;
import java.util.TreeSet;

public class Settings {

    public final static String ANISOTROPY = "anisotropy";
    public final static String BIN_FACTOR = "binFactor";
    public final static String MAX_BIN_LEVEL = "maxBinLevel";
    public final static String MAX_DEEP_CONV_LEVEL = "maxDeepConvLevel";

    public String downSamplingMethod = DownSampler.TRANSFORMJ_SCALE_LINEAR;

    public double anisotropy;

    public int maxDeepConvLevel;

    public int imageBackground;

    public boolean log2 = false;

    public ArrayList< Integer > binFactors;

    public Set< Integer > activeChannels = new TreeSet<>();

    public ArrayList < String > classNames = new ArrayList<>();

    public Set< Integer > boundingBoxExpansions = new TreeSet<>();


    public boolean equals( Settings settings )
    {
        if ( anisotropy != settings.anisotropy ) return false;
        if ( maxDeepConvLevel != settings.maxDeepConvLevel ) return false;
        if ( imageBackground != settings.imageBackground ) return false;
        if ( ! activeChannels.equals( settings.activeChannels ) ) return false;
        if ( ! classNames.equals( settings.classNames ) ) return false;
        if ( ! downSamplingMethod.equals( settings.downSamplingMethod ) ) return false;
        if ( ! boundingBoxExpansions.equals( settings.boundingBoxExpansions ) ) return false;
        if ( ! binFactors.equals( settings.binFactors ) ) return false;

        return true;
    }

    public Settings()
    {
        anisotropy = 1.0;
        binFactors = new ArrayList();
        binFactors.add( 1 );
        binFactors.add( 2 );
        binFactors.add( 3 );
        binFactors.add( 4 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        maxDeepConvLevel = 3;
        imageBackground = 0;
        boundingBoxExpansions.add( 0 );
    }

    public Settings copy()
    {
        Settings settings = new Settings();
        settings.classNames = new ArrayList<>( classNames );
        settings.binFactors = new ArrayList<>( binFactors );
        settings.activeChannels = new TreeSet<>( activeChannels );
        settings.anisotropy = anisotropy;
        settings.log2 = log2;
        settings.maxDeepConvLevel = maxDeepConvLevel;
        settings.imageBackground = imageBackground;
        return settings;
    }

    public void setActiveChannels( String csv )
    {
        String[] ss = csv.split(",");

        this.activeChannels = new TreeSet<>();
        for ( String s : ss)
        {
            this.activeChannels.add( Integer.parseInt(s.trim()) - 1 ); // zero-based
        }
    }

    public void setBoundingBoxExpansions( String csv )
    {
        String[] ss = csv.split(",");

        this.boundingBoxExpansions = new TreeSet<>();
        for ( String s : ss)
        {
            this.boundingBoxExpansions.add( Integer.parseInt(s.trim()) );
        }
    }

    public static String getAsCSVString( Collection< Integer > collection, int add )
    {
        String ss = "";
        for ( int s : collection )
        {
            if ( !ss.equals("") )
                ss += ("," + ( s + add));
            else
                ss += ""+ (s + add);
        }
        return ss;
    }
}
