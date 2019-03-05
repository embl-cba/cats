package de.embl.cba.cats.features.settings;

import de.embl.cba.cats.features.DownSampler;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Set;
import java.util.TreeSet;

public class FeatureSettings
{

    public final static String ANISOTROPY = "anisotropy";
    public final static String BIN_FACTOR = "binFactor";
    public final static String MAX_BIN_LEVEL = "maxBinLevel";
    public final static String MAX_DEEP_CONV_LEVEL = "maxDeepConvLevel";

    public int downSamplingMethod;

    public double anisotropy;

    public int maxDeepConvLevel;

    public int imageBackground = 0;

    public boolean log2 = false;

    public ArrayList< Integer > binFactors = new ArrayList<>(  );

    public Set< Integer > activeChannels = new TreeSet<>();

    public ArrayList < String > classNames = new ArrayList<>();

    public Set< Integer > boundingBoxExpansionsForGeneratingInstancesFromLabels = new TreeSet<>();

    public Set< Integer > smoothingScales = new TreeSet<>();

    public boolean computeGaussian = false;

    public boolean onlyUseDifferenceFeatures;

    public boolean normalize;

	public boolean equals( FeatureSettings featureSettings )
    {
        if ( anisotropy != featureSettings.anisotropy ) return false;
        if ( maxDeepConvLevel != featureSettings.maxDeepConvLevel ) return false;
        if ( imageBackground != featureSettings.imageBackground ) return false;
        if ( ! ( activeChannels.equals( featureSettings.activeChannels ) ) ) return false;
        if ( ! ( classNames.equals( featureSettings.classNames ) ) ) return false;
        if ( ! ( downSamplingMethod == featureSettings.downSamplingMethod ) ) return false;
        if ( ! ( boundingBoxExpansionsForGeneratingInstancesFromLabels.equals( featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels ) ) ) return false;
        if ( ! ( binFactors.equals( featureSettings.binFactors ) ) ) return false;
        if ( ! ( smoothingScales.equals( featureSettings.smoothingScales ) ) ) return false;
        if ( ! ( computeGaussian == featureSettings.computeGaussian ) ) return false;
        if ( ! ( log2 == featureSettings.log2) ) return false;
        if ( ! ( onlyUseDifferenceFeatures == featureSettings.onlyUseDifferenceFeatures) ) return false;
        if ( ! ( normalize == featureSettings.normalize) ) return false;

        return true;
    }

    public FeatureSettings()
    {
        anisotropy = 1.0;
        binFactors.add( 1 );
        binFactors.add( 2 );
        binFactors.add( 3 );
        binFactors.add( 3 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        binFactors.add( -1 );
        maxDeepConvLevel = 3;
        imageBackground = 0;
        boundingBoxExpansionsForGeneratingInstancesFromLabels.add( 0 );
        smoothingScales.add( 1 );
        downSamplingMethod = DownSampler.getID( DownSampler.BIN_AVERAGE );
        onlyUseDifferenceFeatures = false;
        normalize = false;
    }

    public FeatureSettings copy()
    {
        FeatureSettings featureSettings = new FeatureSettings();
        featureSettings.classNames = new ArrayList<>( classNames );
        featureSettings.binFactors = new ArrayList<>( binFactors );
        featureSettings.activeChannels = new TreeSet<>( activeChannels );
        featureSettings.smoothingScales = new TreeSet<>( smoothingScales );
        featureSettings.boundingBoxExpansionsForGeneratingInstancesFromLabels = new TreeSet<>( boundingBoxExpansionsForGeneratingInstancesFromLabels );
        featureSettings.anisotropy = anisotropy;
        featureSettings.log2 = log2;
        featureSettings.maxDeepConvLevel = maxDeepConvLevel;
        featureSettings.imageBackground = imageBackground;
        featureSettings.computeGaussian = computeGaussian;
        featureSettings.downSamplingMethod = downSamplingMethod;
        featureSettings.onlyUseDifferenceFeatures = onlyUseDifferenceFeatures;
        featureSettings.normalize = normalize;

        return featureSettings;
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

    public void setBoundingBoxExpansionsForGeneratingInstancesFromLabels( String csv )
    {
        String[] ss = csv.split(",");

        this.boundingBoxExpansionsForGeneratingInstancesFromLabels = new TreeSet<>();
        for ( String s : ss)
        {
            this.boundingBoxExpansionsForGeneratingInstancesFromLabels.add( Integer.parseInt(s.trim()) );
        }
    }

    public void setSmoothingScales( String csv )
    {
        String[] ss = csv.split(",");

        this.smoothingScales = new TreeSet<>();
        for ( String s : ss)
        {
            this.smoothingScales.add( Integer.parseInt(s.trim()) );
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
