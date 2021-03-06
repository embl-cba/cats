package de.embl.cba.cats.features.settings;

import de.embl.cba.cats.features.DownSampler;
import de.embl.cba.cats.instances.InstancesAndMetadata;

import java.util.TreeSet;

import static de.embl.cba.cats.instances.InstancesAndMetadata.Metadata.*;

public abstract class FeatureSettingsUtils
{

    public static void addSettingsToMetadata( FeatureSettings featureSettings, InstancesAndMetadata instancesAndMetadata)
    {
        instancesAndMetadata.addMetadata( Metadata_Settings_ImageBackground, featureSettings.imageBackground );
        instancesAndMetadata.addMetadata( Metadata_Settings_MaxDeepConvLevel, featureSettings.maxDeepConvLevel );
        instancesAndMetadata.addMetadata( Metadata_Settings_Anisotropy, featureSettings.anisotropy );
        instancesAndMetadata.addMetadata( Metadata_Settings_Log2, featureSettings.log2 == true ? 1 : 0 );
        instancesAndMetadata.addMetadata( Metadata_Settings_DownSamplingMethod, featureSettings.downSamplingMethod );
        instancesAndMetadata.addMetadata( Metadata_Settings_UseOnlyDifferenceFeatures, featureSettings.onlyUseDifferenceFeatures == true ? 1 : 0 );
        instancesAndMetadata.addMetadata( Metadata_Settings_Normalise, featureSettings.normalize == true ? 1 : 0 );

        int i = 0;
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_0, featureSettings.binFactors.get(i++) );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_1, featureSettings.binFactors.get(i++) );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_2, featureSettings.binFactors.get(i++) );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_3, featureSettings.binFactors.get(i++) );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_4, featureSettings.binFactors.get(i++) );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_5, featureSettings.binFactors.get(i++) );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_6, featureSettings.binFactors.get(i++) );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_7, featureSettings.binFactors.get(i++) );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_8, featureSettings.binFactors.get(i++) );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_9, featureSettings.binFactors.get(i++) );

        int useChannels = 0;
        for ( int c : featureSettings.activeChannels )
        {
            useChannels += (int) Math.pow( 2, c );
        }

        instancesAndMetadata.addMetadata( Metadata_Settings_UseChannels, useChannels );



    }


    public static FeatureSettings getFeatureSettingsFromInstancesMetadata( InstancesAndMetadata instancesAndMetadata)
    {

        final FeatureSettings featureSettings = new FeatureSettings();

        featureSettings.imageBackground = ( int ) instancesAndMetadata.getMetadata( Metadata_Settings_ImageBackground, 0 );
        featureSettings.maxDeepConvLevel = ( int ) instancesAndMetadata.getMetadata( Metadata_Settings_MaxDeepConvLevel, 0 );
        featureSettings.anisotropy = (double) instancesAndMetadata.getMetadata( Metadata_Settings_Anisotropy, 0 );
        featureSettings.log2 = (( int ) instancesAndMetadata.getMetadata( Metadata_Settings_Log2, 0 ) == 1 );

        configureDownsampling( instancesAndMetadata, featureSettings );
        configureDifferenceFeatureUsage( instancesAndMetadata, featureSettings );
        configureNormalisation( instancesAndMetadata, featureSettings );


        int iBinFactor = 0;
        featureSettings.binFactors.set( iBinFactor++ , (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_0, 0 ));
        featureSettings.binFactors.set( iBinFactor++ , (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_1, 0 ));
        featureSettings.binFactors.set( iBinFactor++ , (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_2, 0 ));
        featureSettings.binFactors.set( iBinFactor++ , (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_3, 0 ));
        featureSettings.binFactors.set( iBinFactor++ , (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_4, 0 ));
        featureSettings.binFactors.set( iBinFactor++ , (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_5, 0 ));
        featureSettings.binFactors.set( iBinFactor++ , (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_6, 0 ));
        featureSettings.binFactors.set( iBinFactor++ , (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_7, 0 ));
        featureSettings.binFactors.set( iBinFactor++ , (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_8, 0 ));
        featureSettings.binFactors.set( iBinFactor++ , (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_9, 0 ));

        featureSettings.classNames = instancesAndMetadata.getClassNames();

        featureSettings.activeChannels = new TreeSet<>();
        int channels = (int) instancesAndMetadata.getMetadata( Metadata_Settings_UseChannels, 0 );
        final String s1 = String.format("%8s", Integer.toBinaryString(channels & 0xFF)).replace(' ', '0');
        for (int i = 0; i < s1.length(); i++)
        {
            if (s1.charAt( s1.length() - 1 - i ) == '1')
            {
                featureSettings.activeChannels.add( i  );
            }
        }

        return featureSettings;

    }

    private static void configureDownsampling( InstancesAndMetadata instancesAndMetadata, FeatureSettings featureSettings )
    {
        featureSettings.downSamplingMethod = ( int ) instancesAndMetadata.getMetadata( Metadata_Settings_DownSamplingMethod, 0 );
        if ( featureSettings.downSamplingMethod == -1 ) // was not present due to older version
        {
            featureSettings.downSamplingMethod = DownSampler.getID( DownSampler.BIN_AVERAGE ); // this was the old default
        }
    }

    private static void configureNormalisation( InstancesAndMetadata instancesAndMetadata, FeatureSettings featureSettings )
    {
        int normalise = ( int ) instancesAndMetadata.getMetadata( Metadata_Settings_Normalise, 0 );

        if ( normalise == -1 ) // was not present due to older version
        {
            featureSettings.normalize = false; // this was the old default
        }
        else
        {
            featureSettings.normalize = ( normalise == 1 );
        }
    }

    private static void configureDifferenceFeatureUsage( InstancesAndMetadata instancesAndMetadata, FeatureSettings featureSettings )
    {
        int useOnlyDifferences = ( int ) instancesAndMetadata.getMetadata( Metadata_Settings_UseOnlyDifferenceFeatures, 0 );

        if ( useOnlyDifferences == -1 ) // was not present due to older version
        {
            featureSettings.onlyUseDifferenceFeatures = false; // this was the old default
        }
        else
        {
            featureSettings.onlyUseDifferenceFeatures = ( useOnlyDifferences == 1 );
        }
    }

}
