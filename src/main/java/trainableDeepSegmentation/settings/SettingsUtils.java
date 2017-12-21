package trainableDeepSegmentation.settings;

import trainableDeepSegmentation.instances.InstancesAndMetadata;

import java.util.TreeSet;

import static trainableDeepSegmentation.instances.InstancesAndMetadata.Metadata.*;

public abstract class SettingsUtils {

    public static void addSettingsToMetadata( Settings settings,
                                       InstancesAndMetadata instancesAndMetadata)
    {
        instancesAndMetadata.addMetadata( Metadata_Settings_ImageBackground, settings.imageBackground );
        instancesAndMetadata.addMetadata( Metadata_Settings_MaxDeepConvLevel, settings.maxDeepConvLevel );
        instancesAndMetadata.addMetadata( Metadata_Settings_Anisotropy, settings.anisotropy );
        instancesAndMetadata.addMetadata( Metadata_Settings_Log2, settings.log2 == true ? 1 : 0 );

        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_0, settings.binFactors[0] );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_1, settings.binFactors[1] );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_2, settings.binFactors[2] );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_3, settings.binFactors[3] );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_4, settings.binFactors[4] );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_5, settings.binFactors[5] );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_6, settings.binFactors[6] );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_7, settings.binFactors[7] );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_8, settings.binFactors[8] );
        instancesAndMetadata.addMetadata( Metadata_Settings_Binning_9, settings.binFactors[9] );

        int useChannels = 0;
        for ( int c : settings.activeChannels )
        {
            useChannels += (int) Math.pow( 2, c );
        }

        instancesAndMetadata.addMetadata( Metadata_Settings_UseChannels, useChannels );



    }


    public static void setSettingsFromInstancesMetadata( Settings settings,
                                              InstancesAndMetadata instancesAndMetadata)
    {
        settings.imageBackground = ( int ) instancesAndMetadata.getMetadata( Metadata_Settings_ImageBackground, 0 );
        settings.maxDeepConvLevel = ( int ) instancesAndMetadata.getMetadata( Metadata_Settings_MaxDeepConvLevel, 0 );
        settings.anisotropy = ( int ) instancesAndMetadata.getMetadata( Metadata_Settings_Anisotropy, 0 );
        settings.log2 = (( int ) instancesAndMetadata.getMetadata( Metadata_Settings_Log2, 0 ) == 1 );

        settings.binFactors[0] = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_0, 0 );
        settings.binFactors[1] = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_1, 0 );
        settings.binFactors[2] = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_2, 0 );
        settings.binFactors[3] = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_3, 0 );
        settings.binFactors[4] = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_4, 0 );
        settings.binFactors[5] = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_5, 0 );
        settings.binFactors[6] = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_6, 0 );
        settings.binFactors[7] = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_7, 0 );
        settings.binFactors[8] = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_8, 0 );
        settings.binFactors[9] = (int) instancesAndMetadata.getMetadata( Metadata_Settings_Binning_9, 0 );

        settings.classNames = instancesAndMetadata.getClassNames();

        settings.activeChannels = new TreeSet<>();
        int channels = (int) instancesAndMetadata.getMetadata( Metadata_Settings_UseChannels, 0 );
        final String s1 = String.format("%8s", Integer.toBinaryString(channels & 0xFF)).replace(' ', '0');
        for (int i = 0; i < s1.length(); i++)
        {
            if (s1.charAt( s1.length() - 1 - i ) == '1')
            {
                settings.activeChannels.add( i  );
            }
        }


    }

}
