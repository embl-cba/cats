import de.embl.cba.cats.CATS;
import de.embl.cba.cats.ui.LabelButtonsPanel;
import de.embl.cba.cats.ui.Listeners;
import de.embl.cba.cats.ui.Overlays;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

public class ReviewLabels
{
	public static void main( String[] args )
	{
		ImagePlus inputImage = IJ.openImage(
				ResultsExport2D.class.getResource( "boat2d/boat2d.zip" ).getFile() );

		new ImageJ();

		CATS cats = new CATS();
		cats.setInputImage( inputImage );
		cats.setResultImageRAM( );
		cats.loadInstancesAndMetadata(
				ResultsExport2D.class.getResource("boat2d/boat2d.ARFF" ).getFile() );

		inputImage.show();

		Overlays overlays = new Overlays( cats );
		LabelButtonsPanel labelButtonsPanel = new LabelButtonsPanel(
				cats,
				overlays,
				inputImage.getWindow().getLocation(),
				inputImage.getWindow().getWidth() );

		Listeners listeners = new Listeners( cats, overlays, labelButtonsPanel );
		cats.reserveKeyboardShortcuts();

		overlays.reviewLabelsInRoiManagerUI( listeners );


	}
}
