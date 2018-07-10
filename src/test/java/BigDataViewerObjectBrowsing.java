import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvSource;
import bdv.viewer.animate.TranslationAnimator;
import ij.IJ;
import ij.ImagePlus;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.realtransform.AffineTransform3D;

public class BigDataViewerObjectBrowsing
{

    Bdv bdv;


    public BigDataViewerObjectBrowsing( )
    {
        addSourceFromTiffFile( "/Users/tischer/Documents/fiji-plugin-CATS/src/test/resources/fib-sem--cell--8x8x8nm.zip"  );

        double[] center = new double[]{273,74,18};
        double zoom = 2;
        zoomToPoint( bdv, center, zoom );

    }

    private void zoomToPoint( Bdv bdv, double[] spotCoords, double zoom )
    {
        AffineTransform3D affineTransform3D = new AffineTransform3D();

        final AffineTransform3D t = new AffineTransform3D();
        bdv.getBdvHandle().getViewerPanel().getState().getViewerTransform( t );

        double viewerPanelHeight = bdv.getBdvHandle().getViewerPanel().getDisplay().getHeight();
        double viewerPanelWidth = bdv.getBdvHandle().getViewerPanel().getDisplay().getWidth();


        final double dx = viewerPanelWidth / 2 - ( t.get( 0, 0 ) * spotCoords[ 0 ] + t.get( 0, 1 ) * spotCoords[ 1 ] + t.get( 0, 2 ) * spotCoords[ 2 ] );
        final double dy = viewerPanelHeight / 2 - ( t.get( 1, 0 ) * spotCoords[ 0 ] + t.get( 1, 1 ) * spotCoords[ 1 ] + t.get( 1, 2 ) * spotCoords[ 2 ] );
        final double dz = -( t.get( 2, 0 ) * spotCoords[ 0 ] + t.get( 2, 1 ) * spotCoords[ 1 ] + t.get( 2, 2 ) * spotCoords[ 2 ] );

        double[] translation = new double[3];
        translation[ 0 ] = dx;
        translation[ 1 ] = dy;
        translation[ 2 ] = dz;

        final double[] target = new double[] { dx, dy, dz };

        TranslationAnimator translationAnimator = new TranslationAnimator( t, target, 300 );

        bdv.getBdvHandle().getViewerPanel().setTransformAnimator( translationAnimator );
        translationAnimator.setTime( System.currentTimeMillis() );
        bdv.getBdvHandle().getViewerPanel().requestRepaint();

        //affineTransform3D.translate( translation );

        //bdv.getBdvHandle().getViewerPanel().setCurrentViewerTransform( affineTransform3D );


    }

    public BdvSource addSourceFromTiffFile( String filepath )
    {
        ImagePlus imp = IJ.openImage( filepath );
        Img img = ImageJFunctions.wrap( imp );

        //AffineTransform3D prosprScaling = new AffineTransform3D();
        //prosprScaling.scale( PROSPR_SCALING_IN_MICROMETER );

        final BdvSource source = BdvFunctions.show( img, "image" , Bdv.options().addTo( bdv ) );

        bdv = source.getBdvHandle();

        return source;

    }



    public static void main( final String[] args )
    {
        final net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        BigDataViewerObjectBrowsing browsing = new BigDataViewerObjectBrowsing();

    }

}
