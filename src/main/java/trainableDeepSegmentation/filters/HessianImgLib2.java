package trainableDeepSegmentation.filters;

/**
 * Created by tischi on 02/06/17.
 */


import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ij.ImagePlus;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gradient.HessianMatrix;
import net.imglib2.algorithm.linalg.eigen.TensorEigenValues;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.outofbounds.OutOfBoundsBorderFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;



public class HessianImgLib2
{

    public < T extends NumericType< T > & NativeType< T >> void run( ImagePlus imp, final int nThreads ) throws IncompatibleTypeException, InterruptedException, ExecutionException
    {

        Img< T > img = ImageJFunctions.wrap(imp);
        final long[] dim = Intervals.dimensionsAsLongArray(img);


        byte[] pixels = new byte[imp.getHeight()*imp.getWidth()*imp.getNSlices()];
        int pos = 0;
        for ( int z = 0; z < imp.getNSlices(); z ++ )
        {
            byte[] p = (byte[]) imp.getStack().getPixels(z+1);
            System.arraycopy(p, 0, pixels, pos, p.length);
            pos += p.length;
        }

        final ArrayImg< UnsignedByteType, ByteArray> arrayImg = ArrayImgs.unsignedBytes(pixels, dim);

        //ImageJFunctions.show( arrayImg );

        final ExecutorService es = Executors.newFixedThreadPool(nThreads);

        final RandomAccessibleInterval< UnsignedByteType > hessian =
                HessianMatrix.calculateMatrix(
                        Views.extendBorder(arrayImg),
                        ArrayImgs.unsignedBytes(dim[0], dim[1], dim[2], 3),
                        ArrayImgs.unsignedBytes(dim[0], dim[1], dim[2], 6),
                        new OutOfBoundsBorderFactory<>(),
                        nThreads,
                        es);


        //Views.
        ///ArrayImgs.
        //RandomAccessible<T> ra = Views.extendBorder( img );

        //RandomAccessible<T> ra = img.randomAccess();

        //RandomAccessibleInterval< T > view = Views.interval(img, new FinalInterval(imp.getWidth(), imp.getHeight(),
         //       imp.getNSlices()));


        //Views.extendBorder( Views.interval( ImageJFunctions.wrapReal( imp )

        /*
        new ImageConverter( imp ).convertToGray32(); // is this necessary?
        final IntervalView< ByteType > wrapped = Views.interval( ImageJFunctions.wrapByte(imp),
                new FinalInterval(imp.getWidth(), imp.getHeight(), imp.getNSlices()));



        /*
        final double min = imp.getDisplayRangeMin();
        final double max = imp.getDisplayRangeMax();
        final BdvStackSource< FloatType > raw = BdvFunctions.show( wrapped, "raw" );
        final BdvHandle bdv = raw.getBdvHandle();
        raw.setDisplayRange( min, max );
        */

        /*
        final double sig = 2.0;
        final double[] sigma = new double[] { 1.0 * sig, 1.0 * sig, 0.1 * sig };


        final ArrayImg< DoubleType, DoubleArray > gaussian = ArrayImgs.doubles(dim);
        Gauss3.gauss(sigma, Views.extendBorder(wrapped), gaussian);

        final ExecutorService es = Executors.newFixedThreadPool( nThreads );

        final RandomAccessibleInterval< DoubleType > hessian =
                HessianMatrix.calculateMatrix(
                        gaussian,
                        ArrayImgs.doubles(dim[0], dim[1], dim[2], 3),
                        ArrayImgs.doubles(dim[0], dim[1], dim[2], 6),
                        new OutOfBoundsBorderFactory<>(),
                        nThreads,
                        es);*/


        final RandomAccessibleInterval< DoubleType > evs = TensorEigenValues.calculateEigenValuesSymmetric( hessian, TensorEigenValues.createAppropriateResultImg( hessian, new ArrayImgFactory<>(), new DoubleType() ), nThreads, es );

        es.shutdown();

        for ( int d = 0; d < evs.dimension( evs.numDimensions() - 1 ); ++d )
        {
            final IntervalView< DoubleType > hs = Views.hyperSlice( evs, evs.numDimensions() - 1, d );
            ImagePlus imp2 = ImageJFunctions.wrap(hs, "imglib2");
            imp2.show();

            /*
            double minVal = Double.MAX_VALUE;
            double maxVal = -Double.MIN_VALUE;
            for ( final DoubleType h : hs )
            {
                final double dd = h.get();
                minVal = Math.min( dd, minVal );
                maxVal = Math.max( dd, maxVal );
            }

            final DoubleType finalMinVal = new DoubleType( minVal );
            final double norm = 1.0 / ( maxVal - minVal );
            final double maxIntensity = 255;
            final double factor = maxIntensity * norm;

            final ConvertedRandomAccessibleInterval< DoubleType, DoubleType > hsStretched = new ConvertedRandomAccessibleInterval<>( hs, ( s, t ) -> {
                t.set( s );
                t.sub( finalMinVal );
                t.mul( factor );
            }, new DoubleType() );
            */

            //ImageJFunctions.show( hs );

            // wrap back to IJ1

        }




    }
}