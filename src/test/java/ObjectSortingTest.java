import mcib3d.geom.Object3D;
import mcib3d.geom.Object3DPoint;
import mcib3d.geom.Point3D;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class ObjectSortingTest
{
	public static void main( String... args )
	{

		final ArrayList< Object3D > objects = new ArrayList<>();
		final ArrayList< Double > volumes = new ArrayList<>();

		objects.add( new Object3DPoint( 1, new Point3D( 0,0,0 ) ) );
		objects.add( new Object3DPoint( 1, new Point3D( 0,0,0 ) ) );
		objects.add( new Object3DPoint( 1, new Point3D( 0,0,0 ) ) );

		volumes.add( 1000.0 );
		volumes.add( 100.0 );
		volumes.add( 900.0 );

		final ArrayList< Object3D > sorted = new ArrayList<>( objects );

		for ( Object3D item : objects )
		{
			System.out.println( "" + objects.indexOf( item ) );
		}

		Collections.sort( sorted,
				Comparator.comparing( item -> - volumes.get( objects.indexOf( item ) ) ) );

		for ( Object3D item : objects )
		{
			System.out.println( "" + sorted.indexOf( item ) );
		}
	}
}
