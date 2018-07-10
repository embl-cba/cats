package de.embl.cba.cats;

/**
 * Created by de.embl.cba.cats.weka on 12/08/17.
 */
public class UncertaintyRegion implements Comparable<UncertaintyRegion>  {
    public Double maxUncertainty = 0.0;
    public Double avgUncertainty = 0.0;
    public Double sumUncertainty = 0.0;
    public int[] xyzt = new int[4];

    @Override
    public int compareTo( UncertaintyRegion other ){
        return ( this.avgUncertainty.compareTo( other.avgUncertainty ) );
    }

}


