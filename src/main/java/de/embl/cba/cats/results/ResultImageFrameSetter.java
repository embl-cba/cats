package de.embl.cba.cats.results;

public interface ResultImageFrameSetter {
    void set( long x, long y, long z, int classId, double certainty );

    void close();
}
