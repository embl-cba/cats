package trainableDeepSegmentation.resultImage;

public interface ResultImageFrameSetter {
    void set( long x, long y, long z, int classId, double certainty );

    void close();
}
