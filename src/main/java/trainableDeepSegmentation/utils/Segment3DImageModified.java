package trainableDeepSegmentation.utils;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import mcib3d.image3d.ImageByte;
import mcib3d.image3d.ImageHandler;
import mcib3d.image3d.ImageInt;
import mcib3d.image3d.ImageShort;

import java.util.Arrays;

/**
 * Created by tischi on 23/06/17.
 */

public class Segment3DImageModified {
    float lowThreshold;
    float highThreshold;
    ImageHandler imgCopy;
    int[] objID;
    int[] IDcount;
    int[] surfList;
    boolean[] IDisAtEdge;
    boolean[] isSurf;
    int width = 1;
    int height = 1;
    int nbSlices = 1;
    int nbVoxels = 1;
    int depth = 8;
    int minSize;
    int maxSize;
    int nbObj = 0;
    int nbSurfPix = 0;
    boolean sizeFilter = true;
    boolean exclude = false;

    public Segment3DImageModified(ImageHandler img, float lowthr, float highthr) {
        this.init(img, lowthr, highthr);
    }

    private void init(ImageHandler img, float lowthr, float highthr) {
        this.imgCopy = img.duplicate();
        this.width = img.sizeX;
        this.height = img.sizeY;
        this.nbSlices = img.sizeZ;
        this.nbVoxels = this.width * this.height * this.nbSlices;
        this.minSize = 1;
        this.maxSize = this.nbVoxels;
        this.sizeFilter = true;
        this.exclude = false;
        if(lowthr <= highthr) {
            this.lowThreshold = lowthr;
            this.highThreshold = highthr;
        } else {
            this.lowThreshold = highthr;
            this.highThreshold = lowthr;
        }

        if(this.depth != 8 && this.depth != 16) {
            throw new IllegalArgumentException("Counter3D class expects 8- or 16-bits images only");
        } else {
            this.nbObj = this.nbVoxels;
            this.imgArrayModifier();
        }
    }

    public Segment3DImageModified(ImagePlus plus, float lo, float hi) {
        ImageHandler img = ImageHandler.wrap(plus);
        this.init(img, lo, hi);
    }

    public int getMaxSizeObject() {
        return this.maxSize;
    }

    public void setMaxSizeObject(int maxSize) {
        this.maxSize = maxSize;
    }

    public int getMinSizeObject() {
        return this.minSize;
    }

    public void setMinSizeObject(int minSize) {
        this.minSize = minSize;
    }

    public int getNbObj() {
        return this.nbObj;
    }


    public void segmentDiagonalConnected() {
        int currID = 0;
        int currPos = 0;
        int minID = 0;
        long start = System.currentTimeMillis();
        this.objID = new int[this.nbVoxels];

        int newCurrID;
        int i;
        int nbPix;
        for(newCurrID = 1; newCurrID <= this.nbSlices; ++newCurrID) {
            for(i = 0; i < this.height; ++i) {
                for(nbPix = 0; nbPix < this.width; ++nbPix) {
                    if(minID == currID) {
                        ++currID;
                    }

                    if(this.imgCopy.getPixel(currPos) != 0.0F) {
                        minID = this.minAntTag(currID, nbPix, i, newCurrID);
                        this.objID[currPos] = minID;
                    }

                    ++currPos;
                }
            }
        }

        this.IDcount = new int[currID];

        for(newCurrID = 0; newCurrID < this.nbVoxels; ++newCurrID) {
            ++this.IDcount[this.objID[newCurrID]];
        }

        this.IDisAtEdge = new boolean[currID];
        Arrays.fill(this.IDisAtEdge, false);
        this.isSurf = new boolean[this.nbVoxels];
        currPos = 0;

        for(newCurrID = 1; newCurrID <= this.nbSlices; ++newCurrID) {
            for(i = 0; i < this.height; ++i) {
                for(nbPix = 0; nbPix < this.width; ++nbPix) {
                    if(this.imgCopy.getPixel(currPos) != 0.0F) {
                        minID = this.objID[currPos];
                        int surfPix = 0;
                        int neigbNb = 0;
                        int neigbZ = newCurrID - 1;

                        while(true) {
                            int pos;
                            int neigbX;
                            int neigbY;
                            if(neigbZ > newCurrID + 1) {
                                if((surfPix == 6 || this.nbSlices <= 1) && (surfPix == 4 || this.nbSlices != 1)) {
                                    this.isSurf[currPos] = false;
                                } else {
                                    this.isSurf[currPos] = true;
                                    ++this.nbSurfPix;
                                }

                                for(neigbZ = newCurrID - 1; neigbZ <= newCurrID + 1; ++neigbZ) {
                                    for(neigbY = i - 1; neigbY <= i + 1; ++neigbY) {
                                        for(neigbX = nbPix - 1; neigbX <= nbPix + 1; ++neigbX) {
                                            if(neigbX >= 0 && neigbX < this.width && neigbY >= 0 && neigbY < this.height && neigbZ >= 1 && neigbZ <= this.nbSlices) {
                                                pos = this.offset(neigbX, neigbY, neigbZ);
                                                if(this.imgCopy.getPixel(pos) != 0.0F) {
                                                    int currPixID = this.objID[pos];
                                                    if(currPixID > minID) {
                                                        this.replaceID(currPixID, minID);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                if(nbPix == 0 || i == 0 || nbPix == this.width - 1 || i == this.height - 1 || this.nbSlices != 1 && (newCurrID == 1 || newCurrID == this.nbSlices)) {
                                    this.IDisAtEdge[minID] = true;
                                }
                                break;
                            }

                            for(neigbY = i - 1; neigbY <= i + 1; ++neigbY) {
                                for(neigbX = nbPix - 1; neigbX <= nbPix + 1; ++neigbX) {
                                    if(neigbX >= 0 && neigbX < this.width && neigbY >= 0 && neigbY < this.height && neigbZ >= 1 && neigbZ <= this.nbSlices) {
                                        pos = this.offset(neigbX, neigbY, neigbZ);
                                        if(this.imgCopy.getPixel(pos) != 0.0F) {
                                            if(this.nbSlices > 1 && (neigbX == nbPix && neigbY == i && neigbZ == newCurrID - 1 || neigbX == nbPix && neigbY == i && neigbZ == newCurrID + 1) || neigbX == nbPix && neigbY == i - 1 && neigbZ == newCurrID || neigbX == nbPix && neigbY == i + 1 && neigbZ == newCurrID || neigbX == nbPix - 1 && neigbY == i && neigbZ == newCurrID || neigbX == nbPix + 1 && neigbY == i && neigbZ == newCurrID) {
                                                ++surfPix;
                                            }

                                            minID = Math.min(minID, this.objID[pos]);
                                        }

                                        ++neigbNb;
                                    }
                                }
                            }

                            ++neigbZ;
                        }
                    }

                    ++currPos;
                }
            }
        }

        newCurrID = 0;

        // filter the objects
        for(i = 1; i < this.IDcount.length; ++i) {
            if(this.IDcount[i] != 0
                    && ( this.IDcount[i] >= this.minSize || this.IDisAtEdge[i] ) // TISCHI: keep object at edges, because the could be larger!
                    && this.IDcount[i] <= this.maxSize )
            // && (!this.exclude || !this.exclude || !this.IDisAtEdge[i]))
            {
                ++newCurrID;
                nbPix = this.IDcount[i];
                this.replaceID(i, newCurrID);
                this.IDcount[newCurrID] = nbPix;
            } else {
                this.replaceID(i, 0);
            }
        }

        this.nbObj = newCurrID;
    }

    public void segment() {
        int currID = 0;
        int currPos = 0;
        int minID = 0;
        long start = System.currentTimeMillis();
        this.objID = new int[this.nbVoxels];

        int newCurrID;
        int i;
        int nbPix;
        for(newCurrID = 1; newCurrID <= this.nbSlices; ++newCurrID) {
            for(i = 0; i < this.height; ++i) {
                for(nbPix = 0; nbPix < this.width; ++nbPix) {
                    if(minID == currID) {
                        ++currID;
                    }

                    if(this.imgCopy.getPixel(currPos) != 0.0F) {
                        minID = this.minAntTag(currID, nbPix, i, newCurrID);
                        this.objID[currPos] = minID;
                    }

                    ++currPos;
                }
            }
        }

        this.IDcount = new int[currID];

        for(newCurrID = 0; newCurrID < this.nbVoxels; ++newCurrID) {
            ++this.IDcount[this.objID[newCurrID]];
        }

        this.IDisAtEdge = new boolean[currID];
        Arrays.fill(this.IDisAtEdge, false);
        this.isSurf = new boolean[this.nbVoxels];
        currPos = 0;
        int[][] dn = new int[6][3];
        dn[0] = new int[]{0, 0, -1};
        dn[1] = new int[]{0, -1, 0};
        dn[2] = new int[]{-1, 0, 0};
        dn[3] = new int[]{0, +1, 0};
        dn[4] = new int[]{+1, 0, 0};
        dn[5] = new int[]{0, 0, + 1};

        for(newCurrID = 1; newCurrID <= this.nbSlices; ++newCurrID) {
            for(i = 0; i < this.height; ++i) {
                for(nbPix = 0; nbPix < this.width; ++nbPix) {
                    if(this.imgCopy.getPixel(currPos) != 0.0F) {
                        minID = this.objID[currPos];
                        int surfPix = 0;
                        int neigbNb = 0;
                        int neigbZ = newCurrID - 1;

                        while(true) {
                            int pos;
                            int neigbX;
                            int neigbY;
                            if(neigbZ > newCurrID + 1) {
                                if((surfPix == 6 || this.nbSlices <= 1) && (surfPix == 4 || this.nbSlices != 1)) {
                                    this.isSurf[currPos] = false;
                                } else {
                                    this.isSurf[currPos] = true;
                                    ++this.nbSurfPix;
                                }

                                for ( int iNeighbour = 0; iNeighbour < 6; iNeighbour++ )
                                {
                                    neigbX = nbPix + dn[iNeighbour][0];
                                    neigbY = i + dn[iNeighbour][1];
                                    neigbZ = currID + dn[iNeighbour][2];

                                    if(neigbX >= 0 && neigbX < this.width && neigbY >= 0 && neigbY < this.height && neigbZ >= 1 && neigbZ <= this.nbSlices) {
                                        pos = this.offset(neigbX, neigbY, neigbZ);
                                        if(this.imgCopy.getPixel(pos) != 0.0F)
                                        {
                                            int currPixID = this.objID[pos];
                                            if (currPixID > minID)
                                            {
                                                this.replaceID(currPixID, minID);
                                            }
                                        }
                                    }
                                }

                                if(nbPix == 0 || i == 0 || nbPix == this.width - 1 || i == this.height - 1 || this.nbSlices != 1 && (newCurrID == 1 || newCurrID == this.nbSlices)) {
                                    this.IDisAtEdge[minID] = true;
                                }
                                break;
                            }


                            for ( int iNeighbour = 1; iNeighbour < 5; iNeighbour++ )
                            {
                                neigbX = nbPix + dn[iNeighbour][0];
                                neigbY = i + dn[iNeighbour][1];

                                if(neigbX >= 0 && neigbX < this.width && neigbY >= 0 && neigbY < this.height && neigbZ >= 1 && neigbZ <= this.nbSlices) {
                                    pos = this.offset(neigbX, neigbY, neigbZ);
                                    if(this.imgCopy.getPixel(pos) != 0.0F) {
                                        if(this.nbSlices > 1 && (neigbX == nbPix && neigbY == i && neigbZ == newCurrID - 1 || neigbX == nbPix && neigbY == i && neigbZ == newCurrID + 1) || neigbX == nbPix && neigbY == i - 1 && neigbZ == newCurrID || neigbX == nbPix && neigbY == i + 1 && neigbZ == newCurrID || neigbX == nbPix - 1 && neigbY == i && neigbZ == newCurrID || neigbX == nbPix + 1 && neigbY == i && neigbZ == newCurrID) {
                                            ++surfPix;
                                        }

                                        minID = Math.min(minID, this.objID[pos]);
                                    }

                                    ++neigbNb;
                                }
                            }

                            ++neigbZ;
                        }
                    }

                    ++currPos;
                }
            }
        }

        newCurrID = 0;

        // filter the objects
        for(i = 1; i < this.IDcount.length; ++i) {
            if(this.IDcount[i] != 0
                    && ( this.IDcount[i] >= this.minSize || this.IDisAtEdge[i] ) // TISCHI: keep object at edges, because the could be larger!
                    && this.IDcount[i] <= this.maxSize )
                    // && (!this.exclude || !this.exclude || !this.IDisAtEdge[i]))
            {
                ++newCurrID;
                nbPix = this.IDcount[i];
                this.replaceID(i, newCurrID);
                this.IDcount[newCurrID] = nbPix;
            } else {
                this.replaceID(i, 0);
            }
        }

        this.nbObj = newCurrID;
    }

    public ImageInt getLabelledObjectsImage3D() {
        return this.buildImg(this.objID);
    }

    public ImageInt getBinaryObjectsImage3D(int value) {
        return this.buildBinaryImg(this.objID, value);
    }

    public ImageStack getBinaryObjectsStack(int value) {
        return this.buildBinaryImg(this.objID, value).getImageStack();
    }

    public ImageStack getLabelledObjectsStack() {
        return this.getLabelledObjectsImage3D().getImageStack();
    }

    public ImageInt getSurfaceObjectsImage3D() {
        this.surfList = new int[this.nbVoxels];

        for(int i = 0; i < this.nbVoxels; ++i) {
            this.surfList[i] = this.isSurf[i]?this.objID[i]:0;
        }

        return this.buildImg(this.surfList);
    }

    public ImageStack getSurfaceObjectsStack() {
        return this.getSurfaceObjectsImage3D().getImageStack();
    }

    private int minAntTag(int initialValue, int x, int y, int z) {
        int min = initialValue;

        int currPos;
        int neigbX;
        for(neigbX = y - 1; neigbX <= y + 1; ++neigbX) {
            for(int neigbX1 = x - 1; neigbX1 <= x + 1; ++neigbX1) {
                if(neigbX1 >= 0 && neigbX1 < this.width && neigbX >= 0 && neigbX < this.height && z - 1 >= 1 && z - 1 <= this.nbSlices) {
                    currPos = this.offset(neigbX1, neigbX, z - 1);
                    if(this.imgCopy.getPixel(currPos) != 0.0F) {
                        min = Math.min(min, this.objID[currPos]);
                    }
                }
            }
        }

        for(neigbX = x - 1; neigbX <= x + 1; ++neigbX) {
            if(neigbX >= 0 && neigbX < this.width && y - 1 >= 0 && y - 1 < this.height && z >= 1 && z <= this.nbSlices) {
                currPos = this.offset(neigbX, y - 1, z);
                if(this.imgCopy.getPixel(currPos) != 0.0F) {
                    min = Math.min(min, this.objID[currPos]);
                }
            }
        }

        if(x - 1 >= 0 && x - 1 < this.width && y >= 0 && y < this.height && z >= 1 && z <= this.nbSlices) {
            currPos = this.offset(x - 1, y, z);
            if(this.imgCopy.getPixel(currPos) != 0.0F && x >= 1 && y >= 0 && z >= 1) {
                min = Math.min(min, this.objID[currPos]);
            }
        }

        return min;
    }

    private int offset(int m, int n, int o) {
        return m + n * this.width + (o - 1) * this.width * this.height >= this.width * this.height * this.nbSlices?this.width * this.height * this.nbSlices - 1:(m + n * this.width + (o - 1) * this.width * this.height < 0?0:m + n * this.width + (o - 1) * this.width * this.height);
    }

    private void replaceID(int oldVal, int newVal) {
        if(oldVal != newVal) {
            int nbFoundPix = 0;

            for(int i = 0; i < this.objID.length; ++i) {
                if(this.objID[i] == oldVal) {
                    this.objID[i] = newVal;
                    ++nbFoundPix;
                }

                if(nbFoundPix == this.IDcount[oldVal]) {
                    i = this.objID.length;
                }
            }

            this.IDcount[oldVal] = 0;
            this.IDcount[newVal] += nbFoundPix;
        }

    }

    private void imgArrayModifier() {
        int index = 0;

        for(int z = 0; z < this.nbSlices; ++z) {
            for(int y = 0; y < this.height; ++y) {
                for(int x = 0; x < this.width; ++x) {
                    float val = this.imgCopy.getPixel(x, y, z);
                    if(val >= this.lowThreshold && val <= this.highThreshold) {
                        this.imgCopy.setPixel(index, val);
                    } else {
                        this.imgCopy.setPixel(index, 0.0F);
                        --this.nbObj;
                    }

                    ++index;
                }
            }
        }

        if(this.nbObj <= 0) {
            IJ.log("No object found");
        }

    }

    private ImageInt buildImg(int[] IDobj) {
        int index = 0;
        ImageShort ima = new ImageShort("Objects", this.width, this.height, this.nbSlices);

        for(int z = 0; z < this.nbSlices; ++z) {
            for(int y = 0; y < this.height; ++y) {
                for(int x = 0; x < this.width; ++x) {
                    int currVal = IDobj[index];
                    if(currVal != 0) {
                        ima.setPixel(x, y, z, currVal);
                    }

                    ++index;
                }
            }
        }

        return ima;
    }

    private ImageInt buildBinaryImg(int[] IDobj, int value) {
        int index = 0;
        ImageByte ima = new ImageByte("Objects", this.width, this.height, this.nbSlices);

        for(int z = 0; z < this.nbSlices; ++z) {
            for(int y = 0; y < this.height; ++y) {
                for(int x = 0; x < this.width; ++x) {
                    int currVal = IDobj[index];
                    if( currVal != 0 ) {
                        ima.setPixel(x, y, z, value);
                    }
                    ++index;
                }
            }
        }

        return ima;
    }

}

