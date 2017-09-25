
Trainable Deep Weka Segmentation
======================
The **Trainable Deep Weka Segmentation** is a Fiji plugin for trainable segmentation using deep convolution. It is heavily based on the original **Trainable Weka Segmentation** plugin.

## Software requirements

You need Fiji running Java 1.8; older version will run Java 1.6, which does not work for this plugin. The easiest way to have Fiji running Java 1.8 is to reinstall Fiji from scratch.

## Hardware requirements

The plugin should work on any computer however the computations are quite heavy and can thus be slow. The code is multi-threaded such that the execution speed scales basically linearly with the number of cores. 

Examples:
- For a convolution depth of 3 we observed a pixel classification speed of ~100 kiloVoxel / second using a 32 core Linux CentOS 7 machine.
- For a convolution depth of 3 we observed a pixel classification speed of ~10 kiloVoxel / second using a 4 core MacBook Air.

## Installation

Download below files and place them in your Fiji plugins folder:
- https://git.embl.de/grp-almf/fiji-plugin-deep-segmentation/raw/master/out/artifacts/fiji_plugin_trainable_deep_segmentation.jar
- https://github.com/tischi/fiji-plugin-bigDataTools/raw/master/out/artifacts/fiji--bigDataTools_.jar
- install the ImageScience plugins:
	- [Fiji > Help > Update ...] 
	- [Manage update sites]
	- [X] ImageScience
 

The latter plugin enables streaming large data sets from disk.

## Usage

### Open a data set

You can either simply load a data set into Fiji or you can use [Plugins > BigDataTools > DataStreamingTools] in order to stream a data set from disk; this is useful for data sets that come close to or exceed the RAM of your computer. 

Once you opened the data set you launch the segmentation via [Plugins > Segmentation > Trainable Deep Segmentation]; the graphical user interface will appear around your data set.

Supported data types:

- The streaming currenlty only works for Tiff or Hdf5 based data.
- The trainable segmentation supports:
    - 2D+c+t, 3D+c+t
    - spatially anisotropic data 

### Define your classes

- The first class must always be the background class
	- e.g., to mark everything that is outside your sample
- Create a class: [Create new class]
- Rename a class: [Settings]
	- The class names will be there and you can change them

### Set up your results image

- [Create result image]
- For creating a disk-resident results image: `[X] Disk`
	- This is recommended, because you'll have your results even if something crashes
- If you do not check `Disk` the results image will be created in RAM

#### Tip

The following folder structure works well:
Assuming the name of your image is 'cell'

- `cell-segmentation-project`
	- `cell-raw`
		- here is your data
	- `cell-classified`
		- here the tool will store the classification results (see below)
	- `cell-labels`
		- here you can store your labels
	- `cell-for-imaris`

### Logging

Information about what is happening is printed into IJ's log window.
In addition, when you chose to save your classification results to disk (see above), another folder with the ending "--log" will be automatically created next to your results folder. The content of the logging window will be constantly written into a file in this folder.

#### Put labels

- in Fiji select the "Freehand line tool" 
	- adapt line width to your sample: 'double click'

- for each class you have to minimally out one label
	- draw a line on you image
	- add label to class by clicking on the class names or by the keyboard shortcuts [1],[2],...

##### Tips

- don't draw the lables to long, because it will take a lot of memory and time to compute the features for this label

#### Train classifier

- [Train classifier]

#### Apply classifier

- Make sure the classifier is up to date, if in doubt again: [Train classifier]
- Using Fiji's rectangle ROI select a x-y region to be classified
- [Apply classifier]
- This will select a minimal z-range
- You can specify a larger z-region by typing into the "Range" field, e.g.
	- '300,500' will classify all z-slices between these numbers

#### Save labels

- [Save labels]
	- This stores all your labels and also the computed image features for each label
- 


### Reloading an existing project

- Open your image
- Start the Trainable Deep Segmentation
- [Load labels]
- [Create result image]
	- If your result image was disk-resident, selecting the existing folder will reload your previous results


### Keyboard shortcuts

- Arrow key up and down: zoom in and out

 
### Tips and tricks

#### How to put your training labels 

As this tool is able to learn long range context you have to really tell it what you want. I recommend always putting a background label just next to the actual label.


### Settings

Very often you could just leave them as is.

- Minimum tile size:
	- auto: currently the only supported setting; hopefully doing a good job.



### Open your project in Imaris

The first aim is to have a folder with files named like below:

- folder-for-imaris
	- class0--C00--T00000.tif
		- this is your raw data
	- class1--C00--T00000.tif
		- this is the classification result for class1
	- class2--C00--T00000.tif
		- ...
	- ...

1. Create a folder, e.g. cell-for-imaris
2. Copy your raw data into this folder and rename it to class0--C00--T00000.tif
3. Use the DataStreaming tools to generate one tif file per class (see below)
4. Open whole folder in Imaris

Note: You do not need to save the background class

#### Using the DataStreamingTools to generate one Tiff file per class

- Open classification results folder in DataStreamingTools
	- `Streaming`
		- `File naming scheme`: select the one with "classified..."
			- adapt the amount of z-slices to match you raw data: you may have to look this up
	- `Saving`
		- [X] Gate
			- Select the gray values that belong to one (or more) class, e.g.
				- class 2: Min = 20, Max = 39
				- class 2 and 3 together: Min = 20, Max = 59
		- [Save as stacks]

#### Open your files in Imaris

...

Tips and tricks:

- If you are working in a file server enviroment, it might be faster to first save locally, and then later copy everything in one go from your local computer to the server
 
## Additional comments

### Tile size

The tile size determines tThe minimal volume that will be classified, kind of the classification 'chunk-size'

- Considerations:
	- The larger you go the more you risk running out of memory
	- Smaller sizes will give you quicker feedback for classifying really small regions during the training
	- Larger sizes will speed up the classification of a really large volume, such as you whole data set. The reason is that the boundary voxels of each tile cannot be used for classification; as the size of the boundary region is fixed (given by the maximal downsampling), the fraction of boundary voxels compared to the full tile volume decreases with the tile size.






