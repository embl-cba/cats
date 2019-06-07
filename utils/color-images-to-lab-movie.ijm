// INPUT

directoryWithImages = "/Users/tischer/Documents/sam/Resized"

// CODE

run("Close All");
run("Image Sequence...", "open=["+directoryWithImages+"] sort use");
rename("input_images");

setBatchMode(true);
numImages = nSlices;
for (i = 1; i <= numImages; i++) {
	selectWindow("input_images");
	run("Duplicate...", "duplicate range="+i+"-"+i);
	rename("image_"+i);
	selectWindow("image_"+i);
	run("RGB to CIELAB");
	rename("LAB_" + i );	
	selectWindow("image_"+i); 
	run("Make Composite");
	run("Re-order Hyperstack ...", "channels=[Slices (z)] slices=[Channels (c)] frames=[Frames (t)]");
	run("32-bit");
}

selectWindow("input_images"); close();
run("Concatenate...", "all_open");
run("Stack to Hyperstack...", "order=xyczt(default) channels=6 slices=1 frames="+numImages+" display=Grayscale");

// set channel colors 
c = 1;
// LAB for segmentation
Stack.setChannel(c++);run("Grays");
Stack.setChannel(c++);run("Grays");
Stack.setChannel(c++);run("Grays");
// RGB for visualisation
Stack.setChannel(c++);run("Red");setMinAndMax(0, 255);
Stack.setChannel(c++);run("Green");setMinAndMax(0, 255);
Stack.setChannel(c++);run("Blue");setMinAndMax(0, 255);
Stack.setDisplayMode("composite");
// set channel visibility (we do not need to see the LAB) 
Stack.setActiveChannels("000111");
setBatchMode(false);
