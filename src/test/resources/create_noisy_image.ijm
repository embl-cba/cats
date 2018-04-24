newImage("Untitled", "8-bit black", 200, 200, 1);
run("Add Noise");
setThreshold(25, 255);
setOption("BlackBackground", true);
run("Convert to Mask");
makeOval(5, 5, 48, 48);