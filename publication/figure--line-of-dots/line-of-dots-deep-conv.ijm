run("Close All");
open("/Users/de.embl.cba.trainableDeepSegmentation.weka/Documents/fiji-plugin-deep-segmentation/publication/figure--line-of-dots/line-of-dots.tif");
run("FeatureJ Structure", "largest smallest smoothing=1 integration=1");
selectWindow("line-of-dots.tif largest structure eigenvalues"); close();
selectWindow("line-of-dots.tif smallest structure eigenvalues"); rename("Orig_StS"); run("Enhance Contrast", "saturated=0.35");

run("Duplicate...", "title=[Orig_StS_3x3]");
run("Bin...", "x=3 y=3 bin=Average"); run("Enhance Contrast", "saturated=0.35");
run("Duplicate...", "title=[Orig_StS_3x3_3x3]"); 

run("Bin...", "x=3 y=3 bin=Average"); run("Enhance Contrast", "saturated=0.35");
run("Enhance Contrast", "saturated=0.35");

run("FeatureJ Hessian", "largest smallest smoothing=1");
selectWindow("Orig_StS_3x3_3x3 smallest Hessian eigenvalues"); close();

selectWindow("Orig_StS_3x3_3x3 largest Hessian eigenvalues"); 
rename("Orig_StS_3x3_3x3_HeL");

selectWindow("Orig_StS_3x3_3x3_HeL"); 
run("Scale...", "x=9 y=9 interpolation=Bilinear average create title=Orig_StS_3x3_3x3_HeL_UpSample");

selectWindow("Orig_StS_3x3_3x3"); 
run("Scale...", "x=9 y=9 interpolation=Bilinear average create title=Orig_StS_3x3_3x3_UpSample");


selectWindow("Orig_StS_3x3_3x3_UpSample"); 
run("Duplicate...", "title=Orig_StS_3x3_3x3_UpSample_gt100");
setThreshold(100, 100000);
run("Convert to Mask");
run("Invert LUT");

selectWindow("Orig_StS_3x3_3x3_HeL_UpSample"); 
run("Duplicate...", "title=Orig_StS_3x3_3x3_HeL_UpSample_gt-15");
setThreshold(-15, 100000);
run("Convert to Mask");
run("Invert LUT");


