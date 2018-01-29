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

