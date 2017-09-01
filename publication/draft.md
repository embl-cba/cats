# Deep convolutional feature random forest (DCFRF) classification for image segmentation

## Authors

- Christian Tischer
- Ignacio Arganda-Carreras
- Anna Steyer
- Yannick Schwab
- Rainer Pepperkok

## Tables

### Table_TS_comparison
 
| Software | TWS | DCFRF | Ilastik |
| ... | ... | ... | ... |
| Language | Java | Java | Python |
| OS | All | All | All |



## Figure legends

### Figure_Scheme

Schematic depiction of our deep convolutional feature random forest image segmentation algorithm. To go one resolution layer deeper, images are down-sampled by NxN average binning. At each resolution layer hessian matrix and structure tensor eigenvalue images are computed (HS) of all images coming from the previous resolution. For training and classification, binned images are upsampled using bilinear interpolation before they are fed into the random forest classifier. Widths of boxes indicate number of images, numbers inside boxes are the actual number of images for the example scenario of one 3-D input image. Heights of boxes indicate number of pixels per image.  

### Figure_LineOfDots

Examples of how deep convolution using hessian matrix and structure tensor combined with decision trees segments images.
A) Line of dots.




## Abstract

## Introduction

- The quantification and visualization of the content of images often involves the segmentation of certain structures of interest. In fluorescence microscopy images such segmentation can sometimes be achieved using a simple image processing workflow such as local background subtraction followed by global thresholding. In many cases however, especially in electron microscopy images, such simple workflows do not suffice and more complex processing protocols are required. The development of such adavanced processing protocols can easily take many days, without the guarantee of success, such that electron microscopy are currently often still segmented fully manually. While guaranteed to succeed, manual segmentation is very cumbersome and can require hours to days. For example, segmentation of the endoplasmic reticulum covering the DNA in anaphase cells took ?? hours (@Anna: how long?). Thus, machine learning approaches that automatically learn segmentation rules from sparse annotations are of great interest, because a human needs to segment only a subset of the data set (the annotations) while the remainder is automatically segmented by the machine.  

- Current user friendly machine learning solutions comprise the segmentation toolkit ilastik [Ref_ilastik] and the Trainable Weka Segmentation (TWS) Fiji plugin [Ref_TWS] (@Anna: Shall we mentioned something else here?). Both tools compute for each pixel a feature vector, comprising the values of several image features such as, e.g., Gaussian blurring or Hessian matrix eigenvalues. Subsequently, a random forest algorithm [Ref_Breiman] is trained to segment image pixels based on their feature vectors into different structures (classes). While this approach is very powerful and has proven useful in several occasions (@All: Examples?) deep convolutional neural networks (DCNN) seem even more promising in terms of image segmentation capabilities. DCNNs have achieved high ranking results in segmentation challenges and, in fact, seem conceptually more powerful because of the deep convolution capabilities which are currently lacking in ilastik and TWS. However, as far as we know there currently is no user friendly DCNN implementation that could be used by scientist without programming experience. The reason for this may partly be that, given perfectly annotated data, it currently takes hours to days [Ref_3DUNet] until the NN parameters converge during learning by backpropagation. This is in stark contrast to random forest classifier which can be trained in seconds or few minutes.

- We therefore felt that it is useful to develop a novel user-friendly tool that combines the power of deep convolution with the speed of a random forest classifier. Here, we present the implementation of such a tool as well as some example applications.


## Implementation

The DFRF segmentation tool is implemented as a Fiji plugin and can be installed via Fiji's update manager. The DFRF plugin is based on the Trainable Weka Segmentation (TWS) plugin [Ref_TWS]. Just like the TWS our novel tool shares many features with ilastik (@Ignacio: something else?!). To guide potential users and future developers it is very important to outline similarities and differences of these tools, however, for the sake of readabilty we decided to omit those comparisons in the main text, but opted for a tabular form (see Table_CompareImplementations). 

## Deep feature computation

The scheme by which we compute deep features is depicted in figure Fig_DeepFeat. Starting from an input image we compute the eigenvalues of the hessian matrix (HME) and of the structure tensor (STE) at a fixed small smoothing scale (see Section_Anisotropy for details). These images are average binned by a user-selectable factor (typically 2 or 3) and then again HME and STE are computed on the binned images. This process can be repeated a user-selectable number of times. At the highest depth D the images are typically already very small such that we opted to actually not bin from D-1 to D but rather apply a mean filter in order to preserve more spatial information.

Following our scheme for a 3-D input image the number of images at depth N is 7^N. In order to be able to go deep without an explosion of the number of feature images, we introduced a maximal feature depth (MFD) parameter. With this parameter one can control how many "features of features" are allowed. For example, chosing a value of two would only allow features of the following structure: Orig_HME_Bin_Bin_STE or Orig_STE_Bin_HME but not Orig_HME_Bin_STE_Bin_STE. Like this the number of features at depth N is (@TODO: figure out the formula).
 
### Feature importance and subsetting

The classical way to compute feature importances in RFs is to run an out-of-bag sample of the training data through the RF, and compare the classification results when one feature is exchanged by a random other feature [Ref Breiman]. As we typically have aroun2000 features and 400 trees, executing above recipe for all features takes a considerable amount of time and would thus perturb the interactivity of our tool. We thus opted for a different option: During the training we simply count how often each feature was used in the whole forest. The idea being that features which have been selected only at few nodes (in few trees) are probably not very important for the overall classification outcome. Such rarely used features can be deactived. In our current implementation this mainly speeds up the feature upsampling as this now needs to be done for less features. Currently we still compute all features (also deactivated) because features in later resolution levels are derived from features in earlier resolution levels such that it becomes somewhat involved to figure out which features can be left out during the feature computation stage. Moreover, after deactivating rarely used features we run the RF training once more, only taking into account the active features. Here our intuition is that the RF can learn more informative relations between actually useful features (TODO: test this somehow).
   
### Dealing with anisotropic data

The HM and ST feature computation algorithms are intrinsically dealing with anisotropic data [Ref_ImageScience]. In addition, we deal with a potential anisotropy during the downsampling steps of our algorithm. For example, if the resolution of the input data is 200 nm in x/y and 600 nm in z, the first would downsampling - assuming a downsampling factor of 3 -  would be 3x3x1 (instead of 3x3x3), yielding isotropic data with a (600 nm)^3 voxel size in the next resolution layer. The following binnings would be isotropic in this example. 

### Random forest settings

A random forest has the following settings:
- N.. number of trees
- F.. number of random features per node
- ...

We chose F to be on tenth of the number of input features. Our intuition was that 1/10 is high enough to fetch the best features at each node with a decent probability, leading to a good classification strength of each tree, and low enough to have reasonably uncorrelated trees (the probability to have two sucessive nodes in two different trees using the same feature combination only is (1/10)^2 = 1/100). Both high strength and low correlation or important for a random forest classifier to work well [Ref Breiman].  

### Uncertainty display and navigation

As it for instance is possible in ilastik [Ref_Ilastik] one can activate an uncertainty overlay, showing the classification margin, i.e. the difference between the most and second most likely class probabilities. This helps the user to see where more labelling is needed. However, while this is very useful, we found that is is inefficient to manually find and visit regions of high uncertainty in large data sets. We thus implemented an "uncertainty navigation". During classification we keep track of the average uncertainty in each classified image block and store this information in a sorted list. Using keyboard shortcuts the user can navigate through this list and the corresponding region is highlighted on the input image, enabling the user to efficietnly add more class labels in regions of high uncertainty.
 
### N-D support

The DFRF plugin supports multi-channel and multi-time-point data. In terms of multi-channel support the user can choose which channels should be taken into account for the feature computation. Fatures are computed in all channels independenly; we currently do not compute features combining gray values from multiple channels.

### Big image data handling

The DFRF plugin processes the image data in blocks and can thus handle arbitrarily large images. For both the input as well as the classification image the user has the choice to have either of them fully in RAM or stream the data from/to disk. This streaming functionality requires the Fiji 'bigDataTools' plugin [Download_BigDataTools].

### Storage of training data

The user annotations (labels) for each image can be stored and reloaded. The stored file contains the label ROIs, the respective class label and also, if they were computed already, the feature values. 

## User guide 

### Feature naming scheme

The names of the respective features indicate whether either the hessian matrix (He) or the structure tensor (St) was computed and which eigenvalue was computed, the largest (L), the middle (M), or the smallest (S). In addition the name contains information about the current binning relativ to the original image. For example, 9x9x3_StM_3x3x1_StS_Orig means that first the smallest eigenvalue of the structure tensor (StS)  was computed followed by a 3x3x1 average binning, followed by computing the middle eigenvalue of the structure tensor (StM), followed by a 3x3x3 binnning yielding a 9x9x3 binned image with respect to the original image.

#### Down-sample factor

The down-sample factor (DSF) determines how much the images are down-sampled from one resolution level (L) to the next. The optimal choice depends on the spatial structure of the input data. In general, a smaller value captures more details while a lager value faster approaches larger spatial scales. For example choosing a downsample factor of 2 yields binnings 2^(L) (1,2,4,8,...) while 3 yields binnings of 3^(L) (1,3,9,27,...); for the data shown in this article we used a down-sample factor of 3. 

#### Maximal feature depth
  
In our current architecture we have 6 features (4 for 2-D images), namely 3 (2) eigenvalues of the hessian matrix and 3 (2) eigenvalues of the structure tensor. In addition, we allow also simply keep the downsampled version of each image without an additional feature computed. Thus, at each resolution level there are 7 (5) images that could be derived from all images in the preceeding resolution layer, yielding for the 3-D case 7^(L+1) features (L0:7, L1:49, L2:343, L3:2401, L4:16807, ...) at downsampling level L. Obviously this is a fast growing number, which for instance slows down the RF training that has to test all those features for their usefulness given the current classification task. To keep the numbers of features at bay we thus introduced a maximal feature depth (MFD) which determines how many levels of "features of features" are allowed. For example, given a downsample factor of 3 a feature image at level 2 could be 9x9x9_HeL_3x3x3_StS_1x1x1_Orig. Given a MCD of 2, this feature image could not be subjected to any further filters, but could only be further downsampled. However the image 9x9x9_3x3x3_StS_1x1x1_Orig would be subjected to additional filters, because it was so far only subjected to one filter, namely the smallest eigenvalue of the structure tensor (StS) at resolution level 0.

## Applications 

- We present the segmentation of several challenging EM data sets, showing that our approach is
	- interactive
	- accurate
	- fast
	- big data compatible

### Conceptual example application: line of dots

Before tacking on actual data we demonstrate the concept of our strategy on a simple manufactured example, namely a 2-D image comprising several structures including a line of dots (Figure_LineOfDots). Probably there are several ways of segmenting the line of dots using our frame-work, we simply show one that felt intuitive to us. The smallest eigenvalue of the structure tensor highlights dots, corners, and ends of lines (Orig_StS), thereby already eliminating a number of bright pixels that are not part of the line of dots. Two times 3x3 binning connects the dots (Orig_StS_3x3_3x3). To distinguish pixels on the line from the remaining dots, we compute the largest eigenvalue of the hessian matrix (Orig_StS_3x3_3x3_HeL), which has low absolute values along the line. Finally, we upsample the images and mimic a branch of a simple decision tree, namely Orig_StS_3x3_3x3_UpSample > 60 => Orig_StS_3x3_3x3_HeL > -100. As one can see, this simple recipe sucessfully solves this non-trivial segmentation problem.
 
### C.elegans FIB-SEM

- We have manual ground truth => we can give accuracies

### HeLa interphase cell FIB-SEM


### HeLa mitoic cells FIB-SEM

- measurement: fraction of DNA covered with ER

### ISBI data set

- Ignacio

## Discussion

- To go from one resolution layer to the next we opted for an average binning. Binning has the advantage that the image size decreases such that both computation times and memory requirements decrease as well. For example, for a 3-D input image and a downsampling factor of 3 the number of pixels is reduced by 3x3x3=27 at each resolution. This more than compensates the 7-fold increase in the number of images at each resolution such that computation times and memory requirements are in fact decreasing at deeper resolution layers. This is in stark contrast to the current implementations in ilastik and TWS where larger features are computed by increased kernel width, which does not decrease the memory requirements and, depending on implementation details, even increases the computation times. 
- In neural network implementations of deep convolutions all convolutions are learned during the training. This has the advantage that the NN has the chance to learn the optimal convolutional filters for the given segmentation task. However this comes at a cost of many parameters and long training times. For instance, the 3-D U-Net has about 11 million parameters and took 3 days to train [Ref Ronneberger 3-D UNet]. In addition, deep convolutions encoded by NNs are intrinsically not rotationally invariant such that all rotations need to be explicitely learned and encoded by the NN. Especially in 3D this means that a lot of angles to be learned. As most biological data is rotationally invariant we feel that this is a disadvantage of NNs as compared with our approach where we only use rotationally invariant features. 
- Here we chose to use fixed features, namely the eigenvalues of the hessian matrix and the structure tensor. These features have the advantage of being rotationally invariant and being good descriptors of most biologically relevant structures such as membranes, tubes, and vesicles.
- In neural network implementations of deep convolution the convolution kernels are learned by the NN during the training). However this results in many parameters to be learned and thus long training times (typically hours)). 
- Computing features at higher resolution levels is not done by increasing the kernel width but by down-sampling the input image. This has the advantage of an increased speed during feature computation, as well as reduced memory requirements for storing the feature images. For N-D data with a binning factor of B, the reduction in computation time and memory is a factor of B^N for each resolution layer. For classification, the down-sampled feature images are up-sampled again (just as in the 3-D U-Net), this takes time such that some of the gain in speed is lost; we would like to explore in the future whether this up-sampling could be computed on a GPU in order to save time. As the up-sampling is only needed locally at the location of the current instance voxel, the additional memory requirements at this step are relatively small.

## Acknowledgements

We thank ...

## References

- ilastik
- fiji-tws
- imagescience.org

