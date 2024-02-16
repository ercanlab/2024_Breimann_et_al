<div align="center">
  
# Documentation for Breimann et al. 2024

</div>

## _**Analysis of developmental gene expression using smFISH and in silico staging of C. elegans embryos**_

 
Laura Breimann, Ella Bahry, Marwan Zouinkhi, Klim Kolyvanov, Lena Annika Street, Stephan Preibisch and Sevinç Ercan

bioRxiv: _link_ ; doi: 




### Content

* _**1.	Abstract**_
* _**2.	Mask creation**_
* _**3.	Staging**_
* _**4.	smFISH detection**_
* _**5.	Plotting and statistical analysis**_


<br />

<div style="text-align: justify">
 
### 1.	Abstract 


Regulation of transcription during embryogenesis is key to development and differentiation. To study transcript expression throughout Caenorhabditis elegans embryogenesis at single-molecule resolution, we developed a high-throughput single-molecule fluorescence in situ hybridization (smFISH) method that relies on computational methods to developmentally stage embryos and quantify individual mRNA molecules in single embryos. We applied our system to sdc-2, a zygotically transcribed gene essential for hermaphrodite development and dosage compensation. We found that sdc-2 is rapidly activated during early embryogenesis by increasing both the number of mRNAs produced per transcription site and the frequency of sites engaged in transcription. Knockdown of sdc-2 and dpy-27, a subunit of the dosage compensation complex (DCC), increased the frequency of active transcription sites for the X chromosomal gene dpy-23 but not the autosomal gene mdh-1. The temporal resolution from in silico staging of embryos showed that the deletion of a single DCC recruitment element near the dpy-23 gene causes higher dpy-23 mRNA expression after the start of dosage compensation, which could not be resolved using mRNAseq from mixed-stage embryos. In summary, we have established a computational approach to quantify temporal regulation of transcription throughout C. elegans embryogenesis and demonstrated its potential to provide new insights into developmental gene regulation. 
<br />
<br />


### 2. Mask creation

Single fields of view contained several embryos, which were segmented for downstream analysis. An instance segmentation was performed on the max projection of the GFP channel, which contained an autofluorescence signal marking the entire embryo. We used a 2D [StarDist](https://github.com/stardist/stardist) implementation, an approach for nuclei/cell detection with a star convex shape, to extract individual embryos (Schmidt et al., 2018). A ground-truth dataset was created using a [random forest classifier](https://github.com/PreibischLab/image_RF) and a custom ellipsoid fit script implemented in Java, resulting in 1734 fully annotated images. This dataset was split into training (85% - 1168 images) and validation (15% - 206 images) sets. As preprocessing steps, each GFP max projection image was normalized and resized from 1024x1024 to 512x512. StarDist was configured to predict 32 rays (from the pixel to the object outline). All ground-truth (GT) embryo instances were detected (true positives) with a Jaccard similarity score  > 0.75. Instances of false positives, where embryos that were not annotated as GT were detected, occurred in 7 (2.83%) predicted instances. Post-processing for each embryo included predicted label resizing back to the original size and creating a cropped individual embryo image (40 pixels padding). Analysis scripts can be found here: [Stardist prediction](https://github.com/PreibischLab/nd2totif-maskembryos-stagebin-pipeline/blob/master/2_stardist_predict.py)


<br />
<br />

### 3. Staging 

Classification of different embryos into selected age bins was done by training an autoencoder for stage prediction (Figure 2). The developmental bins used for staging (1-4, 5-30, 31-99, 100-149, 150-534, 535-558) were a compromise, selected based on biological meaning and classification prediction capability. All images (training set and full dataset) were preprocessed before training and classification, 250 were withheld for validation. 3D DAPI channel images were first masked for each embryo, then the 21 central slices were extracted, pixel intensities were normalized using only non-zero values. Then, slightly modified copies were generated using small shifts, flips, rotations, shear, and brightness changes (augmentation). From those copies, 750 2D tiles of size 64x64 were extracted from different image regions for each embryo. 
First, an autoencoder was trained to learn a latent representation of the image in the encoder part. For this, the input and output layers were the same during training. The autoencoder loss was mean squared error and the chosen optimizer Adam. The pre-trained encoder part was coupled with a classifier part to predict embryo stages in the next step, where the output layer is the softmax of the stage bins. This new network is initially trained with frozen encoder weights, then finally tuned by training the full network. For the classifier loss, we choose cross entropy with and the optimizer Adam. 

Autoencoder training: [Script]()


A training dataset was annotated using a custom-written ImageJ macro. We annotated ~100 embryos with this approach, but we had a strong class imbalance due to the initial random sampling of embryos. However, using the initial prediction in an active learning approach (the user is asked to provide further training data if prediction performance is bad for specific cases), we avoided random sampling in further iterations and thus annotated specific stages with previously fewer examples to even out the classes. 

Model file: [Script]()

A stage was predicted for each 64x64 tile of an embryo image, and the final stage for each embryo was determined by majority vote of all tiles.

Scripts to predict the stage: [Prediction of developmental stage](https://github.com/PreibischLab/nd2totif-maskembryos-stagebin-pipeline/blob/master/4_stage_prediction.py)

<br />
<br />


### 4. smFISH detection 

Detection of single RNA spots in 3D was performed using the Fiji plugin [RS-FISH](https://github.com/PreibischLab/RS-FISH)(Bahry et al., 2022). For this, images were preprocessed by subtracting a duplicated and median filtered (sigma = 19) image from the raw image to increase single spots and smooth background signals [Median filter](). Spot detection was performed using RS-FISH (Bahry et al., 2022) with the following detection settings: -i0 0, -i1 65535, -a 0.650, -r 1, -s 1.09, -t 0.001, -sr 2, -ir 0.3, -e 0.5116, -it 33000, -bg 0. 


Spots were filtered using binary masks to exclude spots found from neighboring embryos using the Mask filtering option in RS-FISH with masks created using Stardist as described above. [Filtering and Normalization](https://github.com/PreibischLab/RS-FISH/blob/master/src/main/java/corrections/MaskFiltering.java)



Computation was performed in parallel on the cluster and using AWS (Amazon Web Services). 


To correct for z-dependent signal loss, a quadratic function was fitted to the detected spots and used to correct the spot intensity throughout the embryo. [Z correction](https://github.com/PreibischLab/RS-FISH/blob/master/src/main/java/corrections/ZCorrection.java)



A gamma function was fitted to the histogram of all found detections, and the maximum of the curve was set to 1 to normalize intensity detection between different embryos in order to allow quantification of transcript numbers.



[Collecting RS-FISH results]()

<br />
<br /> 


### 5. Plotting and statistical analysis

[Preparing data for plotting]()
[Plotting and statistical test]()

<br />
<br />



