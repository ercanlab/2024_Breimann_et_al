<div align="center">
  
# Documentation for Breimann et al. 2024

</div>

## _**Analysis of developmental gene expression using smFISH and in silico staging of C. elegans embryos**_

 
Laura Breimann, Ella Bahry, Marwan Zouinkhi, Klim Kolyvanov, Lena Annika Street, Stephan Preibisch and Sevinç Ercan

bioRxiv: _link_ ; doi: 

<br /> 

The complete raw imaging data can be found [here](https://data.janelia.org/3Mq4F). Images are split into subfolders of genotypes N2 (wild-type), CB428 (dpy-21 (e428), SEA-12 (ERC41 (ers8[delX:4394846-4396180]))) or RNAi treatment (sdc-2 (C35C5.1), dpy-27 (R13G10.1), ama-1 (F36A4.7), ev (empty-vector), rluc (Renilla luciferase)). Additionally, the embryo masks are provided in the file mask.zip. The manual annotation of the embryo stage and the image tiles used in the autoencoder's training can be found in the manual_count.txt and tiles.zip, respectively. 

<br />

### Content

* [_**1.	Abstract**_](#abstract)
* [_**2.	Mask creation**_](#mask)
* [_**3.	Staging**_](#staging)
* [_**4.	smFISH detection**_](#smFISH)
* [_**5.	Plotting and statistical analysis**_](#plotting)


<br />

<div style="text-align: justify">
 
### 1.	Abstract<a name="abstract">
</a> 

<br />
Regulation of transcription during embryogenesis is key to development and differentiation. To study transcript expression throughout Caenorhabditis elegans embryogenesis at single-molecule resolution, we developed a high-throughput single-molecule fluorescence in situ hybridization (smFISH) method that relies on computational methods to developmentally stage embryos and quantify individual mRNA molecules in single embryos. We applied our system to sdc-2, a zygotically transcribed gene essential for hermaphrodite development and dosage compensation. We found that sdc-2 is rapidly activated during early embryogenesis by increasing both the number of mRNAs produced per transcription site and the frequency of sites engaged in transcription. Knockdown of sdc-2 and dpy-27, a subunit of the dosage compensation complex (DCC), increased the number of active transcription sites for the X chromosomal gene dpy-23 but not the autosomal gene mdh-1, suggesting that the DCC reduces the frequency of dpy-23 transcription. The temporal resolution from in silico staging of embryos showed that the deletion of a single DCC recruitment element near the dpy-23 gene causes higher dpy-23 mRNA expression after the start of dosage compensation, which could not be resolved using mRNAseq from mixed-stage embryos. In summary, we have established a computational approach to quantify temporal regulation of transcription throughout C. elegans embryogenesis and demonstrated its potential to provide new insights into developmental gene regulation. 
<br />


### 2. Mask creation<a name="mask">
</a> 


Single fields of view contained several embryos, which were segmented for downstream analysis. An instance segmentation was performed on the max projection of the GFP channel, which contained an autofluorescence signal marking the entire embryo. We used a 2D [StarDist](https://github.com/stardist/stardist) implementation, an approach for nuclei/cell detection with a star convex shape, to extract individual embryos (Schmidt et al., 2018). A ground-truth dataset was created using a [random forest classifier](https://github.com/PreibischLab/image_RF) and a custom ellipsoid fit script implemented in Java, resulting in 1734 fully annotated images. This dataset was split into training (85% - 1168 images) and validation (15% - 206 images) sets. As preprocessing steps, each GFP max projection image was normalized and resized from 1024x1024 to 512x512. StarDist was configured to predict 32 rays (from the pixel to the object outline). All ground-truth (GT) embryo instances were detected (true positives) with a Jaccard similarity score  > 0.75. Instances of false positives, where embryos that were not annotated as GT were detected, occurred in 7 (2.83%) predicted instances. Post-processing for each embryo included predicted label resizing back to the original size and creating a cropped individual embryo image (40 pixels padding). Analysis scripts can be found here: [Stardist prediction](https://github.com/PreibischLab/nd2totif-maskembryos-stagebin-pipeline/blob/master/2_stardist_predict.py)


<img src="https://github.com/ercanlab/2024_Breimann_et_al/blob/main/images/mask_creation.jpg" alt="Embro segmentation using StarDist" width="800">


<br />

### 3. Staging<a name="staging"> 
</a> 


Classification of different embryos into selected age bins was done by training an autoencoder-based classifier for stage prediction. The developmental bins used for staging (1-4, 5-30, 31-99, 100-149, 150-534, 535-558) were a compromise, selected based on biological meaning and classification prediction capability. A validation set comprising 250 embryos was employed to validate the autoencoder training. The validation of the classifier results was done using a class-stratified leave-one-out approach. For preprocessing of all images (training set and full dataset), 3D DAPI channel images were first masked for each embryo, then the 21 central slices were extracted, and pixel intensities were normalized using only non-zero values. Then, slightly modified copies were generated using small shifts, flips, rotations, shear, and brightness changes (augmentation). From those copies, 750 2D tiles of size 64x64 were extracted from random image regions for each embryo. All tiles of the embryo were used for training and inference.
First, an autoencoder was trained in an auto-associative, self-supervised manner to reconstruct its input, enabling it to learn a latent representation of the images. The autoencoder utilized mean squared error as the loss function and employed the Adam optimizer for training. In the subsequent step, the pre-trained encoder was integrated with a classifier designed to predict embryo stages, where the classification is achieved through a softmax output layer corresponding to the stage bins. This new network was initially trained with frozen encoder weights and then finally tuned by training the full network. Cross-entropy was used as the loss function for the classifier, and the Adam optimizer was used for optimization. A stage was predicted for each 64x64 tile of an embryo image, and the final stage for each embryo was determined by a majority vote of all tiles.


Autoencoder training: [Script](https://github.com/ercanlab/2024_Breimann_et_al/blob/main/scripts/analysis/nuclei_bin_train.py)


A training dataset was manually annotated using a custom-written ImageJ macro. We annotated ~100 embryos with this approach, but we faced a strong class imbalance due to the initial random sampling of embryos. Therefore, an active learning approach (human in the loop) was utilized. Embryos for additional manual annotation were chosen based on low certainty in the classifier's initial predictions, focusing on classes lacking sufficient examples. This method allowed us to move away from random sampling in subsequent iterations, strategically annotating stages with fewer examples to balance the distribution across classes. The training dataset and tiles can be found [here](https://s3proxy.janelia.org/celegans-screen/)

Scripts to predict the stage: [Prediction of the developmental stage](https://github.com/PreibischLab/nd2totif-maskembryos-stagebin-pipeline/blob/master/4_stage_prediction.py)

<img src="https://github.com/ercanlab/2024_Breimann_et_al/blob/main/images/stage_prediction.jpg" alt="Prediction of embryo stage" width="800">



<br />


### 4. smFISH detection<a name="smFISH"> 
</a> 


Detection of single RNA spots in 3D was performed using the Fiji plugin [RS-FISH](https://github.com/PreibischLab/RS-FISH) (Bahry et al., 2022). For this, images were preprocessed by subtracting a duplicated and median filtered (sigma = 19) image from the raw image to increase single spots and smooth background signals [Median filter](https://github.com/PreibischLab/RS-FISH/blob/9f99b29e61ceba594b184a881c9fae8301b32aa2/src/main/java/util/MedianFilter.java) (I). Spot detection was performed using RS-FISH II) (Bahry et al., 2022) with the following detection settings: -i0 0, -i1 65535, -a 0.650, -r 1, -s 1.09, -t 0.001, -sr 2, -ir 0.3, -e 0.5116, -it 33000, -bg 0. 


Spots were filtered using binary masks to exclude spots found from neighboring embryos using the Mask filtering option in RS-FISH with masks created using Stardist as described above (III). [Filtering and Normalization](https://github.com/PreibischLab/RS-FISH/blob/master/src/main/java/corrections/MaskFiltering.java)



Computation was performed in parallel on the cluster and using AWS (Amazon Web Services). 


To correct for z-dependent signal loss, a quadratic function was fitted to the detected spots and used to correct the spot intensity throughout the embryo (IV). [Z correction](https://github.com/PreibischLab/RS-FISH/blob/9f99b29e61ceba594b184a881c9fae8301b32aa2/src/main/java/corrections/MaskFiltering.java)



A gamma function was fitted to the histogram of all found detections, and the maximum of the curve was set to 1 to normalize intensity detection between different embryos in order to allow quantification of transcript numbers (V). [Gamma function](https://github.com/PreibischLab/RS-FISH/blob/master/src/main/java/corrections/ZCorrection.java)


<img src="https://github.com/ercanlab/2024_Breimann_et_al/blob/main/images/spot-detection.jpg" alt="RS_FISH spot detection and processing" width="800">

<br /> 


### 5. Plotting and statistical analysis<a name="plotting">



Example scripts for plotting the data and performing the T-test can be found here: [Plotting and statistical test](https://github.com/ercanlab/2024_Breimann_et_al/tree/main/scripts/plotting).



<br />



