# Comprehensive Femoral Cartilage Segmentation via Atlas-Based Registration and UNet Deep Learning
## Utilizing 3T Sagittal DP Cube MR Imaging for Advanced Cartilage Analysis
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-goldenrod.svg)](https://opensource.org/licenses/GPL-3.0)
![Python 3.8](https://img.shields.io/badge/python-3.8-cornflowerblue)
[![Thesis](https://img.shields.io/badge/thesis-link-palegreen.svg)](https://hdl.handle.net/20.500.12608/62076)

In this thesis conducted at the Computational Bioengineering Laboratory of the Rizzoli Orthopaedic Institute in Bologna, two methods for femoral cartilage segmentation were implemented: one method leverages the creation of an Average Atlas as a reference within the pyKNEEr software, and the other employs a neural network UNet 2D developed using PyTorch.

- The registration-based method includes optimization phases for Elastix parameters and performance evaluation through cross-correlation of results; 

- The neural network approach involves an initial model training phase, validation of outcomes, followed by testing. The 2D segmentations obtained are then reconstructed into 3D volumes and post-processed for enhanced accuracy.

Subsequently, a statistical assessment can be performed using violin plots and Pearson correlation, while the average Hausdorff index is employed for spatial accuracy evaluation.

## Use of the Average Atlas
To utilize the Average Atlas as a new reference in the registration and segmentation of MRIs, it is necessary to first install the pyKNEEr software. [Link pyKNEEr](https://sbonaretti.github.io/pyKNEEr/).

### Average Atlas creation 

The creation of the Average Atlas requires that the files to be used are pre-processed using pyKNEEr

Two types of Average Atlases can be generated:

- Average Atlas of the MRIs and binary masks of the femur and femoral cartilages of the patients;

- Average Atlas of the MRIs with segmentation of the femur and femoral cartilage performed directly from the average MRIs.

[![visualized with ITK-SNAP](https://img.shields.io/badge/visualized%20with-ITK--SNAP-c80000?style=flat)](http://www.itksnap.org)

![Testo alternativo](images/AverageAtlas.gif)

After creating the Average Atlas, it can be utilized within pyKNEEr in the *reference* folder as a new reference.

## Training, testing, and validation of the UNet neural network
For the training, validation, and testing of the 2D UNet neural network, the following percentages of their respective sets were used: training 66.66%, validation 25%, and testing 8.33%.
<br>
![Testo alternativo](images/dataset%20distribution.png)

### Training

### Validation

### Testing

### 3D Volume Reconstruction & Post-processing

## Examples

## References








