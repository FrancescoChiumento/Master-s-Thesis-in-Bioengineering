# Comprehensive Femoral Cartilage Segmentation via Atlas-Based Registration and UNet Deep Learning
## Utilizing 3T Sagittal DP Cube MR Imaging for Advanced Cartilage Analysis
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue)
[![Thesis](https://img.shields.io/badge/thesis-link-<COLOR>.svg)](https://hdl.handle.net/20.500.12608/62076)

In this thesis conducted at the Computational Bioengineering Laboratory of the Rizzoli Orthopaedic Institute in Bologna, two innovative methods for femoral cartilage segmentation were implemented: one method leverages the creation of an Average Atlas as a reference within the pyKNEEr software, and the other employs a neural network UNet developed using PyTorch.

- The registration-based method includes optimization phases for Elastix parameters and performance evaluation through cross-correlation of results. 

- The neural network approach involves an initial model training phase, validation of outcomes, followed by testing. The 2D segmentations obtained are then reconstructed into 3D volumes and post-processed for enhanced accuracy.

Subsequently, a statistical assessment can be performed using violin plots and Pearson correlation, while the average Hausdorff index is employed for spatial accuracy evaluation.


