This folder contains the Jupyter notebooks and the corresponding Python codes for the training, validation, and testing of a 2D UNet neural
network.

For network testing, two approaches have been devised:

- Testing the network involves extracting individual slices from MRI images and, if available, their corresponding binary masks.
  The extracted slices are saved as .png files for segmentation, followed by the reconstruction of three-dimensional volumes and postprocessing.
  The binary masks, when present, are used to calculate the DICE coefficient for each slice compared to the ground truth;

- In the second approach, network testing is performed without saving the slices and their binary masks (when applicable) as .png files.
  This method directly extracts slices for segmentation, proceeding with the saving of three-dimensional segmentations and postprocessing.
  Here too, if binary masks are available, they are employed to compute the DICE coefficient by comparing each slice against the ground truth.
