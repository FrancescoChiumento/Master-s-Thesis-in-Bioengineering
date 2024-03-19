# Francesco Chiumento, 2023

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

def compute_hausdorff_distance(mask_1, mask_2):
    """
    Compute the Hausdorff distance between two binary masks.

    The Hausdorff distance is a measure of the extent to which two subsets of a metric space are close to each other. 
    This function calculates the Hausdorff distance between two binary masks, ensuring that they are of the same pixel type.
    If the pixel types differ, the second mask is cast to the same pixel type as the first mask before the calculation.
    
    Parameters
    ----------
    mask_1 : SimpleITK.Image
        The first input binary mask.
    mask_2 : SimpleITK.Image
        The second input binary mask.

    Returns
    -------
    hausdorff_distance : float
        The Hausdorff distance between the two input masks.

    Notes
    -----
    This function requires the masks to be of the same pixel type. If they are not, it automatically casts `mask_2` to the 
    pixel type of `mask_1` before computing the Hausdorff distance.

    Examples
    --------
    >>> mask1 = sitk.ReadImage('path/to/mask1.nii')
    >>> mask2 = sitk.ReadImage('path/to/mask2.nii')
    >>> hd = compute_hausdorff_distance(mask1, mask2)
    >>> print(f"Hausdorff Distance: {hd}")
    """    
    if mask_1.GetPixelID() != mask_2.GetPixelID():
        mask_2 = sitk.Cast(mask_2, mask_1.GetPixelID())
    
    hausdorffFilter = sitk.HausdorffDistanceImageFilter()
    hausdorffFilter.Execute(mask_1, mask_2)
    hausdorff_distance = hausdorffFilter.GetHausdorffDistance()
    
    return hausdorff_distance

def optimized_compute_average_hausdorff_distance(mask_1, mask_2):
    """
    Calculate the average Hausdorff distance between two binary masks.

    This function computes the average Hausdorff distance between two binary masks, 
    optimizing the calculation by using KD-trees for efficient nearest-neighbor 
    search. The average Hausdorff distance is computed by first converting the masks 
    into point sets, then calculating the mean distance from each point in one set 
    to the nearest point in the other set, and vice versa. The final distance is the 
    average of these two mean distances.

    Parameters
    ----------
    mask_1 : sitk.Image
        The first binary mask as a SimpleITK Image.
    mask_2 : sitk.Image
        The second binary mask as a SimpleITK Image.

    Returns
    -------
    float
        The average Hausdorff distance between the two binary masks.

    Examples
    --------
    >>> mask1 = sitk.ReadImage('path/to/mask1.nii')
    >>> mask2 = sitk.ReadImage('path/to/mask2.nii')
    >>> avg_hausdorff_distance = optimized_compute_average_hausdorff_distance(mask1, mask2)
    >>> print(f"Average Hausdorff Distance: {avg_hausdorff_distance}")

    Notes
    -----
    The binary masks should be of the same dimensions and have binary values 
    (0 and 1, where 1 represents the object of interest). This function assumes 
    that the masks are already preprocessed and ready for comparison.
    """    
    array_1 = sitk.GetArrayFromImage(mask_1)
    array_2 = sitk.GetArrayFromImage(mask_2)

    coords_1 = np.array(np.where(array_1)).T 
    coords_2 = np.array(np.where(array_2)).T
    
    tree_1 = cKDTree(coords_1)
    tree_2 = cKDTree(coords_2)
    
    distances_1_to_2, _ = tree_1.query(coords_2) 
    distances_2_to_1, _ = tree_2.query(coords_1)
    
    avg_hausdorff = (np.mean(distances_1_to_2) + np.mean(distances_2_to_1)) / 2.0
    
    return avg_hausdorff
