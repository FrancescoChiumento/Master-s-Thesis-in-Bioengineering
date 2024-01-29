# Francesco Chiumento, 2023

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

def compute_hausdorff_distance(mask_1, mask_2):
    
    if mask_1.GetPixelID() != mask_2.GetPixelID():
        mask_2 = sitk.Cast(mask_2, mask_1.GetPixelID())
    
    hausdorffFilter = sitk.HausdorffDistanceImageFilter()
    
    hausdorffFilter.Execute(mask_1, mask_2)
    
    hausdorff_distance = hausdorffFilter.GetHausdorffDistance()
    
    return hausdorff_distance

def optimized_compute_average_hausdorff_distance(mask_1, mask_2):
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
