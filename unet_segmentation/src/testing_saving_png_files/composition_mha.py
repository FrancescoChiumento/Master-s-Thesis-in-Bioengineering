import os
import numpy as np
import SimpleITK as sitk
from PIL import Image

def combine_slices_to_mha(input_dir, output_file, spacing, should_flip=False):   
    png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')] 
    png_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) 
    slices = []
    for png in png_files:
        img = Image.open(os.path.join(input_dir, png))
        img_array = np.array(img.convert('L'))     
        if should_flip:
            img_array = np.fliplr(img_array)
        slices.append(img_array)
    img_3d = np.stack(slices, axis=0)
    sitk_img = sitk.GetImageFromArray(img_3d)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin((0, 0, 0))
    sitk_img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))  
    sitk.WriteImage(sitk_img, output_file)
spacing = (0.4121, 0.4121, 0.4)
should_flip = True  

