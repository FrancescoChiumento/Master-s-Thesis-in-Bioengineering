import os
import numpy as np
import SimpleITK as sitk
from PIL import Image

def combine_slices_to_mha(input_dir, output_file, spacing, should_flip=False):
    # List all png files in the input directory
    png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    # Sort files by slice number
    png_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Read and stack images
    slices = []
    for png in png_files:
        img = Image.open(os.path.join(input_dir, png))
        img_array = np.array(img.convert('L'))  # Convert to grayscale

        # Flip the image if the should_flip flag is True
        if should_flip:
            img_array = np.fliplr(img_array)

        slices.append(img_array)

    # Stack slices into a 3D array
    img_3d = np.stack(slices, axis=0)

    # Convert to SimpleITK image
    sitk_img = sitk.GetImageFromArray(img_3d)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin((0, 0, 0))
    
    # Set direction for 3D image
    # Assuming RAI orientation for a 3D image, the direction matrix is identity
    sitk_img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    # Write to MHA file
    sitk.WriteImage(sitk_img, output_file)

# Spacing for the image: assuming the provided spacing is for a 3D image
spacing = (0.4121, 0.4121, 0.4)

# Set to True if you want to flip the images, otherwise set to False
should_flip = True  # Change this to True if flipping is needed

# Paths
input_dir = r'D:\U-Net\U-Net cartilagini_NUOVO_ALLENAMENTO_NON_LESIONATI_SENZA_SAO0\prediction'  # Adjust the path to your PNG files
output_file = r'D:\U-Net\U-Net cartilagini_NUOVO_ALLENAMENTO_NON_LESIONATI_SENZA_SAO0\SAO0_fc_UNET.mha'  # Make sure to include the .mha extension

combine_slices_to_mha(input_dir, output_file, spacing, should_flip)
