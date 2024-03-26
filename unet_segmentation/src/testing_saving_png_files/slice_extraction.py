from PIL import Image, ImageOps

import SimpleITK as sitk
import os
import numpy as np
import shutil

def extract_slices(mha_path, output_dir, prefix, is_mask=False, desired_size=(512, 512)):
  
    itk_image = sitk.ReadImage(mha_path)
    img_array = sitk.GetArrayFromImage(itk_image)
    
    for i in range(img_array.shape[2]):
        slice = img_array[:, :, i]

        if is_mask:
            slice = slice * 255

        slice_padded = pad_image(slice, desired_size)
        img = Image.fromarray(slice_padded.astype('uint8'))
        img = img.rotate(-180)
        img = ImageOps.mirror(img)
        img_filename = f"{prefix}_{str(i).zfill(4)}.png"
        img.save(os.path.join(output_dir, img_filename))


def pad_image(array, desired_size):
 
    delta_width = desired_size[1] - array.shape[1]
    delta_height = desired_size[0] - array.shape[0]
    top, bottom = delta_height // 2, delta_height - (delta_height // 2)
    left, right = delta_width // 2, delta_width - (delta_width // 2)
    

    return np.pad(array, ((top, bottom), (left, right)), 'constant', constant_values=0)

def process_directory(image_mha_path, mask_mha_path, output_image_dir, output_mask_dir, use_masks=True):

    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Processing image: {image_mha_path}")
    extract_slices(image_mha_path, output_image_dir, os.path.splitext(os.path.basename(image_mha_path))[0], is_mask=False)

    if use_masks:
        os.makedirs(output_mask_dir, exist_ok=True)
        print(f"Processing mask: {mask_mha_path}")
        extract_slices(mask_mha_path, output_mask_dir, os.path.splitext(os.path.basename(mask_mha_path))[0], is_mask=True)

    print("Processing completed.")
    
def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
