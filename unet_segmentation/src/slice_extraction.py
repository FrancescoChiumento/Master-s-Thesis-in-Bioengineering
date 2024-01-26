import SimpleITK as sitk
import os
from PIL import Image, ImageOps
import numpy as np

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



images_dir = r"D:\U-Net\U-Net cartilagini_NUOVO_ALLENAMENTO_NON_LESIONATI_SENZA_SAO0\images"
masks_dir = r"D:\U-Net\U-Net cartilagini_NUOVO_ALLENAMENTO_NON_LESIONATI_SENZA_SAO0\masks"

output_image_dir = r"D:\U-Net\U-Net cartilagini_NUOVO_ALLENAMENTO_NON_LESIONATI_SENZA_SAO0\to_segment"
output_mask_dir = r"D:\U-Net\U-Net cartilagini_NUOVO_ALLENAMENTO_NON_LESIONATI_SENZA_SAO0\ground_truth"

image_files = [f for f in os.listdir(images_dir) if f.endswith('.mha')]
mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.mha')]

assert len(image_files) == len(mask_files), "Numero di immagini e maschere non corrisponde."

for img_file, mask_file in zip(image_files, mask_files):
    img_name = os.path.splitext(img_file)[0]
    mask_name = os.path.splitext(mask_file)[0]
    
    print(f"Processing image: {img_name}")
    print(f"Corresponding mask: {mask_name}")

    assert img_name == mask_name, f"Immagine e maschera non corrispondono: {img_name}, {mask_name}"
    
    img_mha_path = os.path.join(images_dir, img_file)
    print(f"Reading image file: {img_mha_path}")
    extract_slices(img_mha_path, output_image_dir, img_name)
    
    mask_mha_path = os.path.join(masks_dir, mask_file)
    print(f"Reading mask file: {mask_mha_path}")
    extract_slices(mask_mha_path, output_mask_dir, mask_name, is_mask=True)