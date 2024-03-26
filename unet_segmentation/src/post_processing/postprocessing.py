import os
import SimpleITK as sitk
import numpy as np

def levelset2binary(mask_LS_itk):
    # Convert the level set mask into a binary mask
    mask_LS_np = sitk.GetArrayFromImage(mask_LS_itk)
    mask_B_np = mask_LS_np > 0.0  # bool
    mask_B_np = mask_B_np.astype(int)  # int
    mask_B_itk = sitk.GetImageFromArray(mask_B_np)
    mask_B_itk.SetSpacing(mask_LS_itk.GetSpacing())
    mask_B_itk.SetOrigin(mask_LS_itk.GetOrigin())
    mask_B_itk.SetDirection(mask_LS_itk.GetDirection())
    # mask_B_itk = sitk.Cast(mask_B_itk, sitk.sitkUInt8)##########

    return mask_B_itk

def process_and_save_image(file_path, output_folder, override_label=None):
    image = sitk.ReadImage(file_path)

    # Perform connected component analysis on the original image
    labels = sitk.ConnectedComponent(image)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labels)

    # Get the dimensions of the labels
    label_sizes = {l: stats.GetNumberOfPixels(l) for l in stats.GetLabels() if l != 0}
    print(f"Dimensioni delle etichette: {label_sizes}")

    # Select the label to overwrite, if specified.
    if override_label and override_label in label_sizes:
        selected_label = override_label
    else:
        selected_label = max(label_sizes, key=label_sizes.get)

    print(f"Selected label: {selected_label} with {label_sizes[selected_label]} pixels")

    # Create a binary image for the selected component
    binary_image = sitk.BinaryThreshold(labels, lowerThreshold=selected_label, upperThreshold=selected_label, insideValue=255, outsideValue=0)

    mask = levelset2binary(binary_image)
    mask = sitk.Cast(mask, sitk.sitkInt16)

    # Retrieve the data as a NumPy array, permute the dimensions, and then convert back to a SimpleITK image
    mask_np = sitk.GetArrayFromImage(mask)
    mask_np_permuted = np.transpose(mask_np, (2, 1, 0))  # Permute the dimensions

    # Create a new SimpleITK image from the permuted array.
    mask_permuted = sitk.GetImageFromArray(mask_np_permuted)
    
    # Set the spacing and origin as in the original mask
    mask_permuted.SetSpacing((0.4121, 0.4121, 0.4))  # Permuted spacing
    mask_permuted.SetOrigin(mask.GetOrigin())

    # Save the image with correct spacing and dimensions
    modified_file_path = os.path.join(output_folder, os.path.basename(file_path).replace('.mha', '_modified.mha'))
    sitk.WriteImage(mask_permuted, modified_file_path)


original_folder = r'your/path/here'
processed_folder = r'your/path/here'

for file_name in os.listdir(original_folder):
    if file_name.endswith('.mha'):
        file_path = os.path.join(original_folder, file_name)
        override_label = None  # Specify the label to overwrite if necessary.
        process_and_save_image(file_path, processed_folder, override_label)
