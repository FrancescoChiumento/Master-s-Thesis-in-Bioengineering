# Francesco Chiumento, 2023

import os
import subprocess
import SimpleITK as sitk
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pykneer import sitk_functions as sitkf

def get_all_paths(base_dir):
    """
    Generate and return a dictionary of paths related to image registration.

    Parameters
    ----------
    base_dir : str
        The base directory from which other paths are derived.

    Returns
    -------
    dict
        Dictionary containing paths for elastix, transformix, reference images, masks, 
        moving images directory, parameters files, and others. 

    Notes
    -----
    The function assumes a specific directory structure and naming conventions 
    for the files and directories based on the provided base_dir.

    Example
    -------
    >>> base_directory = "/path/to/base/directory"
    >>> paths = get_all_paths(base_directory)
    >>> print(paths["elastix_exe_path"])
    "/path/to/base/directory/elastix/elastix.exe"
    """
    elastix_exe_path = os.path.join(base_dir, "elastix", "elastix.exe")
    transformix_exe_path = os.path.join(base_dir, "elastix", "transformix.exe")
    output_folder_path = os.path.join(base_dir, "outputs")
    images_directory = os.path.join(base_dir, "moving_images")
    moving_images_directory = os.path.join(base_dir, "moving_images_directory")

    # Filename of the femur mask
    femur_mask_files = os.listdir(os.path.join(base_dir, "reference_folder", "femur_mask"))
    femur_mask_file = femur_mask_files[0] if femur_mask_files else None

    # Filename of the femur cartilage mask
    cartilage_mask_files = os.listdir(os.path.join(base_dir, "reference_folder", "cartilage_mask"))
    cartilage_mask_file = cartilage_mask_files[0] if cartilage_mask_files else None

    return {
        "base_dir": base_dir,
        "elastix_exe_path": elastix_exe_path,
        "transformix_exe_path": transformix_exe_path,
        "output_folder_path": output_folder_path,
        "images_directory": images_directory,
        "moving_images_directory": moving_images_directory,
        "fixed_image_path": os.path.join(base_dir, "reference_folder", "reference.mha"),
        "fixed_femur_mask_path": os.path.join(base_dir, "reference_folder", "femur_mask", femur_mask_file) if femur_mask_file else None,
        "fixed_cartilage_mask_path": os.path.join(base_dir, "reference_folder", "cartilage_mask", cartilage_mask_file) if cartilage_mask_file else None,
        "f_mask_folder_path": os.path.join(base_dir, "reference_folder", "femur_mask"),
        "fc_mask_folder_path": os.path.join(base_dir, "reference_folder", "cartilage_mask"),
        "parameters_files": [
            os.path.join(base_dir, "MR_param_rigid.txt"),
            os.path.join(base_dir, "MR_param_similarity.txt"),
            os.path.join(base_dir, "MR_param_spline.txt")
        ]
    }

def run_registration(fixed_image_path, moving_image_path, mask_path, output_folder, parameters_files, elastix_exe_path, initial_transform=None):
    """
    Execute the registration using Elastix.

    Parameters
    ----------
    fixed_image_path : str
        Path to the fixed image.
    moving_image_path : str
        Path to the moving image.
    mask_path : str
        Path to the mask.
    output_folder : str
        Directory to save the registration results.
    parameters_files : list of str
        List of parameter files for the registration.
    elastix_exe_path : str
        Path to the Elastix executable.
    initial_transform : str, optional
        Path to the initial transformation file.

    Returns
    -------
    subprocess.CompletedProcess
        Result of the command execution.

    Notes
    -----
    The function calls the Elastix executable to run the registration. If the registration fails,
    it prints an error message along with the standard output and standard error from the command execution.
    """
    cmd = [elastix_exe_path, "-f", fixed_image_path, "-m", moving_image_path, "-fMask", mask_path, "-out", output_folder]
    for param_file in parameters_files:
        cmd.extend(["-p", param_file])

    if initial_transform:
        cmd.extend(["-t0", initial_transform])

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Error during registration with elastix")
        print(result.stdout)
        print(result.stderr)

    return result

def dilate_mask_function(base_dir):
    """
    Dilate the binary masks of the fixed image and save them.

    This function reads a fixed image from the provided base directory, casts it to 8-bit, and 
    then applies a binary dilation operation. The dilated mask is then saved to both femur and 
    cartilage mask folders.

    Parameters
    ----------
    base_dir : str
        Base directory path from which all other paths are derived.

    Returns
    -------
    None

    Notes
    -----
    The function assumes that the directory structure and filenames are consistent with the 
    definitions in `get_all_paths` function. It saves the dilated masks with the filenames 
    "dilated_femur_mask.mha" and "dilated_cartilage_mask.mha" respectively.
    """
    paths = get_all_paths(base_dir)
    
    # Dilation of the femur mask
    femur_mask_image = sitk.ReadImage(paths["fixed_femur_mask_path"])
    femur_mask_image_8bit = sitk.Cast(femur_mask_image, sitk.sitkUInt8)
    dilated_femur_mask_image = sitk.BinaryDilate(femur_mask_image_8bit, [5]*3)
    dilated_femur_mask_image = sitk.Cast(dilated_femur_mask_image, femur_mask_image.GetPixelID())
    sitk.WriteImage(dilated_femur_mask_image, os.path.join(paths["f_mask_folder_path"], "dilated_femur_mask.mha"))

    # Dilation of the cartilage mask
    cartilage_mask_image = sitk.ReadImage(paths["fixed_cartilage_mask_path"])
    cartilage_mask_image_8bit = sitk.Cast(cartilage_mask_image, sitk.sitkUInt8)
    dilated_cartilage_mask_image = sitk.BinaryDilate(cartilage_mask_image_8bit, [5]*3)
    dilated_cartilage_mask_image = sitk.Cast(dilated_cartilage_mask_image, cartilage_mask_image.GetPixelID())
    sitk.WriteImage(dilated_cartilage_mask_image, os.path.join(paths["fc_mask_folder_path"], "dilated_cartilage_mask.mha"))

    print("Dilated binary masks")

    
def register_moving_images_function(base_dir):
    """
    Execute a two-phase registration process for all moving images in the specified directory.

    This function performs image registration using Elastix. In the first phase, moving images
    are registered to the fixed image using the femur mask. In the second phase, the output of 
    the first phase is registered again using the cartilage mask. The function also handles 
    the saving of the transformed images and masks.

    Parameters
    ----------
    base_dir : str
        Base directory path from which all other paths are derived.

    Returns
    -------
    registered_image_paths : list
        List of paths to the registered images.
    moving_files_path : list
        List of paths to the moving images.
    patient_folder_name : str
        Name of the patient folder.

    Notes
    -----
    The function assumes that the directory structure and filenames are consistent with the 
    definitions in `get_all_paths` function. It saves the registered images and masks in 
    respective directories under the "outputs" folder.
    """
    paths = get_all_paths(base_dir)
    transform_parameters_dict = {}
    moving_files_path = [os.path.join(paths["images_directory"], f) for f in os.listdir(paths["images_directory"]) if f.endswith('.mha')]
    dilated_f_mask_file_path = os.path.join(paths["f_mask_folder_path"], "dilated_femur_mask.mha")
    dilated_fc_mask_file_path = os.path.join(paths["fc_mask_folder_path"], "dilated_cartilage_mask.mha")
    moving_images_dir = paths["moving_images_directory"]
    base_output_folder = paths["output_folder_path"]
    elastix_exe_path = paths["elastix_exe_path"]
    parameters_files = paths["parameters_files"]
    fixed_image_path = paths["fixed_image_path"]
    registered_image_paths = []
    
    for moving_image_path in moving_files_path:
        file_name = os.path.basename(moving_image_path).replace('.mha', '')
        patient_folder_name = os.path.basename(os.path.dirname(moving_image_path))
        moving_femur_mask_path = os.path.join(moving_images_dir, file_name, file_name + '_femur_mask.mha')
        moving_cartilage_mask_path = os.path.join(moving_images_dir, file_name, file_name + '_cartilage_mask.mha')
        fase1_femur_folder = os.path.join(base_output_folder, file_name, "fase1_femur")
        os.makedirs(fase1_femur_folder, exist_ok=True)
        fase2_femur_folder = os.path.join(base_output_folder, file_name, "fase2_femur")
        os.makedirs(fase2_femur_folder, exist_ok=True)
        fase2_cartilage_folder = os.path.join(base_output_folder, file_name, "fase2_cartilage")
        os.makedirs(fase2_cartilage_folder, exist_ok=True)
        
        print(f"\nStarting registration for {moving_image_path}")
        print()
        start_time = time.time()
        
        result = run_registration(fixed_image_path, moving_image_path, dilated_f_mask_file_path, fase1_femur_folder, parameters_files, elastix_exe_path)
        elapsed_time = time.time() - start_time
        if result.returncode != 0:
            print(f"Error during the registration of femur for {moving_image_path}. Error code: {result.returncode}")
            print()
            exit()
        print(f"Registration of the femur completed for {moving_image_path} in {elapsed_time:.2f} seconds.")
        print()

        new_moving_image_path = os.path.join(fase1_femur_folder, "result.2.mha")
        
        # Second phase registration
        print(f"Starting second phase registration for {moving_image_path}")
        print()
        start_time = time.time()
        result = run_registration(fixed_image_path, new_moving_image_path, dilated_fc_mask_file_path, fase2_cartilage_folder, parameters_files, elastix_exe_path)
        if result.returncode != 0:
            print(f"Error during the registration of the femural cartilage for {moving_image_path}. Error code: {result.returncode}")
            print()
            exit()
        print(f"Registration of the femural cartilage completed for {moving_image_path} in {elapsed_time:.2f} seconds.")
        print()
        
        # Rename the transformed image to avoid name conflicts
        original_image_output_path = os.path.join(fase2_cartilage_folder, "result.2.mha")
        renamed_image_output_path = os.path.join(fase2_cartilage_folder, "final_registered_moving_image.mha")
        os.rename(original_image_output_path, renamed_image_output_path)
        registered_image_paths.append(renamed_image_output_path)
        
         # Final transform of the femur mask using Transformix
        transform_parameters_file2_fase1 = os.path.join(fase1_femur_folder, "TransformParameters.2.txt")
        transform_parameters_file2_fase2 = os.path.join(fase2_cartilage_folder, "TransformParameters.2.txt")

        all_transform_parameters_files = [
        transform_parameters_file2_fase1, 
        transform_parameters_file2_fase2 
        ]
        transform_parameters_dict[file_name] = all_transform_parameters_files
        
    with open(os.path.join(paths["base_dir"], "transform_parameters_dict.pkl"), 'wb') as f:
        pickle.dump(transform_parameters_dict, f)

    return registered_image_paths, moving_files_path, patient_folder_name, transform_parameters_dict


def register_femur_mask_function(base_dir, moving_femur_mask_path, all_transform_parameters_files, moving_image_path, patient_folder_name,file_name):
    """
    Execute the registration process for the femur mask.

    This function applies transformations to the femur mask using the provided transform parameter files.
    The mask undergoes a series of transformations, with the result of each transformation being used as
    the input for the next. The function also handles the conversion between binary and level set representations
    of the mask.

    Parameters
    ----------
    base_dir : str
        Base directory path from which all other paths are derived.
    moving_femur_mask_path : str
        Path to the moving femur mask.
    all_transform_parameters_files : list
        List of paths to the transform parameter files.
    moving_image_path : str
        Path to the moving image.
    patient_folder_name : str
        Name of the patient folder.

    Notes
    -----
    The function assumes that the directory structure and filenames are consistent with the 
    definitions in `get_all_paths` function. It saves the transformed masks in the respective 
    directories under the "outputs" folder.
    """
    paths = get_all_paths(base_dir)
    transformix_exe_path = paths["transformix_exe_path"]

    fase2_femur_folder = os.path.join(base_dir, "outputs", file_name, "fase2_femur")


    paths = get_all_paths(base_dir)

    moving_files_path = [os.path.join(paths["images_directory"], f) for f in os.listdir(paths["images_directory"]) if f.endswith('.mha')]
    base_output_folder = paths["output_folder_path"]
    transformix_exe_path = paths["transformix_exe_path"]
    moving_images_dir = paths["moving_images_directory"]

    # Apply the transformations to the femur mask
    input_mask = moving_femur_mask_path
    input_image = sitk.ReadImage(input_mask)
    input_mask_levelset = sitkf.binary2levelset(input_image)
    temp_levelset_path = os.path.join(fase2_femur_folder, "temp_levelset.mha")
    sitk.WriteImage(input_mask_levelset, temp_levelset_path)

    for idx, transform_file in enumerate(all_transform_parameters_files):
        print(f"Starting registration {idx + 1} of the femural mask {moving_image_path}")
        print()
        start_time = time.time()

        output_file_name = "result.mha"
        output_path = os.path.join(fase2_femur_folder, output_file_name)
    
        result = subprocess.run([transformix_exe_path, "-in", temp_levelset_path, "-out", fase2_femur_folder, "-tp", transform_file])
        if result.returncode != 0:
            print(f"Error during the registration {idx + 1} of the femural mask {moving_image_path}. error code: {result.returncode}")
            print()
            exit()

        elapsed_time = time.time() - start_time
        print(f"Transformation {idx + 1} of the femural mask{moving_image_path} completed in  {elapsed_time:.2f} seconds.")
        print()

        # Post-transformation cleaning
        output_image = sitk.ReadImage(output_path)
        binary_output = sitkf.levelset2binary(output_image)
        cleaned_levelset_output = sitkf.binary2levelset(binary_output)
        temp_levelset_path_next = os.path.join(fase2_femur_folder, f"temp_levelset_{idx}.mha")
        sitk.WriteImage(cleaned_levelset_output, temp_levelset_path_next)
        temp_levelset_path = temp_levelset_path_next

    output_image = sitk.ReadImage(output_path)
    output_image_binary = sitkf.levelset2binary(output_image)    
    renamed_image_output_path = os.path.join(fase2_femur_folder, "registered_femur_mask.mha")
    sitk.WriteImage(output_image_binary, renamed_image_output_path)
    return renamed_image_output_path

def register_cartilage_mask_function(base_dir, moving_cartilage_mask_path, all_transform_parameters_files, moving_image_path, patient_folder_name,file_name):
    """
    Execute the registration process for the femoral cartilage mask.

    This function applies transformations to the femoral cartilage mask using the provided transform parameter files.
    The mask undergoes a series of transformations, with the result of each transformation being used as
    the input for the next. The function also handles the conversion between binary and level set representations
    of the mask.

    Parameters
    ----------
    base_dir : str
        Base directory path from which all other paths are derived.
    moving_cartilage_mask_path : str
        Path to the moving femoral cartilage mask.
    all_transform_parameters_files : list
        List of paths to the transform parameter files.
    moving_image_path : str
        Path to the moving image.
    patient_folder_name : str
        Name of the patient folder.

    Notes
    -----
    The function assumes that the directory structure and filenames are consistent with the 
    definitions in `get_all_paths` function. It saves the transformed masks in the respective 
    directories under the "outputs" folder.
    """    
    fase2_cartilage_folder = os.path.join(base_dir, "outputs", file_name, "fase2_cartilage")

    # Apply the transformations to the femur cartilage mask 
    input_mask_fc_image = sitk.ReadImage(moving_cartilage_mask_path)
    input_mask_fc_levelset = sitkf.binary2levelset(input_mask_fc_image)
    temp_levelset_fc_path = os.path.join(fase2_cartilage_folder, "temp_levelset_fc.mha")
    sitk.WriteImage(input_mask_fc_levelset, temp_levelset_fc_path)
    
    paths = get_all_paths(base_dir)
    transformix_exe_path = paths["transformix_exe_path"]

    moving_files_path = [os.path.join(paths["images_directory"], f) for f in os.listdir(paths["images_directory"]) if f.endswith('.mha')]
    base_output_folder = paths["output_folder_path"]
    transformix_exe_path = paths["transformix_exe_path"]
    moving_images_dir = paths["moving_images_directory"]

    for idx, transform_file in enumerate(all_transform_parameters_files):
        print(f"Starting registration {idx + 1} of the femural cartilage mask {moving_image_path}")
        print()
        start_time = time.time()

        output_file_name = "result.mha"
        output_path = os.path.join(fase2_cartilage_folder, output_file_name)

        result = subprocess.run([transformix_exe_path, "-in", temp_levelset_fc_path, "-out", fase2_cartilage_folder, "-tp", transform_file])
        if result.returncode != 0:
            print(f"Error during the transformation {idx + 1} of the femural cartilage mask{moving_image_path}. Error code: {result.returncode}")
            print()
            exit()

        elapsed_time = time.time() - start_time
        print(f"Transformation {idx + 1} of the femural cartilage maskk {moving_image_path} completed in {elapsed_time:.2f} seconds.")
        print()

        # Post-transformation cleaning
        output_image = sitk.ReadImage(output_path)
        binary_output = sitkf.levelset2binary(output_image)
        cleaned_levelset_output = sitkf.binary2levelset(binary_output)
        temp_levelset_fc_path_next = os.path.join(fase2_cartilage_folder, f"temp_levelset_fc_{idx}.mha")
        sitk.WriteImage(cleaned_levelset_output, temp_levelset_fc_path_next)
        temp_levelset_fc_path = temp_levelset_fc_path_next

    output_image = sitk.ReadImage(output_path)
    output_image_binary = sitkf.levelset2binary(output_image)  
    renamed_image_output_path = os.path.join(fase2_cartilage_folder, "registered_femoral_cartilage_mask.mha")
    sitk.WriteImage(output_image_binary, renamed_image_output_path)

    return renamed_image_output_path

def get_registered_image_paths(base_directory):
    """
    Retrieve paths of registered images from the specified directory.

    This function searches for finalized registered images in the "fase2_cartilage" subdirectory 
    of each patient folder within the "outputs" folder and returns a list of their paths.

    Parameters
    ----------
    base_directory : str
        The root directory from which to search for patient folders and their corresponding 
        registered images.

    Returns333
    -------
    list
        A list of tuples containing the patient folder name and the path to the registered images 
        found under each patient's "fase2_cartilage" subdirectory.

    Notes
    -----
    The function assumes a specific directory structure where each patient has a subdirectory named 
    "fase2_cartilage" containing the registered images with the filename 
    "final_registered_moving_image.mha" within the "outputs" folder.
    """
    registered_image_paths = []
    outputs_directory = os.path.join(base_directory, "outputs")
    patient_folders = [d for d in os.listdir(outputs_directory) if os.path.isdir(os.path.join(outputs_directory, d))]

    for patient_folder in patient_folders:
        fase2_folder = os.path.join(outputs_directory, patient_folder, "fase2_cartilage")
        image_path = os.path.join(fase2_folder, "final_registered_moving_image.mha")
        if os.path.exists(image_path):
            registered_image_paths.append((patient_folder, image_path))

    return registered_image_paths

def register_masks_for_image(base_dir, moving_image_path, all_transform_parameters_files):
    """
    Register femur and femoral cartilage masks for a given moving image.

    This function registers the femur and femoral cartilage masks using the transform parameters
    obtained from the previous registration steps. It then calculates and prints the volume of 
    the registered masks.

    Parameters
    ----------
    base_dir : str
        Base directory path from which all other paths are derived.
    moving_image_path : str
        Path to the moving image for which the masks need to be registered.
    all_transform_parameters_files : list of str
        List of paths to the transform parameters files to be used for the registration.

    Returns
    -------
    None

    Notes
    -----
    The function assumes that the directory structure and filenames are consistent with the 
    definitions in `base_dir`. The function saves the registered masks in respective directories 
    and prints their volumes.
    """
    file_name = os.path.basename(moving_image_path).replace('.mha', '')
    moving_femur_mask_path = os.path.join(base_dir, "moving_images_directory", file_name, file_name + '_femur_mask.mha')
    moving_cartilage_mask_path = os.path.join(base_dir, "moving_images_directory", file_name, file_name + '_cartilage_mask.mha')
    patient_folder_name = os.path.basename(os.path.dirname(moving_image_path))
    
    # Registration of the femur masks
    renamed_femur_mask_output_path = register_femur_mask_function(base_dir, moving_femur_mask_path, all_transform_parameters_files, moving_image_path, patient_folder_name, file_name)
    sitk_image = sitk.ReadImage(renamed_femur_mask_output_path)
    img_array = sitk.GetArrayFromImage(sitk_image)

    # Registration of the femural cartilage masks
    renamed_cartilage_mask_output_path = register_cartilage_mask_function(base_dir, moving_cartilage_mask_path, all_transform_parameters_files, moving_image_path, patient_folder_name, file_name)
    sitk_image = sitk.ReadImage(renamed_cartilage_mask_output_path)
    img_array = sitk.GetArrayFromImage(sitk_image)
    print()
    
def get_registered_mask_paths(base_directory, mask_filename, subfolder="fase2_cartilage"):
    """
    Retrieve paths of registered masks from the specified directory.

    This function searches for registered masks within a specific subdirectory of each patient folder 
    and returns a list of their paths.

    Parameters
    ----------
    base_directory : str
        The root directory from which to search for patient folders and their corresponding 
        registered masks.
    mask_filename : str
        The filename of the mask to search for within the subdirectory.
    subfolder : str, optional
        The name of the subdirectory under each patient folder where the mask is expected to be found. 
        Default is "fase2_cartilage".

    Returns
    -------
    list
        A list of paths to the registered masks found under each patient's specific subdirectory.

    Notes
    -----
    The function assumes a particular directory structure where each patient has a subdirectory (as 
    specified by the `subfolder` parameter) containing the registered masks with the filename as provided 
    by `mask_filename`.
    """
    mask_paths = []
    outputs_directory = os.path.join(base_directory, "outputs")
    patient_folders = [d for d in os.listdir(outputs_directory) if os.path.isdir(os.path.join(outputs_directory, d))]
    
    for patient_folder in patient_folders:
        specific_folder = os.path.join(outputs_directory, patient_folder, subfolder)
        mask_path = os.path.join(specific_folder, mask_filename)
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)      
    return mask_paths

def compute_average_image(image_paths):
    """
    Compute the average of a list of images.

    This function reads a list of image paths, computes their average on a pixel-by-pixel basis, 
    and returns the resulting averaged image.

    Parameters
    ----------
    image_paths : list of tuple
        A list of tuples, where each tuple contains a patient folder name and the path to the image.

    Returns
    -------
    SimpleITK.Image
        The averaged image computed from the list of input images.

    Notes
    -----
    The function assumes all images in `image_paths` have the same dimensions and are compatible 
    for arithmetic operations.
    """
    
    first_path = image_paths[0][1] if isinstance(image_paths[0], tuple) else image_paths[0]
    accumulator_image = sitk.ReadImage(first_path, sitk.sitkFloat32)
    
    for thing in image_paths[1:]:
        img_path = thing[1] if isinstance(thing, tuple) else thing
        image = sitk.ReadImage(img_path, sitk.sitkFloat32)
        accumulator_image = sitk.Add(accumulator_image, image)
        
    average_image = sitk.Divide(accumulator_image, len(image_paths))
    return average_image

def binarize_image(image, threshold=0.5):
    """
    Binarize an image based on a given threshold.

    This function converts the input image to a binary image where pixel values above the threshold 
    are set to 1 and pixel values below or equal to the threshold are set to 0.

    Parameters
    ----------
    image : SimpleITK.Image
        The input image to be binarized.
    threshold : float, optional
        The threshold value used for binarization. Default is 0.5.

    Returns
    -------
    SimpleITK.Image
        The binarized image.

    Notes
    -----
    The function assumes that the input image pixel values are normalized between 0 and 1.
    """
    binary_image = sitk.BinaryThreshold(image, lowerThreshold=threshold, upperThreshold=1.0, insideValue=1, outsideValue=0)
    return binary_image

def compute_and_save_averages(base_directory, output_directory):
    """
    Compute average images and masks, then save them to the specified directory.

    This function calculates the average image and masks (femur and cartilage) from registered 
    images and masks located in the base_directory. It then saves these averages to the 
    output_directory. The average masks are also binarized before saving.

    Parameters
    ----------
    base_directory : str
        The directory containing the registered images and masks.
    output_directory : str
        The directory where the averaged and binarized images and masks will be saved.

    Returns
    -------
    None

    Notes
    -----
    This function assumes that the registered images and masks are organized in a 
    specific directory structure. The structure and naming convention for the files 
    are determined by the helper functions `get_registered_image_paths` and `get_registered_mask_paths`.
    """

    registered_image_paths = get_registered_image_paths(base_directory)
    print("Calculation of the average image of the registered images...")
    print()
    average_image = compute_average_image(registered_image_paths)
    print("Average image of the registered images calculated")
    print()
    sitk.WriteImage(average_image, os.path.join(output_directory, "average_atlas.mha"))
    print("Average image saved")
    print()
    
    femur_mask_paths = get_registered_mask_paths(base_directory, "registered_femur_mask.mha", "fase2_femur")
    print("Calculation of then average image for the femural mask...")
    print()
    average_femur_mask = compute_average_image(femur_mask_paths)
    print("Average image for the femural mask calculated")
    print()
    binary_average_femur_mask = binarize_image(average_femur_mask)
    sitk.WriteImage(binary_average_femur_mask, os.path.join(output_directory, "binary_average_femur_mask.mha"))
    
    cartilage_mask_paths = get_registered_mask_paths(base_directory, "registered_femoral_cartilage_mask.mha", "fase2_cartilage")
    print("Calculation of then average image for the femural cartilage mask...")
    print()
    average_cartilage_mask = compute_average_image(cartilage_mask_paths)
    print("Average image for the femural cartilage mask calculated")
    print()
    binary_average_cartilage_mask = binarize_image(average_cartilage_mask)
    sitk.WriteImage(binary_average_cartilage_mask, os.path.join(output_directory, "binary_average_cartilage_mask.mha"))
    
    print("Calculated and saved binarized average mask.")


def compute_atlas_function(base_dir):
    """
    Compute and save the average atlas based on registered images and masks.

    This function calculates the average atlas (both images and masks) from registered 
    images and masks located in the base_dir. The average atlas is then saved to an 
    "atlas_output" subdirectory within the base_dir.

    Parameters
    ----------
    base_dir : str
        The root directory containing all the necessary files and subdirectories 
        for atlas computation, including registered images and masks.

    Returns
    -------
    None

    Notes
    -----
    This function assumes a specific directory structure and naming conventions 
    for the files. The structure is determined by the helper functions 
    `get_all_paths` and `compute_and_save_averages`.
    The output directory can be modified within the function if desired.
    """
    paths = get_all_paths(base_dir)
    output_directory = os.path.join(base_dir, "atlas_output")  

    compute_and_save_averages(base_dir, output_directory)
    
    return

def show_average_atlas(base_dir, slice_index=None, flip_axis=None):
    """
    Display the average atlas from a specified base directory after rotating it by 180 degrees and flipping horizontally.

    Load, rotate, flip and display the average atlas image from the "atlas_output" sub-directory 
    located within the provided base directory. A specific or the central slice of the atlas 
    is visualized after rotation and horizontal flipping.

    Parameters
    ----------
    base_dir : str
        Base directory path where the "atlas_output" directory is located.
    slice_index : int, optional
        Index of the sagittal slice to be displayed. If not provided, the central slice is shown.
    flip_axis : int, optional
        Axis to flip the image along. 0 for vertical flip, 1 for horizontal flip.

    Returns
    -------
    None

    Notes
    -----
    The function uses SimpleITK for image loading, numpy for rotation and flipping, and matplotlib for visualization.

    Examples
    --------
    >>> show_average_atlas("/path/to/directory")  # shows central slice rotated by 180 degrees and flipped horizontally
    >>> show_average_atlas("/path/to/directory", slice_index=50)  # shows slice at index 50 rotated by 180 degrees and flipped horizontally
    >>> show_average_atlas("/path/to/directory", slice_index=50, flip_axis=0)  # vertical flip and then horizontal flip

    """
    atlas_path = os.path.join(base_dir, "atlas_output", "average_atlas.mha")  
    atlas_image = sitk.ReadImage(atlas_path)

    atlas_array = sitk.GetArrayFromImage(atlas_image)

    # If no specific slice_index is provided, show the central slice.
    if slice_index is None:
        slice_index = int(atlas_array.shape[2] / 2)

    processed_slice = np.rot90(atlas_array[:, :, slice_index], 2)  # Rotate the image by 180 degrees

    # Flip the image if an axis is specified
    if flip_axis is not None:
        processed_slice = np.flip(processed_slice, axis=flip_axis)

    # Additional horizontal flip to mirror the image
    processed_slice = np.flip(processed_slice, axis=1)

    plt.figure(figsize=(4, 4))
    plt.imshow(processed_slice, cmap='gray')  # Display the specified sagittal slice, rotated by 180 degrees and flipped horizontally
    plt.title(f"Average Atlas - Sagittal Slice at Index {slice_index}")
    plt.axis('off')
    plt.show()

