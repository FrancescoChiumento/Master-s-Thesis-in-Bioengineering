# Francesco Chiumento, 2023

import os
import shutil
import itertools
import pandas as pd

from pykneer import pykneer_io as io
from pykneer import segmentation_sa_for_nb as segm
from pykneer import segmentation_quality_for_nb as sq
from average_atlas import get_registered_image_paths, compute_and_save_averages

dice_scores = {}

base_dir = os.getcwd()
segmented_path = os.path.join(base_dir, "segmented")
registered_path = os.path.join(base_dir, "registered")

def copy_images_to_preprocessed(base_dir):
    """
    Copy images from the 'moving_images' directory to the 'preprocessed' directory.
    
    Parameters
    ----------
    base_dir : str
        The base directory containing both the source and destination folders.
    
    Returns
    -------
    int
        Number of images copied.

    Example
    -------
    >>> copy_images_to_preprocessed("/path/to/base/directory")
    5 images copied from 'moving_images' to 'preprocessed'
    5
    """
    
    # Path to the source and destination directories
    source_dir = os.path.join(base_dir, "moving_images")
    dest_dir = os.path.join(base_dir, "preprocessed")

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # List all files in the source directory
    image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Copy each file to the destination directory
    for image_file in image_files:
        shutil.copy2(os.path.join(source_dir, image_file), dest_dir)

    num_images_copied = len(image_files)
    print(f"{num_images_copied} images copied from 'moving_images' to 'preprocessed'")

    return num_images_copied

def create_combination_directories(base_dir, registered_images):
    """
    Create directories for combinations of registered images and copy the images into them.
    
    Parameters
    ----------
    base_dir : str
        The base directory where the "temp_combination" directory will be created.
    registered_images : list of str
        List of paths to registered images to be combined.
    
    Returns
    -------
    list of tuple
        List of generated combinations of registered images.

    Example
    -------
    >>> registered_images_paths = ["path1.mha", "path2.mha", "path3.mha"]
    >>> create_combination_directories("/path/to/base/directory", registered_images_paths)
    Generated 3 clusters.
    Creating directory for combination: ('path1.mha', 'path2.mha')...
    ...
    [('path1.mha', 'path2.mha'), ('path1.mha', 'path3.mha'), ('path2.mha', 'path3.mha')]
    """
    temp_combination_base = os.path.join(base_dir, "temp_combination")
    # The number of clusters can be chosen, with each cluster containing all possible combinations of recorded images.
    clusters = list(itertools.combinations(registered_images, 4))
    print(f"Generated {len(clusters)} clusters.")
    print()
    
    for idx, combination in enumerate(clusters):
        # For each cluster, a folder is created, and within this folder, the combinations of images will be copied
        temp_combination = os.path.join(temp_combination_base, f"combination_{idx}")
        
        try:
            print(f"Creating directory for combination: {combination}")
            print()
            
            if os.path.exists(temp_combination):
                shutil.rmtree(temp_combination)  
            os.makedirs(temp_combination)
            
            for img_path in combination:
                directory_name = os.path.dirname(img_path)  
                parts = directory_name.split('_Cube_')
                if len(parts) > 1:
                    name_part = parts[1].split('_prep')[0]
                    destination_path = os.path.join(temp_combination, f"{name_part}.mha")
                    # Only the names of the images are extracted and copied into each folder for every combination
                    print(f"Copying image: {img_path} to {destination_path}")
                    print()
                    shutil.copy2(img_path, destination_path)

                    # Verifying the copy
                    if os.path.exists(destination_path):
                        print(f"Successfully copied to {destination_path}")
                        print()
                    else:
                        print(f"Failed to copy {img_path}")
                        print()
                else:
                    print(f"Unexpected file format for {img_path}. Skipping.")
                    print()

        except Exception as e:
            print(f"Error creating directory for combination {combination}: {str(e)}")
            print()

    return clusters

def analyze_combinations(base_dir, clusters):
    """
    Analyze combinations of images to compute the DICE score for each combination.

    Parameters
    ----------
    base_dir : str
        The base directory where the "temp_combination" and other relevant directories are located.
    clusters : list of tuple
        List of tuples, where each tuple contains paths to two images that form a combination.

    Returns
    -------
    dict
        Dictionary with combinations as keys and their corresponding DICE scores as values.

    Example
    -------
    >>> base_directory = "/path/to/base/directory"
    >>> clusters_combinations = [('path1.mha', 'path2.mha'), ('path1.mha', 'path3.mha')]
    >>> analyze_combinations(base_directory, clusters_combinations)
    {('path1.mha', 'path2.mha'): 0.85, ('path1.mha', 'path3.mha'): 0.78}

    Notes
    -----
    This function assumes that other required functions like `compute_and_save_averages`,
    `segment_images`, and `update_image_list_file` are defined elsewhere in the code.
    """
    temp_combination_base = os.path.join(base_dir, "temp_combination")
    
    for idx, combination in enumerate(clusters):
        temp_combination = os.path.join(temp_combination_base, f"combination_{idx}")
        
        try:
            print(f"Analyzing combination: {combination}")
            print()

            print(f"Processing combination {idx+1} of {len(clusters)} has started ")
            print()
            
            # Calculate the average atlas for the current combination using the temporary directory
            iteration_output_dir = os.path.join(temp_combination, "outputs")
            os.makedirs(iteration_output_dir, exist_ok=True)
            compute_and_save_averages(base_dir, iteration_output_dir)
            print(f"Atlas computed for combination {idx+1}. Starting segmentation.")
            print()

            # The average atlas created and its corresponding binary average masks are copied to the "reference" folder with the following names: "reference.mha," "reference_f.mha," and "referencef_fc.mha." The "reference" folder will indeed be used for the subsequent segmentation step
            print("Starting copying files into the segmentation folder")
            reference_dir = os.path.join(base_dir, "reference", "newsubject")
            shutil.copy2(os.path.join(iteration_output_dir, "average_atlas.mha"), os.path.join(reference_dir, "reference.mha"))
            shutil.copy2(os.path.join(iteration_output_dir, "binary_average_femur_mask.mha"), os.path.join(reference_dir, "reference_f.mha"))
            shutil.copy2(os.path.join(iteration_output_dir, "binary_average_cartilage_mask.mha"), os.path.join(reference_dir, "reference_fc.mha"))

            # Get the names of the images in the cluster
            image_names_in_clusters = [os.path.basename(os.path.dirname(os.path.dirname(img_path))) for img_path in combination]
            print()
            print("Image names in current clusters:", image_names_in_clusters)
            print()

            # Get a list of all images in `preprocessed`
            preprocessed_dir = os.path.join(base_dir, "preprocessed")
            all_images = [os.path.splitext(img)[0] for img in os.listdir(preprocessed_dir) if img.endswith('.mha')]
            print("All images:", all_images)
            
            # Find the images that are not part of the cluster
            images_to_segment = [img for img in all_images if img not in image_names_in_clusters]

            # Update the image_list_newsubject.txt file using the names of the images that did not participate in the atlas creation, the average atlas generated from each cluster will be tested only on these images
            print("Images to segment:", images_to_segment)
            update_image_list_file(images_to_segment, base_dir)

            # Now the segment_images function can be called using the images_to_segment dataset
            segmented_images = segment_images(images_to_segment, base_dir)

            # In this case as well, the .txt file is updated with the names of the images that have been segmented. This file will be used in the subsequent step for calculating the DICE index
            segmentation_quality_file_path = os.path.join(base_dir, "segmentation_quality.txt")
            image_names_to_update = [img + "_fc.mha" for img in images_to_segment]
            update_segmentation_quality_file(image_names_to_update, segmentation_quality_file_path)

            # Calculate the DICE coefficient for segmented images
            input_file_name = "segmentation_quality.txt"
            image_data_quality = io.load_image_data_segmentation_quality(input_file_name)
            dice_coeff, _, _ = sq.compute_overlap(image_data_quality) 
            print("DICE coefficient for this combination:", dice_coeff)
            print()

            mean_dice_coeff = sum(dice_coeff) / len(dice_coeff)
            print("Average DICE: ", mean_dice_coeff)
            dice_scores[str(combination)] = mean_dice_coeff
            # Remove the temporary directory
            shutil.rmtree(temp_combination)
            
        except Exception as e:
            print(f"Error analyzing combination {combination}: {str(e)}")
            print()
    # In this step, the results of the DICE coefficients obtained from each segmentation using the atlases created by different clusters are saved
    with open("dice_scores_results.txt", "w") as f:
        for combination, score in dice_scores.items():
            directory_names = [os.path.basename(os.path.dirname(os.path.dirname(path))) for path in eval(combination)]
            f.write(f"{directory_names}: {score:.4f}\n")
        
        # The highest DICE index and its respective combination of images are saved
        best_combination = max(dice_scores, key=dice_scores.get)
        best_directory_names = [os.path.basename(os.path.dirname(os.path.dirname(path))) for path in eval(best_combination)]
        f.write("\n")
        f.write(f"The best combination is {best_directory_names} with a DICE score of {dice_scores[best_combination]}")
    
    return dice_scores
def segment_images(image_list, base_dir, n_of_cores=22):
    """
    Segment a list of images using the provided base directory and number of cores.

    Parameters
    ----------
    image_list : list of str
        List of image names to be segmented.
    base_dir : str
        The base directory where the relevant directories and files are located.
    n_of_cores : int, optional
        Number of cores to be used for segmentation. Default is 4.

    Returns
    -------
    None

    Example
    -------
    >>> images = ["image1.mha", "image2.mha"]
    >>> base_directory = "/path/to/base/directory"
    >>> segment_images(images, base_directory, n_of_cores=2)

    Notes
    -----
    This function assumes that other required functions like `empty_folder`, 
    `update_image_list_file`, and functions from `io` and `segm` modules are 
    defined elsewhere in the code.
    """
    
    empty_folder(segmented_path)
    empty_folder(registered_path)
    print(f"Segmenting {len(image_list)} images.")
    try:
        update_image_list_file(image_list, base_dir)

        modality = "newsubject"
        image_data = io.load_image_data_segmentation(modality, os.path.join(base_dir, "image_list_newsubject.txt"))

        segm.prepare_reference(image_data)
        print("Start bone registration to reference")
        segm.register_bone_to_reference(image_data, n_of_cores)
        segm.invert_bone_transformations(image_data, n_of_cores)
        segm.warp_bone_mask(image_data, n_of_cores)
        segm.register_cartilage_to_reference(image_data, n_of_cores)
        segm.invert_cartilage_transformations(image_data, n_of_cores)
        segm.warp_cartilage_mask(image_data, n_of_cores)
        print("Segmentation completed")
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")

def update_image_list_file(image_list, base_dir):
    """
    Update the image list file with the provided list of image names.

    Parameters
    ----------
    image_list : list of str
        List of image names to be added to the image list file.
    base_dir : str
        The base directory where the "image_list_newsubject.txt" file is located.

    Returns
    -------
    None

    Example
    -------
    >>> images = ["image1", "image2"]
    >>> base_directory = "/path/to/base/directory"
    >>> update_image_list_file(images, base_directory)

    Notes
    -----
    This function assumes the structure of the "image_list_newsubject.txt" file 
    where the first three lines are headers and the subsequent lines list the images.
    """
    print(f"Updating image list file with {len(image_list)} images.")
    try:
        # Read the first three lines of the original file
        with open(os.path.join(base_dir, "image_list_newsubject.txt"), "r") as f:
            header_lines = [f.readline().strip() for _ in range(3)]

        # Write the three lines in the file and add the new images
        with open(os.path.join(base_dir, "image_list_newsubject.txt"), "w") as f:
            for line in header_lines:
                f.write(line + "\n")

            for img_name in image_list:
                f.write(f"m {img_name}.mha\n")  # Aggiunto .mha qui
    except Exception as e:
        print(f"Error updating image list file: {str(e)}")


def update_segmentation_quality_file(image_names, output_file_path):
    """
    Update the segmentation_quality.txt file based on segmented images.

    Parameters
    ----------
    image_names : list of str
        List of segmented image names (without _fc).
    output_file_path : str
        Path to the segmentation_quality.txt file to be updated.

    Returns
    -------
    None

    Notes
    -----
    The function assumes each image name follows the pattern "01_Sag_DP_Cube_GF0_prep.mha".
    """

    with open(output_file_path, "w") as f:
        # Write the fixed header lines
        f.write("./segmented\n")
        f.write("./segmented_groundTruth\n")
        
        for img_name in image_names:
            # Assuming image names already have _fc before .mha extension
            img_name_fc = img_name
            
            # Extract the tag from the image name (e.g., "GF0" from "01_Sag_DP_Cube_GF0_prep.mha")
            tag = img_name.split("_Cube_")[1].split("_prep")[0]
            
            # Write the two lines for this image
            f.write(f"s {img_name_fc}\n")
            f.write(f"g {tag}_fc.mha\n")
            
def empty_folder(folder_path):
    """
    Remove all files and subdirectories within a specified folder.

    This function deletes all files and subdirectories within the specified folder.
    If any error occurs during the process, it will print the error message.

    Parameters
    ----------
    folder_path : str
        Path to the folder that needs to be emptied.

    Notes
    -----
    - The function uses both the `os` and `shutil` modules for file and directory 
      operations.
    - If a file or directory cannot be removed, an exception will be caught and the 
      error message will be printed.

    Example
    -------
    folder_path = "/path/to/folder"
    empty_folder(folder_path)

    """
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            
def extract_image_name_from_path(path):
    """
    Extract the unique image name from a given path.

    This function assumes that the unique image name is located two directories above
    the file specified in the path. For instance, if given a path like 
    `/some/directory/unique_image_name/subdir/file.ext`, it will return `unique_image_name`.

    Parameters
    ----------
    path : str
        The full path from which the image name should be extracted.

    Returns
    -------
    str
        The extracted image name.

    Examples
    --------
    >>> path = "/some/directory/unique_image_name/subdir/file.ext"
    >>> extract_image_name_from_path(path)
    'unique_image_name'

    """
    # Extracts the unique image name from the path
    return os.path.basename(os.path.dirname(os.path.dirname(path)))

def create_results_table(dice_scores):
    """
    Convert the dice_scores dictionary into a sorted pandas DataFrame.

    Parameters
    ----------
    dice_scores : dict
        Dictionary with combinations as keys and their corresponding DICE scores as values.

    Returns
    -------
    pd.DataFrame
        A DataFrame sorted by DICE scores and containing the top 3 combinations.

    Example
    -------
    >>> dice_scores = {('image1', 'image2'): 0.85, ('image1', 'image3'): 0.78}
    >>> df = create_results_table(dice_scores)
    >>> print(df)
    Combinazione        DICE
    0  image1, image2  0.85
    1  image1, image3  0.78
    """
    # Extract image names and scores
    combinations = [tuple(map(extract_image_name_from_path, eval(key))) for key in dice_scores.keys()]
    scores = list(dice_scores.values())
    
    # Create DataFrame
    df = pd.DataFrame({
        'Combinazione': [", ".join(comb) for comb in combinations],
        'DICE': scores
    })
    
    # Sort by DICE scores and select top 3
    df = df.sort_values(by='DICE', ascending=False).head(3)
    return df

def get_best_combination_from_table(dice_scores):
    """
    Returns and prints the best combination of images based on DICE scores using the results table.

    Parameters
    ----------
    dice_scores : dict
        Dictionary with combinations as keys and their corresponding DICE scores as values.

    Returns
    -------
    tuple
        A tuple containing the best combination of image names and its DICE score.

    Example
    -------
    >>> dice_scores = {('image1', 'image2'): 0.85, ('image1', 'image3'): 0.78}
    >>> best_combination, best_dice = get_best_combination_from_table(dice_scores)
    >>> print(f"The best combination is {best_combination} with a DICE score of {best_dice:.2f}")
    The best combination is ('image1', 'image2') with a DICE score of 0.85
    """
    df = create_results_table(dice_scores)
    best_row = df.iloc[0]
    best_combination = best_row["Combinazione"]
    best_dice = best_row["DICE"]
    
    return (best_combination, best_dice)