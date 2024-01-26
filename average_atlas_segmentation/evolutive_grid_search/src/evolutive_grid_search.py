#Francesco Chiumento, 2023

print("Step 1: Begin Configuration File Generation")
print()
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import shutil
import json
import datetime

# Get the current path
base_path = os.getcwd()

# Combine with the subdirectory 'pykneer'
pykneer_path = os.path.join(base_path, "pykneer")

# Add the combined path to sys.path
sys.path.append(pykneer_path)

from pykneer import pykneer_io             as io
from pykneer import segmentation_sa_for_nb as segm
from pykneer import segmentation_quality_for_nb as sq

from segmentation_functions import create_MR_param_rigid
from segmentation_functions import create_MR_param_similarity
from segmentation_functions import create_MR_param_spline
from segmentation_functions import create_MR_iparam_similarity
from segmentation_functions import create_MR_iparam_rigid
from segmentation_functions import create_MR_iparam_spline

intermediate_results = []

rigid_counter = 0
similarity_counter = 0
spline_counter = 0

best_dice_coeff_rigid = 0
best_params_rigid = None

best_dice_coeff_similarity = 0
best_params_similarity = None

best_dice_coeff_spline = 0
best_params_spline = None

base_path = os.getcwd()
pykneer_path = os.path.join(base_path, "pykneer")
segmented_path = os.path.join(base_path, "segmented")
registered_path = os.path.join(base_path, "registered")

def save_parameters(parameters, transformation_type, iteration, dice_coeff, jacc_coeff, vol_simil):
    
    """
    Save optimization parameters and evaluation results to a CSV file.

    Args:
        parameters (list): List of optimization parameters.
        transformation_type (str): Type of transformation applied.
        iteration (int): Iteration number.
        dice_coeff (list): List of DICE coefficients.
        jacc_coeff (list): List of Jaccard coefficients.
        vol_simil (list): List of volume similarity scores.

    Returns:
        None

    Saves the provided parameters, transformation type, iteration, and the mean values
    of DICE coefficient, Jaccard coefficient, and volume similarity to a CSV file.
    The data is stored in a tabular format with appropriate column names.

    The CSV file is created or appended to, depending on whether it already exists.

    If an error occurs during the file-saving process, an error message is printed.

    Global Variables:
        rigid_evaluation_counter (int): Counter for rigid transformation evaluations.
        similarity_evaluation_counter (int): Counter for similarity transformation evaluations.
        spline_evaluation_counter (int): Counter for spline transformation evaluations.
        function_evalutation_limit (int): Maximum allowed function evaluations.
    """
    
    filename = "optimal_parameters.csv"
    base_path = os.getcwd()
    parameters_optimal_path = os.path.join(base_path, "pykneer", "optimal_parameters")
    filepath = os.path.join(parameters_optimal_path, filename)

    # Calculate the average of the coefficients and volume similarity
    mean_dice_coeff = sum(dice_coeff)/len(dice_coeff)
    mean_jacc_coeff = sum(jacc_coeff)/len(jacc_coeff)
    mean_vol_simil = sum(vol_simil)/len(vol_simil)
    
    if transformation_type == "rigid":
        relative_iteration = (rigid_evaluation_counter % function_evalutation_limit) or function_evalutation_limit
    elif transformation_type == "similarity":
        relative_iteration = (similarity_evaluation_counter % function_evalutation_limit) or function_evalutation_limit
    else:  # spline
        relative_iteration = (spline_evaluation_counter % function_evalutation_limit) or function_evalutation_limit


    # Create a DataFrame with the parameters and results
    data = pd.DataFrame([list(parameters) + [transformation_type, relative_iteration, mean_dice_coeff, mean_jacc_coeff, mean_vol_simil]],
                        columns=[f"param_{i}" for i in range(len(parameters))] + ["transformation_type", "iteration", "dice_coeff", "jacc_coeff", "vol_simil"])

    try:
        # Save the data to a CSV file, appending a new row if the file already exists
        if not os.path.exists(filepath):
            data.to_csv(filepath, index=False)
        else:
            data.to_csv(filepath, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error in saving the data: {e}")

rigid_evaluation_counter = 0
similarity_evaluation_counter = 0
spline_evaluation_counter = 0
function_evalutation_limit= 10

def rigid_optimization_function(param_rigid, param_similarity_fixed, param_spline_fixed):
    
    """
    Perform rigid transformation optimization and evaluation.

    Args:
        param_rigid (list): Rigid transformation parameters.
        param_similarity_fixed (list): Fixed similarity transformation parameters.
        param_spline_fixed (list): Fixed spline transformation parameters.

    Returns:
        float: Negative mean DICE coefficient.

    This function performs optimization of rigid transformation parameters and evaluates
    the results using the DICE coefficient. It also handles the cleaning of folders,
    configuration file creation, segmentation, DICE coefficient calculation,
    and saving of parameters and metrics.

    The function raises an exception if the evaluation limit is reached.

    Global Variables:
        rigid_evaluation_counter (int): Counter for rigid transformation evaluations.
        best_dice_coeff_rigid (float): Best DICE coefficient achieved so far.
        best_params_rigid (list): Parameters corresponding to the best DICE coefficient.
        rigid_counter (int): Counter for rigid transformation iterations.
        function_evalutation_limit (int): Maximum allowed function evaluations.
        segmented_path (str): Path to segmented data.
        registered_path (str): Path to registered data.

    """

    global rigid_evaluation_counter
    global best_dice_coeff_rigid
    global best_params_rigid
    global rigid_counter
    rigid_evaluation_counter += 1
    rigid_counter += 1
    
    if rigid_counter == 1:
        print(f"Beginning of the rigid optimization phase: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    if rigid_evaluation_counter > function_evalutation_limit:
        print("Best parameters for rigid transformation:", best_params_rigid)
        print(f"End of the rigid optimization phase: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        raise Exception("Objective function evaluation limit reached")
        
    print(f"Rigid Iteration: {rigid_counter}")
    print()
    print(f"Rigid Parameters: {param_rigid}")
    print()
    
   # Parameter preparation, direct transformation parameters are set equal to inverse transformation parameters
    param_irigid = param_rigid.copy()
    param_isimilarity = param_similarity_fixed.copy()
    param_ispline = param_spline_fixed.copy()

    # Cleaning up folders at each new iteration
    empty_folder(segmented_path)
    empty_folder(registered_path)
    
    # Image configuration
    input_file_name = "./image_list_newsubject.txt"
    modality        = "newsubject"
    n_of_cores      = 16
 
    # Configuration file creation, a pause time has been added to ensure the writing process to files is not skipped
    create_MR_param_rigid(*param_rigid)
    time.sleep(1)
    create_MR_param_similarity(*param_similarity_fixed)
    time.sleep(1)
    create_MR_param_spline(*param_spline_fixed)
    time.sleep(1)
    create_MR_iparam_rigid(*param_irigid)
    time.sleep(1)
    create_MR_iparam_similarity(*param_isimilarity)
    time.sleep(1)
    create_MR_iparam_spline(*param_ispline)
    time.sleep(1)
    
    # Segmentation
    image_data = io.load_image_data_segmentation(modality, input_file_name)
    segm.prepare_reference(image_data)
    print("Start bone registration to reference")
    print()
    segm.register_bone_to_reference(image_data, n_of_cores)
    segm.invert_bone_transformations(image_data, n_of_cores)
    segm.warp_bone_mask(image_data, n_of_cores)
    segm.register_cartilage_to_reference(image_data, n_of_cores)
    segm.invert_cartilage_transformations(image_data, n_of_cores)
    segm.warp_cartilage_mask(image_data, n_of_cores)
    
    print("Segmentation completed")
    print()
    # Calculation of the DICE coefficient
    input_file_name            = "segmentation_quality.txt"
    image_data_quality = io.load_image_data_segmentation_quality(input_file_name)
    dice_coeff, jacc_coeff, vol_simil = sq.compute_overlap(image_data_quality)
    
    print("DICE coefficient:", dice_coeff)
    print()
    
    # Calculation of the mean DICE coefficient
    mean_dice_coeff = sum(dice_coeff) / len(dice_coeff)
    print("Mean DICE:", mean_dice_coeff)
    print()
    if mean_dice_coeff > best_dice_coeff_rigid:
        best_dice_coeff_rigid = mean_dice_coeff
        best_params_rigid = param_rigid # If a higher mean DICE is achieved, save the parameters corresponding to the iteration
        
    # Saving parameters and metrics
    parameters = param_rigid + param_similarity_fixed + param_spline_fixed
    iteration = len(intermediate_results) + 1
    save_parameters(parameters, "rigid", iteration, dice_coeff, jacc_coeff, vol_simil)
    
    # Adding the metric to the list of intermediate results
    intermediate_results.append(-mean_dice_coeff)
    
    return -mean_dice_coeff  # Returns the negative of the DICE coefficient

def similarity_optimization_function(param_similarity, param_rigid_fixed, param_spline_fixed):
    
    """
    Perform similarity transformation optimization and evaluation.

    Args:
        param_similarity (list): Similarity transformation parameters.
        param_rigid_fixed (list): Fixed rigid transformation parameters.
        param_spline_fixed (list): Fixed spline transformation parameters.

    Returns:
        float: Negative mean DICE coefficient.

    This function performs optimization of similarity transformation parameters and evaluates
    the results using the DICE coefficient. It also handles the cleaning of folders,
    configuration file creation, segmentation, DICE coefficient calculation,
    and saving of parameters and metrics.

    The function raises an exception if the evaluation limit is reached.

    Global Variables:
        similarity_evaluation_counter (int): Counter for similarity transformation evaluations.
        best_dice_coeff_similarity (float): Best DICE coefficient achieved so far.
        best_params_similarity (list): Parameters corresponding to the best DICE coefficient.
        similarity_counter (int): Counter for similarity transformation iterations.
        function_evalutation_limit (int): Maximum allowed function evaluations.
        segmented_path (str): Path to segmented data.
        registered_path (str): Path to registered data.

    """
    
    global similarity_evaluation_counter
    global best_dice_coeff_similarity
    global best_params_similarity  
    global similarity_counter  
   
    similarity_evaluation_counter += 1
    similarity_counter += 1 
    
    if similarity_counter == 1:
        print(f"Beginning of the similarity optimization phase:{time.strftime('%Y-%m-%d %H:%M:%S')}")  # Record the start time
        print()
        
    if similarity_evaluation_counter > function_evalutation_limit:
        print("Best parameters for similarity transformation:", best_params_similarity)
        print()
        print(f"End of the similarity optimization phase: {time.strftime('%Y-%m-%d %H:%M:%S')}") 
        raise Exception("Objective function evaluation limit reached")
        
    print(f"Similarity Iteration: {similarity_counter}")
    print()
    print(f"Similarity Parameters: {param_similarity}")
    print()
    
    # Parameter preparation
    param_irigid = param_rigid_fixed.copy()
    param_isimilarity = param_similarity.copy()
    param_ispline = param_spline_fixed.copy()

    empty_folder(segmented_path)
    empty_folder(registered_path)
    
    input_file_name = "./image_list_newsubject.txt"
    modality        = "newsubject"
    n_of_cores      = 16
 
    create_MR_param_rigid(*param_rigid_fixed)
    time.sleep(1)
    create_MR_param_similarity(*param_similarity)
    time.sleep(1)
    create_MR_param_spline(*param_spline_fixed)
    time.sleep(1)
    create_MR_iparam_rigid(*param_irigid)
    time.sleep(1)
    create_MR_iparam_similarity(*param_isimilarity)
    time.sleep(1)
    create_MR_iparam_spline(*param_ispline)
    time.sleep(1)

    image_data = io.load_image_data_segmentation(modality, input_file_name)
    segm.prepare_reference(image_data)
    print("Start bone registration to reference")
    print()
    segm.register_bone_to_reference(image_data, n_of_cores)
    segm.invert_bone_transformations(image_data, n_of_cores)
    segm.warp_bone_mask(image_data, n_of_cores)
    segm.register_cartilage_to_reference(image_data, n_of_cores)
    segm.invert_cartilage_transformations(image_data, n_of_cores)
    segm.warp_cartilage_mask(image_data, n_of_cores)
    
    print("Segmentation completed")
    print()
    
    # Calculation of the DICE coefficient
    input_file_name            = "segmentation_quality.txt"
    image_data_quality = io.load_image_data_segmentation_quality(input_file_name)
    dice_coeff, jacc_coeff, vol_simil = sq.compute_overlap(image_data_quality)
    
    print("DICE coefficient:", dice_coeff)
    print()
    
    mean_dice_coeff = sum(dice_coeff) / len(dice_coeff)
    if mean_dice_coeff > best_dice_coeff_similarity:
        best_dice_coeff_similarity = mean_dice_coeff
        best_params_similarity = param_similarity
    
    print("Mean DICE:", mean_dice_coeff)
    print()
   
    parameters = param_rigid_fixed + param_similarity + param_spline_fixed
    iteration = len(intermediate_results) + 1
    save_parameters(parameters, "similarity", iteration, dice_coeff, jacc_coeff, vol_simil)
    
    intermediate_results.append(-mean_dice_coeff)
    
    return -mean_dice_coeff 

def spline_optimization_function(param_spline, param_rigid_fixed, param_similarity_fixed):
    
    """
    Perform spline transformation optimization and evaluation.

    Args:
        param_spline (list): Spline transformation parameters.
        param_rigid_fixed (list): Fixed rigid transformation parameters.
        param_similarity_fixed (list): Fixed similarity transformation parameters.

    Returns:
        float: Negative mean DICE coefficient.

    This function performs optimization of spline transformation parameters and evaluates
    the results using the DICE coefficient. It also handles the cleaning of folders,
    configuration file creation, segmentation, DICE coefficient calculation,
    and saving of parameters and metrics.

    The function raises an exception if the evaluation limit is reached.

    Global Variables:
        spline_evaluation_counter (int): Counter for spline transformation evaluations.
        best_dice_coeff_spline (float): Best DICE coefficient achieved so far.
        best_params_spline (list): Parameters corresponding to the best DICE coefficient.
        spline_counter (int): Counter for spline transformation iterations.
        function_evalutation_limit (int): Maximum allowed function evaluations.
        segmented_path (str): Path to segmented data.
        registered_path (str): Path to registered data.

    """

    global spline_evaluation_counter
    global best_dice_coeff_spline
    global best_params_spline
    global spline_counter  

    spline_evaluation_counter += 1
    spline_counter += 1

    if spline_counter == 1:
        print(f"Beginning of the spline optimization phase:{time.strftime('%Y-%m-%d %H:%M:%S')}") 
        print()   

    if spline_evaluation_counter > function_evalutation_limit:
        print("Best parameters for spline transformation:", best_params_spline)
        print()
        print(f"End of the spline optimization phase: {time.strftime('%Y-%m-%d %H:%M:%S')}")  
        raise Exception("Objective function evaluation limit reached")
        
    print(f"Spline Iteration: {spline_counter}")
    print()
    print(f"Spline Parameters: {param_spline}")
    print()
    
    param_irigid = param_rigid_fixed.copy()
    param_isimilarity = param_similarity_fixed.copy()
    param_ispline = param_spline.copy()
    
    empty_folder(segmented_path)
    empty_folder(registered_path)
    
    input_file_name = "./image_list_newsubject.txt"
    modality        = "newsubject"
    n_of_cores      = 16
 
    create_MR_param_rigid(*param_rigid_fixed)
    time.sleep(1)
    create_MR_param_similarity(*param_similarity_fixed)
    time.sleep(1)
    create_MR_param_spline(*param_spline)
    time.sleep(1)
    create_MR_iparam_rigid(*param_irigid)
    time.sleep(1)
    create_MR_iparam_similarity(*param_isimilarity)
    time.sleep(1)
    create_MR_iparam_spline(*param_ispline)
    time.sleep(1)
    
    image_data = io.load_image_data_segmentation(modality, input_file_name)
    segm.prepare_reference(image_data)
    print("Start bone registration to reference")
    print()
    segm.register_bone_to_reference(image_data, n_of_cores)
    segm.invert_bone_transformations(image_data, n_of_cores)
    segm.warp_bone_mask(image_data, n_of_cores)
    segm.register_cartilage_to_reference(image_data, n_of_cores)
    segm.invert_cartilage_transformations(image_data, n_of_cores)
    segm.warp_cartilage_mask(image_data, n_of_cores)
    
    print("Segmentation completed")
    print()
    
    input_file_name            = "segmentation_quality.txt"
    image_data_quality = io.load_image_data_segmentation_quality(input_file_name)
    dice_coeff, jacc_coeff, vol_simil = sq.compute_overlap(image_data_quality)
    
    print("DICE coefficient:", dice_coeff)
    print()
    
    mean_dice_coeff = sum(dice_coeff) / len(dice_coeff)
    if mean_dice_coeff > best_dice_coeff_spline:
        best_dice_coeff_spline = mean_dice_coeff
        best_params_spline = param_spline
        
    print("Mean DICE:", mean_dice_coeff)
    print()

    parameters = param_rigid_fixed + param_similarity_fixed + param_spline
    iteration = len(intermediate_results) + 1
    save_parameters(parameters,"spline", iteration, dice_coeff, jacc_coeff, vol_simil)
    
    intermediate_results.append(-mean_dice_coeff)
    
    return -mean_dice_coeff 

def statistical_results_analysis(filepath, best_params_rigid=None, best_params_similarity=None, best_params_spline=None):

    """
    Analyze the statistical results from an optimization process.
    
    This function reads the results from a CSV file and provides insights such as:
    - The iteration number with the highest DICE value for each transformation.
    - Prints the best parameters for rigid, similarity, and spline transformations.
    
    Parameters
    ----------
    filepath : str
        The path to the CSV file containing the results.
    best_params_rigid : list or None, optional
        The best parameters for the rigid transformation. If not provided, they are extracted from the CSV file.
    best_params_similarity : list or None, optional
        The best parameters for the similarity transformation. If not provided, they are extracted from the CSV file.
    best_params_spline : list or None, optional
        The best parameters for the spline transformation. If not provided, they are extracted from the CSV file.
    
    Returns
    -------
    DataFrame
        A DataFrame containing descriptive statistics of the results in the CSV file.

    Examples
    --------
    >>> statistical_results_analysis("path_to_file.csv")
    Iteration with highest DICE for rigid transformation: 1
    Iteration with highest DICE for similarity transformation: 3
    Iteration with highest DICE for spline transformation: 5
    
    Best parameters for rigid transformation:
    ...
    Best parameters for similarity transformation:
    ...
    Best parameters for spline transformation:
    ...
    
    """
        
    # Read data from the CSV file
    data = pd.read_csv(filepath)

    # Calculate descriptive statistics
    descr_stats = data.describe()

    # Find the row with the highest DICE value for each type of transformation
    rigid_max_row = data[data['transformation_type'] == 'rigid'].nlargest(1, 'dice_coeff')
    similarity_max_row = data[data['transformation_type'] == 'similarity'].nlargest(1, 'dice_coeff')
    spline_max_row = data[data['transformation_type'] == 'spline'].nlargest(1, 'dice_coeff')

    # Print the iteration number with the highest DICE value for each transformation
    print(f"Iteration with highest DICE for rigid transformation: {rigid_max_row['iteration'].values[0]}")
    print(f"Iteration with highest DICE for similarity transformation: {similarity_max_row['iteration'].values[0]}")
    print(f"Iteration with highest DICE for spline transformation: {spline_max_row['iteration'].values[0]}")

    # Mapping for values to their corresponding strings
    metric_values = {
        1: "AdvancedMattesMutualInformation",
        2: "AdvancedNormalizedCorrelation",
        3: "AdvancedMeanSquares"
    }

    sampler_values = {
        1: "RandomCoordinate",
        2: "RandomSparseMask",
        3: "Grid",
        4: "Full"
    }

    combine_transform_values = {
        1: "Compose",
        2: "Add"
    }

    # Extract the best parameters for each transformation
    best_rigid_params = best_params_rigid
    best_similarity_params = best_params_similarity
    best_spline_params = best_params_spline

    parameters_dict = {
        "NumberOfResolutions": [],
        "Metric": [],
        "NumberOfHistogramBins": [],
        "Sampler": [],
        "NumberOfSpatialSamples": [],
        "BSplineInterpolationOrder": [],
        "HowToCombineTransforms": [],
        "MaximumNumberOfIterations": [],
        "FinalBSplineInterpolationOrder": []
    }

    # Map indices to parameter names
    parameter_names = list(parameters_dict.keys())

    # Fill the dictionary with values
    print(f"\nOptimal parameters for transformations: rigid, similarity, and spline:")
    for transform_type, best_params in zip(['rigid', 'similarity', 'spline'], 
                                           [best_rigid_params, best_similarity_params, best_spline_params]):
        for param, value in enumerate(best_params):
            if param == 1:  # For Metric
                parameters_dict[parameter_names[param]].append(metric_values.get(value, value))
            elif param == 3:  # For Sampler
                parameters_dict[parameter_names[param]].append(sampler_values.get(value, value))
            elif param == 6:  # For HowToCombineTransforms
                parameters_dict[parameter_names[param]].append(combine_transform_values.get(value, value))
            else:
                parameters_dict[parameter_names[param]].append(value)

    # Convert dictionary to DataFrame
    df_params = pd.DataFrame(parameters_dict)
    df_params.index = ['rigid', 'similarity', 'spline']

    # Display the table
    display(df_params)

    return descr_stats

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
        