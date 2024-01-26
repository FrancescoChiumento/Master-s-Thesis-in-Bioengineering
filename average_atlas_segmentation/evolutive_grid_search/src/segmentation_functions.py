# Francesco Chiumento, 2023

import os
import sys

base_path = os.getcwd()
pykneer_path = os.path.join(base_path, "pykneer")
sys.path.append(pykneer_path)

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

combine_transform_values= {
    1: "Compose",
    2: "Add"
}

def create_MR_param_rigid(NumberOfResolutions, Metric, NumberOfHistogramBins,Sampler, NumberOfSpatialSamples, BSplineInterpolationOrder, HowToCombineTransforms, MaximumNumberOfIterations, FinalBSplineInterpolationOrder):
    print(f"Received parameters:  param_rigid: {NumberOfResolutions}, {Metric}, {NumberOfHistogramBins}, {Sampler}, {NumberOfSpatialSamples}, {BSplineInterpolationOrder}, {HowToCombineTransforms}, {MaximumNumberOfIterations}, {FinalBSplineInterpolationOrder}")
    Metric = round(Metric)
    Sampler = round(Sampler)
    HowToCombineTransforms = round(HowToCombineTransforms)
    Metric = metric_values.get(Metric, Metric)
    Sampler = sampler_values.get(Sampler, Sampler)
    HowToCombineTransforms = combine_transform_values.get(HowToCombineTransforms, HowToCombineTransforms)
    filepath = os.path.join(base_path, "pykneer", "parameterFiles", "MR_param_rigid.txt")
    with open(filepath, "w") as file:
        file.write(f"""
// Parameter file for rigid registration - Serena Bonaretti

// *********************** Images ***********************
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")

// ******************** Registration ********************
(Registration "MultiResolutionRegistration")
(NumberOfResolutions {round(NumberOfResolutions)})
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

// *********************** Metric ***********************
(Metric "{Metric}")
(NumberOfHistogramBins {round(NumberOfHistogramBins)})

// *********************** Sampler **********************
(ImageSampler "{Sampler}")
(NumberOfSpatialSamples {round(NumberOfSpatialSamples)})
(NewSamplesEveryIteration "true")

// ******************** Interpolator ********************
(Interpolator "BSplineInterpolator")
(BSplineInterpolationOrder {round(BSplineInterpolationOrder)})

// ******************* Transformation *******************
(Transform "EulerTransform")
(AutomaticTransformInitialization "true")
(AutomaticScalesEstimation "true")
(HowToCombineTransforms "{HowToCombineTransforms}")

// ********************* Optimizer **********************
(Optimizer "AdaptiveStochasticGradientDescent")
(MaximumNumberOfIterations {round(MaximumNumberOfIterations)})

// *********************** Masks ************************
(ErodeMask "false")

// ********************** Resampler *********************
(Resampler "DefaultResampler")
(DefaultPixelValue 0)

// **************** ResampleInterpolator ****************
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder {round(FinalBSplineInterpolationOrder)})

// ******************* Writing image ********************
(WriteResultImage "true")
(ResultImagePixelType "float")
(ResultImageFormat "mha")
""")


def create_MR_param_similarity(NumberOfResolutions, Metric, NumberOfHistogramBins,Sampler, NumberOfSpatialSamples, BSplineInterpolationOrder, HowToCombineTransforms, MaximumNumberOfIterations, FinalBSplineInterpolationOrder):
    print(f"Received parameters:  param_similarity: {NumberOfResolutions}, {Metric}, {NumberOfHistogramBins}, {Sampler}, {NumberOfSpatialSamples}, {BSplineInterpolationOrder}, {HowToCombineTransforms}, {MaximumNumberOfIterations}, {FinalBSplineInterpolationOrder}")
    Metric = round(Metric)
    Sampler = round(Sampler)
    HowToCombineTransforms = round(HowToCombineTransforms)
    Metric = metric_values.get(Metric, Metric)
    Sampler = sampler_values.get(Sampler, Sampler)
    HowToCombineTransforms = combine_transform_values.get(HowToCombineTransforms, HowToCombineTransforms)
    filepath = os.path.join(base_path, "pykneer", "parameterFiles", "MR_param_similarity.txt")
    with open(filepath, "w") as file:
        file.write(f"""     
// Parameter file to invert similarity registration - Serena Bonaretti

// *********************** Images ***********************
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")

// ******************** Registration ********************
(Registration "MultiResolutionRegistration")
(NumberOfResolutions {round(NumberOfResolutions)})
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

// *********************** Metric ***********************
(Metric "{Metric}")
(NumberOfHistogramBins {round(NumberOfHistogramBins)})

// *********************** Sampler **********************
(ImageSampler "{Sampler}")
(NumberOfSpatialSamples {round(NumberOfSpatialSamples)})
(NewSamplesEveryIteration "true")

// ******************** Interpolator ********************
(Interpolator "BSplineInterpolator")
(BSplineInterpolationOrder {round(BSplineInterpolationOrder)})

// ******************* Transformation *******************
(Transform "SimilarityTransform")
(AutomaticTransformInitialization "true")
(AutomaticScalesEstimation "true")
(HowToCombineTransforms "{HowToCombineTransforms}")

// ********************* Optimizer **********************
(Optimizer "AdaptiveStochasticGradientDescent")
(MaximumNumberOfIterations {round(MaximumNumberOfIterations)})

// *********************** Masks ************************
(ErodeMask "false")

// ********************** Resampler *********************
(Resampler "DefaultResampler")
(DefaultPixelValue 0)

// **************** ResampleInterpolator ****************
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder {int(FinalBSplineInterpolationOrder)})

// ******************* Writing image ********************
(WriteResultImage "true")
(ResultImagePixelType "float")
(ResultImageFormat "mha")
""")


def create_MR_param_spline(NumberOfResolutions, Metric, NumberOfHistogramBins,Sampler, NumberOfSpatialSamples, BSplineInterpolationOrder, HowToCombineTransforms, MaximumNumberOfIterations, FinalBSplineInterpolationOrder):
    print(f"Received parameters:  param_spline: {NumberOfResolutions}, {Metric}, {NumberOfHistogramBins}, {Sampler}, {NumberOfSpatialSamples}, {BSplineInterpolationOrder}, {HowToCombineTransforms}, {MaximumNumberOfIterations}, {FinalBSplineInterpolationOrder}")
    Metric = round(Metric)
    Sampler = round(Sampler)
    HowToCombineTransforms = round(HowToCombineTransforms)
    Metric = metric_values.get(Metric, Metric)
    Sampler = sampler_values.get(Sampler, Sampler)
    HowToCombineTransforms = combine_transform_values.get(HowToCombineTransforms, HowToCombineTransforms)
    filepath = os.path.join(base_path, "pykneer", "parameterFiles", "MR_param_spline.txt")
    with open(filepath, "w") as file:
        file.write(f"""
// Parameter file to invert B-spline registration - Serena Bonaretti
// *********************** Images ***********************
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")

// ******************** Registration ********************
(Registration "MultiResolutionRegistration")
(NumberOfResolutions {round(NumberOfResolutions)})
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

// *********************** Metric ***********************
(Metric "{Metric}")
(NumberOfHistogramBins {round(NumberOfHistogramBins)})

// *********************** Sampler **********************
(ImageSampler "{Sampler}")
(NumberOfSpatialSamples {round(NumberOfSpatialSamples)})
(NewSamplesEveryIteration "true")

// ******************** Interpolator ********************
(Interpolator "BSplineInterpolator")
(BSplineInterpolationOrder {round(BSplineInterpolationOrder)})

// ******************* Transformation *******************
(Transform "BSplineTransform")
(HowToCombineTransforms "{HowToCombineTransforms}")

// ********************* Optimizer **********************
(Optimizer "AdaptiveStochasticGradientDescent")
(MaximumNumberOfIterations {round(MaximumNumberOfIterations)})

// *********************** Masks ************************
(ErodeMask "false")

// ********************** Resampler *********************
(Resampler "DefaultResampler")
(DefaultPixelValue 0)

// **************** ResampleInterpolator ****************
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder {round(FinalBSplineInterpolationOrder)})

// ******************* Writing image ********************
(WriteResultImage "true")
(ResultImagePixelType "float")
(ResultImageFormat "mha")
""")

def create_MR_iparam_rigid(NumberOfResolutions, Metric, NumberOfHistogramBins,Sampler, NumberOfSpatialSamples, BSplineInterpolationOrder, HowToCombineTransforms, MaximumNumberOfIterations, FinalBSplineInterpolationOrder):
    print(f"Received parameters:  iparam_rigid: {NumberOfResolutions}, {Metric}, {NumberOfHistogramBins}, {Sampler}, {NumberOfSpatialSamples}, {BSplineInterpolationOrder}, {HowToCombineTransforms}, {MaximumNumberOfIterations}, {FinalBSplineInterpolationOrder}")
    Metric = round(Metric)
    Sampler = round(Sampler)
    HowToCombineTransforms = round(HowToCombineTransforms)
    Metric = metric_values.get(Metric, Metric)
    Sampler = sampler_values.get(Sampler, Sampler)
    HowToCombineTransforms = combine_transform_values.get(HowToCombineTransforms, HowToCombineTransforms)
    filepath = os.path.join(base_path, "pykneer", "parameterFiles", "MR_iparam_rigid.txt")
    with open(filepath, "w") as file:
        file.write(f"""
// Parameter file to invert rigid registration - Serena Bonaretti
// *********************** Images ***********************
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")

// ******************** Registration ********************
(Registration "MultiResolutionRegistration")
(NumberOfResolutions {round(NumberOfResolutions)})
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

// *********************** Metric ***********************
(Metric "{Metric}")
(NumberOfHistogramBins {round(NumberOfHistogramBins)})

// *********************** Sampler **********************
(ImageSampler "{Sampler}")
(NumberOfSpatialSamples {round(NumberOfSpatialSamples)})
(NewSamplesEveryIteration "true")

// ******************** Interpolator ********************
(Interpolator "BSplineInterpolator")
(BSplineInterpolationOrder {round(BSplineInterpolationOrder)})

// ******************* Transformation *******************
(Transform "EulerTransform")
(AutomaticTransformInitialization "true")
(AutomaticScalesEstimation "true")
(HowToCombineTransforms "{HowToCombineTransforms}")

// ********************* Optimizer **********************
(Optimizer "AdaptiveStochasticGradientDescent")
(MaximumNumberOfIterations {round(MaximumNumberOfIterations)})

// *********************** Masks ************************
(ErodeMask "false")

// ********************** Resampler *********************
(Resampler "DefaultResampler")
(DefaultPixelValue 0)

// **************** ResampleInterpolator ****************
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder {round(FinalBSplineInterpolationOrder)})

// ******************* Writing image ********************
(WriteResultImage "false")
(ResultImagePixelType "float")
(ResultImageFormat "mha")
""")
        
def create_MR_iparam_similarity(NumberOfResolutions, Metric, NumberOfHistogramBins,Sampler, NumberOfSpatialSamples, BSplineInterpolationOrder, HowToCombineTransforms, MaximumNumberOfIterations, FinalBSplineInterpolationOrder):
    print(f"Received parameters:  iparam_similarity: {NumberOfResolutions}, {Metric}, {NumberOfHistogramBins}, {Sampler}, {NumberOfSpatialSamples}, {BSplineInterpolationOrder}, {HowToCombineTransforms}, {MaximumNumberOfIterations}, {FinalBSplineInterpolationOrder}")
    Metric = round(Metric)
    Sampler = round(Sampler)
    HowToCombineTransforms = round(HowToCombineTransforms)
    Metric = metric_values.get(Metric, Metric)
    Sampler = sampler_values.get(Sampler, Sampler)
    HowToCombineTransforms = combine_transform_values.get(HowToCombineTransforms, HowToCombineTransforms)
    filepath = os.path.join(base_path, "pykneer", "parameterFiles", "MR_iparam_similarity.txt")
    with open(filepath, "w") as file:
        file.write(f"""
        
// Parameter file to invert similarity registration - Serena Bonaretti
// *********************** Images ***********************
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")

// ******************** Registration ********************
(Registration "MultiResolutionRegistration")
(NumberOfResolutions {round(NumberOfResolutions)})
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

// *********************** Metric ***********************
(Metric "{Metric}")
(NumberOfHistogramBins {round(NumberOfHistogramBins)})

// *********************** Sampler **********************
(ImageSampler "{Sampler}")
(NumberOfSpatialSamples {round(NumberOfSpatialSamples)})
(NewSamplesEveryIteration "true")

// ******************** Interpolator ********************
(Interpolator "BSplineInterpolator")
(BSplineInterpolationOrder {round(BSplineInterpolationOrder)})

// ******************* Transformation *******************
(Transform "SimilarityTransform")
(AutomaticTransformInitialization "true")
(AutomaticScalesEstimation "true")
(HowToCombineTransforms "{HowToCombineTransforms}")

// ********************* Optimizer **********************
(Optimizer "AdaptiveStochasticGradientDescent")
(MaximumNumberOfIterations {round(MaximumNumberOfIterations)})

// *********************** Masks ************************
(ErodeMask "false")

// ********************** Resampler *********************
(Resampler "DefaultResampler")
(DefaultPixelValue 0)

// **************** ResampleInterpolator ****************
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder {round(FinalBSplineInterpolationOrder)})

// ******************* Writing image ********************
(WriteResultImage "false")
(ResultImagePixelType "float")
(ResultImageFormat "mha")
""")
        
def create_MR_iparam_spline(NumberOfResolutions, Metric, NumberOfHistogramBins,Sampler, NumberOfSpatialSamples, BSplineInterpolationOrder, HowToCombineTransforms, MaximumNumberOfIterations, FinalBSplineInterpolationOrder):
    print(f"Received parameters:  iparam_spline: {NumberOfResolutions}, {Metric}, {NumberOfHistogramBins}, {Sampler}, {NumberOfSpatialSamples}, {BSplineInterpolationOrder}, {HowToCombineTransforms}, {MaximumNumberOfIterations}, {FinalBSplineInterpolationOrder}")
    Metric = round(Metric)
    Sampler = round(Sampler)
    HowToCombineTransforms = round(HowToCombineTransforms)
    Metric = metric_values.get(Metric, Metric)
    Sampler = sampler_values.get(Sampler, Sampler)
    HowToCombineTransforms = combine_transform_values.get(HowToCombineTransforms, HowToCombineTransforms)
    filepath = os.path.join(base_path, "pykneer", "parameterFiles", "MR_iparam_spline.txt")
    with open(filepath, "w") as file:
        file.write(f"""
// Parameter file to invert B-spline registration - Serena Bonaretti
// *********************** Images ***********************
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")

// ******************** Registration ********************
(Registration "MultiResolutionRegistration")
(NumberOfResolutions {round(NumberOfResolutions)})
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")

// *********************** Metric ***********************
(Metric "{Metric}")
(NumberOfHistogramBins {round(NumberOfHistogramBins)})

// *********************** Sampler **********************
(ImageSampler "{Sampler}")
(NumberOfSpatialSamples {round(NumberOfSpatialSamples)})
(NewSamplesEveryIteration "true")

// ******************** Interpolator ********************
(Interpolator "BSplineInterpolator")
(BSplineInterpolationOrder {round(BSplineInterpolationOrder)})

// ******************* Transformation *******************
(Transform "BSplineTransform")
(HowToCombineTransforms "{HowToCombineTransforms}")

// ********************* Optimizer **********************
(Optimizer "AdaptiveStochasticGradientDescent")
(MaximumNumberOfIterations {round(MaximumNumberOfIterations)})

// *********************** Masks ************************
(ErodeMask "false")

// ********************** Resampler *********************
(Resampler "DefaultResampler")
(DefaultPixelValue 0)

// **************** ResampleInterpolator ****************
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder {round(FinalBSplineInterpolationOrder)})

// ******************* Writing image ********************
(WriteResultImage "false")
(ResultImagePixelType "float")
(ResultImageFormat "mha")
""")