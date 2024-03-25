import os
import SimpleITK as sitk
import numpy as np

def levelset2binary(mask_LS_itk):
    # Trasforma la maschera level set in una maschera binaria
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

    # Esegui l'analisi dei componenti connessi sull'immagine originale
    labels = sitk.ConnectedComponent(image)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labels)

    # Ottieni le dimensioni delle etichette
    label_sizes = {l: stats.GetNumberOfPixels(l) for l in stats.GetLabels() if l != 0}
    print(f"Dimensioni delle etichette: {label_sizes}")

    # Seleziona l'etichetta da sovrascrivere, se specificata
    if override_label and override_label in label_sizes:
        selected_label = override_label
    else:
        selected_label = max(label_sizes, key=label_sizes.get)

    print(f"Etichetta selezionata: {selected_label} con {label_sizes[selected_label]} pixel")

    # Crea un'immagine binaria per la componente selezionata
    binary_image = sitk.BinaryThreshold(labels, lowerThreshold=selected_label, upperThreshold=selected_label, insideValue=255, outsideValue=0)

    mask = levelset2binary(binary_image)
    mask = sitk.Cast(mask, sitk.sitkInt16)

    # Ottieni i dati come array NumPy, permuta le dimensioni e poi riconverti in immagine SimpleITK
    mask_np = sitk.GetArrayFromImage(mask)
    mask_np_permuted = np.transpose(mask_np, (2, 1, 0))  # Permuta le dimensioni

    # Crea una nuova immagine SimpleITK dall'array permutato
    mask_permuted = sitk.GetImageFromArray(mask_np_permuted)
    
    # Imposta la spaziatura e l'origine come nella maschera originale
    mask_permuted.SetSpacing((0.4121, 0.4121, 0.4))  # Spaziatura permutata
    mask_permuted.SetOrigin(mask.GetOrigin())

    # Salva l'immagine con la spaziatura e le dimensioni corrette
    modified_file_path = os.path.join(output_folder, os.path.basename(file_path).replace('.mha', '_modified.mha'))
    sitk.WriteImage(mask_permuted, modified_file_path)


original_folder = r'E:\U-Net\SEGMENTAZIONI FINALI UNET\originali'
processed_folder = r'E:\U-Net\SEGMENTAZIONI FINALI UNET\saving_for_morphology'

for file_name in os.listdir(original_folder):
    if file_name.endswith('.mha'):
        file_path = os.path.join(original_folder, file_name)
        override_label = None  # Specifica l'etichetta da sovrascrivere se necessario
        process_and_save_image(file_path, processed_folder, override_label)
