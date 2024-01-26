import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len (self.images)
    
    def __getitem__(self, index): #stiamo accedendo agli elementi di un oggetto, in particolare prende l'indice dell'elemento che si vuole ottenere dal dataset
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg","_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        #carica l'immagine e la maschera utilizzando PIL (Python Imaging Library). L'immagine viene convertita in RGB e la maschera in scala di grigi (L). Le immagini vengono poi convertite in array Numpy.
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0 #usiamo una sigmoide nella nostra ultima attivazione. In particolare questa linea trasforma la maschera da valori di 255 (bianco) a 1.0. Questo è comune quando le maschere sono immagini in bianco e nero dove il bianco rappresenta l'oggetto di interesse. Poichè si prevede di usare una funzione di attivazione sigmoide nell'output della rete, i valori della maschera devono essere 0 o 1.

        #La funzione sigmoide è una funzione matematica che trasforma l'input in un valore compreso fra 0 e 1. E' comunemente usata nelle reti neurali, in particolare quando si vuole ottenere un outout che rappresenti una probabilità o una classificazione binaria. Per ogni valore in input x, la sigmoide produce un valore compreso fra 0 e 1. Se x è molto piccolo, la sigmoide si avvicina a 0; quando x è molto grande, si avvicina a 1.

        #Essa è spesso utilizzata come funzione di attivazione nell'ultimo strato di una rete neurale per problemi di classificazione binaria, dove l'obiettivo è prevedere una delle due classi (si o no).

        if self.transform is not None: # se viene fornita una trasformazione ( come per l'augentation dei dati o la preelaborazione), questa viene applicata sia all'immagine che alla maschera. La trasformazione potrebbe includere operazioni come normalizzazione, ridimnensionamento, ritagliatura o rotazione.
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    