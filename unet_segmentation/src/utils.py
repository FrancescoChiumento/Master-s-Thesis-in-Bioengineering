import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename): 
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(  # crea e restituisce DataLoader per i set di addestramento e validazione.
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers = 4,
        pin_memory = True,
):
    train_ds = CarvanaDataset(
        image_dir =train_dir,
        mask_dir = train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle = True,
    )

    val_ds = CarvanaDataset(
        image_dir = val_dir,
        mask_dir = val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        num_workers= num_workers,
        pin_memory = pin_memory,
        shuffle =False,
    )

    return train_loader,val_loader

def check_accuracy(loader, model ,device="cuda"): # si prende come input un DataLoader(loader), il modello (model) e il dispositivo su cui eseguire la valutazione (device, di default impostato su cuda)

    num_correct = 0 #Conta il numero totale di pixel correttamente segmentati
    num_pixels = 0 #conta il numero totale di pixel in tutte le immagini nel "loader"
    dice_score = 0
    model.eval() #pone il modello in modalità valutazione, disabilitando specifche caratteristiche come il dropout e la normalizzazione batch, che sono state utilizzate solo durante l'addestramento.

    with torch.no_grad(): #durante la valutazione non è necessario calcolare i gradienti, quindi si disabilita il loro calcolo per risparmiare memoria e velocizzare il processo.
        for x , y in loader: #x è l'immagine in input, y la maschera di segmentazione corretta.
            x = x.to(device) #sposta i dati su GPU
            y = y.to(device)
            preds = torch.sigmoid(model(x)) # il modello fa una predizione sull'immagine. La funzione sigmoid è applicata alla predizione perchè la rete U-net utilizza lafunzione di persita BCEWithLogitLoss.
            preds = (preds > 0.5).float() #converte le previsioni in valori binari (0 o 1) basandosi su una soglia di 0.5.
            num_correct += (preds == y).sum() #conta quanti pixel predetti corrispondono ai pixel reali della maschera di segmentazione
            num_pixels += torch.numel(preds) #aggiorna il conteggio totale dei pixel
            dice_score += (2 * (preds *y).sum()) / (
                (preds+y).sum() +1e-8
            )

        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
        )
        print(f"Dice score: {dice_score/len(loader)}")
        model.train()

def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")  # Salva le previsioni
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/target_{idx}.png")  # Salva le maschere di verità di base

    model.train()



