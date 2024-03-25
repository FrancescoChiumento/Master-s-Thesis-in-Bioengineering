from model import UNET
from utils import load_checkpoint
from utils import check_accuracy
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import os

class NewDataDataset(CarvanaDataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__(image_dir, mask_dir, transform)
        self.masks = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        # Check that there is a corresponding mask for each image.
        assert len(self.images) == len(self.masks), "The number of images and masks does not match."
        
        # Optional print statement for debugging.
        for img, mask in zip(sorted(self.images), sorted(self.masks)):
            print(f"Image: {img}, Mask: {mask}")
            assert os.path.splitext(img)[0] == os.path.splitext(mask)[0], f"Image and mask do not match: {img} != {mask}"

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

def load_model(checkpoint_path, device):
    model = UNET(in_channels=3, out_channels=1).to(device)
    load_checkpoint(torch.load(checkpoint_path, map_location=device), model)
    model.eval()
    return model

def predict(model, loader, device):
    model.eval()
    saved_predictions = []
    dice_scores = []

    with torch.no_grad():
        for image, mask in loader:
            image = image.to(device)
            mask = mask.to(device).float()
            mask = (mask > 0.5).float()
            preds = torch.sigmoid(model(image))
            preds_bin = (preds > 0.5).float()  
            saved_predictions.append(preds_bin)
            
            preds_flat = preds_bin.view(-1)
            mask_flat = mask.view(-1)
            dice_score = dice_coefficient(preds_flat, mask_flat)
            dice_scores.append(dice_score.item())
    
    return saved_predictions, dice_scores

def dice_coefficient(pred, target):
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum())
    
    return dice

BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = True 

def run_segmentation(checkpoint_path, images_dir, masks_dir, output_folder, device):
    print("Loading the model...")
    model = load_model(checkpoint_path, device)  # Load the UNET model from checkpoint

    # Define the image transformations
    transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    print("Preparing dataset...")
    dataset = NewDataDataset(images_dir, masks_dir, transform=transform)  # Prepare the dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)  # Prepare the DataLoader
    
    print("Starting the prediction process...")
    predictions, dice_scores = predict(model, loader, device)  # Make predictions using the model
    
    if not os.path.exists(output_folder):
        print(f"Creating the output directory at {output_folder}")
        os.makedirs(output_folder)
    
    print("Saving the predicted images...")
    for idx, (pred, dice_score) in enumerate(zip(predictions, dice_scores)):
        formatted_idx = str(idx).zfill(3)
        save_path = os.path.join(output_folder, f"prediction_{formatted_idx}.png")
        torchvision.utils.save_image(pred.float(), save_path)
        print(f"Saved prediction {formatted_idx} with Dice score: {dice_score}")

    # Calculate and print the mean Dice score
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    print(f"Mean Dice score: {mean_dice:.4f}")
