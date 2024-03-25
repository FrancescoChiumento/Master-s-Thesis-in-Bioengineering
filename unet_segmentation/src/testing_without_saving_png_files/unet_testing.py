import SimpleITK as sitk
import os
from PIL import Image, ImageOps
import numpy as np
from model import UNET
from utils import load_checkpoint
from utils import check_accuracy
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import os
from torchvision.transforms.functional import to_tensor
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader




def extract_slices(mha_path, is_mask=False, desired_size=(512, 512)):
    itk_image = sitk.ReadImage(mha_path)
    img_array = sitk.GetArrayFromImage(itk_image)
    slices = []
    
    for i in range(img_array.shape[2]):
        slice = img_array[:, :, i]

        if is_mask:
            slice = slice * 255

        slice_padded = pad_image(slice, desired_size)
        # Convert the slice to RGB by replicating the channel three times.
        img = Image.fromarray(slice_padded.astype('uint8')).convert("RGB")
        img = img.rotate(-180)
        img = ImageOps.mirror(img)
        slices.append(img)
    
    return slices

def process_slice(idx, img_slice, mask_slice, model, device, transform, predictions_dir, img_file):
    augmented = transform(image=np.array(img_slice), mask=np.array(mask_slice))
    img_transformed = augmented['image'].unsqueeze(0)

    with autocast():
        preds = model(img_transformed.to(device))
    preds_bin = (torch.sigmoid(preds) > 0.5).float()
    
    mask_array = np.array(mask_slice) / 255.0
    if mask_array.ndim == 3 and mask_array.shape[2] > 1:
        mask_array = mask_array[:, :, 0]  # If the mask is multichannel, take only one channel.
    
    mask_tensor = torch.tensor(mask_array, dtype=torch.float32).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    
    preds_bin_2d = preds_bin.squeeze().view(-1)
    mask_tensor_2d = mask_tensor.squeeze().view(-1)
    
    assert preds_bin_2d.shape[0] == mask_tensor_2d.shape[0], "The tensor dimensions do not match."
    
    dice_score = dice_coefficient(preds_bin_2d, mask_tensor_2d)
    
    # Print the value of the Dice score for this slice
    print(f"Dice score for {img_file} slice {idx}: {dice_score.item()}")
    
    preds_bin_np = preds_bin.cpu().squeeze().numpy()
    pred_img = Image.fromarray((preds_bin_np * 255).astype(np.uint8))
    pred_img.save(os.path.join(predictions_dir, f"pred_{img_file[:-4]}_slice_{idx}.png"))
    
    return dice_score.item()


def pad_image(array, desired_size):
 
    delta_width = desired_size[1] - array.shape[1]
    delta_height = desired_size[0] - array.shape[0]
    top, bottom = delta_height // 2, delta_height - (delta_height // 2)
    left, right = delta_width // 2, delta_width - (delta_width // 2)
    

    return np.pad(array, ((top, bottom), (left, right)), 'constant', constant_values=0)

class NewDataDataset(CarvanaDataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__(image_dir, mask_dir, transform)
        self.masks = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        # Check that there is a corresponding mask for each image
        assert len(self.images) == len(self.masks), "The number of images and masks does not match"
        
        # Optional print for debugging purposes
        for img, mask in zip(sorted(self.images), sorted(self.masks)):
            print(f"Image: {img}, Mask: {mask}")
            assert os.path.splitext(img)[0] == os.path.splitext(mask)[0], f"Image and mask do not match {img} != {mask}"

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
    # Load the weights directly into the model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}  # Remove the prefix "module."
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel AFTER loading the weights.
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
NUM_WORKERS = 8
PIN_MEMORY = True 


def run_segmentation(checkpoint_path, images_dir, masks_dir, device, predictions_dir):
    print("Loading the model...")
    model = load_model(checkpoint_path, device)

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # Change: Use a list to maintain order
    results = []

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.mha')]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.mha')]
    assert len(image_files) == len(mask_files), "The number of images and masks does not match"

    for img_file, mask_file in zip(image_files, mask_files):
        img_mha_path = os.path.join(images_dir, img_file)
        mask_mha_path = os.path.join(masks_dir, mask_file)

        image_slices = extract_slices(img_mha_path, is_mask=False)
        mask_slices = extract_slices(mask_mha_path, is_mask=True)

        for idx, (img_slice, mask_slice) in enumerate(zip(image_slices, mask_slices)):
            # Sequential execution to maintain order
            dice_score = process_slice(idx, img_slice, mask_slice, model, device, transform, predictions_dir, img_file)
            results.append((img_file, idx, dice_score))

    # Sort and print the results sequentially
    for img_file, idx, dice_score in sorted(results, key=lambda x: (x[0], x[1])):
        formatted_idx = str(idx).zfill(4) 
        print(f"Dice score for {img_file} slice {formatted_idx}: {dice_score:.4f}")
        pred_img_path = os.path.join(predictions_dir, f"pred_{img_file[:-4]}_slice_{formatted_idx}.png")
        pred_img = Image.fromarray((preds_bin_np * 255).astype(np.uint8))
        pred_img.save(pred_img_path)

    mean_dice_score = sum([x[2] for x in results]) / len(results) if results else 0
    print(f"Mean Dice score: {mean_dice_score:.4f}")
    print("Segmentation process completed.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_segmentation(checkpoint_path= r"C:\Users\franc\OneDrive\Desktop\Test\UNet\my_checkpoint_BA0.pth.tar",
                 images_dir=r"C:\Users\franc\OneDrive\Desktop\Test\UNet\images",
                 masks_dir=r"C:\Users\franc\OneDrive\Desktop\Test\UNet\masks",
                 device=device,
                 predictions_dir=r"C:\Users\franc\OneDrive\Desktop\Test\UNet\predictions")
