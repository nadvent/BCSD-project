import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F
from pathlib import Path


# Load Dataset

class VisionDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "original")
        self.masks_dir = os.path.join(data_dir, "mask")
        self.is_train = is_train

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory 'self.images_dir' not found")

        self.filenames = sorted(
            [f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.png', '.jpg'))]
        )

        self.normalize = transforms.Normalize(
            # Using ImageNet calculated values for RGB 
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )


    def __len__(self):
        return len(self.filenames)

    # get the images into a readable format and apply transformations
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.images_dir,filename)

        stem = Path(filename).stem

        mask_path = os.path.join(self.masks_dir, f"{stem}.png")  # Assuming masks are in PNG format

        if not os.path.exists(mask_path):
            fallback_jpg = os.path.join(self.masks_dir, f"{stem}.jpg")
            if os.path.exists(fallback_jpg):
                mask_path = fallback_jpg
            else:
                raise FileNotFoundError(f"Mask file for '{filename}' not found in '{self.masks_dir}'")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # greyscale

        # tensor
        image = F.to_image(image)
        mask = F.to_image(mask)

        mask = F.resize(mask, (416, 416), interpolation=transforms.InterpolationMode.NEAREST)
        image = F.resize(image, (416, 416), interpolation=transforms.InterpolationMode.BILINEAR)

        # we flip the image and mask horizontally because
        # in medical training, the orientation of the image is actually valid for both sides.. 
        # This is a data augmentation technique to increase the size of the dataset.
        if self.is_train:
            if torch.rand(1) < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
        
        # torch datatype and scale the pixels to float32 
        image = F.to_dtype(image, torch.float32, scale=True) # [0,1]
        image = self.normalize(image)

        mask = F.to_dtype(mask, torch.float32, scale=False)
        mask = (mask > 0).float() # in {0, 1}

        return image, mask 

if __name__ == "__main__":
    train_dataset = VisionDataset(data_dir="./data/train", is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    images, masks = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")  # [4, 3, 416, 416]
    print(f"Batch mask shape:  {masks.shape}")   # [4, 1, 416, 416]
