from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import torch
import torchvision.transforms.v2 as transforms  # pyright: ignore[reportMissingTypeStubs]
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import functional as F  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import override


class VisionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):

    data_dir: str
    images_dir: str
    masks_dir: str
    is_train: bool
    filenames: list[str]
    normalize: transforms.Normalize

    def __init__(self, data_dir: str, is_train: bool = True) -> None:
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "original")
        self.masks_dir = os.path.join(data_dir, "mask")
        self.is_train = is_train

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory '{self.images_dir}' not found")

        self.filenames = sorted(
            [
                f
                for f in os.listdir(self.images_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        # ImageNet calculations
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.filenames)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename: str = self.filenames[idx]
        img_path: str = os.path.join(self.images_dir, filename)

        stem: str = Path(filename).stem
        mask_path: str = os.path.join(self.masks_dir, f"{stem}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.masks_dir, f"{stem}.jpg")

        image: Image.Image = Image.open(img_path).convert("RGB")
        mask: Image.Image = Image.open(mask_path).convert("L")

        image_tensor = cast(torch.Tensor, F.to_image(image))  # pyright: ignore[reportUnknownMemberType]
        mask_tensor = cast(torch.Tensor, F.to_image(mask))  # pyright: ignore[reportUnknownMemberType]

        mask_tensor = F.resize(
            mask_tensor,
            [416, 416],
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        image_tensor = F.resize(
            image_tensor,
            [416, 416],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

        # In medical imaging, the orientation of the images are valid even if
        # it's flipped horizontally. Hence, we can get augment our data better
        if self.is_train and bool(torch.rand(1).item() < 0.5):
            image_tensor = F.hflip(image_tensor)
            mask_tensor = F.hflip(mask_tensor)

        # Scale RGB image to float32 [0.0, 1.0] and normalize
        image_tensor = F.to_dtype(image_tensor, torch.float32, scale=True)
        image_tensor = cast(torch.Tensor, self.normalize(image_tensor))

        # Convert mask to binary float32 {0, 1}
        # 0 for black, 1 for white pixels in mask
        mask_tensor = F.to_dtype(mask_tensor, torch.float32, scale=False)
        mask_tensor = (mask_tensor > 0).float()

        return image_tensor, mask_tensor


if __name__ == "__main__":
    train_dataset = VisionDataset(data_dir="./data/train", is_train=True)
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset, batch_size=4, shuffle=True
    )

    batch: tuple[torch.Tensor, torch.Tensor] = next(iter(train_loader))  # pyright: ignore[reportAny]
    images, masks = batch
    print(f"Batch image shape: {images.shape}")
    print(f"Batch mask shape:  {masks.shape}")
