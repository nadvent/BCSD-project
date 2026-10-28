from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from config_utils import load_config
from dataset import VisionDataset
from evaluate import (
    BCEDiceLoss,
    EvaluationMetrics,
    compute_batch_metrics,
    evaluate_model,
)
from model import UNet
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm  # pyright: ignore[reportMissingTypeStubs]


def select_device() -> torch.device:
    """Selects the best available hardware accelerator (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Executes a single training epoch and returns (mean_loss, mean_dice)."""
    _ = model.train()

    running_loss: float = 0.0
    running_dice: float = 0.0
    total_samples: int = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:02d}/{total_epochs:02d}]",
        leave=False,
    )

    for images, masks in progress_bar:
        batch_size: int = images.shape[0]
        images_dev: torch.Tensor = images.to(device, non_blocking=True)
        masks_dev: torch.Tensor = masks.to(device, non_blocking=True)

        # 1. Clear previous gradients
        optimizer.zero_grad(set_to_none=True)

        # 2. Forward pass
        logits: torch.Tensor = cast(torch.Tensor, model(images_dev))

        # 3. Compute combined BCE + Soft Dice loss
        loss: torch.Tensor = cast(torch.Tensor, criterion(logits, masks_dev))

        # 4. Backward pass & weight update
        loss.backward()
        _ = optimizer.step()

        # 5. Track batch metrics without gradient overhead
        with torch.no_grad():
            batch_dice, _, _ = compute_batch_metrics(logits, masks_dev)

        batch_loss_val: float = float(loss.item())
        running_loss += batch_loss_val * batch_size
        running_dice += batch_dice * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{batch_loss_val:.4f}",
            dice=f"{batch_dice:.4f}",
        )

    denom: float = float(max(total_samples, 1))
    return running_loss / denom, running_dice / denom


def main() -> None:
    # 0. Check for optional --config argument before full parsing
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    conf_args, _ = conf_parser.parse_known_args()

    cfg = load_config(conf_args.config)
    cfg_data = cfg.get("data", {})
    cfg_model = cfg.get("model", {})
    cfg_train = cfg.get("training", {})

    parser = argparse.ArgumentParser(
        description="Train U-Net on the Blood Cell Segmentation Dataset.",
        parents=[conf_parser],
    )
    _ = parser.add_argument(
        "--train_dir",
        type=str,
        default=cfg_data.get("train_dir", "./data/train"),
        help="Path to training dataset directory.",
    )
    _ = parser.add_argument(
        "--val_dir",
        type=str,
        default=cfg_data.get("val_dir", "./data/test"),
        help="Path to validation/test dataset directory.",
    )
    _ = parser.add_argument(
        "--output_dir",
        type=str,
        default=cfg_train.get("output_dir", "./outputs"),
        help="Directory to save model checkpoints.",
    )
    _ = parser.add_argument(
        "--epochs",
        type=int,
        default=int(cfg_train.get("epochs", 20)),
        help="Number of training epochs.",
    )
    _ = parser.add_argument(
        "--batch_size",
        type=int,
        default=int(cfg_train.get("batch_size", 4)),
        help="Mini-batch size.",
    )
    _ = parser.add_argument(
        "--lr",
        type=float,
        default=float(cfg_train.get("learning_rate", 1e-4)),
        help="Initial learning rate for AdamW.",
    )
    _ = parser.add_argument(
        "--weight_decay",
        type=float,
        default=float(cfg_train.get("weight_decay", 1e-4)),
        help="Weight decay for AdamW.",
    )
    _ = parser.add_argument(
        "--bce_weight",
        type=float,
        default=float(cfg_train.get("bce_weight", 0.5)),
        help="Weight assigned to BCE in combined loss (remainder goes to Soft Dice).",
    )
    _ = parser.add_argument(
        "--padding",
        type=int,
        default=int(cfg_model.get("padding", 0)),
        choices=[0, 1],
        help="Convolution padding in U-Net (0=unpadded 228x228 output, 1=padded 416x416 output).",
    )
    _ = parser.add_argument(
        "--height",
        type=int,
        default=int(cfg_data.get("input_height", 416)),
        help="Input image height.",
    )
    _ = parser.add_argument(
        "--width",
        type=int,
        default=int(cfg_data.get("input_width", 416)),
        help="Input image width.",
    )
    args = parser.parse_args()

    train_dir = cast(str, args.train_dir)
    val_dir = cast(str, args.val_dir)
    output_dir = Path(cast(str, args.output_dir))
    epochs = cast(int, args.epochs)
    batch_size = cast(int, args.batch_size)
    learning_rate = cast(float, args.lr)
    weight_decay = cast(float, args.weight_decay)
    bce_weight = cast(float, args.bce_weight)
    padding = cast(int, args.padding)
    input_height = cast(int, args.height)
    input_width = cast(int, args.width)

    output_dir.mkdir(parents=True, exist_ok=True)
    device: torch.device = select_device()
    print(f"Using compute device: {device}")

    # 1. Initialize Datasets and DataLoaders
    train_dataset = VisionDataset(
        data_dir=train_dir,
        is_train=True,
        input_height=input_height,
        input_width=input_width,
    )
    val_dataset = VisionDataset(
        data_dir=val_dir,
        is_train=False,
        input_height=input_height,
        input_width=input_width,
    )

    use_pin_memory: bool = device.type == "cuda"
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
    )

    print(
        f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples."
    )

    # 2. Initialize Model, Loss, Optimizer, and Scheduler
    model = UNet(in_channels=3, out_channels=1, padding=padding).to(device)
    criterion = BCEDiceLoss(bce_weight=bce_weight)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_dice: float = 0.0
    best_ckpt_path: Path = output_dir / "best_model.pth"
    last_ckpt_path: Path = output_dir / "last_model.pth"

    # 3. Training & Validation Loop
    print("-" * 88)
    for epoch in range(1, epochs + 1):
        start_time: float = time.time()

        train_loss, train_dice = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )

        val_metrics: EvaluationMetrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_metrics.dice)
        current_lr: float = float(optimizer.param_groups[0]["lr"])
        elapsed: float = time.time() - start_time

        saved_marker: str = ""
        if val_metrics.dice > best_val_dice:
            best_val_dice = val_metrics.dice
            torch.save(model.state_dict(), best_ckpt_path)
            saved_marker = " [*Best*]"

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_metrics.loss:.4f} | Val Dice: {val_metrics.dice:.4f} | "
            f"Val IoU: {val_metrics.iou:.4f} | LR: {current_lr:.1e}{saved_marker}"
        )

    # Save final epoch weights
    torch.save(model.state_dict(), last_ckpt_path)
    print("-" * 88)
    print(
        f"Training complete. Best Validation Dice: {best_val_dice:.4f} saved to '{best_ckpt_path}'."
    )


if __name__ == "__main__":
    main()
