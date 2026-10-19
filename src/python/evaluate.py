from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast, final

import torch
import torch.nn as nn
from dataset import VisionDataset
from model import UNet
from torch.utils.data import DataLoader
from typing_extensions import override


def align_target_mask(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Center-crops the ground-truth mask to match the spatial dimensions of the model logits.

    When using unpadded U-Net (padding=0), a 416x416 input produces 228x228 logits while
    the dataset mask is 416x416. When using padded U-Net (padding=1), dimensions already
    match (416x416) and this function acts as a no-op.
    """
    _, _, h_pred, w_pred = logits.shape
    _, _, h_target, w_target = targets.shape

    if h_pred == h_target and w_pred == w_target:
        return targets

    diff_y: int = (h_target - h_pred) // 2
    diff_x: int = (w_target - w_pred) // 2
    return targets[
        :,
        :,
        diff_y : diff_y + h_pred,
        diff_x : diff_x + w_pred,
    ]


@final
class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy (with logits) and Soft Dice Loss."""

    bce_weight: float
    smooth: float
    bce_loss: nn.BCEWithLogitsLoss

    def __init__(
        self,
        bce_weight: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    @override
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        targets_aligned: torch.Tensor = align_target_mask(logits, targets)

        # 1. Numerically stable Binary Cross-Entropy on raw logits (Log-Sum-Exp trick)
        bce: torch.Tensor = cast(torch.Tensor, self.bce_loss(logits, targets_aligned))

        # 2. Differentiable Soft Dice Loss on continuous probabilities
        probs: torch.Tensor = torch.sigmoid(logits)
        batch_size: int = probs.shape[0]

        probs_flat: torch.Tensor = probs.contiguous().view(batch_size, -1)
        targets_flat: torch.Tensor = targets_aligned.contiguous().view(batch_size, -1)

        intersection: torch.Tensor = (probs_flat * targets_flat).sum(dim=1)
        cardinality: torch.Tensor = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        soft_dice: torch.Tensor = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )
        dice_loss: torch.Tensor = 1.0 - soft_dice.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


@final
@dataclass(frozen=True)
class EvaluationMetrics:
    """Container for validation and test segmentation metrics."""

    loss: float
    dice: float
    iou: float
    pixel_acc: float


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[float, float, float]:
    """Computes discrete Dice Coefficient (F1), IoU (Jaccard Index), and Pixel Accuracy."""
    targets_aligned: torch.Tensor = align_target_mask(logits, targets)

    probs: torch.Tensor = torch.sigmoid(logits)
    preds: torch.Tensor = (probs > threshold).float()

    batch_size: int = preds.shape[0]
    preds_flat: torch.Tensor = preds.contiguous().view(batch_size, -1)
    targets_flat: torch.Tensor = targets_aligned.contiguous().view(batch_size, -1)

    intersection: torch.Tensor = (preds_flat * targets_flat).sum(dim=1)
    preds_sum: torch.Tensor = preds_flat.sum(dim=1)
    targets_sum: torch.Tensor = targets_flat.sum(dim=1)

    # Dice = 2 * |P n G| / (|P| + |G|)
    dice: torch.Tensor = (2.0 * intersection + eps) / (preds_sum + targets_sum + eps)

    # IoU = |P n G| / (|P| + |G| - |P n G|)
    union: torch.Tensor = preds_sum + targets_sum - intersection
    iou: torch.Tensor = (intersection + eps) / (union + eps)

    # Pixel Accuracy
    pixel_acc: torch.Tensor = (preds_flat == targets_flat).float().mean(dim=1)

    return (
        float(dice.mean().item()),
        float(iou.mean().item()),
        float(pixel_acc.mean().item()),
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """Runs inference evaluation over a DataLoader without tracking gradients."""
    model.eval()

    total_loss: float = 0.0
    total_dice: float = 0.0
    total_iou: float = 0.0
    total_acc: float = 0.0
    total_samples: int = 0

    with torch.no_grad():
        for images, masks in dataloader:
            batch_size: int = images.shape[0]
            images_dev: torch.Tensor = images.to(device, non_blocking=True)
            masks_dev: torch.Tensor = masks.to(device, non_blocking=True)

            logits: torch.Tensor = cast(torch.Tensor, model(images_dev))
            loss: torch.Tensor = cast(torch.Tensor, criterion(logits, masks_dev))

            batch_dice, batch_iou, batch_acc = compute_batch_metrics(
                logits, masks_dev, threshold=threshold
            )

            total_loss += float(loss.item()) * batch_size
            total_dice += batch_dice * batch_size
            total_iou += batch_iou * batch_size
            total_acc += batch_acc * batch_size
            total_samples += batch_size

    denom: float = float(max(total_samples, 1))
    return EvaluationMetrics(
        loss=total_loss / denom,
        dice=total_dice / denom,
        iou=total_iou / denom,
        pixel_acc=total_acc / denom,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained U-Net on the test split."
    )
    _ = parser.add_argument(
        "--weights",
        type=str,
        default="./outputs/best_model.pth",
        help="Path to saved model state_dict (.pth).",
    )
    _ = parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/test",
        help="Path to test dataset directory.",
    )
    _ = parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation.",
    )
    _ = parser.add_argument(
        "--padding",
        type=int,
        default=0,
        choices=[0, 1],
        help="Convolution padding used in U-Net (0=unpadded, 1=padded).",
    )
    args = parser.parse_args()

    weights_path = Path(cast(str, args.weights))
    data_dir = cast(str, args.data_dir)
    batch_size = cast(int, args.batch_size)
    padding = cast(int, args.padding)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = VisionDataset(data_dir=data_dir, is_train=False)
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = UNet(in_channels=3, out_channels=1, padding=padding).to(device)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{weights_path}'. Train the model first."
        )

    state_dict: dict[str, torch.Tensor] = cast(
        dict[str, torch.Tensor],
        torch.load(weights_path, map_location=device, weights_only=True),
    )
    _ = model.load_state_dict(state_dict)

    criterion = BCEDiceLoss(bce_weight=0.5)
    metrics: EvaluationMetrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("-" * 55)
    print(f"Evaluation Results on '{data_dir}' ({len(test_dataset)} samples)")
    print("-" * 55)
    print(f"Combined Loss : {metrics.loss:.4f}")
    print(f"Dice Score    : {metrics.dice:.4f}")
    print(f"IoU (Jaccard) : {metrics.iou:.4f}")
    print(f"Pixel Acc     : {metrics.pixel_acc:.4f}")
    print("-" * 55)


if __name__ == "__main__":
    main()
