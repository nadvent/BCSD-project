from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

# Ensure src/python is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src" / "python"))

from config_utils import load_config
from dataset import VisionDataset
from evaluate import (
    BCEDiceLoss,
    align_target_mask,
    compute_batch_metrics,
)
from model import UNet


class TestPipeline(unittest.TestCase):

    def test_unet_shapes_unpadded(self) -> None:
        """Test unpadded U-Net (padding=0) produces [B, 1, 228, 228] output from [B, 3, 416, 416]."""
        model = UNet(in_channels=3, out_channels=1, padding=0)
        x = torch.randn(2, 3, 416, 416)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 1, 228, 228))

    def test_unet_shapes_padded(self) -> None:
        """Test padded U-Net (padding=1) preserves exact [B, 1, 416, 416] spatial dimensions."""
        model = UNet(in_channels=3, out_channels=1, padding=1)
        x = torch.randn(2, 3, 416, 416)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 1, 416, 416))

    def test_align_target_mask_unpadded(self) -> None:
        """Test that align_target_mask correctly crops a 416x416 target to 228x228 centered."""
        logits = torch.zeros(1, 1, 228, 228)
        targets = torch.arange(416 * 416, dtype=torch.float32).view(1, 1, 416, 416)
        cropped = align_target_mask(logits, targets)

        self.assertEqual(cropped.shape, (1, 1, 228, 228))
        # Check center offset (416 - 228) // 2 = 94
        expected = targets[:, :, 94 : 94 + 228, 94 : 94 + 228]
        self.assertTrue(torch.equal(cropped, expected))

    def test_align_target_mask_noop(self) -> None:
        """Test that align_target_mask is a no-op when shapes already match."""
        logits = torch.zeros(1, 1, 416, 416)
        targets = torch.ones(1, 1, 416, 416)
        aligned = align_target_mask(logits, targets)
        self.assertEqual(aligned.shape, (1, 1, 416, 416))
        self.assertTrue(torch.equal(aligned, targets))

    def test_bce_dice_loss_and_backward(self) -> None:
        """Test BCEDiceLoss scalar output and gradient flow."""
        criterion = BCEDiceLoss(bce_weight=0.5)
        logits = torch.randn(2, 1, 228, 228, requires_grad=True)
        targets = (torch.rand(2, 1, 416, 416) > 0.5).float()

        loss = criterion(logits, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.item()), 0.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        assert logits.grad is not None
        self.assertTrue(torch.any(logits.grad != 0.0))

    def test_compute_batch_metrics_perfect(self) -> None:
        """Test metrics for an exact prediction match."""
        logits = torch.full((1, 1, 100, 100), 10.0)
        targets = torch.ones(1, 1, 100, 100)

        dice, iou, acc = compute_batch_metrics(logits, targets, threshold=0.5)
        self.assertAlmostEqual(dice, 1.0, places=3)
        self.assertAlmostEqual(iou, 1.0, places=3)
        self.assertAlmostEqual(acc, 1.0, places=3)

    def test_compute_batch_metrics_disjoint(self) -> None:
        """Test metrics for completely non-overlapping predictions and targets."""
        logits = torch.full((1, 1, 100, 100), -10.0)
        targets = torch.ones(1, 1, 100, 100)

        dice, iou, acc = compute_batch_metrics(logits, targets, threshold=0.5)
        self.assertAlmostEqual(dice, 0.0, places=3)
        self.assertAlmostEqual(iou, 0.0, places=3)
        self.assertAlmostEqual(acc, 0.0, places=3)

    def test_load_default_config(self) -> None:
        """Test default.yaml loading and schema sanity."""
        cfg = load_config()
        self.assertIsInstance(cfg, dict)
        self.assertIn("data", cfg)
        self.assertIn("model", cfg)
        self.assertIn("training", cfg)
        self.assertIn("export", cfg)
        self.assertIn("inference", cfg)

        self.assertEqual(cfg["model"]["padding"], 0)
        self.assertEqual(cfg["data"]["input_height"], 416)
        self.assertEqual(cfg["training"]["epochs"], 20)

    def test_vision_dataset_smoke(self) -> None:
        """Test VisionDataset loading on available test split if present."""
        test_dir = repo_root / "data" / "test"
        if not (test_dir / "original").exists():
            self.skipTest("data/test/original not found, skipping dataset smoke test.")

        dataset = VisionDataset(str(test_dir), is_train=False)
        self.assertGreater(len(dataset), 0)
        img, mask = dataset[0]
        self.assertEqual(img.shape, (3, 416, 416))
        self.assertEqual(mask.shape, (1, 416, 416))
        self.assertTrue(set(mask.unique().tolist()).issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
