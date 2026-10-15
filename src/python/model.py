from __future__ import annotations

from typing import cast, final

import torch
from torch import nn
from typing_extensions import override


# two convolutional layers, followed by batchNorm (weights and bias) and ReLU.
@final
class DoubleConv(nn.Module):
    conv_block: nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1, # Configure for same or valid padding see README.md
    ) -> None:
        super().__init__()

        # Sequential is similar to torch.nn.ModuleList but provides a container
        # for running...you guess it, sequences.
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3, # f
                stride = 1, # s
                padding=padding,# p
                bias = False, # unneeded due to BatchNorm2d normalizing it after
            ),
            nn.BatchNorm2d(out_channels), # to reduce covariance between each "channel"
            nn.ReLU(inplace=True),

            # Repeat the same steps
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=padding,
                bias=False
            ),
            # Batch Normalization: Accelerating Deep Network Training by
            # Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @override
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv_block(x))


# Building UNet from scratch
# see https://arxiv.org/abs/1505.04597 for reference
@final
class UNet(nn.Module):
    padding: int
    pool: nn.MaxPool2d # using maxpool

    # Encode Blocks 1-4
    enc1: DoubleConv
    enc2: DoubleConv
    enc3: DoubleConv
    enc4: DoubleConv

    # Block 5
    bottleneck: DoubleConv

    # Decoder
    up4: nn.ConvTranspose2d
    dec4: DoubleConv
    up3: nn.ConvTranspose2d
    dec3: DoubleConv
    up2: nn.ConvTranspose2d
    dec2: DoubleConv
    up1: nn.ConvTranspose2d
    dec1: DoubleConv

    final_conv: nn.Conv2d

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1, # greyscale
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.enc1 = DoubleConv(in_channels, 64, padding=padding)
        self.enc2 = DoubleConv(64, 128, padding=padding)
        self.enc3 = DoubleConv(128, 256, padding=padding)
        self.enc4 = DoubleConv(256, 512, padding=padding)

        self.bottleneck = DoubleConv(512, 1024, padding=padding)

        '''
        Up-Conv or Transpose-convolutions 2x2,
        Upsample the image, decrease channels.
        it's pretty accurate because the information from
        the (feature maps in) encoder helps "skip connections" that provide spatial data
        We doubleconv again to combine the information from the upsampled layer
        It's also for refining data and avoiding artifacts caused by a 2x2 up-conv
        '''

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512, padding=padding)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256, padding=padding)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128, padding=padding)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, padding=padding)

        # 1x1 Convolution for raw logits rather than probabilities (e.g. torch.sigmoid).
        # More in BCEWithLogitsLoss in training.py
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    # This method is only relevant if we
    # DON'T use padding for our convolution layer (see README.md)

    def _center_crop(
        self,
        encoder_tensor: torch.Tensor,
        target_height: int,
        target_width: int,
    )->torch.Tensor:

        _, _, h, w = encoder_tensor.shape
        diff_y: int 
        diff_x: int

        diff_y = (h - target_height) // 2
        diff_x = (w - target_width) // 2

        return encoder_tensor[:, :, diff_y : diff_y + target_height, diff_x : diff_x + target_width]

    # Forward propagation (output) and linking all the layers
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = cast(torch.Tensor, self.enc1(x))
        p1 = cast(torch.Tensor, self.pool(e1))

        e2 = cast(torch.Tensor, self.enc2(p1))
        p2 = cast(torch.Tensor, self.pool(e2))

        e3 = cast(torch.Tensor, self.enc3(p2))
        p3 = cast(torch.Tensor, self.pool(e3))

        e4 = cast(torch.Tensor, self.enc4(p3))
        p4 = cast(torch.Tensor, self.pool(e4))

        b = cast(torch.Tensor, self.bottleneck(p4))

        d4 = cast(torch.Tensor, self.up4(b))
        e4_cropped = self._center_crop(e4, d4.shape[2], d4.shape[3])
        d4 = torch.cat([e4_cropped, d4], dim=1)
        d4 = cast(torch.Tensor, self.dec4(d4))

        d3 = cast(torch.Tensor, self.up3(d4))
        e3_cropped = self._center_crop(e3, d3.shape[2], d3.shape[3])
        d3 = torch.cat([e3_cropped, d3], dim=1)
        d3 = cast(torch.Tensor, self.dec3(d3))

        d2 = cast(torch.Tensor, self.up2(d3))
        e2_cropped = self._center_crop(e2, d2.shape[2], d2.shape[3])
        d2 = torch.cat([e2_cropped, d2], dim=1)
        d2 = cast(torch.Tensor, self.dec2(d2))

        d1 = cast(torch.Tensor, self.up1(d2))
        e1_cropped = self._center_crop(e1, d1.shape[2], d1.shape[3])
        d1 = torch.cat([e1_cropped, d1], dim=1)
        d1 = cast(torch.Tensor, self.dec1(d1))

        return cast(torch.Tensor, self.final_conv(d1))

if __name__ == "__main__":
    dummy_input: torch.Tensor = torch.randn(4, 3, 416, 416)

    # 1. Test Unpadded U-Net (padding=0, original 2015 paper behavior)
    unpadded_model = UNet(in_channels=3, out_channels=1, padding=0)
    unpadded_out: torch.Tensor = unpadded_model(dummy_input)  # pyright: ignore[reportAny]
    print(f"Input shape:           {dummy_input.shape}")
    print(f"Unpadded output shape: {unpadded_out.shape}  # [4, 1, 228, 228]")

    # 2. Test Padded U-Net (padding=1, preserves 416x416 mask dimensions)
    padded_model = UNet(in_channels=3, out_channels=1, padding=1)
    padded_out: torch.Tensor = padded_model(dummy_input)  # pyright: ignore[reportAny]
    print(f"Padded output shape:   {padded_out.shape}  # [4, 1, 416, 416]")

