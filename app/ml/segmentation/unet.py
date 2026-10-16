"""
Standard UNet architecture for Dermoscopic Lesion Segmentation (Rule 11).
Trained on ISIC2018 Task 1 data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Classic UNet with 4 downsampling steps and skip connections.
    Takes 3-channel RGB image -> outputs 1-channel probability mask.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_features: int = 32):
        super().__init__()
        f = base_features

        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))

        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.conv_up1 = DoubleConv(f * 16, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.conv_up2 = DoubleConv(f * 8, f * 4)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.conv_up3 = DoubleConv(f * 4, f * 2)

        self.up4 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.conv_up4 = DoubleConv(f * 2, f)

        self.outc = nn.Conv2d(f, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        # Pad if dimensions differ slightly due to odd resolutions
        diff_y = x4.size()[2] - x.size()[2]
        diff_x = x4.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = self.conv_up1(torch.cat([x4, x], dim=1))

        x = self.up2(x)
        diff_y = x3.size()[2] - x.size()[2]
        diff_x = x3.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = self.conv_up2(torch.cat([x3, x], dim=1))

        x = self.up3(x)
        diff_y = x2.size()[2] - x.size()[2]
        diff_x = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = self.conv_up3(torch.cat([x2, x], dim=1))

        x = self.up4(x)
        diff_y = x1.size()[2] - x.size()[2]
        diff_x = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = self.conv_up4(torch.cat([x1, x], dim=1))

        logits = self.outc(x)
        return logits
