"""Reusable convolutional building blocks for the untrained decoder.

We deliberately keep the architecture small (≈100k–300k params per decoder)
so that the implicit DIP regularization is strong: large overparameterized
networks tend to memorize noise and weaken the prior.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → LeakyReLU. Optional reflection padding."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        use_bn: bool = True,
        act_slope: float = 0.2,
        reflection_pad: bool = True,
    ):
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        if reflection_pad:
            layers.append(nn.ReflectionPad2d(pad))
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0, bias=not use_bn)
            )
        else:
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=not use_bn)
            )
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(act_slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Bilinear upsample x2 followed by two ConvBlocks (no skips)."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBlock(in_ch, mid_ch)
        self.conv2 = ConvBlock(mid_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(self.up(x)))
