"""Untrained convolutional decoder used by the double-decoder ensemble.

A decoder consumes a small latent code z ∈ R^{1 × Cz × Hz × Wz} and outputs
a (1 × 1 × H × W) feature map that is finally squashed to either:
  * amplitude A ∈ [0, 1] via sigmoid, or
  * phase     Φ ∈ [−π, π] via π · tanh.

The number of upsample stages is chosen so that 2^stages · Hz = H.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .blocks import ConvBlock, UpsampleBlock


class UntrainedDecoder(nn.Module):
    """Conv + upsample decoder producing a single-channel real output.

    Parameters
    ----------
    out_h, out_w : output spatial dimensions
    z_channels  : channel count of the latent code (e.g. 128)
    z_hw        : spatial side of the latent (must divide out_h, out_w by a
                  power of 2)
    base_ch     : channel count of the first layer
    output_activation : 'sigmoid' for amplitude, 'tanh_pi' for phase, or
                  'none' for raw real output.
    """

    def __init__(
        self,
        out_h: int,
        out_w: int,
        z_channels: int = 128,
        z_hw: int = 4,
        base_ch: int = 128,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        assert out_h == out_w, "Square outputs assumed; relax if needed."
        n_up = int(math.log2(out_h // z_hw))
        assert 2 ** n_up * z_hw == out_h, (
            f"out_h={out_h} must be z_hw={z_hw} times a power of two"
        )
        self.out_h, self.out_w = out_h, out_w
        self.z_channels, self.z_hw = z_channels, z_hw

        # Decreasing channel pyramid
        chs = [z_channels] + [max(base_ch // (2 ** i), 16) for i in range(n_up)]
        ups = []
        for i in range(n_up):
            ups.append(UpsampleBlock(chs[i], chs[i + 1]))
        self.ups = nn.Sequential(*ups)
        self.head = ConvBlock(chs[-1], 16)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        self.output_activation = output_activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.ups(z)
        h = self.head(h)
        h = self.out_conv(h)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(h)
        if self.output_activation == "tanh_pi":
            return math.pi * torch.tanh(h)
        if self.output_activation == "none":
            return h
        raise ValueError(f"Unknown output activation: {self.output_activation!r}")


def make_double_decoder(
    out_h: int,
    out_w: int,
    z_channels: int = 128,
    z_hw: int = 4,
    base_ch: int = 128,
) -> tuple[UntrainedDecoder, UntrainedDecoder]:
    """Construct a (g_A, g_Φ) pair with shared hyperparameters."""
    g_A = UntrainedDecoder(
        out_h, out_w, z_channels, z_hw, base_ch, output_activation="sigmoid"
    )
    g_phi = UntrainedDecoder(
        out_h, out_w, z_channels, z_hw, base_ch, output_activation="tanh_pi"
    )
    return g_A, g_phi


def latent_init(
    z_channels: int,
    z_hw: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Initialize a latent code z ~ N(0, 1) of shape (1, Cz, z_hw, z_hw)."""
    return torch.randn(
        (1, z_channels, z_hw, z_hw),
        generator=generator,
        device=device,
    )
