"""Single-member and ensemble wrappers for the double-decoder generator.

A single `EnsembleMember` holds:
  * two decoders (amplitude, phase),
  * two trainable latent codes (z_A, z_phi),
  * the random seed that initialized everything (for provenance).

The `Ensemble` is a list of K members. We train each member independently
(sequentially by default; parallel optional) and aggregate at evaluation
time. Independence of initialization is what gives us *diverse* members,
the engine that powers the variance-based uncertainty estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn

from .decoder import UntrainedDecoder, make_double_decoder, latent_init


@dataclass
class MemberSpec:
    """Hyperparameters that uniquely determine an ensemble member."""

    out_h: int
    out_w: int
    z_channels: int = 128
    z_hw: int = 4
    base_ch: int = 128
    seed: int = 0


class EnsembleMember(nn.Module):
    """One (g_A, g_Φ, z_A, z_Φ) tuple — a single untrained reconstruction."""

    def __init__(self, spec: MemberSpec, device: torch.device | str = "cpu"):
        super().__init__()
        self.spec = spec
        # Per-member RNG so seeds are reproducible regardless of caller state
        gen = torch.Generator(device="cpu").manual_seed(spec.seed)
        torch.manual_seed(spec.seed)  # ensure conv/bn init also reproducible
        self.g_A, self.g_phi = make_double_decoder(
            spec.out_h, spec.out_w, spec.z_channels, spec.z_hw, spec.base_ch
        )
        z_A = latent_init(spec.z_channels, spec.z_hw, device="cpu", generator=gen)
        z_phi = latent_init(spec.z_channels, spec.z_hw, device="cpu", generator=gen)
        # Latents are trainable parameters (the "z-update" idea)
        self.z_A = nn.Parameter(z_A)
        self.z_phi = nn.Parameter(z_phi)
        self.to(device)

    @property
    def net_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters of the two decoders (not including z)."""
        return list(self.g_A.parameters()) + list(self.g_phi.parameters())

    @property
    def latent_parameters(self) -> list[nn.Parameter]:
        return [self.z_A, self.z_phi]

    def amplitude(self) -> torch.Tensor:
        return self.g_A(self.z_A)

    def phase(self) -> torch.Tensor:
        return self.g_phi(self.z_phi)

    def complex_image(self) -> torch.Tensor:
        """Construct x = A · exp(j Φ). Returns (1, 1, H, W) complex tensor."""
        A = self.amplitude()
        phi = self.phase()
        return (A * torch.cos(phi)).to(torch.complex64) + 1j * (A * torch.sin(phi)).to(
            torch.complex64
        )

    def forward(self) -> torch.Tensor:
        return self.complex_image()


@dataclass
class Ensemble:
    """A list of trained ensemble members + their stored reconstructions."""

    members: list[EnsembleMember] = field(default_factory=list)
    reconstructions: list[torch.Tensor] = field(default_factory=list)
    """Final complex reconstructions (1, 1, H, W) for each member."""

    losses: list[float] = field(default_factory=list)
    """Final data loss for each member (used for tie-breaking)."""

    @property
    def K(self) -> int:
        return len(self.members)

    def stack_recon(self) -> torch.Tensor:
        """Stack all reconstructions into (K, 1, H, W) complex tensor."""
        if not self.reconstructions:
            raise RuntimeError("Ensemble has no reconstructions yet.")
        return torch.cat(self.reconstructions, dim=0)

    def mean_recon(self) -> torch.Tensor:
        """Ensemble mean of complex reconstructions.

        Important: we must align members to a common phase/shift convention
        before averaging — otherwise the mean is meaningless. The alignment
        happens at evaluation time via `pr_uq.metrics.alignment.align_ensemble`.
        """
        stack = self.stack_recon()
        return stack.mean(dim=0, keepdim=True)
