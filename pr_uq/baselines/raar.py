"""Relaxed Averaged Alternating Reflections (RAAR) for FPR.

Reference: D. R. Luke, "Relaxed averaged alternating reflections for
diffraction imaging", Inverse Problems 21:1 (2005).

RAAR update:
    R_F = 2 P_F − I    (reflection through magnitude set)
    R_S = 2 P_S − I    (reflection through support set)
    x_{n+1} = (β / 2) [R_S R_F + I] x_n  +  (1 − β) P_F x_n

We share the API with `run_hio` for easy interchangeability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..physics.forward import forward_field, forward_intensity, inverse_field
from .hio import _binary_support


@dataclass
class RAARResult:
    recon: torch.Tensor
    runtime_s: float
    iters: int
    restarts: int
    final_data_loss: float


def _project_magnitude_complex(
    x: torch.Tensor, y_mag: torch.Tensor, oversample: int, image_hw: tuple[int, int]
) -> torch.Tensor:
    X = forward_field(x, oversample=oversample)
    X_proj = y_mag * X / X.abs().clamp_min(1e-12)
    return inverse_field(X_proj, signal_hw=image_hw)


def _project_support(x: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
    """Real, non-negative inside support; zero outside."""
    x_real = x.real.clamp_min(0)
    return (x_real * support_mask).to(torch.complex64)


def run_raar(
    y_intensity: torch.Tensor,
    image_hw: tuple[int, int],
    iters: int = 2000,
    beta: float = 0.87,
    restarts: int = 3,
    support_area_frac: float = 0.5,
    oversample: int = 2,
    seed: int = 0,
    device: torch.device | str = "cpu",
    support_mask: torch.Tensor | None = None,
) -> RAARResult:
    """Run RAAR with `restarts` random initializations; return best."""
    H, W = image_hw
    y_mag = y_intensity.clamp_min(0).sqrt()
    if support_mask is None:
        support_mask = _binary_support(H, W, support_area_frac, device=device)
    else:
        support_mask = (support_mask > 0.5).float().to(device)

    best_loss = float("inf")
    best_x: torch.Tensor | None = None
    t0 = time.perf_counter()
    for r in range(restarts):
        gen = torch.Generator(device="cpu").manual_seed(seed + r)
        phase0 = torch.rand(y_mag.shape, generator=gen) * 2 * torch.pi
        X0 = (y_mag.cpu() * torch.exp(1j * phase0)).to(device).to(torch.complex64)
        x = inverse_field(X0, signal_hw=image_hw).to(device)

        for it in range(iters):
            Pf = _project_magnitude_complex(x, y_mag, oversample, image_hw)
            Rf = 2 * Pf - x
            Ps_Rf = _project_support(Rf, support_mask)
            Rs_Rf = 2 * Ps_Rf - Rf
            x = 0.5 * beta * (Rs_Rf + x) + (1.0 - beta) * Pf

        intensity_pred = forward_intensity(x, oversample=oversample)
        loss = float((intensity_pred - y_intensity).pow(2).mean().item())
        if loss < best_loss:
            best_loss = loss
            best_x = x.detach().clone()

    runtime = time.perf_counter() - t0
    assert best_x is not None
    return RAARResult(
        recon=best_x, runtime_s=runtime, iters=iters, restarts=restarts, final_data_loss=best_loss
    )
