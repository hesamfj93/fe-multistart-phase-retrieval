"""Classical Fienup HIO (Hybrid Input-Output) baseline for FPR.

Reference: J. R. Fienup, "Phase retrieval algorithms: a comparison",
Applied Optics 21:15 (1982).

Notes
-----
* Operates entirely in the oversampled domain assumed by the rest of the
  codebase (centered FFT with `norm='ortho'`).
* Uses random-phase initialization with the measured magnitude.
* Support is provided externally as a binary mask of the image-sized
  region; default = central rectangle of half the area.
* Returns the best of several random restarts (by training-domain
  data consistency, not by GT — kept fair).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..physics.forward import forward_field, forward_intensity, inverse_field


@dataclass
class HIOResult:
    recon: torch.Tensor      # (1, 1, H, W) complex
    runtime_s: float
    iters: int
    restarts: int
    final_data_loss: float


def _binary_support(h: int, w: int, area_frac: float, device) -> torch.Tensor:
    """Centered binary support occupying `area_frac` of the area."""
    if area_frac >= 1.0:
        return torch.ones(1, 1, h, w, device=device)
    side = (area_frac ** 0.5)
    hh = int(h * side)
    ww = int(w * side)
    mask = torch.zeros(1, 1, h, w, device=device)
    h0 = (h - hh) // 2
    w0 = (w - ww) // 2
    mask[..., h0 : h0 + hh, w0 : w0 + ww] = 1.0
    return mask


def run_hio(
    y_intensity: torch.Tensor,
    image_hw: tuple[int, int],
    iters: int = 2000,
    beta: float = 0.9,
    restarts: int = 3,
    support_area_frac: float = 0.5,
    oversample: int = 2,
    seed: int = 0,
    device: torch.device | str = "cpu",
    support_mask: torch.Tensor | None = None,
) -> HIOResult:
    """Run HIO; return the best of `restarts` random initializations.

    Parameters
    ----------
    y_intensity : (1, 1, oH, oW) measured intensity.
    image_hw    : (H, W) of the unpadded image.
    iters       : iterations per restart.
    beta        : HIO feedback constant (typical: 0.7–0.9).
    restarts    : number of random restarts; we keep the lowest-final-loss one.
    support_area_frac : if `support_mask` is None, build a centered binary
        support of this area fraction.
    """
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
        # Random-phase initialization
        gen = torch.Generator(device="cpu").manual_seed(seed + r)
        phase0 = torch.rand(y_mag.shape, generator=gen) * 2 * torch.pi
        X0 = (y_mag.cpu() * torch.exp(1j * phase0)).to(device).to(torch.complex64)
        x_prev = inverse_field(X0, signal_hw=image_hw).to(device)

        for it in range(iters):
            # Fourier magnitude projection
            X = forward_field(x_prev, oversample=oversample)
            X_proj = y_mag * X / X.abs().clamp_min(1e-12)
            x_curr = inverse_field(X_proj, signal_hw=image_hw)

            # Real-image enforcement (assume real ground truth)
            x_real = x_curr.real

            # HIO update inside / outside support
            inside = support_mask > 0.5
            x_new = torch.where(
                inside, x_real.clamp_min(0), x_prev.real - beta * x_real
            )
            x_prev = x_new.to(torch.complex64)

        # Final data loss
        intensity_pred = forward_intensity(x_prev, oversample=oversample)
        loss = float((intensity_pred - y_intensity).pow(2).mean().item())
        if loss < best_loss:
            best_loss = loss
            best_x = x_prev.detach().clone()

    runtime = time.perf_counter() - t0
    assert best_x is not None
    return HIOResult(
        recon=best_x, runtime_s=runtime, iters=iters, restarts=restarts, final_data_loss=best_loss
    )
