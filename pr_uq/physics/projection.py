"""Magnitude-constraint projection used in the β-mixing step.

The classical magnitude projection replaces |F(x)| with the measured
sqrt(Y) while preserving the phase of F(x). When the measurement is noisy,
the projection is the maximum-likelihood estimate of the true field's
phase under Gaussian assumptions, given fixed magnitude.

We provide:

  * `project_magnitude(x, y_intensity)` — exact projection.
  * `project_intensity(x, y_intensity, eps)` — projection with a small
    floor to avoid division by zero in dark regions.

Both return a complex image tensor of the same spatial shape as `x`.
"""

from __future__ import annotations

import torch

from .forward import forward_field, inverse_field


def project_magnitude(
    x: torch.Tensor, y_intensity: torch.Tensor, oversample: int = 2, eps: float = 1e-8
) -> torch.Tensor:
    """Project x onto the set { z : |F(z)|^2 == y_intensity }.

    Parameters
    ----------
    x : (B, 1, H, W) complex image.
    y_intensity : (B, 1, oH, oW) measured intensity (NOT magnitude).
    oversample : must match the one used to generate y_intensity.
    eps : magnitude floor.
    """
    H, W = x.shape[-2:]
    X = forward_field(x, oversample=oversample)
    mag_meas = y_intensity.clamp_min(0).sqrt()
    mag_cur = X.abs().clamp_min(eps)
    # Replace magnitude with measured, keep phase
    X_proj = X * (mag_meas / mag_cur)
    return inverse_field(X_proj, signal_hw=(H, W))


def beta_mix(
    x_net: torch.Tensor,
    y_intensity: torch.Tensor,
    beta: float,
    oversample: int = 2,
) -> torch.Tensor:
    """One iteration of physics-projected β-mixing.

    Returns x_use = β · P(x_net) + (1 − β) · x_net where P is the magnitude
    projection. β should be annealed from ~0.7 (early, trust physics) to
    ~0.1 (late, trust the prior).
    """
    x_proj = project_magnitude(x_net, y_intensity, oversample=oversample)
    return beta * x_proj + (1.0 - beta) * x_net


def physics_residual(
    x_recon: torch.Tensor,
    y_intensity: torch.Tensor,
    oversample: int = 2,
    return_image_domain: bool = True,
) -> torch.Tensor:
    """Per-pixel physics residual r_phys.

    Computes |F(x)|^2 - Y at the detector and, if `return_image_domain=True`,
    pulls the residual back to image domain by replacing the field's
    magnitude with the residual and inverse-FFTing. The image-domain
    residual is what gets fused with ensemble variance for uncertainty.

    Returns a *real, non-negative* tensor of shape (B, 1, H, W) if
    `return_image_domain` else (B, 1, oH, oW).
    """
    H, W = x_recon.shape[-2:]
    X = forward_field(x_recon, oversample=oversample)
    intensity_pred = X.real.pow(2) + X.imag.pow(2)
    res = (intensity_pred - y_intensity).abs()
    if not return_image_domain:
        return res
    # Carry the residual back to the image domain by treating it as a magnitude
    # spectrum with zero phase. This is heuristic but commonly used for
    # diagnostic uncertainty visualization.
    phase = torch.angle(X)
    R = res.sqrt().to(torch.complex64) * torch.exp(1j * phase)
    r_image = inverse_field(R, signal_hw=(H, W))
    return r_image.abs()
