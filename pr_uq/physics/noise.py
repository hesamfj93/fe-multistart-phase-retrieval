"""Noise simulators for synthetic measurements.

We support:

  * Gaussian additive noise on intensity, parameterized by σ relative to
    the maximum intensity.
  * Poisson (shot) noise on intensity, parameterized by photon budget
    N_p (mean photons per pixel at the brightest measurement pixel).

Both functions return a noisy *intensity* tensor (not magnitude).
"""

from __future__ import annotations

import torch


def add_gaussian_noise(
    intensity: torch.Tensor, sigma: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Add zero-mean Gaussian noise scaled by σ · max(intensity)."""
    if sigma <= 0:
        return intensity.clone()
    scale = sigma * intensity.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    noise = torch.randn(intensity.shape, generator=generator, device=intensity.device)
    return (intensity + scale * noise).clamp_min(0.0)


def add_poisson_noise(
    intensity: torch.Tensor, photon_budget: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Apply shot noise with `photon_budget` photons at the peak.

    The intensity is first normalized so that its max = N_p, sampled from a
    Poisson distribution, then rescaled back to the original peak so PSNR
    metrics remain comparable across N_p.
    """
    if photon_budget <= 0 or not torch.isfinite(torch.tensor(photon_budget)):
        return intensity.clone()
    peak = intensity.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    scaled = intensity / peak * photon_budget
    # torch.poisson supports float tensors; ignore CPU/GPU generator nuance
    noisy = torch.poisson(scaled, generator=generator)
    return noisy / photon_budget * peak


def apply_noise(
    intensity: torch.Tensor,
    noise_model: str,
    noise_level: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Dispatch to Gaussian or Poisson noise."""
    if noise_model == "none" or noise_level == 0:
        return intensity.clone()
    if noise_model == "gaussian":
        return add_gaussian_noise(intensity, sigma=noise_level, generator=generator)
    if noise_model == "poisson":
        return add_poisson_noise(intensity, photon_budget=noise_level, generator=generator)
    raise ValueError(f"Unknown noise model: {noise_model!r}")
