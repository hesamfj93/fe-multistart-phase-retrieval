"""Data-fidelity losses for Fourier phase retrieval.

We support three flavors:

  * `magnitude_l2`: standard 0.5 ||F(x)|² − Y||² (Gaussian noise).
  * `log_magnitude_l2`: |log|F(x)| − log Y| — dynamic-range robust.
  * `poisson_nll`: |F(x)|² − Y · log|F(x)|² (Poisson likelihood, up to consts).

All losses accept an optional per-pixel weight `w` of shape (1, 1, oH, oW),
typically produced by the spectral annealing schedule.
"""

from __future__ import annotations

import torch

from ..physics.forward import forward_intensity


def _weighted_mean(t: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    if w is None:
        return t.mean()
    return (t * w).sum() / w.sum().clamp_min(1e-12)


def magnitude_l2(
    x: torch.Tensor,
    y_intensity: torch.Tensor,
    weight: torch.Tensor | None = None,
    oversample: int = 2,
) -> torch.Tensor:
    """0.5 · weighted mean of (|F(x)|² − Y)²."""
    intensity_pred = forward_intensity(x, oversample=oversample)
    diff = (intensity_pred - y_intensity).pow(2)
    return 0.5 * _weighted_mean(diff, weight)


def log_magnitude_l2(
    x: torch.Tensor,
    y_intensity: torch.Tensor,
    weight: torch.Tensor | None = None,
    oversample: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted mean of (log|F(x)| − log√Y)²."""
    intensity_pred = forward_intensity(x, oversample=oversample)
    log_pred = 0.5 * torch.log(intensity_pred.clamp_min(eps))
    log_meas = 0.5 * torch.log(y_intensity.clamp_min(eps))
    diff = (log_pred - log_meas).pow(2)
    return _weighted_mean(diff, weight)


def poisson_nll(
    x: torch.Tensor,
    y_intensity: torch.Tensor,
    weight: torch.Tensor | None = None,
    oversample: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Poisson negative log-likelihood, up to data-only constants.

      NLL = sum_i  ( λ_i − y_i · log λ_i )

    where λ_i = |F(x)|²_i. The data-only term `Σ y log y - y` is dropped
    since it does not depend on x.
    """
    lam = forward_intensity(x, oversample=oversample).clamp_min(eps)
    nll = lam - y_intensity * torch.log(lam)
    return _weighted_mean(nll, weight)


def build_data_loss(name: str):
    """Factory for selecting a data loss by string name."""
    if name in {"magnitude_l2", "mag_l2"}:
        return magnitude_l2
    if name in {"log_magnitude", "log_mag", "log_magnitude_l2"}:
        return log_magnitude_l2
    if name in {"poisson_nll", "poisson"}:
        return poisson_nll
    raise ValueError(f"Unknown data loss: {name!r}")
