"""Calibration metrics for pixel-wise uncertainty in image reconstruction.

We adapt the standard expected calibration error (ECE) to continuous
outputs: pixels are binned by predicted uncertainty quantile, and within
each bin we compare the mean predicted uncertainty to the mean *actual*
reconstruction error.

A perfectly calibrated uncertainty estimator produces a 45° reliability
diagram and ECE = 0.

We also compute:

  * Pearson and Spearman correlation between uncertainty and absolute error.
  * Gaussian negative log-likelihood (NLL) assuming a per-pixel Gaussian
    likelihood with σ given by the predicted uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CalibrationResult:
    """Container for calibration measurements on one image."""

    ece: float                   # expected calibration error
    pearson_r: float             # uncertainty ↔ absolute error
    spearman_r: float            # rank correlation
    nll: float                   # Gaussian NLL per pixel
    bin_pred: list[float]        # mean predicted uncertainty per bin
    bin_err: list[float]         # mean absolute error per bin
    bin_counts: list[int]        # pixels per bin


# -----------------------------------------------------------------------------
# Reliability + ECE
# -----------------------------------------------------------------------------

def _bin_indices(values: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Quantile bin indices (equal-mass bins) for a flat tensor."""
    flat = values.flatten()
    qs = torch.linspace(0, 1, n_bins + 1, device=flat.device)
    edges = torch.quantile(flat, qs)
    # bucketize: returns index in [1, n_bins]; subtract 1
    idx = torch.bucketize(flat, edges[1:-1])
    return idx.clamp(0, n_bins - 1)


def reliability_diagram(
    uncertainty: torch.Tensor,
    abs_error: torch.Tensor,
    n_bins: int = 15,
) -> tuple[list[float], list[float], list[int]]:
    """Compute reliability bins: per-bin (mean uncertainty, mean error, count)."""
    u_flat = uncertainty.flatten()
    e_flat = abs_error.flatten()
    if u_flat.numel() != e_flat.numel():
        raise ValueError("uncertainty and abs_error must match in size")
    idx = _bin_indices(u_flat, n_bins)
    bin_pred = []
    bin_err = []
    bin_counts = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum().item())
        if n == 0:
            bin_pred.append(float("nan"))
            bin_err.append(float("nan"))
            bin_counts.append(0)
        else:
            bin_pred.append(float(u_flat[mask].mean().item()))
            bin_err.append(float(e_flat[mask].mean().item()))
            bin_counts.append(n)
    return bin_pred, bin_err, bin_counts


def expected_calibration_error(
    uncertainty: torch.Tensor,
    abs_error: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Weighted L1 gap between predicted uncertainty and observed error."""
    bin_pred, bin_err, bin_counts = reliability_diagram(uncertainty, abs_error, n_bins)
    total = sum(bin_counts)
    if total == 0:
        return 0.0
    ece = 0.0
    for p, e, n in zip(bin_pred, bin_err, bin_counts):
        if n == 0:
            continue
        ece += (n / total) * abs(p - e)
    return float(ece)


# -----------------------------------------------------------------------------
# Correlations
# -----------------------------------------------------------------------------

def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    xm = x - x.mean()
    ym = y - y.mean()
    den = xm.pow(2).sum().sqrt() * ym.pow(2).sum().sqrt()
    if den.item() < 1e-12:
        return 0.0
    return float((xm * ym).sum().item() / den.item())


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation."""
    x = x.flatten().float()
    y = y.flatten().float()
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    return _pearson(rx, ry)


# -----------------------------------------------------------------------------
# Gaussian NLL
# -----------------------------------------------------------------------------

def gaussian_nll(
    uncertainty: torch.Tensor,
    abs_error: torch.Tensor,
    sigma_floor: float = 1e-3,
) -> float:
    """Per-pixel Gaussian NLL: 0.5 (log(2πσ²) + (err/σ)²).

    `uncertainty` is interpreted as σ (predicted std).
    """
    sigma = uncertainty.clamp_min(sigma_floor)
    err = abs_error
    nll = 0.5 * (torch.log(2 * torch.pi * sigma.pow(2)) + (err / sigma).pow(2))
    return float(nll.mean().item())


# -----------------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------------

def calibrate(
    uncertainty: torch.Tensor,
    abs_error: torch.Tensor,
    n_bins: int = 15,
    sigma_floor: float = 1e-3,
) -> CalibrationResult:
    """Compute all calibration metrics for one image."""
    bin_pred, bin_err, bin_counts = reliability_diagram(uncertainty, abs_error, n_bins)
    return CalibrationResult(
        ece=expected_calibration_error(uncertainty, abs_error, n_bins),
        pearson_r=_pearson(uncertainty, abs_error),
        spearman_r=_spearman(uncertainty, abs_error),
        nll=gaussian_nll(uncertainty, abs_error, sigma_floor=sigma_floor),
        bin_pred=bin_pred,
        bin_err=bin_err,
        bin_counts=bin_counts,
    )
