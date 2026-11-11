"""Reconstruction quality metrics: PSNR, SSIM, MSE, Fourier magnitude error.

All metrics expect aligned reconstructions (see `pr_uq.metrics.alignment`).
The convenience wrapper `score_against_gt` performs alignment for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import torch
import torch.nn.functional as F

from ..physics.forward import forward_magnitude
from .alignment import AlignmentResult, align_to_gt


@dataclass
class ReconScores:
    """All metrics for one reconstruction vs ground truth."""

    psnr: float
    ssim: float
    mse: float
    nmse: float
    fourier_mag_error: float
    variant: int = 0
    shift: tuple[int, int] = (0, 0)
    raw_psnr: float | None = None
    runtime_s: float | None = None
    extra: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Base metrics
# -----------------------------------------------------------------------------

def _to_real(x: torch.Tensor) -> torch.Tensor:
    return x.real if torch.is_complex(x) else x


def mse(recon: torch.Tensor, gt: torch.Tensor) -> float:
    r = _to_real(recon)
    g = _to_real(gt)
    return float(F.mse_loss(r, g).item())


def nmse(recon: torch.Tensor, gt: torch.Tensor) -> float:
    r = _to_real(recon)
    g = _to_real(gt)
    num = (r - g).pow(2).sum()
    den = g.pow(2).sum().clamp_min(1e-12)
    return float((num / den).item())


def psnr(recon: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak-signal-to-noise ratio in dB, assuming `data_range` is the peak.

    For normalized images in [0, 1] use data_range=1.0.
    """
    m = mse(recon, gt)
    if m <= 1e-20:
        return 100.0
    return float(10.0 * torch.log10(torch.tensor(data_range ** 2 / m)).item())


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device | str) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)


def ssim(
    recon: torch.Tensor,
    gt: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Simple SSIM implementation, single-channel."""
    r = _to_real(recon).float()
    g = _to_real(gt).float()
    device = r.device
    kernel = _gaussian_kernel(window_size, sigma, device)
    pad = window_size // 2

    def conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad)

    mu_r = conv(r)
    mu_g = conv(g)
    mu_r2 = mu_r * mu_r
    mu_g2 = mu_g * mu_g
    mu_rg = mu_r * mu_g
    sig_r2 = conv(r * r) - mu_r2
    sig_g2 = conv(g * g) - mu_g2
    sig_rg = conv(r * g) - mu_rg
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_rg + c1) * (2 * sig_rg + c2)
    den = (mu_r2 + mu_g2 + c1) * (sig_r2 + sig_g2 + c2)
    return float(num.div(den.clamp_min(1e-12)).mean().item())


def fourier_magnitude_error(
    recon: torch.Tensor,
    gt: torch.Tensor,
    oversample: int = 2,
) -> float:
    """Normalized error between |F(recon)| and |F(gt)|.

    This metric is *invariant to PR ambiguities* — it does not require
    alignment — and is a useful sanity check (a method may produce wrong-
    looking images but with matching Fourier magnitudes; this would flag
    such cases).
    """
    if not torch.is_complex(recon):
        recon = recon.to(torch.complex64)
    if not torch.is_complex(gt):
        gt = gt.to(torch.complex64)
    Mr = forward_magnitude(recon, oversample=oversample)
    Mg = forward_magnitude(gt, oversample=oversample)
    num = (Mr - Mg).pow(2).sum()
    den = Mg.pow(2).sum().clamp_min(1e-12)
    return float((num / den).sqrt().item())


# -----------------------------------------------------------------------------
# Top-level: score (with alignment) against ground truth
# -----------------------------------------------------------------------------

def score_against_gt(
    recon: torch.Tensor,
    gt: torch.Tensor,
    data_range: float = 1.0,
    oversample: int = 2,
    runtime_s: float | None = None,
) -> ReconScores:
    """Align `recon` to `gt` and compute all reconstruction metrics."""
    # Raw (un-aligned) PSNR for reference
    raw = psnr(recon, gt, data_range=data_range)
    res = align_to_gt(recon, gt)
    aligned = res.aligned_recon
    return ReconScores(
        psnr=psnr(aligned, gt, data_range=data_range),
        ssim=ssim(aligned, gt, data_range=data_range),
        mse=mse(aligned, gt),
        nmse=nmse(aligned, gt),
        fourier_mag_error=fourier_magnitude_error(aligned, gt, oversample=oversample),
        variant=res.variant,
        shift=res.shift,
        raw_psnr=raw,
        runtime_s=runtime_s,
    )


# ---------------------------------------------------------------------------
# Complex-field metrics (stage 3, complex-valued objects)
# ---------------------------------------------------------------------------

def psnr_complex(recon: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR of the complex field: 10 log10(range^2 / mean |recon - gt|^2). Exact -> 100 dB."""
    d = recon.to(torch.complex64) - gt.to(torch.complex64)
    m = float(d.abs().pow(2).mean().item())
    if m <= 1e-20:
        return 100.0
    return float(10.0 * math.log10(data_range ** 2 / m))


def nrmse_complex(recon: torch.Tensor, gt: torch.Tensor) -> float:
    g = gt.to(torch.complex64)
    return float((recon.to(torch.complex64) - g).abs().pow(2).sum().sqrt() / g.abs().pow(2).sum().sqrt().clamp_min(1e-12))


def phase_error(recon: torch.Tensor, gt: torch.Tensor, thresh: float = 0.05) -> float:
    """Amplitude-weighted mean absolute phase difference (radians) where |gt| > thresh."""
    g = gt.to(torch.complex64)
    w = g.abs()
    m = w > thresh
    if not bool(m.any()):
        return float("nan")
    d = torch.angle(recon.to(torch.complex64) * g.conj()).abs()
    return float((w[m] * d[m]).sum() / w[m].sum())
