"""Textbook HIO and RAAR on the full oversampled grid (Stage-2 classical baselines).

The older `hio.py` and `raar.py` crop every iterate back to the H x W image window, so the
zero-padded region of the oversampled field never receives the HIO feedback or the RAAR
reflection. These functions keep the iterate on the full oversampled grid, which is the
standard formulation for oversampled Fourier phase retrieval.

Two object constraint sets S are provided:
- "support_only": zero outside the central H x W window where the image sits before zero
  padding, complex values allowed inside. This is the same prior the untrained method uses
  (it estimates a complex object; the window is the extent implied by 2x oversampling).
- "real_nonneg": additionally real and non-negative inside the window. These are the usual
  assumptions of classical CDI. They hold for MNIST, and with them noiseless oversampled
  MNIST is typically recovered almost exactly, so this variant is a strong reference rather
  than a like-for-like comparison.

References
- J. R. Fienup, "Phase retrieval algorithms: a comparison", Applied Optics 21, 2758 (1982).
- D. R. Luke, "Relaxed averaged alternating reflections for diffraction imaging",
  Inverse Problems 21, 37 (2005).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from ..physics.forward import forward_field, inverse_field


@dataclass
class ClassicalResult:
    recon: torch.Tensor        # (1, 1, H, W) complex64, cropped object window
    runtime_s: float
    algorithm: str
    seed: int


def object_window(image_hw: tuple[int, int], oversample: int, device) -> torch.Tensor:
    """Boolean (1, 1, oH, oW) mask of the central H x W window (same placement as the forward model)."""
    H, W = image_hw
    oH, oW = H * oversample, W * oversample
    m = torch.zeros(1, 1, oH, oW, dtype=torch.bool, device=device)
    h0, w0 = (oH - H) // 2, (oW - W) // 2
    m[..., h0:h0 + H, w0:w0 + W] = True
    return m


def _crop(z: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    H, W = image_hw
    oH, oW = z.shape[-2:]
    h0, w0 = (oH - H) // 2, (oW - W) // 2
    return z[..., h0:h0 + H, w0:w0 + W]


def _proj_magnitude(z: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
    """P_M: keep the phase of F(z), impose the measured magnitude. Full grid in and out."""
    Z = forward_field(z.to(torch.complex64), oversample=1)
    Zp = y_mag * Z / Z.abs().clamp_min(1e-12)
    return inverse_field(Zp, signal_hw=tuple(z.shape[-2:]))


CONSTRAINTS = ("real_nonneg", "support_only", "shrinkwrap")

# Shrinkwrap (Marchesini et al. 2003): every SW_EVERY iterations the support becomes the part of
# the object window where the Gaussian-blurred amplitude exceeds SW_THRESH times its maximum. The
# blur starts at SW_SIGMA0 pixels and shrinks by 1% per update down to SW_SIGMA_MIN. Inside the
# support the amplitude is bounded by 1, the bound the untrained network has through its sigmoid.
SW_EVERY, SW_THRESH, SW_SIGMA0, SW_SIGMA_DECAY, SW_SIGMA_MIN, AMP_MAX = 20, 0.2, 3.0, 0.99, 1.5, 1.0


def _clip_magnitude(v: torch.Tensor, vmax: float = AMP_MAX) -> torch.Tensor:
    mag = v.abs()
    return torch.where(mag > vmax, v * (vmax / mag.clamp_min(1e-12)), v)


def _shrinkwrap_support(z: torch.Tensor, window: torch.Tensor, sigma: float) -> torch.Tensor:
    """Window AND (Gaussian-blurred |z| > SW_THRESH * max over the window)."""
    import torch.nn.functional as F
    r = int(math.ceil(3 * sigma))
    xs = torch.arange(-r, r + 1, dtype=torch.float32, device=z.device)
    g = torch.exp(-xs ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    a = z.abs().float()
    a = F.conv2d(a, g.view(1, 1, 1, -1), padding=(0, r))
    a = F.conv2d(a, g.view(1, 1, -1, 1), padding=(r, 0))
    peak = a[window].max()
    return window & (a > SW_THRESH * peak)


def _proj_object(z: torch.Tensor, window: torch.Tensor, constraint: str = "real_nonneg") -> torch.Tensor:
    """P_S. real_nonneg: real part clipped at zero inside the window. support_only: keep
    complex values inside the window. Both: zero outside."""
    z = z.to(torch.complex64)
    if constraint == "real_nonneg":
        v = z.real.clamp_min(0).to(torch.complex64)
    elif constraint == "support_only":
        v = z
    elif constraint == "shrinkwrap":
        v = _clip_magnitude(z)          # `window` is the current shrinkwrap support here
    else:
        raise ValueError(f"unknown constraint {constraint!r}; expected one of {CONSTRAINTS}")
    return torch.where(window, v, torch.zeros_like(v))


def _init(y_mag: torch.Tensor, seed: int) -> torch.Tensor:
    """Random-phase start with the measured magnitude (phase drawn on the CPU for reproducibility)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    phase = (torch.rand(tuple(y_mag.shape), generator=g) * 2 * torch.pi).to(y_mag.device)
    Z0 = y_mag.to(torch.complex64) * torch.exp(1j * phase)
    return inverse_field(Z0, signal_hw=tuple(y_mag.shape[-2:])).real.to(torch.complex64)


def _error_reduction(z, y_mag, window, iters, constraint):
    for _ in range(iters):
        z = _proj_object(_proj_magnitude(z, y_mag), window, constraint)
    return z


def run_hio_full(y_intensity, image_hw, iters=5000, beta=0.9, er_iters=200, seed=0, oversample=2,
                 device="cpu", constraint="real_nonneg") -> ClassicalResult:
    """Fienup HIO: z <- P_M z where it satisfies S, else z - beta * P_M z; then an ER tail."""
    if constraint not in CONSTRAINTS:
        raise ValueError(f"unknown constraint {constraint!r}")
    t0 = time.perf_counter()
    y_mag = y_intensity.clamp_min(0).sqrt().to(device)
    window = object_window(image_hw, oversample, device)
    if constraint == "real_nonneg":
        z = _init(y_mag, seed).real
        for _ in range(iters):
            zp = _proj_magnitude(z, y_mag).real
            ok = window & (zp >= 0)
            z = torch.where(ok, zp, z - beta * zp)
        z = z.to(torch.complex64)
    elif constraint == "shrinkwrap":
        z = _init(y_mag, seed)
        support, sigma = window.clone(), SW_SIGMA0
        for it in range(iters):
            zp = _proj_magnitude(z, y_mag)
            ok = support & (zp.abs() <= AMP_MAX)
            z = torch.where(ok, zp, z - beta * zp)
            if (it + 1) % SW_EVERY == 0:
                support = _shrinkwrap_support(z, window, sigma)
                sigma = max(SW_SIGMA_MIN, sigma * SW_SIGMA_DECAY)
        window = support
    else:
        z = _init(y_mag, seed)
        for _ in range(iters):
            zp = _proj_magnitude(z, y_mag)
            z = torch.where(window, zp, z - beta * zp)
    z = _error_reduction(z, y_mag, window, er_iters, constraint)
    if z.is_cuda:
        torch.cuda.synchronize()
    return ClassicalResult(_crop(z, image_hw).to(torch.complex64), time.perf_counter() - t0, f"hio_{constraint}", seed)


def run_raar_full(y_intensity, image_hw, iters=5000, beta=0.87, er_iters=200, seed=0, oversample=2,
                  device="cpu", constraint="real_nonneg") -> ClassicalResult:
    """Luke RAAR: z <- (beta/2)(R_S R_M + I) z + (1 - beta) P_M z; then an ER tail."""
    if constraint not in CONSTRAINTS:
        raise ValueError(f"unknown constraint {constraint!r}")
    t0 = time.perf_counter()
    y_mag = y_intensity.clamp_min(0).sqrt().to(device)
    window = object_window(image_hw, oversample, device)
    z = _init(y_mag, seed)
    support, sigma = window.clone(), SW_SIGMA0
    for it in range(iters):
        pm = _proj_magnitude(z, y_mag)
        rm = 2 * pm - z
        rs_rm = 2 * _proj_object(rm, support, constraint) - rm
        z = 0.5 * beta * (rs_rm + z) + (1.0 - beta) * pm
        if constraint == "shrinkwrap" and (it + 1) % SW_EVERY == 0:
            support = _shrinkwrap_support(_proj_magnitude(z, y_mag), window, sigma)
            sigma = max(SW_SIGMA_MIN, sigma * SW_SIGMA_DECAY)
    z = _error_reduction(_proj_object(z, support, constraint), y_mag, support, er_iters, constraint)
    if z.is_cuda:
        torch.cuda.synchronize()
    return ClassicalResult(_crop(z, image_hw).to(torch.complex64), time.perf_counter() - t0, f"raar_{constraint}", seed)
