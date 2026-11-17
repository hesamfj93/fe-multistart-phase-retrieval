"""Simulated twin of the CXIDB entry 28 object, for the development phase of the real-data demonstration.

The measured object (Marchesini et al., Phys. Rev. B 68, 140101, 2003) is a layer of 50 nm gold
balls on a silicon nitride membrane. The twin is built only from what the measured pattern shows
without a reconstruction: the autocorrelation of the 1024 x 1024 pattern (scripts/cxidb28_preprocess.py)
has a central peak and two side peaks about 1.6 um apart, so the object is two clusters about
0.4 um across, offset by about 90 and 15 pixels of the 17.9 nm grid.

Model
- Balls of 50 nm diameter (2.79 px), placed by random sequential addition without overlap inside
  two ellipses with semi-axes CLUSTER_SEMI_AXES_PX, until FILL of the ellipse area is covered or
  MAX_TRIES placements fail. The cluster centres are +-CLUSTER_OFFSET_PX / 2 from the field centre.
- Projected thickness t of each ball (sphere chord), summed over balls, sampled on a grid
  SUPERSAMPLE times finer and averaged back to 17.9 nm pixels after forming the exit wave.
- Exit wave relative to the unscattered beam: x = exp(-k (beta + i delta) t) - 1 with k = 2 pi / lambda.
  delta and beta are approximate values for gold near 590 eV, used to make the twin complex in a
  realistic way (amplitude about 0.55 and phase about 3.8 rad for a full ball), not to model the
  experiment exactly.

Measurement: |F x|^2 with the forward model of the code base (128 x 128 field, oversampling 2),
set to zero where the real pattern is unmeasured, with optional noise:
- "lognormal": each intensity multiplied by exp(sigma e - sigma^2 / 2), e ~ N(0, 1). In the real
  pattern the difference between centrosymmetric pixel pairs has a relative SD of 12 to 16% of the
  intensity at every intensity level from 1e3 to 1e7 (not the 1 / sqrt(I) of photon noise), and a
  relative centrosymmetry error of 0.012 on the magnitudes; sigma = 0.15 reproduces both, while
  the noiseless twin has 0.001.
- "poisson": N_p photons at the brightest measured pixel.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..physics.forward import forward_intensity

PIXEL_NM = 17.944335428224846          # object pixel of the 256 x 256 binned pattern (preprocess.json)
WAVELENGTH_NM = 2.1
BALL_DIAMETER_NM = 50.0
DELTA_AU, BETA_AU = 3.5e-3, 2.9e-3      # approximate, gold near 590 eV
CLUSTER_SEMI_AXES_PX = (11.0, 7.0)      # (vertical, horizontal)
CLUSTER_OFFSET_PX = (90.0, 15.0)        # second cluster minus first cluster, (vertical, horizontal)
FILL = 0.55
MAX_TRIES = 20000
SUPERSAMPLE = 4
FIELD = 128


def ball_centres(seed: int) -> np.ndarray:
    """(n, 2) ball centres in pixels of the FIELD x FIELD grid, (row, col)."""
    rng = np.random.default_rng(int(seed))
    d = BALL_DIAMETER_NM / PIXEL_NM
    a, b = CLUSTER_SEMI_AXES_PX
    c0 = np.array([FIELD / 2, FIELD / 2])
    off = np.array(CLUSTER_OFFSET_PX) / 2
    centres = []
    for cc in (c0 - off, c0 + off):
        target = FILL * math.pi * a * b / (math.pi * d * d / 4)
        placed, tries = [], 0
        while len(placed) < target and tries < MAX_TRIES:
            tries += 1
            u = rng.uniform(-1, 1, size=2)
            if u[0] ** 2 + u[1] ** 2 > 1:
                continue
            p = cc + u * np.array([a - d / 2, b - d / 2])
            if all(np.hypot(*(p - q)) >= d for q in placed):
                placed.append(p)
        centres.extend(placed)
    return np.array(centres)


def thickness_nm(centres: np.ndarray, supersample: int = SUPERSAMPLE) -> np.ndarray:
    """Projected gold thickness on the fine grid, (FIELD*s, FIELD*s), in nm."""
    s = supersample
    n = FIELD * s
    r = BALL_DIAMETER_NM / 2
    coords = (np.arange(n) + 0.5) / s * PIXEL_NM          # pixel centres in nm
    t = np.zeros((n, n))
    for cy, cx in centres:
        y0, x0 = cy * PIXEL_NM, cx * PIXEL_NM
        iy = np.where(np.abs(coords - y0) <= r)[0]
        ix = np.where(np.abs(coords - x0) <= r)[0]
        if iy.size == 0 or ix.size == 0:
            continue
        dy = coords[iy][:, None] - y0
        dx = coords[ix][None, :] - x0
        rho2 = dy ** 2 + dx ** 2
        t[np.ix_(iy, ix)] += 2 * np.sqrt(np.clip(r * r - rho2, 0, None))
    return t


def exit_wave(seed: int) -> torch.Tensor:
    """Complex (1, 1, FIELD, FIELD) twin object x = exp(-k (beta + i delta) t) - 1."""
    t = thickness_nm(ball_centres(seed))
    k = 2 * math.pi / WAVELENGTH_NM
    xf = np.exp(-k * (BETA_AU + 1j * DELTA_AU) * t) - 1.0
    s = SUPERSAMPLE
    x = xf.reshape(FIELD, s, FIELD, s).mean(axis=(1, 3))
    return torch.from_numpy(x.astype(np.complex64)).view(1, 1, FIELD, FIELD)


def measurement(x: torch.Tensor, mask: torch.Tensor, noise: str = "none", level: float | None = None,
                seed: int = 0) -> torch.Tensor:
    """|F x|^2 (oversampling 2), zero where `mask` is False, with noise "none", "lognormal" (level =
    sigma) or "poisson" (level = photons at the brightest measured pixel). The noise is drawn on the
    CPU from torch.Generator(seed)."""
    y = forward_intensity(x.to(torch.complex64), oversample=2).float()
    mask = mask.to(torch.bool)
    y = torch.where(mask, y, torch.zeros_like(y))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    if noise == "lognormal":
        sig = float(level)
        e = torch.randn(tuple(y.shape), generator=g)
        y = y * torch.exp(sig * e - sig * sig / 2)
    elif noise == "poisson":
        peak = y[mask].max()
        y = torch.poisson(y * (float(level) / peak), generator=g) * (peak / float(level))
    elif noise != "none":
        raise ValueError(f"unknown noise {noise!r}")
    return torch.where(mask, y, torch.zeros_like(y))
