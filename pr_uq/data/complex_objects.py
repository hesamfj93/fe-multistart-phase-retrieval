"""Complex-valued test objects for the stage-3 experiment.

An MNIST digit gives the amplitude A in [0, 1]. The phase is a smooth random field: white
Gaussian noise filtered with a Gaussian of `sigma_px` pixels, centered and scaled so that its
largest absolute value over the support (A > `support_thresh`) is `max_phase`. The object is
x = A * exp(i * phi), so it is zero wherever A is zero.

For such objects the real and nonnegative constraint of classical CDI is wrong, which is the
case the stage-3 experiment is meant to test. The construction depends only on the image and
on a per-image seed, `phase_seed_offset + mnist_test_index`, so every run sees the same objects.
All parameters are fixed in the stage-3 configuration files before any held-out run.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

PHASE_SEED_OFFSET = 1_000_003
SUPPORT_THRESH = 0.05


def smooth_phase(amplitude: np.ndarray, seed: int, sigma_px: float = 3.0,
                 max_phase: float = math.pi / 2, support_thresh: float = SUPPORT_THRESH) -> np.ndarray:
    """Smooth random phase field of the shape of `amplitude` (2-D), scaled on the support."""
    rng = np.random.default_rng(int(seed))
    field = gaussian_filter(rng.standard_normal(amplitude.shape), sigma=float(sigma_px), mode="reflect")
    s = amplitude > support_thresh
    if s.any():
        field = field - field[s].mean()
        peak = np.abs(field[s]).max()
        if peak > 0:
            field = field * (float(max_phase) / peak)
    return field


def to_complex_object(img: torch.Tensor, seed: int, sigma_px: float = 3.0,
                      max_phase: float = math.pi / 2) -> torch.Tensor:
    """(1, 1, H, W) real image in [0, 1] -> (1, 1, H, W) complex64 object A * exp(i phi)."""
    amp = img.detach().float().cpu().numpy()[0, 0]
    phi = smooth_phase(amp, seed, sigma_px, max_phase)
    x = (amp * np.exp(1j * phi)).astype(np.complex64)
    return torch.from_numpy(x[None, None]).to(img.device)


def object_class(dataset_cfg: dict) -> str:
    return str((dataset_cfg.get("object") or {}).get("class", "real"))


def object_from_cfg(img: torch.Tensor, dataset_cfg: dict, ds_idx: int) -> torch.Tensor:
    """Apply the object model of the configuration. 'real' returns the image unchanged."""
    spec = dataset_cfg.get("object") or {}
    cls = str(spec.get("class", "real"))
    if cls == "real":
        return img
    if cls != "complex":
        raise ValueError(f"unknown object class {cls!r}; expected 'real' or 'complex'")
    return to_complex_object(
        img,
        seed=int(spec.get("phase_seed_offset", PHASE_SEED_OFFSET)) + int(ds_idx),
        sigma_px=float(spec.get("phase_sigma_px", 3.0)),
        max_phase=float(spec.get("phase_max_rad", math.pi / 2)),
    )
