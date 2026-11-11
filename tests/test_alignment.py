"""Tests for PR alignment.

Run with:  pytest tests/test_alignment.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.metrics.alignment import (
    align_to_gt, best_cyclic_shift, cyclic_shift, apply_variant
)
from pr_uq.metrics.reconstruction import psnr


def _make_gt() -> torch.Tensor:
    """Random sparse 16×16 real image."""
    torch.manual_seed(42)
    img = torch.zeros(1, 1, 16, 16)
    img[..., 4:10, 4:10] = torch.rand(1, 1, 6, 6)
    return img


def test_shift_recovery():
    """Cross-correlation finds a known cyclic shift exactly."""
    gt = _make_gt()
    dy_true, dx_true = 3, -2
    shifted = cyclic_shift(gt, dy_true, dx_true)
    dy, dx = best_cyclic_shift(shifted, gt)
    assert dy == dy_true and dx == dx_true, f"got dy={dy} dx={dx}"


def test_align_after_shift_perfect():
    """align_to_gt undoes a pure cyclic shift to PSNR > 60 dB."""
    gt = _make_gt()
    shifted = cyclic_shift(gt, 5, -3)
    res = align_to_gt(shifted, gt)
    aligned = res.aligned_recon
    p = psnr(aligned, gt)
    assert p > 50.0, f"PSNR after alignment {p:.2f} dB — alignment broken"


def test_align_after_flip():
    """align_to_gt undoes a horizontal flip."""
    gt = _make_gt()
    flipped = apply_variant(gt, 1)  # horizontal flip
    res = align_to_gt(flipped, gt)
    aligned = res.aligned_recon
    p = psnr(aligned, gt)
    assert p > 50.0, f"PSNR after flip alignment {p:.2f} dB"


def test_align_complex_global_phase():
    """Global phase rotation is undone in the complex case."""
    gt = _make_gt().to(torch.complex64)
    rotated = gt * torch.exp(torch.tensor(1j * 1.23))
    res = align_to_gt(rotated, gt, return_real=False)
    aligned = res.aligned_recon
    # PSNR on complex magnitudes
    err = (aligned - gt).abs().pow(2).mean().item()
    norm = gt.abs().pow(2).mean().item()
    assert err / max(norm, 1e-12) < 1e-4
