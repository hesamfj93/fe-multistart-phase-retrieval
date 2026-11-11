"""Tests for the forward model and projection.

Run with:  pytest tests/test_forward.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.physics.forward import (
    forward_field, forward_intensity, forward_magnitude, inverse_field
)
from pr_uq.physics.projection import project_magnitude, beta_mix


@pytest.fixture
def x_image() -> torch.Tensor:
    """A small real test image embedded in a complex tensor."""
    torch.manual_seed(0)
    img = torch.zeros(1, 1, 16, 16, dtype=torch.complex64)
    img[..., 4:12, 4:12] = torch.rand(1, 1, 8, 8, dtype=torch.float32).to(torch.complex64)
    return img


def test_fft_roundtrip_unitary(x_image):
    """ortho FFT is unitary: ||F(x)|| = ||x|| (up to numerical eps)."""
    X = forward_field(x_image, oversample=1)
    n_in = x_image.abs().pow(2).sum().item()
    n_out = X.abs().pow(2).sum().item()
    assert abs(n_in - n_out) / max(n_in, 1e-12) < 1e-5


def test_inverse_recovers_image(x_image):
    """inverse_field(forward_field(x)) ≈ x (within signal support)."""
    X = forward_field(x_image, oversample=2)
    x_rec = inverse_field(X, signal_hw=(16, 16))
    err = (x_rec - x_image).abs().pow(2).sum().item()
    norm = x_image.abs().pow(2).sum().item()
    assert err / max(norm, 1e-12) < 1e-5


def test_intensity_nonneg(x_image):
    """|F(x)|^2 ≥ 0 everywhere."""
    Y = forward_intensity(x_image, oversample=2)
    assert (Y >= -1e-7).all()


def test_magnitude_consistency(x_image):
    Y = forward_intensity(x_image, oversample=2)
    M = forward_magnitude(x_image, oversample=2)
    assert torch.allclose(M.pow(2), Y, atol=1e-5)


def test_projection_reduces_residual(x_image):
    """After magnitude projection, |F(proj(x))| should be much closer to
    sqrt(Y) than |F(x_perturbed)|.

    Note: the projection is exact in the oversampled (2H, 2W) domain, but
    we crop back to (H, W) which enforces a compact-support constraint —
    after re-forward-FFT the magnitude is no longer exactly sqrt(Y). What
    we can test is that the projection *reduces* the residual substantially.
    """
    Y = forward_intensity(x_image, oversample=2)
    x_perturbed = x_image + 0.1 * torch.randn_like(x_image.real).to(torch.complex64)
    Y_before = forward_intensity(x_perturbed, oversample=2)
    res_before = (Y_before - Y).abs().mean().item() / Y.mean().item()

    x_proj = project_magnitude(x_perturbed, Y, oversample=2)
    Y_after = forward_intensity(x_proj, oversample=2)
    res_after = (Y_after - Y).abs().mean().item() / Y.mean().item()
    assert res_after < res_before, f"projection did not reduce residual: {res_before:.3e} -> {res_after:.3e}"


def test_projection_exact_before_crop(x_image):
    """Without crop (oversample=1 = no padding), projection is exact."""
    Y = forward_intensity(x_image, oversample=1)
    x_perturbed = x_image + 0.1 * torch.randn_like(x_image.real).to(torch.complex64)
    x_proj = project_magnitude(x_perturbed, Y, oversample=1)
    Y_after = forward_intensity(x_proj, oversample=1)
    rel_err = (Y_after - Y).abs().mean().item() / Y.mean().item()
    assert rel_err < 1e-5, f"projection at oversample=1 should be exact, got {rel_err:.2e}"


def test_beta_mix_endpoints(x_image):
    """β=0 returns x_net unchanged; β=1 returns the projection."""
    Y = forward_intensity(x_image, oversample=2)
    out0 = beta_mix(x_image, Y, beta=0.0, oversample=2)
    out1 = beta_mix(x_image, Y, beta=1.0, oversample=2)
    assert torch.allclose(out0, x_image, atol=1e-5)
    proj = project_magnitude(x_image, Y, oversample=2)
    assert torch.allclose(out1, proj, atol=1e-5)
