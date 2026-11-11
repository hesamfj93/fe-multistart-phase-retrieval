"""Tests for fidelity-weighted ensemble aggregation.

Run with:  pytest tests/test_aggregation.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.uncertainty.ensemble_uq import (
    ensemble_aggregate, fidelity_weights, select_anchor, member_fourier_errors,
)
from pr_uq.physics.forward import forward_intensity
from pr_uq.metrics.reconstruction import psnr


def _signal(seed=0):
    torch.manual_seed(seed)
    x = torch.zeros(1, 1, 16, 16)
    x[..., 4:12, 4:12] = torch.rand(1, 1, 8, 8)
    return x.to(torch.complex64)


def test_fidelity_weights_sum_to_one():
    fe = torch.tensor([0.1, 0.2, 0.5, 0.3])
    for method in ["uniform", "softmax", "inverse", "top_k"]:
        w = fidelity_weights(fe, method=method, top_k=2)
        assert abs(float(w.sum()) - 1.0) < 1e-5
        assert (w >= 0).all()


def test_fidelity_weights_downweight_bad():
    """Softmax must give the lowest-FE member the highest weight."""
    fe = torch.tensor([0.10, 0.40, 0.15])
    w = fidelity_weights(fe, method="softmax", temperature=0.1)
    assert int(torch.argmax(w)) == 0            # lowest FE gets most weight
    assert w[1] < w[2] < w[0]                    # ordering follows FE


def test_top_k_selection_zeros_the_rest():
    fe = torch.tensor([0.1, 0.9, 0.2, 0.8])
    w = fidelity_weights(fe, method="top_k", top_k=2)
    assert (w > 0).sum() == 2
    assert w[0] > 0 and w[2] > 0                 # the two lowest-FE
    assert w[1] == 0 and w[3] == 0


def test_uniform_equals_legacy_mean():
    """ensemble_aggregate(uniform, first) == plain aligned mean."""
    stack = torch.stack([_signal(s).squeeze(0) for s in range(3)], dim=0)
    y = forward_intensity(_signal(0), oversample=2)
    stats = ensemble_aggregate(stack, y_intensity=y, anchor_method="first",
                               weight_method="uniform", align=False)
    # With align=False and uniform weights, mean is the plain average
    expected = stack.mean(dim=0, keepdim=True)
    assert torch.allclose(stats.mean_complex, expected, atol=1e-5)


def test_best_fe_anchor_picks_lowest_error_member():
    """The member equal to the true signal has FE≈0 and must be the anchor."""
    true = _signal(0)
    y = forward_intensity(true, oversample=2)
    # stack: member 0 = true signal, members 1-2 = noise-corrupted
    bad1 = true + 0.5 * torch.randn_like(true.real).to(torch.complex64)
    bad2 = true + 0.8 * torch.randn_like(true.real).to(torch.complex64)
    stack = torch.cat([true, bad1, bad2], dim=0)
    fe = member_fourier_errors(stack, y, oversample=2)
    assert int(torch.argmin(fe)) == 0
    assert select_anchor(stack, method="best_fe", y_intensity=y) == 0


def test_downweighting_beats_uniform_with_one_bad_member():
    """A stack of 2 good + 1 bad member: softmax mean should be closer to GT
    than the uniform mean."""
    true = _signal(0)
    y = forward_intensity(true, oversample=2)
    good1 = true + 0.02 * torch.randn_like(true.real).to(torch.complex64)
    good2 = true + 0.02 * torch.randn_like(true.real).to(torch.complex64)
    bad = 0.3 * torch.randn_like(true.real).to(torch.complex64)  # unrelated noise
    stack = torch.cat([good1, good2, bad], dim=0)
    uni = ensemble_aggregate(stack, y_intensity=y, anchor_method="best_fe",
                             weight_method="uniform")
    sof = ensemble_aggregate(stack, y_intensity=y, anchor_method="best_fe",
                             weight_method="softmax", temperature=0.1)
    p_uni = psnr(uni.mean_complex, true)
    p_sof = psnr(sof.mean_complex, true)
    assert p_sof > p_uni, f"softmax {p_sof:.2f} should beat uniform {p_uni:.2f}"


def test_k1_returns_zero_std():
    stack = _signal(0)  # (1,1,H,W)
    y = forward_intensity(stack, oversample=2)
    stats = ensemble_aggregate(stack, y_intensity=y)
    assert stats.n_members == 1
    assert float(stats.std_pixelwise.abs().max()) == 0.0
