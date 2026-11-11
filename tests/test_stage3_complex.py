"""Stage 3 (complex-valued objects): object model, invariances, scoring, splits, baselines, configs."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pr_uq.data.complex_objects import object_from_cfg, smooth_phase, to_complex_object
from pr_uq.physics.forward import forward_intensity

REPO = Path(__file__).resolve().parent.parent


def _digit():
    img = torch.zeros(1, 1, 32, 32)
    img[..., 8:24, 12:15] = 1.0
    img[..., 8:11, 12:22] = 0.7
    return img


def _runner():
    spec = importlib.util.spec_from_file_location("runner_s3", REPO / "scripts" / "run_phase_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_complex_object_is_deterministic_and_bounded():
    img = _digit()
    a, b = to_complex_object(img, seed=7), to_complex_object(img, seed=7)
    c = to_complex_object(img, seed=8)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.allclose(a.abs(), img, atol=1e-6)               # amplitude is the digit
    assert torch.all(a[img == 0] == 0)                            # zero off the support
    phi = torch.angle(a)[img > 0.05]
    assert abs(float(phi.abs().max()) - math.pi / 2) < 1e-5       # phase range on the support


def test_object_from_cfg_real_is_identity():
    img = _digit()
    assert object_from_cfg(img, {}, 5) is img
    assert object_from_cfg(img, {"object": {"class": "real"}}, 5) is img
    with pytest.raises(ValueError):
        object_from_cfg(img, {"object": {"class": "nope"}}, 5)


def test_smooth_phase_is_smooth():
    amp = np.ones((32, 32))
    phi = smooth_phase(amp, seed=1, sigma_px=3.0)
    rough = smooth_phase(amp, seed=1, sigma_px=0.5)
    assert np.abs(np.diff(phi, axis=0)).mean() < np.abs(np.diff(rough, axis=0)).mean()


def test_intensity_invariant_to_trivial_ambiguities_for_complex_objects():
    x = to_complex_object(_digit(), seed=3)
    y = forward_intensity(x, oversample=2)
    twin = torch.flip(x.conj(), dims=(-2, -1))
    shifted = torch.roll(x, shifts=(2, -3), dims=(-2, -1))       # object stays inside the window
    for z in (x * torch.exp(torch.tensor(1j * 0.9)), twin, shifted):
        assert torch.allclose(forward_intensity(z, oversample=2), y, rtol=1e-4, atol=1e-5 * float(y.max()))


def test_complex_scoring_recovers_ambiguous_copies_and_penalizes_wrong_phase():
    R = _runner()
    x = to_complex_object(_digit(), seed=3)
    y = forward_intensity(x, oversample=2)
    std = torch.zeros(1, 1, 32, 32)
    twin = torch.flip(x.conj(), dims=(-2, -1)) * torch.exp(torch.tensor(1j * 0.4))
    rec, aligned, u, r, err = R.score_reconstruction(twin, std, x, y, 2, complex_object=True)
    assert rec["psnr"] >= 60 and rec["phase_err_rad"] < 1e-3 and rec["nrmse_c"] < 1e-3
    wrong = x.abs().to(torch.complex64)                          # right amplitude, zero phase
    rec2, *_ = R.score_reconstruction(wrong, std, x, y, 2, complex_object=True)
    assert rec2["psnr"] < 30 and rec2["psnr_amp"] >= 60 and rec2["phase_err_rad"] > 0.1


def test_rank_offset_selects_rows_100_to_199():
    from pr_uq.data.splits import load_protocol_indices, read_split_csv, split_path
    all_idx = read_split_csv(split_path(REPO / "splits", "heldout"))
    sel = load_protocol_indices(REPO / "splits", "heldout", 100, start=100)
    assert sel == all_idx[100:200]
    assert not set(sel) & set(load_protocol_indices(REPO / "splits", "heldout", 100))
    with pytest.raises(ValueError):
        load_protocol_indices(REPO / "splits", "heldout", 101, start=100)


def test_shrinkwrap_baseline_runs_and_bounds_amplitude():
    from pr_uq.baselines.oversampled import _shrinkwrap_support, object_window, run_hio_full, run_raar_full
    x = to_complex_object(_digit(), seed=3)
    y = forward_intensity(x, oversample=2)
    for fn in (run_hio_full, run_raar_full):
        r = fn(y, (32, 32), iters=60, er_iters=10, seed=0, constraint="shrinkwrap")
        assert r.recon.shape == (1, 1, 32, 32) and float(r.recon.abs().max()) <= 1.0 + 1e-5
    win = object_window((32, 32), 2, "cpu")
    z = torch.zeros(1, 1, 64, 64, dtype=torch.complex64)
    z[..., 28:36, 28:36] = 1
    s = _shrinkwrap_support(z, win, 1.5)
    assert bool((s & ~win).sum() == 0) and 0 < int(s.sum()) < int(win.sum())


@pytest.mark.parametrize("s2,s3,allowed", [
    ("stage2_mnist_heldout.yaml", "stage3_complex_heldout.yaml", set()),
    ("stage2_mnist_heldout_k1long.yaml", "stage3_complex_heldout_k1long.yaml", set()),
    ("stage2_mnist_heldout_baselines.yaml", "stage3_complex_heldout_baselines.yaml", {"constraints"}),
])
def test_stage3_design_equals_frozen_stage2_design(s2, s3, allowed):
    a = yaml.safe_load((REPO / "configs" / s2).read_text(encoding="utf-8"))
    b = yaml.safe_load((REPO / "configs" / s3).read_text(encoding="utf-8"))
    assert b["dataset"]["object"]["class"] == "complex" and b["dataset"]["rank_start"] == 100
    assert b["dataset"]["n_images"] == 100 and b["dataset"]["protocol_split"] == "heldout"
    for c in (a, b):
        c.pop("experiment")
        for k in ("object", "rank_start", "n_images"):
            c["dataset"].pop(k, None)
        for k in allowed:
            c.get("baselines", {}).pop(k, None)
    assert a == b
