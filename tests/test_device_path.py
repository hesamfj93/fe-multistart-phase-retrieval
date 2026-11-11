"""Device-path tests: the full per-image pipeline on CPU and, when available, on CUDA.

A CPU-only smoke test cannot catch device mismatches, because every tensor is on the
CPU. These tests run training, alignment, all five methods, uncertainty, metrics,
saving and resume on each available device, with a few iterations so they take
seconds. The CUDA cases are skipped automatically on machines without a GPU.

Run with:  pytest tests/test_device_path.py -v
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
METHODS = ["k1", "uniform", "median", "best_single_by_fe", "fe_softmax"]


def _runner():
    spec = importlib.util.spec_from_file_location("rpr_device_test", REPO / "scripts" / "run_phase_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_stack(device: str, k: int = 4):
    from pr_uq.physics.forward import forward_intensity
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(1, 1, 16, 16)
    x[..., 4:12, 4:12] = torch.rand(1, 1, 8, 8, generator=g)
    x = x.to(device=device, dtype=torch.complex64)
    y = forward_intensity(x, oversample=2)
    noise = [0.05 * torch.randn(1, 1, 16, 16, generator=g) for _ in range(k)]
    stack = torch.cat([x + n.to(device=device, dtype=torch.complex64) for n in noise], dim=0)
    return stack, y


@pytest.mark.parametrize("device", DEVICES)
def test_aggregation_helpers_run_on_device(device):
    from pr_uq.uncertainty.ensemble_uq import ensemble_aggregate, fidelity_weights, member_fourier_errors
    stack, y = _toy_stack(device)
    fe = member_fourier_errors(stack, y, oversample=2)
    for method in ["uniform", "softmax", "inverse", "top_k"]:
        w = fidelity_weights(fe, method=method, top_k=2)
        assert w.device == fe.device and abs(float(w.sum()) - 1.0) < 1e-5
    for anchor in ["first", "best_fe", "medoid"]:
        for weight in ["uniform", "softmax", "inverse", "top_k"]:
            st = ensemble_aggregate(stack, y_intensity=y, anchor_method=anchor,
                                    weight_method=weight, top_k=2)
            assert st.mean_complex.device.type == device
            assert torch.isfinite(st.std_pixelwise).all()


@pytest.mark.parametrize("device", DEVICES)
def test_runner_strategies_run_on_device(device):
    runner = _runner()
    from pr_uq.metrics.alignment import align_ensemble_to_anchor
    from pr_uq.uncertainty.ensemble_uq import member_fourier_errors, select_anchor
    stack, y = _toy_stack(device, k=5)
    gt = stack[0:1].clone()
    fe = member_fourier_errors(stack, y, oversample=2)
    anchor = select_anchor(stack, method="best_fe", y_intensity=y, oversample=2)
    aligned = align_ensemble_to_anchor(stack, anchor_index=anchor)
    strategies, best = runner.compute_strategies(stack, y, fe, anchor, aligned, 0.05, 2)
    assert set(strategies) == set(METHODS)
    for name, (mean_c, std) in strategies.items():
        assert mean_c.device.type == device, name
        rec, *_ = runner.score_reconstruction(mean_c, std, gt, y, 2)
        assert all(math.isfinite(rec[k]) for k in ["psnr", "ssim", "fe"]), name


@pytest.mark.parametrize("device", DEVICES)
def test_full_image_pipeline_and_resume_on_device(device, tmp_path):
    """Train K=5 members for one smoke image (few iterations), then everything after."""
    runner = _runner()
    from pr_uq.data.datasets import build_dataset
    from pr_uq.utils.config import load_config
    from pr_uq.utils.logging import get_logger

    cfg = load_config(REPO / "configs" / "smoke_test.yaml")
    cfg["dataset"].update({"n_images": 1, "root": str(REPO / "data"), "splits_dir": str(REPO / "splits")})
    cfg["trainer"].update({"iterations": 30, "log_every": 10, "support_update_every": 10})
    cfg["experiment"]["output_dir"] = str(tmp_path)
    sample = build_dataset(cfg["dataset"])[0]
    logger = get_logger(f"device_test_{device}", output_dir=tmp_path / "logs")

    res = runner.process_image(sample, 0, cfg, device, tmp_path, logger)
    assert res["status"] == "ok", f"pipeline failed on {device}: {res.get('error')}\n{res.get('traceback')}"
    assert res["device_used"] == device
    assert set(res["methods"]) == set(METHODS)
    for m in METHODS:
        assert all(math.isfinite(res["methods"][m][k]) for k in ["psnr", "ssim", "fe"])
    for m in METHODS[1:]:
        assert all(k in res["methods"][m] for k in ["unc_pearson", "unc_spearman", "unc_ece"])
    assert abs(sum(res["softmax_weights"]) - 1.0) < 1e-5
    assert 1.0 <= res["effective_sample_size"] <= 5.0

    img_dir = tmp_path / "per_image" / res["image_id"]
    m = np.load(img_dir / "members.npz")
    a = np.load(img_dir / "aggregates.npz")
    assert m["stack_raw__real"].shape[0] == 5 and m["stack_aligned__real"].shape[0] == 5
    assert all(f"recon_{k}__real" in a.files for k in METHODS)
    assert json.loads((img_dir / "metrics.json").read_text(encoding="utf-8"))["status"] == "ok"
    assert not (img_dir / "_members").exists()            # cache removed after success

    again = runner.process_image(sample, 0, cfg, device, tmp_path, logger)
    assert again.get("_skipped") is True                     # image-level resume


@pytest.mark.parametrize("device", DEVICES)
def test_failed_image_keeps_members_and_resumes_from_them(device, tmp_path, monkeypatch):
    """A crash after training must keep the trained members; the retry must reuse them."""
    runner = _runner()
    from pr_uq.data.datasets import build_dataset
    from pr_uq.utils.config import load_config
    from pr_uq.utils.logging import get_logger

    cfg = load_config(REPO / "configs" / "smoke_test.yaml")
    cfg["dataset"].update({"n_images": 1, "root": str(REPO / "data"), "splits_dir": str(REPO / "splits")})
    cfg["trainer"].update({"iterations": 20, "log_every": 10, "support_update_every": 10})
    sample = build_dataset(cfg["dataset"])[0]
    logger = get_logger(f"device_retry_{device}", output_dir=tmp_path / "logs")

    real = runner.compute_strategies
    def boom(*a, **k):
        raise RuntimeError("simulated post-training failure")
    monkeypatch.setattr(runner, "compute_strategies", boom)
    first = runner.process_image(sample, 0, cfg, device, tmp_path, logger)
    assert first["status"] == "failed"
    cache = tmp_path / "per_image" / first["image_id"] / "_members"
    assert sorted(p.name for p in cache.glob("member_*.pt")) == [f"member_{k:02d}.pt" for k in range(5)]

    cached_psnr = [float(torch.load(cache / f"member_{k:02d}.pt", map_location="cpu")["psnr"]) for k in range(5)]

    monkeypatch.setattr(runner, "compute_strategies", real)
    calls = {"n": 0}
    real_train = runner.train_member
    def counting_train(*a, **k):
        calls["n"] += 1
        return real_train(*a, **k)
    monkeypatch.setattr(runner, "train_member", counting_train)
    second = runner.process_image(sample, 0, cfg, device, tmp_path, logger)
    assert second["status"] == "ok", second.get("error")
    assert calls["n"] == 0                                    # no member was retrained
    assert second["member_psnr"] == pytest.approx(cached_psnr)  # results come from the cached members
    assert second["n_members_resumed"] == 5
    assert second["runtime_total_s"] >= sum(second["runtime_per_member_s"])  # training time not lost
