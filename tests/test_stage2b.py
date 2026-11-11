"""Tests for the pre-registered Stage-2b additions: full-grid HIO/RAAR baselines, the
compute-matched single-start control, extra-run merging, the PSNR cap, Holm adjustment and
the confirmatory image-level test.

Run with:  pytest tests/test_stage2b.py -v
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.baselines.oversampled import (  # noqa: E402
    _crop, _proj_object, object_window, run_hio_full, run_raar_full,
)
from pr_uq.data.splits import DEV_INDICES, load_protocol_indices  # noqa: E402
from pr_uq.physics.forward import forward_field, forward_intensity, forward_magnitude  # noqa: E402

SPLITS = REPO / "splits"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_t2b", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy(device="cpu"):
    g = torch.Generator().manual_seed(1)
    x = torch.zeros(1, 1, 16, 16)
    x[..., 5:11, 4:12] = torch.rand(1, 1, 6, 8, generator=g)
    return x.to(device=device, dtype=torch.complex64)


def _fe(rec, y):
    m = y.clamp_min(0).sqrt()
    return float((forward_magnitude(rec, 2) - m).norm() / m.norm())


# ---------------------------------------------------------------- classical baselines

def test_full_grid_transform_matches_forward_model():
    x = torch.rand(1, 1, 16, 16).to(torch.complex64)
    w = object_window((16, 16), 2, "cpu")
    z = torch.zeros(1, 1, 32, 32, dtype=torch.complex64)
    z[w] = x.flatten()
    assert torch.allclose(forward_field(z, 1), forward_field(x, 2), atol=1e-5)
    assert torch.equal(_crop(z, (16, 16)), x)


def test_object_projection_variants():
    w = object_window((4, 4), 2, "cpu")
    z = (torch.randn(1, 1, 8, 8) + 1j * torch.randn(1, 1, 8, 8)).to(torch.complex64)
    rp = _proj_object(z, w, "real_nonneg")
    so = _proj_object(z, w, "support_only")
    assert torch.all(rp[~w] == 0) and torch.all(so[~w] == 0)
    assert torch.all(rp.imag == 0) and torch.all(rp.real >= 0)
    assert torch.equal(so[w], z[w])                      # complex values kept inside the window
    with pytest.raises(ValueError):
        _proj_object(z, w, "nonsense")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", [run_hio_full, run_raar_full])
@pytest.mark.parametrize("constraint", ["real_nonneg", "support_only"])
def test_classical_algorithms_reduce_fourier_error(device, fn, constraint):
    x = _toy(device)
    y = forward_intensity(x, oversample=2)
    res = fn(y, (16, 16), iters=300, er_iters=20, seed=0, device=device, constraint=constraint)
    assert res.recon.shape == (1, 1, 16, 16) and res.recon.dtype == torch.complex64
    assert res.recon.device.type == device
    init_fe = 1.0                                        # a random-phase start has FE of order 1
    assert _fe(res.recon, y) < 0.5 * init_fe
    again = fn(y, (16, 16), iters=300, er_iters=20, seed=0, device=device, constraint=constraint)
    assert torch.allclose(res.recon, again.recon)        # deterministic for a fixed seed


def test_baseline_runner_on_one_image_with_resume(tmp_path):
    bl = _load("run_baselines")
    from pr_uq.data.datasets import build_dataset
    from pr_uq.utils.config import load_config
    from pr_uq.utils.logging import get_logger
    cfg = load_config(REPO / "configs" / "stage2_mnist_heldout_baselines.yaml")
    cfg["dataset"].update({"protocol_split": "smoke", "n_images": 1, "root": str(REPO / "data"),
                           "splits_dir": str(SPLITS)})
    cfg["baselines"].update({"iterations": 40, "er_iterations": 5, "starts": 2})
    sample = build_dataset(cfg["dataset"])[0]
    logger = get_logger("t2b_baselines", output_dir=tmp_path / "logs")
    r = bl.process_image(sample, 0, cfg, "cpu", tmp_path, logger)
    assert r["status"] == "ok", r.get("error")
    assert set(r["methods"]) == set(bl.method_names(cfg))
    assert set(bl.method_names(cfg)) == {f"{a}_{p}_{s}" for a in ("hio", "raar")
                                         for p in ("supp", "realpos") for s in ("s1", "best2")}
    for v in r["methods"].values():
        assert all(k in v for k in ["psnr", "ssim", "fe", "runtime_s"])
    for tag, info in r["starts"].items():
        assert info["best_index"] == int(np.argmin(info["start_fe"]))
    assert bl.process_image(sample, 0, cfg, "cpu", tmp_path, logger).get("_skipped") is True


# ---------------------------------------------------------------- compute-matched control

def test_k1long_config_differs_only_in_intended_keys():
    from pr_uq.utils.config import load_config
    a = load_config(REPO / "configs" / "stage2_mnist_heldout.yaml")
    b = load_config(REPO / "configs" / "stage2_mnist_heldout_k1long.yaml")
    assert b["ensemble"]["K"] == 1 and b["trainer"]["iterations"] == 5 * a["trainer"]["iterations"] == 15000
    assert b["experiment"]["single_method_name"] == "k1_long" and b["dataset"]["n_images"] == 100
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    for c in (a2, b2):
        c["ensemble"].pop("K"); c["trainer"].pop("iterations"); c["dataset"].pop("n_images")
        c["experiment"] = {k: v for k, v in c["experiment"].items()
                           if k not in ("name", "output_dir", "single_method_name")}
    assert a2 == b2                                       # nothing else differs


def test_single_start_run_uses_its_own_method_name(tmp_path):
    runner = _load("run_phase_retrieval")
    from pr_uq.data.datasets import build_dataset
    from pr_uq.utils.config import load_config
    from pr_uq.utils.logging import get_logger
    cfg = load_config(REPO / "configs" / "stage2_mnist_heldout_k1long.yaml")
    cfg["dataset"].update({"protocol_split": "smoke", "n_images": 1, "root": str(REPO / "data"),
                           "splits_dir": str(SPLITS)})
    cfg["trainer"].update({"iterations": 20, "log_every": 10, "support_update_every": 10})
    cfg["model"].update({"z_channels": 64, "base_ch": 64})
    sample = build_dataset(cfg["dataset"])[0]
    r = runner.process_image(sample, 0, cfg, "cpu", tmp_path, get_logger("t2b_k1", output_dir=tmp_path / "logs"))
    assert r["status"] == "ok", r.get("error")
    assert list(r["methods"]) == ["k1_long"] and r["K"] == 1


# ---------------------------------------------------------------- report additions

def _row(idx, rank, psnr, fscore=0.2):
    methods = {m: {"psnr": psnr, "ssim": 0.8, "fe": 0.2, "unc_pearson": 0.3, "unc_spearman": 0.3,
                   "unc_ece": 0.1, "failure_score": fscore}
               for m in ["k1", "uniform", "median", "best_single_by_fe", "fe_softmax"]}
    return {"status": "ok", "image_id": f"image_{idx:05d}", "name": f"mnist_{idx:06d}",
            "mnist_test_index": idx, "_heldout_rank": rank, "K": 5, "member_fe": [0.1, 0.2],
            "member_psnr": [psnr, psnr - 1], "runtime_per_member_s": [1.0] * 5,
            "effective_sample_size": 3.0, "methods": methods}


def _write_run(dirpath, rows):
    for r in rows:
        d = dirpath / "per_image" / r["image_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}),
                                        encoding="utf-8")


def test_holm_adjustment_known_values():
    rep = _load("generate_report")
    adj = rep.holm([0.01, 0.04, None, 0.03])
    assert adj[2] is None
    assert adj[0] == pytest.approx(0.03) and adj[3] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)


def test_psnr_cap_applies_to_summaries_not_per_image(tmp_path):
    rep = _load("generate_report")
    rows = [_row(i, k, 20.0) for k, i in enumerate(load_protocol_indices(SPLITS, "heldout", 4))]
    rows[0]["methods"]["fe_softmax"]["psnr"] = 140.0
    res = rep.analyse(rows, 15.0, 5, rep.METHODS, rep.PAIRS)
    assert res["methods"]["fe_softmax"]["psnr"]["max"] == rep.PSNR_CAP_DB
    assert res["failure"]["fe_softmax"]["n_exact"] == 1
    rep.write_tables(rows, res, tmp_path, primary=True)
    assert "140.0000" in (tmp_path / "per_image.csv").read_text(encoding="utf-8")


def test_extra_runs_merge_by_index_and_names_cannot_collide(tmp_path):
    rep = _load("generate_report")
    h = load_protocol_indices(SPLITS, "heldout", 3)
    main = [_row(i, k, 20.0 + k) for k, i in enumerate(h)]
    ex = tmp_path / "ex"
    _write_run(ex, [{"status": "ok", "image_id": f"image_{i:05d}", "name": f"mnist_{i:06d}", "mnist_test_index": i,
                     "device_used": "cuda", "runtime_per_member_s": [5.0],
                     "methods": {"k1_long": {"psnr": 19.0, "ssim": 0.7, "fe": 0.3}}} for i in h[:2]])
    extras, recs = rep.merge_extra_runs(main, [str(ex)])
    assert extras == ["k1_long"] and recs[0]["merged"] == 2
    assert "k1_long" in main[0]["methods"] and "k1_long" not in main[2]["methods"]
    assert main[0]["methods"]["k1_long"]["runtime_s"] == 5.0
    pairs = rep.PAIRS + [(t, "k1_long") for t in rep.EXTRA_PAIR_TARGETS]
    res = rep.analyse(main, 15.0, 5, rep.METHODS + extras, pairs)
    e = [p for p in res["paired"] if p["baseline"] == "k1_long" and p["metric"] == "psnr"][0]
    assert e["n"] == 2                                    # only images that have both methods
    bad = tmp_path / "bad"
    _write_run(bad, [{"status": "ok", "image_id": f"image_{h[0]:05d}", "name": "x", "mnist_test_index": h[0],
                      "methods": {"fe_softmax": {"psnr": 1, "ssim": 0, "fe": 1}}}])
    with pytest.raises(SystemExit):
        rep.merge_extra_runs([_row(h[0], 0, 20.0)], [str(bad)])


def test_confirmatory_test_uses_only_ranks_50_to_99():
    rep = _load("generate_report")
    h = load_protocol_indices(SPLITS, "heldout", 100)
    assert not set(h) & set(DEV_INDICES)
    rows = []
    for rank, i in enumerate(h):
        psnr = 10.0 + (rank % 50) * 0.3
        if rank < 50:
            fscore = psnr            # wrong direction on ranks 0-49: must be ignored
        else:
            fscore = -psnr           # perfectly right direction on ranks 50-99
        rows.append(_row(i, rank, psnr, fscore))
    out = rep.confirmatory_test(rows)
    assert out["status"] == "computed" and out["n_available"] == 50
    assert out["results"]["fe_softmax"]["rho"] == pytest.approx(1.0)
    assert out["primary_pass"] is True
    partial = rep.confirmatory_test(rows[:80])
    assert partial["status"].startswith("not computed") and partial["n_available"] == 30
