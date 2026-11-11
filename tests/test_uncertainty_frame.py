"""The pixel-wise uncertainty metrics must compare u and the error in the same (ground-truth) frame.

Run with:  pytest tests/test_uncertainty_frame.py -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.metrics.alignment import align_to_gt, apply_variant, cyclic_shift, map_to_gt_frame  # noqa: E402
from pr_uq.physics.forward import forward_intensity  # noqa: E402

SPLITS = REPO / "splits"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tuf", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _digit(n=16):
    x = torch.zeros(1, 1, n, n)
    x[..., 3:13, 6:9] = 1.0          # vertical stroke
    x[..., 3:6, 6:12] = 0.8          # top bar, breaks the flip symmetry
    return x


@pytest.mark.parametrize("variant", [0, 1, 2, 3])
def test_map_to_gt_frame_undoes_flip_and_shift(variant):
    gt = _digit().to(torch.complex64)
    recon = cyclic_shift(apply_variant(gt, variant), 2, -3)          # same object in another frame
    res = align_to_gt(recon, gt)
    moved = map_to_gt_frame(recon.abs(), res)                          # a real map in the recon frame
    assert torch.allclose(moved, gt.abs(), atol=1e-5)


def test_score_reconstruction_compares_u_and_error_in_one_frame():
    R = _load("run_phase_retrieval")
    gt = _digit().to(torch.complex64)
    y = forward_intensity(gt, oversample=2)
    blob = torch.zeros(1, 1, 16, 16); blob[..., 9:12, 7:9] = 0.6       # local error on the stroke
    recon_gt_frame = gt + blob
    recon = cyclic_shift(apply_variant(recon_gt_frame, 3), 3, 2)        # twin + shift
    std = cyclic_shift(apply_variant(blob.to(torch.complex64), 3), 3, 2).real  # disagreement where the error is
    rec, aligned, u, r_phys, err = R.score_reconstruction(recon, std, gt, y, 2)
    assert u.shape == err.shape == (1, 1, 16, 16)
    # u is returned in the ground-truth frame, so its peak region coincides with the error
    assert rec["unc_pearson"] > 0.5
    # the misregistered comparison (u left in the recon frame) is much weaker
    from pr_uq.uncertainty.calibration import calibrate
    res = align_to_gt(recon, gt)
    assert res.variant != 0 or res.shift != (0, 0)
    # invert map_to_gt_frame (flips are involutions): u back in the frame of the reconstruction
    u_recon_frame = apply_variant(cyclic_shift(u.to(torch.complex64), res.shift[0], res.shift[1]), res.variant).real
    assert calibrate(u_recon_frame, err).pearson_r < rec["unc_pearson"] - 0.3
    assert R.UNCERTAINTY_EVAL_VERSION == 2


def test_rescore_reproduces_stored_metrics_and_writes_v2(tmp_path):
    R = _load("run_phase_retrieval")
    S = _load("rescore_uncertainty")
    from pr_uq.data.datasets import build_dataset
    from pr_uq.utils.config import load_config, save_config
    from pr_uq.utils.logging import get_logger
    cfg = load_config(REPO / "configs" / "stage2_mnist_heldout.yaml")
    cfg["dataset"].update({"protocol_split": "smoke", "n_images": 1, "root": str(REPO / "data"),
                           "splits_dir": str(SPLITS)})
    cfg["trainer"].update({"iterations": 20, "log_every": 10, "support_update_every": 10})
    cfg["model"].update({"z_channels": 16, "base_ch": 16})
    cfg["ensemble"]["K"] = 3
    sample = build_dataset(cfg["dataset"])[0]
    r = R.process_image(sample, 0, cfg, "cpu", tmp_path, get_logger("tuf", output_dir=tmp_path / "logs"))
    assert r["status"] == "ok", r.get("error")
    assert r["uncertainty_eval_version"] == 2
    save_config(cfg, tmp_path / "config.yaml")
    img_dir = next((tmp_path / "per_image").iterdir())
    metrics_before = (img_dir / "metrics.json").read_text(encoding="utf-8")

    rec = S.rescore_image(img_dir, R, 0.05, 2, 15)
    assert rec["consistent_with_metrics_json"], rec["consistency_max_abs_dev"]
    assert rec["anchor_index_recomputed"] == rec["anchor_index_stored"]
    stored = json.loads(metrics_before)["methods"]
    for m, vals in rec["methods"].items():         # this run was already scored with version 2
        for k, v in vals.items():
            assert v == pytest.approx(stored[m][k], abs=1e-4)
    assert (img_dir / "uncertainty_v2.json").exists() and (img_dir / "aggregates_v2.npz").exists()
    assert (img_dir / "metrics.json").read_text(encoding="utf-8") == metrics_before   # never modified
    ag = np.load(img_dir / "aggregates_v2.npz")
    assert "unc_fe_softmax" in ag.files and "recon_fe_softmax__real" in ag.files


def test_report_prefers_uncertainty_v2(tmp_path):
    rep = _load("generate_report")
    d = tmp_path / "per_image" / "image_00001"; d.mkdir(parents=True)
    row = {"status": "ok", "K": 5, "methods": {"fe_softmax": {"psnr": 20.0, "unc_spearman": 0.1}}}
    (d / "metrics.json").write_text(json.dumps(row), encoding="utf-8")
    (d / "uncertainty_v2.json").write_text(json.dumps(
        {"uncertainty_eval_version": 2, "methods": {"fe_softmax": {"unc_spearman": 0.6}}}), encoding="utf-8")
    rows, _ = rep.load_run(tmp_path)
    assert rows[0]["methods"]["fe_softmax"]["unc_spearman"] == 0.6
    assert rows[0]["methods"]["fe_softmax"]["psnr"] == 20.0
    assert rows[0]["_unc_eval_version"] == 2
