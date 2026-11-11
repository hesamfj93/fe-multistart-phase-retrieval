"""Evaluation-protocol tests: split integrity, dev exclusion, prefix stability,
report partitioning, and the frozen-design fingerprint.

Run with:  pytest tests/test_protocol.py -v
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.data.splits import (  # noqa: E402
    DEV_INDICES, generate_split_lists, load_protocol_indices, read_split_csv, split_path,
)

SPLITS = REPO / "splits"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- split files

def test_split_files_match_deterministic_regeneration():
    expected = generate_split_lists()
    for role, idx in expected.items():
        assert read_split_csv(split_path(SPLITS, role)) == idx, role


def test_dev_file_is_exactly_the_eight_development_images():
    assert read_split_csv(split_path(SPLITS, "dev")) == list(DEV_INDICES)
    assert list(DEV_INDICES) == [429, 1729, 1917, 2890, 4919, 6044, 7532, 9399]


@pytest.mark.parametrize("n", [50, 100, 200])
def test_heldout_has_zero_overlap_with_dev(n):
    h = load_protocol_indices(SPLITS, "heldout", n)
    assert len(h) == n == len(set(h))
    assert not set(h) & set(DEV_INDICES)


def test_smoke_disjoint_from_dev_and_full_heldout_list():
    smoke = read_split_csv(split_path(SPLITS, "smoke"))
    heldout = read_split_csv(split_path(SPLITS, "heldout"))
    assert not set(smoke) & set(DEV_INDICES)
    assert not set(smoke) & set(heldout)


def test_extension_50_to_100_keeps_first_50():
    assert load_protocol_indices(SPLITS, "heldout", 100)[:50] == load_protocol_indices(SPLITS, "heldout", 50)


def test_requesting_more_rows_than_file_raises():
    with pytest.raises(ValueError):
        load_protocol_indices(SPLITS, "heldout", 201)


def test_dev_index_injected_into_heldout_file_is_rejected(tmp_path):
    shutil.copytree(SPLITS, tmp_path / "splits")
    p = split_path(tmp_path / "splits", "heldout")
    lines = p.read_text(encoding="utf-8").splitlines()
    # replace the index in rank 3 with a development index
    parts = lines[5].split(","); parts[1] = str(DEV_INDICES[0]); lines[5] = ",".join(parts)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION"):
        load_protocol_indices(tmp_path / "splits", "heldout", 50)


def test_protocol_dataset_returns_requested_images_in_order():
    from pr_uq.data.datasets import build_dataset
    spec = {"name": "mnist", "protocol_split": "heldout", "n_images": 4, "size": 32,
            "root": str(REPO / "data"), "splits_dir": str(SPLITS)}
    names = [s.name for s in build_dataset(spec)]
    assert names == [f"mnist_{i:06d}" for i in load_protocol_indices(SPLITS, "heldout", 4)]


# ---------------------------------------------------------------- report partition

def _fake_image(run, idx, rank, split):
    d = run / "per_image" / f"image_{idx:05d}"
    d.mkdir(parents=True)
    methods = {m: {"psnr": 20.0 + rank, "ssim": 0.8, "fe": 0.2, "unc_pearson": 0.3,
                   "unc_spearman": 0.3, "unc_ece": 0.1, "failure_score": 0.2}
               for m in ["k1", "uniform", "median", "best_single_by_fe", "fe_softmax"]}
    (d / "metrics.json").write_text(json.dumps({
        "status": "ok", "image_id": f"image_{idx:05d}", "name": f"mnist_{idx:06d}",
        "mnist_test_index": idx, "protocol_split": split, "split_rank": rank, "K": 5,
        "anchor_index": 0, "best_single_index": 0, "effective_sample_size": 3.0,
        "member_fe": [0.1, 0.2, 0.3, 0.4, 0.5], "member_psnr": [25, 24, 23, 22, 21],
        "runtime_per_member_s": [1, 1, 1, 1, 1], "runtime_total_s": 5, "methods": methods}), encoding="utf-8")


def test_report_partitions_by_index_and_never_mixes_dev_into_primary(tmp_path):
    rep = _load_script("generate_report")
    h = load_protocol_indices(SPLITS, "heldout", 3)
    run = tmp_path / "run"
    for r, i in enumerate(h):
        _fake_image(run, i, r, "heldout")
    _fake_image(run, DEV_INDICES[0], 0, "heldout")          # mislabelled on purpose
    rows, failed = rep.load_run(run)
    sec = rep.partition(rows, rep.split_lists(SPLITS))
    assert [r["mnist_test_index"] for r in sec["heldout"]] == h
    assert [r["mnist_test_index"] for r in sec["development"]] == [DEV_INDICES[0]]
    sec2 = rep.partition(rows, rep.split_lists(SPLITS), n_heldout=2)
    assert [r["mnist_test_index"] for r in sec2["heldout"]] == h[:2]
    assert [r["mnist_test_index"] for r in sec2["heldout_beyond_n"]] == h[2:]


def test_report_end_to_end_writes_primary_table_only_from_heldout(tmp_path, monkeypatch):
    rep = _load_script("generate_report")
    h = load_protocol_indices(SPLITS, "heldout", 6)
    run = tmp_path / "run"
    for r, i in enumerate(h):
        _fake_image(run, i, r, "heldout")
    _fake_image(run, DEV_INDICES[1], 0, "dev")
    monkeypatch.setattr(sys, "argv", ["x", "--run-dir", str(run), "--splits-dir", str(SPLITS), "--no-figures"])
    assert rep.main() == 0
    summary = json.loads((run / "report_summary.json").read_text(encoding="utf-8"))
    assert summary["sections"]["heldout"]["mnist_test_indices"] == h
    assert summary["sections"]["development"]["mnist_test_indices"] == [DEV_INDICES[1]]
    assert (run / "tables" / "heldout" / "primary_table.csv").exists()
    assert not (run / "tables" / "development" / "primary_table.csv").exists()


# ---------------------------------------------------------------- frozen design

def test_fingerprint_ignores_session_keys_but_detects_design_changes():
    runner = _load_script("run_phase_retrieval")
    from pr_uq.utils.config import load_config
    cfg = load_config(REPO / "configs" / "stage2_mnist_heldout.yaml")
    fp = runner._fingerprint(cfg)
    c2 = copy.deepcopy(cfg)
    c2["dataset"]["n_images"] = 100; c2["experiment"]["device"] = "cuda"
    c2["experiment"]["output_dir"] = "elsewhere"
    assert runner._fingerprint(c2) == fp                     # extension / device / location allowed
    for sec, key, val in [("uncertainty", "fidelity_temperature", 0.1), ("trainer", "iterations", 2000),
                          ("ensemble", "K", 3), ("model", "z_channels", 64),
                          ("evaluation", "failure_psnr_db", 12.0)]:
        c3 = copy.deepcopy(cfg); c3[sec][key] = val
        assert runner._fingerprint(c3) != fp, f"{sec}.{key} change not detected"


def test_stage2_config_values_are_the_frozen_design():
    from pr_uq.utils.config import load_config
    cfg = load_config(REPO / "configs" / "stage2_mnist_heldout.yaml")
    assert cfg["dataset"]["protocol_split"] == "heldout" and cfg["dataset"]["n_images"] == 50
    assert cfg["model"]["z_channels"] == 128 and cfg["trainer"]["iterations"] == 3000
    assert cfg["ensemble"]["K"] == 5 and cfg["uncertainty"]["fidelity_temperature"] == 0.05


def test_reports_for_different_heldout_sizes_do_not_overwrite_each_other(tmp_path, monkeypatch):
    rep = _load_script("generate_report")
    h = load_protocol_indices(SPLITS, "heldout", 6)
    run = tmp_path / "run"
    for r, i in enumerate(h):
        _fake_image(run, i, r, "heldout")
    for n in (3, 6):
        monkeypatch.setattr(sys, "argv", ["x", "--run-dir", str(run), "--splits-dir", str(SPLITS),
                                          "--n-heldout", str(n), "--no-figures"])
        assert rep.main() == 0
    for n in (3, 6):
        s = json.loads((run / f"report_n{n}" / "report_summary.json").read_text(encoding="utf-8"))
        assert s["sections"]["heldout"]["mnist_test_indices"] == h[:n]
        rows = (run / f"report_n{n}" / "tables" / "heldout" / "primary_table.csv").read_text(encoding="utf-8").splitlines()
        assert rows[1].split(",")[1] == str(n)          # n column of the K=1 row
