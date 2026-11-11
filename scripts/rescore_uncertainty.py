#!/usr/bin/env python
"""Recompute the pixel-wise uncertainty metrics of a finished K>1 run in the ground-truth frame.

Why. Up to uncertainty-evaluation version 1, `score_reconstruction` compared the fused
uncertainty map u, which is in the frame of the reconstruction, with the error map, which is
in the frame of the ground truth, without moving u into the ground-truth frame. Pixel-wise
Pearson, Spearman and ECE were therefore computed on misregistered maps whenever the
reconstruction was shifted or flipped. Version 2 applies the same flip and shift to u first.

What this script does, per completed image:
  * loads the stored members (members.npz); nothing is retrained;
  * recomputes all aggregation methods with the frozen code (anchor, alignment, T);
  * scores them with the corrected `score_reconstruction`;
  * checks that PSNR, SSIM, FE and the image-level failure score agree with metrics.json
    (these do not depend on the correction), and records the largest deviation;
  * writes per_image/<id>/uncertainty_v2.json (corrected values plus the old ones) and
    per_image/<id>/aggregates_v2.npz (reconstructions, u and residual in the ground-truth frame).

metrics.json and aggregates.npz are never modified. generate_report.py uses
uncertainty_v2.json when it is present.

    python scripts/rescore_uncertainty.py --run-dir results/stage2_mnist_heldout
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.metrics.alignment import align_ensemble_to_anchor  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors, select_anchor  # noqa: E402

UNC_KEYS = ("unc_pearson", "unc_spearman", "unc_ece")
CHECK_KEYS = ("psnr", "ssim", "fe", "failure_score")
# Recomputing on another device or another PyTorch build gives tiny float differences.
TOLERANCE = {"psnr": 0.05, "ssim": 1e-3, "fe": 1e-4, "failure_score": 1e-3}


def _runner():
    spec = importlib.util.spec_from_file_location("runner_rescore", REPO / "scripts" / "run_phase_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _complex(npz, key):
    return torch.from_numpy(npz[f"{key}__real"] + 1j * npz[f"{key}__imag"]).to(torch.complex64)


def _real(npz, key):
    return torch.from_numpy(np.asarray(npz[key])).float()


def rescore_image(img_dir: Path, R, temperature: float, oversample: int, n_bins: int) -> dict:
    metrics = json.loads((img_dir / "metrics.json").read_text(encoding="utf-8"))
    b = np.load(img_dir / "members.npz")
    stack = _complex(b, "stack_raw")
    y = _real(b, "measurement")
    gt = _real(b, "gt").to(torch.complex64)

    fe = member_fourier_errors(stack, y, oversample=oversample)
    anchor = select_anchor(stack, method="best_fe", y_intensity=y, oversample=oversample)
    aligned = align_ensemble_to_anchor(stack, anchor_index=anchor)
    strategies, best_idx = R.compute_strategies(stack, y, fe, anchor, aligned, temperature, oversample)

    out_methods, old_methods, max_dev, arrays = {}, {}, {k: 0.0 for k in CHECK_KEYS}, {}
    for name, (mean_c, std) in strategies.items():
        rec, aligned_recon, u, r_phys, _ = R.score_reconstruction(mean_c, std, gt, y, oversample, n_bins)
        stored = metrics["methods"].get(name, {})
        for k in CHECK_KEYS:
            if k in rec and k in stored and stored[k] is not None:
                max_dev[k] = max(max_dev[k], abs(float(rec[k]) - float(stored[k])))
        arrays[f"recon_{name}"] = aligned_recon
        if u is not None:
            out_methods[name] = {k: float(rec[k]) for k in UNC_KEYS}
            old_methods[name] = {k: stored.get(k) for k in UNC_KEYS}
            arrays[f"unc_{name}"] = u
            arrays[f"rphys_{name}"] = r_phys

    consistent = all(max_dev[k] <= TOLERANCE[k] for k in CHECK_KEYS)
    record = {
        "uncertainty_eval_version": R.UNCERTAINTY_EVAL_VERSION,
        "image_id": metrics.get("image_id"),
        "mnist_test_index": metrics.get("mnist_test_index"),
        "methods": out_methods,
        "previous_values": old_methods,
        "previous_version": metrics.get("uncertainty_eval_version", 1),
        "consistency_max_abs_dev": max_dev,
        "consistent_with_metrics_json": consistent,
        "anchor_index_recomputed": int(anchor),
        "anchor_index_stored": metrics.get("anchor_index"),
        "best_single_index_recomputed": int(best_idx),
        "best_single_index_stored": metrics.get("best_single_index"),
    }
    (img_dir / "uncertainty_v2.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    R._save_npz(img_dir / "aggregates_v2.npz", **arrays)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    temperature = float(cfg["uncertainty"].get("fidelity_temperature", 0.05))
    oversample = int(cfg["measurement"].get("oversample", 2))
    n_bins = int(cfg["uncertainty"].get("n_bins_ece", 15))
    R = _runner()

    done, skipped, inconsistent = 0, 0, []
    for mj in sorted((run_dir / "per_image").glob("image_*/metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        if m.get("status") != "ok" or int(m.get("K", 1)) < 2 or not (mj.parent / "members.npz").exists():
            skipped += 1
            continue
        rec = rescore_image(mj.parent, R, temperature, oversample, n_bins)
        done += 1
        if not rec["consistent_with_metrics_json"]:
            inconsistent.append((rec["image_id"], rec["consistency_max_abs_dev"]))

    summary = {"run_dir": str(run_dir), "rescored": done, "skipped": skipped,
               "temperature": temperature, "oversample": oversample,
               "inconsistent_images": inconsistent}
    (run_dir / "uncertainty_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"rescored {done} images, skipped {skipped}, inconsistent {len(inconsistent)}")
    for img_id, dev in inconsistent:
        print(f"  INCONSISTENT {img_id}: {dev}")
    return 1 if inconsistent else 0


if __name__ == "__main__":
    raise SystemExit(main())
