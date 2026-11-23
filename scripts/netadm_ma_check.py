#!/usr/bin/env python
"""Check of our Net-ADM re-implementation in the MNIST setting of Ma et al. (arXiv:2210.12646v1).

Their MNIST experiment uses 6 grayscale 28 x 28 images, sampling rate r = m/n per dimension,
the network {25, 15, 10}, Adam, learning rate 0.005 decayed every 500 outer steps, 1000 outer
iterations with 5 inner steps, rho = 1, eps = 0.001, and averages PSNR over ten runs. The paper
does not name the 6 images, define the 2D padding or the PSNR, or say whether the sampling-rate
experiment has noise, and gives its PSNR values only in a figure. This script therefore fixes
its own choices in advance: the first 6 of our 8 development images (never used for held-out
evaluation) at their native 28 x 28 size, r = 2 per dimension (zero padding to 56 x 56, the
forward model of our studies), no noise, seeds 0 to 9 for each image, and Net-ADM with the
settings of our studies, which are the published ones. HIO with the real and nonnegative
constraint (5000 + 200 iterations, one start per seed) gives a classical reference.

PSNR: data range 1, after alignment to the ground truth for global phase, shift and twin, as in
our studies; "PSNR, no alignment" compares the raw real part, clipped to [0, 1], with the ground
truth and is reported only because Ma et al. remove the flip ambiguity by setting two border
pixels instead of aligning.

    python scripts/netadm_ma_check.py --out results/netadm_ma_check
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pr_uq.baselines.netadm import run_netadm  # noqa: E402
from pr_uq.baselines.oversampled import run_hio_full  # noqa: E402
from pr_uq.data.datasets import make_mnist_offline  # noqa: E402
from pr_uq.metrics.alignment import align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import psnr  # noqa: E402
from pr_uq.physics.forward import forward_intensity  # noqa: E402

N_IMAGES, SEEDS, OVERSAMPLE = 6, range(10), 2


def dev_indices() -> list[int]:
    rows = [r for r in open("splits/mnist_test_dev_indices.csv") if r[0].isdigit()]
    return [int(r.split(",")[1]) for r in rows][:N_IMAGES]


def scores(recon: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    aligned = align_to_gt(recon, gt.to(torch.complex64)).aligned_recon
    raw = recon.real.clamp(0, 1)
    return psnr(aligned.real, gt), psnr(raw, gt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    idx = dev_indices()
    samples = make_mnist_offline(size=28, root="./data", indices=idx)
    rows = []
    for mnist_index, s in zip(idx, samples):
        gt = s.image.float()
        y = forward_intensity(gt.to(torch.complex64), oversample=OVERSAMPLE)
        for seed in SEEDS:
            r = run_netadm(y, (28, 28), seed=seed, oversample=OVERSAMPLE)
            p, p_raw = scores(r.recon, gt)
            rows.append({"method": "Net-ADM", "mnist_index": mnist_index, "seed": seed, "psnr": p, "psnr_no_alignment": p_raw, "runtime_s": r.runtime_s})
            h = run_hio_full(y, (28, 28), seed=seed, oversample=OVERSAMPLE, constraint="real_nonneg")
            p, p_raw = scores(h.recon, gt)
            rows.append({"method": "HIO real nonnegative", "mnist_index": mnist_index, "seed": seed, "psnr": p, "psnr_no_alignment": p_raw, "runtime_s": h.runtime_s})
            print(mnist_index, seed, *(f"{x['method']} {x['psnr']:.2f}" for x in rows[-2:]), flush=True)
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    lines = ["# Net-ADM in the MNIST setting of Ma et al.: check", "",
             f"6 development images (MNIST test indices {idx}), 28 x 28, r = 2 per dimension, no noise, seeds 0 to 9.", "",
             "| method | mean PSNR (dB) | SD over 60 runs | per-image means | mean PSNR, no alignment | runs above 30 dB | mean runtime (s) |",
             "|---|--:|--:|---|--:|--:|--:|"]
    for m in ("Net-ADM", "HIO real nonnegative"):
        v = [r for r in rows if r["method"] == m]
        p = np.array([r["psnr"] for r in v])
        per = [float(np.mean([r["psnr"] for r in v if r["mnist_index"] == i])) for i in idx]
        summary[m] = {"mean_psnr": float(p.mean()), "sd": float(p.std(ddof=1)), "per_image": per,
                      "mean_psnr_no_alignment": float(np.mean([r["psnr_no_alignment"] for r in v])),
                      "runs_above_30dB": int((p > 30).sum()), "mean_runtime_s": float(np.mean([r["runtime_s"] for r in v]))}
        s = summary[m]
        lines.append(f"| {m} | {s['mean_psnr']:.2f} | {s['sd']:.2f} | {', '.join(f'{x:.1f}' for x in per)} | "
                     f"{s['mean_psnr_no_alignment']:.2f} | {s['runs_above_30dB']} of 60 | {s['mean_runtime_s']:.1f} |")
    (out / "summary.json").write_text(json.dumps({"mnist_indices": idx, "methods": summary}, indent=2))
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
