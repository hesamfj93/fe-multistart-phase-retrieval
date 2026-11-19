#!/usr/bin/env python
"""Post hoc: how well the FE of the FE-softmax reconstruction flags failures (PSNR < 15 dB).

AUROC per untrained run with a percentile bootstrap 95% interval over images (10,000 resamples,
seed 0; resamples without a failure or without a success are drawn again). Reads the stored
per-image results; writes paper/tmlr_auroc.json and paper/tmlr_supp/tab_auroc.tex.

    python paper/tmlr_auroc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = {
    "Study 1, real, ours": "results/stage2_mnist_heldout",
    "Study 2, complex, ours": "results/stage3_complex_heldout",
    "Study 3, real, moderate, ours": "results/study3/real_ours_moderate",
    "Study 3, real, strong, ours": "results/study3/real_ours_strong",
    "Study 3, complex, moderate, ours": "results/study3/complex_ours_moderate",
    "Study 3, complex, strong, ours": "results/study3/complex_ours_strong",
    "Study 3, real, moderate, Net-ADM": "results/study3/real_netadm_moderate",
    "Study 3, real, strong, Net-ADM": "results/study3/real_netadm_strong",
}
FAIL_DB = 15.0
B = 10_000


def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    """P(score of a failure > score of a success), ties counted one half."""
    s1, s0 = score[pos], score[~pos]
    gt = (s1[:, None] > s0[None, :]).sum()
    eq = (s1[:, None] == s0[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(s1) * len(s0)))


def main() -> None:
    rng = np.random.default_rng(0)
    res, rows = {}, []
    for name, run in RUNS.items():
        ms = [json.loads(p.read_text()) for p in sorted((REPO / run / "per_image").glob("*/metrics.json"))]
        fe = np.array([m["methods"]["fe_softmax"]["fe"] for m in ms])
        ps = np.array([min(m["methods"]["fe_softmax"]["psnr"], 60.0) for m in ms])
        fail = ps < FAIL_DB
        n, k = len(ps), int(fail.sum())
        if k == 0 or k == n:
            res[name] = {"n": n, "failures": k, "auroc": None}
            rows.append(f"{name} & {n} & {k} & -- & -- \\\\")
            continue
        a = auroc(fe, fail)
        boots = []
        while len(boots) < B:
            idx = rng.integers(0, n, n)
            f = fail[idx]
            if 0 < f.sum() < n:
                boots.append(auroc(fe[idx], f))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[name] = {"n": n, "failures": k, "auroc": a, "ci95": [float(lo), float(hi)]}
        rows.append(f"{name} & {n} & {k} & {a:.2f} & [{lo:.2f}, {hi:.2f}] \\\\")
        print(f"{name:34s} n={n:3d} failures={k:2d} AUROC={a:.2f} [{lo:.2f}, {hi:.2f}]")
    (REPO / "paper" / "tmlr_auroc.json").write_text(json.dumps(res, indent=2))
    (REPO / "paper" / "tmlr_supp").mkdir(exist_ok=True)
    (REPO / "paper" / "tmlr_supp" / "tab_auroc.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
