#!/usr/bin/env python
"""Cross-study numbers for the TMLR manuscript, from the stored per-image results.

Descriptive summaries only; the confirmatory tests are in the frozen reports of each study.
The FE of the FE-softmax result is the error against the measurement (paper/tmlr_fe.py).
Writes paper/tmlr_numbers.json.

    python paper/tmlr_numbers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tmlr_fe import REPO, fe_softmax_measured
CAP = 60.0
RUNS = {
    "s1_real_ours": "results/stage2_mnist_heldout",
    "s2_complex_ours": "results/stage3_complex_heldout",
    "s3_real_ours_moderate": "results/study3/real_ours_moderate",
    "s3_real_ours_strong": "results/study3/real_ours_strong",
    "s3_complex_ours_moderate": "results/study3/complex_ours_moderate",
    "s3_complex_ours_strong": "results/study3/complex_ours_strong",
    "s3_real_netadm_moderate": "results/study3/real_netadm_moderate",
    "s3_real_netadm_strong": "results/study3/real_netadm_strong",
    "s3_real_netadm_none": "results/study3/real_netadm_none",
}
RULES = ("k1", "uniform", "median", "best_single_by_fe", "fe_softmax")


def load(run: str) -> list[dict]:
    out = []
    for p in sorted((REPO / run / "per_image").glob("*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("status", "ok") == "ok":
            m["fe_softmax_measured"] = fe_softmax_measured(p.parent)
            out.append(m)
    return out


def summarize(ms: list[dict]) -> dict:
    cap = lambda v: min(float(v), CAP)  # noqa: E731
    psnr = {r: np.array([cap(m["methods"][r]["psnr"]) for m in ms]) for r in RULES}
    mem = [np.minimum(np.array(m["member_psnr"], float), CAP) for m in ms]
    fe = [np.array(m["member_fe"], float) for m in ms]
    best = np.array([x.max() for x in mem])
    lowest_is_best = sum(int(np.argmin(f) == np.argmax(p)) for f, p in zip(fe, mem))
    fe_fused = np.array([float(m["fe_softmax_measured"]) for m in ms])
    rho = spearmanr(fe_fused, -psnr["fe_softmax"]).correlation
    within = [spearmanr(-f, p).correlation for f, p in zip(fe, mem)]
    return {
        "n": len(ms),
        "mean": {r: float(v.mean()) for r, v in psnr.items()},
        "failures": {r: int((v < 15).sum()) for r, v in psnr.items()},
        "best_start_mean": float(best.mean()),
        "best_start_failures": int((best < 15).sum()),
        "selection_minus_fesoftmax": float((psnr["best_single_by_fe"] - psnr["fe_softmax"]).mean()),
        "fesoftmax_minus_k1": float((psnr["fe_softmax"] - psnr["k1"]).mean()),
        "lowest_fe_is_best_start": lowest_is_best,
        "spearman_fe_fused_vs_negpsnr": float(rho),
        "mean_within_image_spearman": float(np.nanmean(within)),
        "median_start_fe": float(np.median(np.concatenate(fe))),
        "start_failures": int(sum((x < 15).sum() for x in mem)),
        "n_starts": int(sum(len(x) for x in mem)),
    }


def main() -> None:
    res = {k: summarize(load(v)) for k, v in RUNS.items()}
    (REPO / "paper" / "tmlr_numbers.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, v in res.items():
        mm = v["mean"]
        print(f"{k:28s} n={v['n']:3d} K1 {mm['k1']:.2f} uni {mm['uniform']:.2f} med {mm['median']:.2f} "
              f"sel {mm['best_single_by_fe']:.2f} fes {mm['fe_softmax']:.2f} best {v['best_start_mean']:.2f} | "
              f"fail K1 {v['failures']['k1']} fes {v['failures']['fe_softmax']} | sel-fes {v['selection_minus_fesoftmax']:+.2f} | "
              f"lowFE=best {v['lowest_fe_is_best_start']} | rho {v['spearman_fe_fused_vs_negpsnr']:.2f} | "
              f"within {v['mean_within_image_spearman']:.2f} | medFE {v['median_start_fe']:.3f}")


if __name__ == "__main__":
    main()
