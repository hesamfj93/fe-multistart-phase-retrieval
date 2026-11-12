#!/usr/bin/env python
"""Post-hoc analyses of the stage-3 experiment (complex-valued objects). Not part of the plan.

These analyses were written after the pre-specified report (scripts/stage3_report.py) had been
run. They test nothing and change no decision of STAGE3_COMPLEX_PLAN.md. They describe:

1. FE by method family: classical and untrained reconstructions reach very different FE levels,
   so FE is compared only within one family.
2. FE within RAAR with shrinkwrap (best of five): how well it separates exact recoveries from
   failures, and how the untrained FE-softmax reconstruction does on the RAAR failures.
3. A fallback rule that was not planned: use RAAR-shrinkwrap best5 when its FE is at most t,
   otherwise FE-softmax. Reported over a grid of t, without choosing a value.

    python scripts/stage3_posthoc.py --main results/stage3_complex_heldout \
        --baselines results/stage3_complex_heldout_baselines \
        --out results/stage3_complex_heldout/report/posthoc_stage3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

CAP, FAIL = 60.0, 15.0
THRESHOLDS = (0.001, 0.003, 0.01, 0.02, 0.03)
NET = ("k1", "uniform", "median", "best_single_by_fe", "fe_softmax")
CLASSICAL = ("hio_supp_best5", "hio_sw_best5", "hio_realpos_best5",
             "raar_supp_best5", "raar_sw_best5", "raar_realpos_best5")


def load_run(run_dir: Path) -> dict:
    """Per-image method metrics of a run, keyed by MNIST test index; failed images are skipped."""
    out = {}
    for mj in sorted((run_dir / "per_image").glob("image_*/metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        if m.get("status") == "ok":
            out[int(m["mnist_test_index"])] = m["methods"]
    return out


def summarize(p: np.ndarray) -> dict:
    """Mean, median, failures (< 15 dB) and exact recoveries (>= 60 dB) of capped PSNR values."""
    return {"mean": float(p.mean()), "median": float(np.median(p)),
            "failures": int((p < FAIL).sum()), "exact": int((p >= CAP).sum())}


def main() -> None:
    """Compute the three post-hoc summaries and write them as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--main", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    net, bl = load_run(Path(a.main)), load_run(Path(a.baselines))
    ids = sorted(set(net) & set(bl))
    get = lambda run, m, k: np.array([run[i][m][k] for i in ids], dtype=float)  # noqa: E731

    fe_levels = {}
    for m in NET:
        f = get(net, m, "fe")
        fe_levels[m] = {"median": float(np.median(f)), "min": float(f.min()), "max": float(f.max())}
    for m in CLASSICAL:
        f = get(bl, m, "fe")
        fe_levels[m] = {"median": float(np.median(f)), "min": float(f.min()), "max": float(f.max())}

    p_net = np.minimum(get(net, "fe_softmax", "psnr"), CAP)
    p_raar = np.minimum(get(bl, "raar_sw_best5", "psnr"), CAP)
    f_raar = get(bl, "raar_sw_best5", "fe")
    exact, fail = p_raar >= CAP, p_raar < FAIL
    mid = ~exact & ~fail
    rho = stats.spearmanr(f_raar, -p_raar)[0]
    within_raar = {
        "spearman_fe_vs_minus_psnr": float(rho),
        "n_exact": int(exact.sum()), "fe_max_exact": float(f_raar[exact].max()),
        "n_fail": int(fail.sum()), "fe_min_fail": float(f_raar[fail].min()),
        "n_between": int(mid.sum()), "fe_median_between": float(np.median(f_raar[mid])),
        "fe_softmax_better_on_raar_failures": int((p_net[fail] > p_raar[fail]).sum()),
        "fe_softmax_better_on_raar_between": int((p_net[mid] > p_raar[mid]).sum()),
        "fe_softmax_better_overall": int((p_net > p_raar).sum()),
        "median_difference_db": float(np.median(p_net - p_raar)),
    }

    fallback = []
    for t in THRESHOLDS:
        use = f_raar <= t
        h = np.where(use, p_raar, p_net)
        fallback.append({"t": t, "raar_used": int(use.sum()), **summarize(h)})

    out = {
        "note": "post hoc; not part of STAGE3_COMPLEX_PLAN.md; no test, no chosen threshold",
        "n": len(ids),
        "fe_levels": fe_levels,
        "within_raar_sw_best5": within_raar,
        "fallback_raar_sw_then_fe_softmax": fallback,
        "reference": {"fe_softmax": summarize(p_net), "raar_sw_best5": summarize(p_raar),
                      "per_image_oracle_mean": float(np.maximum(p_net, p_raar).mean())},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
