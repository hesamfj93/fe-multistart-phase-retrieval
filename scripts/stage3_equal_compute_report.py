#!/usr/bin/env python
"""Post hoc: RAAR with shrinkwrap at the wall time of FE-softmax (complex objects, ranks 100-199).

Not part of STAGE3_COMPLEX_PLAN.md. The confirmatory comparison C8 (FE-softmax against RAAR
with shrinkwrap, best of 5 starts) stays as reported. FE-softmax used about nine times the wall
time of that baseline, so this analysis gives RAAR with shrinkwrap 45 starts, the same wall time
(configs/stage3_complex_heldout_raar45.yaml), and describes:

1. Reproduction: seeds 0-4 of the new run against the stored best5 run.
2. Best of the first K starts, selected by FE against the measurement, for K = 1 ... 45.
3. FE-softmax against RAAR with shrinkwrap best of 45: paired differences, wins, failures,
   and how FE-softmax does on the images where RAAR fails. Descriptive; the Wilcoxon p-value
   is reported without multiplicity adjustment.
4. The fallback rule of the post-hoc analysis (posthoc_stage3.json) with best of 45.

PSNR is the complex PSNR, capped at 60 dB; < 15 dB is a failure, >= 60 dB an exact recovery.

    python scripts/stage3_equal_compute_report.py --main results/stage3_complex_heldout \
        --best5 results/stage3_complex_heldout_baselines --raar45 results/stage3_complex_heldout_raar45 \
        --out results/stage3_complex_heldout_raar45/report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

CAP, FAIL = 60.0, 15.0
KS = (1, 2, 5, 10, 15, 20, 30, 45)
THRESHOLDS = (0.001, 0.003, 0.01, 0.02, 0.03)


def load(run_dir: Path) -> dict:
    """metrics.json of completed images, keyed by MNIST test index."""
    out = {}
    for mj in sorted((run_dir / "per_image").glob("image_*/metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        if m.get("status") == "ok":
            out[int(m["mnist_test_index"])] = m
    return out


def summarize(p: np.ndarray) -> dict:
    """Mean, median, failures and exact recoveries of capped PSNR values."""
    return {"mean": float(p.mean()), "median": float(np.median(p)),
            "failures": int((p < FAIL).sum()), "exact": int((p >= CAP).sum())}


def main() -> int:
    """Compute the four summaries and write summary.json and report.md."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--main", required=True)
    ap.add_argument("--best5", required=True)
    ap.add_argument("--raar45", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    net, b5, r45 = load(Path(a.main)), load(Path(a.best5)), load(Path(a.raar45))
    ids = sorted(set(net) & set(b5) & set(r45))
    if len(ids) != 100:
        print(f"Incomplete: {len(ids)} images in all three runs")
        return 1

    cap = lambda v: np.minimum(np.asarray(v, dtype=float), CAP)  # noqa: E731
    fe45 = np.array([r45[i]["starts"]["raar_sw"]["start_fe"] for i in ids])        # (100, 45)
    ps45 = cap([r45[i]["starts"]["raar_sw"]["start_psnr"] for i in ids])           # (100, 45)
    fe5o = np.array([b5[i]["starts"]["raar_sw"]["start_fe"] for i in ids])
    ps5o = cap([b5[i]["starts"]["raar_sw"]["start_psnr"] for i in ids])
    p_net = cap([net[i]["methods"]["fe_softmax"]["psnr"] for i in ids])
    p_b5 = cap([b5[i]["methods"]["raar_sw_best5"]["psnr"] for i in ids])

    # 1. reproduction of seeds 0-4
    same_best = int((fe45[:, :5].argmin(1) == fe5o.argmin(1)).sum())
    best5_new = ps45[np.arange(len(ids)), fe45[:, :5].argmin(1)]
    repro = {
        "devices": {"best5_run": sorted({b5[i]["device_used"] for i in ids}),
                    "raar45_run": sorted({r45[i]["device_used"] for i in ids})},
        "same_selected_start": same_best,
        "max_abs_start_psnr_diff_db": float(np.abs(ps45[:, :5] - ps5o).max()),
        "starts_within_0p1_db": int((np.abs(ps45[:, :5] - ps5o) <= 0.1).sum()),
        "starts_compared": int(ps5o.size),
        "best5_new": summarize(best5_new), "best5_stored": summarize(p_b5),
    }

    # 2. best of the first K starts, selected by FE
    curve = []
    for k in KS:
        sel = fe45[:, :k].argmin(1)
        p = ps45[np.arange(len(ids)), sel]
        curve.append({"K": k, **summarize(p), "any_exact_among_K": int((ps45[:, :k] >= CAP).any(1).sum())})
    sel45 = fe45.argmin(1)
    p45 = ps45[np.arange(len(ids)), sel45]
    f45 = fe45[np.arange(len(ids)), sel45]

    # 3. FE-softmax against best of 45
    d = p_net - p45
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    fail45 = p45 < FAIL
    nz = d[d != 0]
    comp = {
        "mean_difference_db": float(d.mean()), "t_ci95": [float(ci[0]), float(ci[1])],
        "median_difference_db": float(np.median(d)),
        "fe_softmax_better": int((d > 0).sum()), "raar45_better": int((d < 0).sum()), "ties": int((d == 0).sum()),
        "wilcoxon_p_two_sided_unadjusted": float(stats.wilcoxon(nz).pvalue) if nz.size else None,
        "raar45_failures": int(fail45.sum()),
        "fe_softmax_better_on_raar45_failures": int((p_net[fail45] > p45[fail45]).sum()),
        "fe_softmax_failures_on_raar45_failures": int((p_net[fail45] < FAIL).sum()),
        "both_fail": int(((p_net < FAIL) & fail45).sum()),
        "fe_softmax": summarize(p_net), "raar45": summarize(p45),
        "per_image_oracle_mean": float(np.maximum(p_net, p45).mean()),
    }

    # 4. fallback rule with best of 45
    fallback = []
    for t in THRESHOLDS:
        use = f45 <= t
        fallback.append({"t": t, "raar_used": int(use.sum()), **summarize(np.where(use, p45, p_net))})

    # FE separation within best of 45
    exact, fail = p45 >= CAP, p45 < FAIL
    sep = {"fe_max_exact": float(f45[exact].max()) if exact.any() else None,
           "fe_min_fail": float(f45[fail].min()) if fail.any() else None}

    out = {"note": "post hoc; not part of STAGE3_COMPLEX_PLAN.md; C8 unchanged; no multiplicity adjustment",
           "n": len(ids), "equal_wall_time": {"fe_softmax_s_per_image": 128.7, "raar_sw_s_per_start": 2.86, "starts": 45},
           "reproduction_seeds_0_4": repro, "best_of_first_K": curve,
           "fe_softmax_vs_raar45": comp, "fallback_raar45_then_fe_softmax": fallback,
           "fe_separation_raar45": sep}
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    L = ["# Post hoc: RAAR with shrinkwrap at equal wall time (45 starts), complex objects, n = 100", "",
         f"Reproduction of seeds 0-4: same selected start on {same_best}/100 images; "
         f"{repro['starts_within_0p1_db']}/{repro['starts_compared']} starts within 0.1 dB.", "",
         "| K | mean | median | failures | exact | images with an exact start |", "|--:|--:|--:|--:|--:|--:|"]
    L += [f"| {c['K']} | {c['mean']:.2f} | {c['median']:.2f} | {c['failures']} | {c['exact']} | {c['any_exact_among_K']} |"
          for c in curve]
    c = comp
    L += ["", f"FE-softmax: mean {c['fe_softmax']['mean']:.2f}, failures {c['fe_softmax']['failures']}. "
          f"RAAR best45: mean {c['raar45']['mean']:.2f}, failures {c['raar45']['failures']}, exact {c['raar45']['exact']}.",
          f"Difference FE-softmax minus RAAR best45: mean {c['mean_difference_db']:+.2f} dB "
          f"[{c['t_ci95'][0]:.2f}, {c['t_ci95'][1]:.2f}], median {c['median_difference_db']:+.2f} dB; "
          f"FE-softmax better on {c['fe_softmax_better']}, RAAR on {c['raar45_better']}.",
          f"On the {c['raar45_failures']} RAAR failures FE-softmax was better on {c['fe_softmax_better_on_raar45_failures']}.",
          "", "| t | RAAR used | mean | failures | exact |", "|--:|--:|--:|--:|--:|"]
    L += [f"| {f['t']} | {f['raar_used']} | {f['mean']:.2f} | {f['failures']} | {f['exact']} |" for f in fallback]
    (od / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
