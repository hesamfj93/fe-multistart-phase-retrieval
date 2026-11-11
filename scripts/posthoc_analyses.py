#!/usr/bin/env python
"""Post-hoc analyses of the n = 100 held-out run (evaluation only, nothing is retrained).

These analyses were NOT part of the frozen analysis plan. They answer questions raised after the
plan was executed and are reported as exploratory or as sensitivity analyses. None of them
changes a pre-specified decision.

  1. Uncertainty ablation (FE-softmax): pixel-wise Spearman between a map and |error|, over the
     whole window and inside the object region, and the area under the error-retention curve,
     for the fused map u, its two components, and two maps that need no ensemble (gradient
     magnitude and amplitude of the reconstruction).
  2. Image-level reliability scores: Spearman(score, -PSNR) for the failure score s and for
     ensemble-free alternatives (FE of the fused reconstruction, lowest start FE).
  3. Paired comparisons of the pre-specified hypotheses split by rank half (0-49 / 50-99).
  4. Percentile-bootstrap 95% CIs of the mean capped PSNR for every method.

    python scripts/posthoc_analyses.py --run-dir results/stage2_mnist_heldout \
        --k1long results/stage2_mnist_heldout_k1long --baselines results/stage2_mnist_heldout_baselines

Writes <run-dir>/report_n100/posthoc/posthoc.json and posthoc.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.ndimage import binary_dilation

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.metrics.alignment import align_ensemble_to_anchor, align_to_gt, map_to_gt_frame  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors, select_anchor  # noqa: E402

CAP = 60.0
N_BOOT = 10000


def _runner():
    spec = importlib.util.spec_from_file_location("runner_posthoc", REPO / "scripts" / "run_phase_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _order(n):
    lines = [l for l in (REPO / "splits" / "mnist_test_heldout_indices.csv").read_text(encoding="utf-8").splitlines()
             if l and not l.startswith("#")][1:]
    return [int(l.split(",")[1]) for l in lines][:n]


def _load(run: Path):
    out = {}
    for mj in (run / "per_image").glob("image_*/metrics.json"):
        r = json.loads(mj.read_text(encoding="utf-8"))
        if r.get("status") == "ok":
            out[int(r["mnist_test_index"])] = r
    return out


def _nu(m):
    lo, hi = np.percentile(m, 1), np.percentile(m, 99)
    return (np.clip(m, lo, hi) - lo) / (hi - lo + 1e-12)


def _retention_area(score, err):
    """Mean fraction of total |error| left after removing the highest-scoring pixels (lower is better)."""
    o = np.argsort(-score, kind="stable")
    tot = err.sum()
    if tot <= 0:
        return float("nan")
    return float((1.0 - np.cumsum(err[o]) / tot).mean())


def _t_ci(d):
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, len(d) - 1)
    return float(d.mean() - t * se), float(d.mean() + t * se)


def uq_ablation(run: Path, order, R):
    names = ["u (fused)", "sigma_w only", "residual only", "gradient magnitude", "amplitude"]
    full = {k: [] for k in names}; inner = {k: [] for k in names}; ret = {k: [] for k in names + ["oracle"]}
    for idx in order:
        b = np.load(run / "per_image" / f"image_{idx:05d}" / "members.npz")
        stack = torch.from_numpy(b["stack_raw__real"] + 1j * b["stack_raw__imag"]).to(torch.complex64)
        y = torch.from_numpy(b["measurement"]).float()
        gt = torch.from_numpy(b["gt"]).to(torch.complex64)
        fe = member_fourier_errors(stack, y, oversample=2)
        a = select_anchor(stack, method="best_fe", y_intensity=y, oversample=2)
        al = align_ensemble_to_anchor(stack, anchor_index=a)
        st, _ = R.compute_strategies(stack, y, fe, a, al, 0.05, 2)
        mean_c, std = st["fe_softmax"]
        _, aligned, u, r_gt, err = R.score_reconstruction(mean_c, std, gt, y, 2)
        sig = map_to_gt_frame(std, align_to_gt(mean_c, gt)).squeeze().numpy()
        xr = aligned.real.squeeze().numpy()
        gy, gx = np.gradient(xr)
        maps = {"u (fused)": u.squeeze().numpy(), "sigma_w only": _nu(sig), "residual only": _nu(r_gt.squeeze().numpy()),
                "gradient magnitude": np.hypot(gy, gx), "amplitude": np.abs(xr)}
        e = err.squeeze().numpy()
        region = binary_dilation((gt.real.squeeze().numpy() > 0.05) | (np.abs(xr) > 0.05), iterations=1)
        for k, m in maps.items():
            full[k].append(stats.spearmanr(m.ravel(), e.ravel()).correlation)
            inner[k].append(stats.spearmanr(m[region], e[region]).correlation)
            ret[k].append(_retention_area(m.ravel(), e.ravel()))
        ret["oracle"].append(_retention_area(e.ravel(), e.ravel()))
    summ = lambda v: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))}
    out = {"spearman_full_window": {k: summ(v) for k, v in full.items()},
           "spearman_object_region": {k: summ(v) for k, v in inner.items()},
           "error_retention_area": {k: summ(v) for k, v in ret.items()},
           "note": "region = 1-pixel dilation of (gt > 0.05) or (|recon| > 0.05); retention area: random = 0.5"}
    paired = {}
    for k in ["u (fused)", "sigma_w only"]:
        d = np.array(inner[k]) - np.array(inner["gradient magnitude"])
        lo, hi = _t_ci(d)
        paired[f"{k} minus gradient magnitude (object region)"] = {
            "mean": float(d.mean()), "ci95": [lo, hi], "n_higher": int((d > 0).sum()),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue)}
    out["paired_vs_gradient"] = paired
    return out


def image_level(main, order):
    out = {}
    for label, sel in [("ranks 50-99", order[50:100]), ("ranks 0-99", order)]:
        p = np.array([main[i]["methods"]["fe_softmax"]["psnr"] for i in sel])
        scores = {"failure score s (pre-specified)": [main[i]["methods"]["fe_softmax"]["failure_score"] for i in sel],
                  "FE of the FE-softmax reconstruction": [main[i]["methods"]["fe_softmax"]["fe"] for i in sel],
                  "lowest start FE": [min(main[i]["member_fe"]) for i in sel]}
        res = {}
        for k, v in scores.items():
            rho = stats.spearmanr(v, -p).correlation
            rng = np.random.default_rng(0); n = len(sel); bs = []
            v = np.asarray(v)
            for _ in range(2000):
                ix = rng.integers(0, n, n)
                bs.append(stats.spearmanr(v[ix], -p[ix]).correlation)
            res[k] = {"rho": float(rho), "bootstrap95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        out[label] = res
    return out


def split_halves(main, k1long, base, order):
    comps = [("H1a", "fe_softmax", "k1_long", k1long), ("H1b", "best_single_by_fe", "k1_long", k1long),
             ("H2a", "fe_softmax", "hio_supp_best5", base), ("H2b", "fe_softmax", "raar_supp_best5", base)]
    out = {}
    for label, sel in [("ranks 0-49", order[:50]), ("ranks 50-99", order[50:100]), ("ranks 0-99", order)]:
        res = {}
        for h, a, b, other in comps:
            d = np.array([min(main[i]["methods"][a]["psnr"], CAP) - min(other[i]["methods"][b]["psnr"], CAP) for i in sel])
            lo, hi = _t_ci(d)
            res[h] = {"mean": float(d.mean()), "ci95": [lo, hi], "n_better": int((d > 0).sum()), "n": len(d),
                      "wilcoxon_p_uncorrected": float(stats.wilcoxon(d).pvalue)}
        out[label] = res
    return out


def bootstrap_cis(main, k1long, base, order):
    rng = np.random.default_rng(0)
    methods = {m: [main[i]["methods"][m]["psnr"] for i in order]
               for m in ["k1", "uniform", "median", "best_single_by_fe", "fe_softmax"]}
    methods["k1_long"] = [k1long[i]["methods"]["k1_long"]["psnr"] for i in order]
    for m in base[order[0]]["methods"]:
        methods[m] = [base[i]["methods"][m]["psnr"] for i in order]
    out = {}
    for m, v in methods.items():
        v = np.minimum(np.asarray(v, dtype=float), CAP); n = len(v)
        bs = v[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
        out[m] = {"mean": float(v.mean()), "boot95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--k1long", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    order = _order(a.n)
    main_rows, k1l, base = _load(run), _load(Path(a.k1long)), _load(Path(a.baselines))
    R = _runner()
    res = {"status": "post hoc, not part of the frozen analysis plan",
           "uncertainty_ablation": uq_ablation(run, order, R),
           "image_level_scores": image_level(main_rows, order),
           "hypotheses_by_rank_half": split_halves(main_rows, k1l, base, order),
           "bootstrap_ci_capped_psnr": bootstrap_cis(main_rows, k1l, base, order)}
    out = Path(a.out_dir) if a.out_dir else run / f"report_n{a.n}" / "posthoc"
    out.mkdir(parents=True, exist_ok=True)
    (out / "posthoc.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
