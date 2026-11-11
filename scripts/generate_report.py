#!/usr/bin/env python
"""Protocol-aware reporting for phase-retrieval runs.

Every completed image is assigned to exactly one section by its MNIST test index:

  heldout      index in splits/mnist_test_heldout_indices.csv  -> PRIMARY paper table
  development  index in splits/mnist_test_dev_indices.csv      -> separate section only
  smoke        index in splits/mnist_test_smoke_indices.csv    -> software verification only
  other        anything else (legacy runs)                     -> listed, never summarised as primary

Only held-out images contribute to the primary table. Development images can come from
the same run directory or from a separate one (--dev-run-dir); they are never mixed.

Extra runs (--extra-run, repeatable) add comparison methods computed on the same images by
other runners: the compute-matched single start (k1_long) and the classical HIO/RAAR
baselines. Their methods are merged per image by MNIST test index and compared in pairs
with the two lead methods (FE-softmax and best-single-by-FE).

Pre-registered rules applied here (PREREGISTRATION_STAGE2_100.md):
  * PSNR is capped at 60 dB in every summary statistic and paired difference; a value at or
    above 60 dB counts as exact recovery. Per-image CSVs keep the raw values.
  * Holm adjustment over all paired PSNR comparisons listed in the report.
  * Confirmatory image-level test on held-out ranks 50 to 99 only: Spearman(image failure
    score, -PSNR) for FE-softmax, one-sided, alpha 0.05.

Usage
  python scripts/generate_report.py --run-dir results/stage2_mnist_heldout --n-heldout 50
  python scripts/generate_report.py --run-dir results/stage2_mnist_heldout --n-heldout 100 ^
      --extra-run results/stage2_mnist_heldout_k1long --extra-run results/stage2_mnist_heldout_baselines

Outputs (in --report-dir; default <run-dir>/report_n<N> when --n-heldout is given, else <run-dir>)
  tables/<section>/per_image.csv, summary.csv, paired_comparison.csv, fe_psnr.csv,
                   uncertainty.csv, failure.csv, effective_size.csv
  tables/heldout/primary_table.csv           (the paper table; held-out only)
  tables/heldout/confirmatory_image_level.csv (when ranks 50 to 99 are complete)
  figures/<section>/*.png
  report.md, report_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.data.splits import DEV_INDICES, read_split_csv, split_path  # noqa: E402

METHODS = ["k1", "uniform", "median", "best_single_by_fe", "fe_softmax"]
LABEL = {"k1": "K=1", "uniform": "K=5 uniform", "median": "K=5 median",
         "best_single_by_fe": "K=5 best-single-by-FE", "fe_softmax": "K=5 FE-softmax (T=0.05)",
         "k1_long": "K=1, 15000 it (compute-matched)"}
ENSEMBLE = ["uniform", "median", "best_single_by_fe", "fe_softmax"]
UNC_SOURCE = {"k1": "none", "uniform": "ensemble (intrinsic)", "median": "ensemble (intrinsic)",
              "best_single_by_fe": "aligned K=5 stack (borrowed, not intrinsic)",
              "fe_softmax": "ensemble (intrinsic)"}
PAIRS = [("best_single_by_fe", "k1"), ("fe_softmax", "k1"), ("uniform", "k1"), ("median", "k1"),
         ("fe_softmax", "uniform"), ("fe_softmax", "median"), ("best_single_by_fe", "fe_softmax")]
EXTRA_PAIR_TARGETS = ["fe_softmax", "best_single_by_fe"]

PSNR_CAP_DB = 60.0
CONFIRM_RANKS = range(50, 100)
CONFIRM_PRIMARY = "fe_softmax"
CONFIRM_SECONDARY = ["uniform", "median", "best_single_by_fe"]
CONFIRM_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 10000

BEST_SINGLE_NOTE = ("Best-single-by-FE has no intrinsic ensemble uncertainty. Its uncertainty "
                    "values are derived from the full aligned K=5 stack and are not a "
                    "like-for-like uncertainty output of the selected single reconstruction.")
WORDING_NOTE = ("The plain-mean degradation observed during development is a uniform "
                "aggregation failure. FE is a practical, empirically evaluated, "
                "ground-truth-free reliability proxy, not a guaranteed best-member estimator. "
                "Selection and fidelity-weighted fusion are two complementary operating modes.")
CAP_NOTE = (f"PSNR is capped at {PSNR_CAP_DB:g} dB in all summary statistics and paired differences "
            f"(a reconstruction at or above {PSNR_CAP_DB:g} dB is exact up to floating-point error); "
            "per-image CSVs keep raw values.")


def label(m: str) -> str:
    if m in LABEL:
        return LABEL[m]
    mm = re.match(r"(hio|raar)_(supp|realpos)_(s1|best(\d+))$", m)
    if mm:
        prior = "support only" if mm.group(2) == "supp" else "real+nonneg"
        starts = "1 start" if mm.group(3) == "s1" else f"best of {mm.group(4)} by FE"
        return f"{mm.group(1).upper()}, {prior}, {starts}"
    return m


def unc_source(m: str) -> str:
    return UNC_SOURCE.get(m, "none")


def cap(p: float) -> float:
    return min(float(p), PSNR_CAP_DB)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _t975(n: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1))
    except Exception:
        return 1.96


def describe(xs) -> dict:
    xs = np.asarray([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    n = len(xs)
    if n == 0:
        return {"n": 0}
    sd = float(xs.std(ddof=1)) if n > 1 else 0.0
    half = _t975(n) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": float(xs.mean()), "std": sd, "median": float(np.median(xs)),
            "min": float(xs.min()), "max": float(xs.max()),
            "ci95_low": float(xs.mean() - half), "ci95_high": float(xs.mean() + half)}


def spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)
    except Exception:
        rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        return float((rx * ry).sum() / math.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def auroc(scores, labels) -> float:
    """Probability that a random failure scores higher than a random success (ties count 1/2)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def holm(pvalues):
    """Holm step-down adjustment; None entries are passed through."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    out = [None] * len(pvalues)
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvalues[j])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        out[i] = running
    return out


def bootstrap_spearman_ci(x, y, resamples=BOOTSTRAP_RESAMPLES, seed=0):
    from scipy.stats import rankdata
    x = np.asarray(x, float); y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(resamples, len(x)))
    rx = rankdata(x[idx], axis=1); ry = rankdata(y[idx], axis=1)
    rx -= rx.mean(axis=1, keepdims=True); ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    rho = np.where(den > 0, (rx * ry).sum(axis=1) / np.where(den > 0, den, 1), np.nan)
    rho = rho[~np.isnan(rho)]
    return float(np.percentile(rho, 2.5)), float(np.percentile(rho, 97.5))


# --------------------------------------------------------------------------
# Loading, merging and partitioning
# --------------------------------------------------------------------------

def apply_uncertainty_v2(r) -> None:
    """Use the frame-corrected pixel-wise uncertainty metrics when they exist.

    scripts/rescore_uncertainty.py writes per_image/<id>/uncertainty_v2.json for runs scored
    before the correction. Runs scored after it carry uncertainty_eval_version 2 in
    metrics.json already. The version used is recorded in r["_unc_eval_version"].
    """
    v2 = r["_dir"] / "uncertainty_v2.json"
    if r.get("status") == "ok" and v2.exists():
        rec = json.loads(v2.read_text(encoding="utf-8"))
        for m, vals in rec.get("methods", {}).items():
            if m in r.get("methods", {}):
                r["methods"][m].update(vals)
        r["_unc_eval_version"] = int(rec.get("uncertainty_eval_version", 2))
    else:
        r["_unc_eval_version"] = r.get("uncertainty_eval_version") or (1 if int(r.get("K", 1) or 1) > 1 else None)


def load_run(run_dir: Path):
    rows, failed = [], []
    for mj in sorted((run_dir / "per_image").glob("image_*/metrics.json")):
        r = json.loads(mj.read_text(encoding="utf-8"))
        r["_dir"] = mj.parent
        r["_run"] = str(run_dir)
        apply_uncertainty_v2(r)
        if "mnist_test_index" not in r:
            try:
                r["mnist_test_index"] = int(str(r.get("name", "")).rsplit("_", 1)[1])
            except Exception:
                r["mnist_test_index"] = None
        (rows if r.get("status") == "ok" else failed).append(r)
    return rows, failed


def merge_extra_runs(rows, extra_dirs):
    """Merge methods from extra runs into the main rows, by MNIST test index."""
    by_idx = {r["mnist_test_index"]: r for r in rows}
    extras, records = [], []
    for d in extra_dirs:
        ex_rows, ex_failed = load_run(Path(d))
        rec = {"run_dir": str(Path(d).resolve()), "n_ok": len(ex_rows), "n_failed": len(ex_failed),
               "methods": [], "devices": {}, "merged": 0, "extra_only_indices": []}
        for er in ex_rows:
            rec["devices"][er.get("device_used", "unknown")] = rec["devices"].get(er.get("device_used", "unknown"), 0) + 1
            main = by_idx.get(er["mnist_test_index"])
            if main is None:
                rec["extra_only_indices"].append(er["mnist_test_index"])
                continue
            for m, v in er["methods"].items():
                if m in METHODS or (m in main["methods"] and main["methods"][m].get("_source") != str(d)):
                    raise SystemExit(f"method name {m!r} from extra run {d} collides with an existing method")
                v = dict(v)
                if "runtime_s" not in v:
                    v["runtime_s"] = float(sum(er.get("runtime_per_member_s", [])))
                v["_source"] = str(d)
                v["_device"] = er.get("device_used")
                main["methods"][m] = v
                if m not in extras:
                    extras.append(m)
                if m not in rec["methods"]:
                    rec["methods"].append(m)
            rec["merged"] += 1
        records.append(rec)
    return extras, records


def split_lists(splits_dir: Path) -> dict:
    out = {}
    for role in ["dev", "heldout", "smoke"]:
        p = split_path(splits_dir, role)
        out[role] = read_split_csv(p) if p.exists() else []
    out["dev"] = sorted(set(out["dev"]) | set(DEV_INDICES))
    return out


def partition(rows, lists, n_heldout=None) -> dict:
    """Assign rows to sections by index membership. Development wins over everything."""
    heldout_rank = {i: r for r, i in enumerate(lists["heldout"])}
    dev, smoke = set(lists["dev"]), set(lists["smoke"])
    sec = {"heldout": [], "development": [], "smoke": [], "other": [], "heldout_beyond_n": []}
    for r in rows:
        i = r["mnist_test_index"]
        if i in dev:
            sec["development"].append(r)
        elif i in heldout_rank:
            r["_heldout_rank"] = heldout_rank[i]
            if n_heldout is not None and heldout_rank[i] >= n_heldout:
                sec["heldout_beyond_n"].append(r)
            else:
                sec["heldout"].append(r)
        elif i in smoke:
            sec["smoke"].append(r)
        else:
            sec["other"].append(r)
    sec["heldout"].sort(key=lambda r: r["_heldout_rank"])
    return sec


# --------------------------------------------------------------------------
# Section analysis
# --------------------------------------------------------------------------

def method_runtime(r, m) -> float:
    v = r["methods"][m]
    if "runtime_s" in v:
        return float(v["runtime_s"])
    rt = r.get("runtime_per_member_s", [])
    return float(rt[0]) if m == "k1" else float(sum(rt))


def _has(r, m):
    return m in r["methods"]


def analyse(rows, failure_db: float, auroc_min: int, methods_all, pairs) -> dict:
    res = {"n_images": len(rows), "methods_order": list(methods_all), "methods": {}, "paired": [],
           "uncertainty": {}, "failure": {}, "fe_psnr": {}, "effective_size": {}}
    for m in methods_all:
        rs = [r for r in rows if _has(r, m)]
        res["methods"][m] = {
            "psnr": describe([cap(r["methods"][m]["psnr"]) for r in rs]),
            "ssim": describe([r["methods"][m]["ssim"] for r in rs]),
            "fe": describe([r["methods"][m]["fe"] for r in rs]),
            "runtime_s": describe([method_runtime(r, m) for r in rs]),
        }
    for a, b in pairs:
        rs = [r for r in rows if _has(r, a) and _has(r, b)]
        for metric, higher_better in [("psnr", True), ("ssim", True), ("fe", False)]:
            if metric == "psnr":
                d = np.array([cap(r["methods"][a]["psnr"]) - cap(r["methods"][b]["psnr"]) for r in rs], float)
            else:
                d = np.array([r["methods"][a][metric] - r["methods"][b][metric] for r in rs], float)
            if len(d) == 0:
                continue
            better = d > 1e-9 if higher_better else d < -1e-9
            worse = d < -1e-9 if higher_better else d > 1e-9
            entry = {"method": a, "baseline": b, "metric": metric, **describe(d),
                     "n_better": int(better.sum()), "n_worse": int(worse.sum()),
                     "n_equal": int(len(d) - better.sum() - worse.sum())}
            try:
                from scipy.stats import trim_mean, wilcoxon
                entry["trimmed_mean_10pct"] = float(trim_mean(d, 0.1))
                nz = d[np.abs(d) > 1e-9]
                entry["wilcoxon_p_uncorrected"] = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else None
            except Exception:
                entry["wilcoxon_p_uncorrected"] = None
            res["paired"].append(entry)
    psnr_entries = [e for e in res["paired"] if e["metric"] == "psnr"]
    for e, h in zip(psnr_entries, holm([e.get("wilcoxon_p_uncorrected") for e in psnr_entries])):
        e["wilcoxon_p_holm"] = h
    for m in [x for x in ENSEMBLE if x in methods_all]:
        res["uncertainty"][m] = {
            "source": unc_source(m),
            "pearson": describe([r["methods"][m].get("unc_pearson") for r in rows]),
            "spearman": describe([r["methods"][m].get("unc_spearman") for r in rows]),
            "ece": describe([r["methods"][m].get("unc_ece") for r in rows]),
        }
    for m in methods_all:
        rs = [r for r in rows if _has(r, m)]
        labels = [int(r["methods"][m]["psnr"] < failure_db) for r in rs]
        nf = sum(labels)
        f = {"threshold_psnr_db": failure_db, "n": len(rs), "n_failures": nf, "n_success": len(rs) - nf,
             "failure_rate": nf / len(rs) if rs else float("nan"),
             "n_exact": int(sum(r["methods"][m]["psnr"] >= PSNR_CAP_DB for r in rs))}
        if m in ENSEMBLE:
            if nf >= auroc_min and (len(rs) - nf) >= auroc_min:
                f["detection_auroc"] = auroc([r["methods"][m].get("failure_score", 0.0) for r in rs], labels)
            else:
                f["detection_auroc"] = None
                f["detection_note"] = (f"not computed: needs >= {auroc_min} failures and successes "
                                       f"(have {nf} / {len(rs) - nf})")
        res["failure"][m] = f
    pooled_fe, pooled_ps, within, top1, top2 = [], [], [], 0, 0
    per_img = []
    for r in rows:
        fe = np.asarray(r.get("member_fe", []), float)
        ps = np.asarray(r.get("member_psnr", []), float)
        if len(fe) < 2:
            continue
        pooled_fe += fe.tolist(); pooled_ps += ps.tolist()
        rho = spearman(-fe, ps)
        within.append(rho)
        b_fe, order_ps = int(fe.argmin()), list((-ps).argsort())
        t1, t2 = int(b_fe == order_ps[0]), int(b_fe in order_ps[:2])
        top1 += t1; top2 += t2
        per_img.append({"image_id": r["image_id"], "within_rho": rho, "top1": t1, "top2": t2})
    n = len(within)
    w = [x for x in within if not math.isnan(x)]
    res["fe_psnr"] = {
        "n_images": n, "n_members": len(pooled_fe),
        "pooled_spearman": spearman(-np.asarray(pooled_fe), np.asarray(pooled_ps)) if pooled_fe else float("nan"),
        "within_image_spearman_mean": float(np.mean(w)) if w else float("nan"),
        "within_image_spearman_median": float(np.median(w)) if w else float("nan"),
        "n_images_negative_within_rho": int(sum(1 for x in w if x < 0)),
        "top1_agreement": top1, "top1_rate": top1 / n if n else float("nan"),
        "top2_agreement": top2, "top2_rate": top2 / n if n else float("nan"),
        "per_image": per_img,
    }
    res["effective_size"] = describe([r.get("effective_sample_size") for r in rows if r.get("K", 1) > 1])
    devices = {}
    for r in rows:
        devices[r.get("device_used", "unknown")] = devices.get(r.get("device_used", "unknown"), 0) + 1
    res["devices"] = devices
    return res


def confirmatory_test(heldout_rows) -> dict:
    """Pre-registered: on held-out ranks 50..99 only, Spearman(image failure score, -PSNR) > 0."""
    sub = [r for r in heldout_rows if r.get("_heldout_rank") in CONFIRM_RANKS]
    out = {"ranks": f"{CONFIRM_RANKS.start}-{CONFIRM_RANKS.stop - 1}", "n_required": len(CONFIRM_RANKS),
           "n_available": len(sub), "primary_method": CONFIRM_PRIMARY, "alpha": CONFIRM_ALPHA,
           "hypothesis": "Spearman(failure_score, -PSNR) > 0, one-sided"}
    if len(sub) < len(CONFIRM_RANKS):
        out["status"] = "not computed (ranks 50 to 99 incomplete)"
        return out
    from scipy.stats import spearmanr
    out["status"] = "computed"
    out["results"] = {}
    for m in [CONFIRM_PRIMARY] + CONFIRM_SECONDARY:
        fs = [r["methods"][m]["failure_score"] for r in sub]
        neg_psnr = [-r["methods"][m]["psnr"] for r in sub]
        t = spearmanr(fs, neg_psnr, alternative="greater")
        lo, hi = bootstrap_spearman_ci(fs, neg_psnr)
        out["results"][m] = {"rho": float(t.statistic), "p_one_sided": float(t.pvalue),
                             "ci95_low": lo, "ci95_high": hi,
                             "role": "primary" if m == CONFIRM_PRIMARY else "secondary (descriptive)"}
    out["primary_pass"] = bool(out["results"][CONFIRM_PRIMARY]["p_one_sided"] < CONFIRM_ALPHA)
    return out


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def _fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_tables(rows, res, out: Path, primary: bool):
    out.mkdir(parents=True, exist_ok=True)
    methods_all = res["methods_order"]
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["image_id", "mnist_test_index", "label", "heldout_rank", "anchor_index",
                "best_single_index", "effective_sample_size", "device_used", "runtime_total_s"]
        for m in methods_all:
            head += [f"{m}_psnr", f"{m}_ssim", f"{m}_fe"]
            if m in ENSEMBLE:
                head += [f"{m}_unc_pearson", f"{m}_unc_spearman", f"{m}_unc_ece", f"{m}_failure_score"]
        w.writerow(head)
        for r in rows:
            row = [r["image_id"], r["mnist_test_index"], r.get("label"), r.get("_heldout_rank", ""),
                   r.get("anchor_index"), r.get("best_single_index"),
                   _fmt(r.get("effective_sample_size")), r.get("device_used", ""),
                   _fmt(r.get("runtime_total_s"), 2)]
            for m in methods_all:
                mm = r["methods"].get(m)
                if mm is None:
                    row += ["", "", ""] + (["", "", "", ""] if m in ENSEMBLE else [])
                    continue
                row += [_fmt(mm["psnr"]), _fmt(mm["ssim"]), _fmt(mm["fe"])]
                if m in ENSEMBLE:
                    row += [_fmt(mm.get("unc_pearson")), _fmt(mm.get("unc_spearman")),
                            _fmt(mm.get("unc_ece")), _fmt(mm.get("failure_score"))]
            w.writerow(row)

    stat_cols = ["n", "mean", "std", "median", "min", "max", "ci95_low", "ci95_high"]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric"] + stat_cols)
        for m in methods_all:
            for k in ["psnr", "ssim", "fe", "runtime_s"]:
                d = res["methods"][m][k]
                w.writerow([label(m), "psnr_capped" if k == "psnr" else k]
                           + [_fmt(d.get(c)) if c != "n" else d.get("n") for c in stat_cols])

    with open(out / "paired_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "baseline", "metric", "n", "mean_delta", "std_delta", "median_delta",
                    "trimmed_mean_10pct", "ci95_low", "ci95_high", "n_better", "n_worse", "n_equal",
                    "wilcoxon_p_uncorrected", "wilcoxon_p_holm"])
        for e in res["paired"]:
            w.writerow([label(e["method"]), label(e["baseline"]), e["metric"], e.get("n"),
                        _fmt(e.get("mean")), _fmt(e.get("std")), _fmt(e.get("median")),
                        _fmt(e.get("trimmed_mean_10pct")), _fmt(e.get("ci95_low")),
                        _fmt(e.get("ci95_high")), e["n_better"], e["n_worse"], e["n_equal"],
                        _fmt(e.get("wilcoxon_p_uncorrected"), 6), _fmt(e.get("wilcoxon_p_holm"), 6)])

    with open(out / "uncertainty.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "uncertainty_source", "metric"] + stat_cols)
        for m, u in res["uncertainty"].items():
            for k in ["pearson", "spearman", "ece"]:
                d = u[k]
                w.writerow([label(m), u["source"], k] + [_fmt(d.get(c)) if c != "n" else d.get("n") for c in stat_cols])

    with open(out / "failure.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "threshold_psnr_db", "n", "n_failures", "n_success", "failure_rate",
                    "n_exact_ge_cap", "detection_auroc", "note"])
        for m in methods_all:
            fr = res["failure"][m]
            w.writerow([label(m), fr["threshold_psnr_db"], fr["n"], fr["n_failures"], fr["n_success"],
                        _fmt(fr["failure_rate"]), fr["n_exact"], _fmt(fr.get("detection_auroc")),
                        fr.get("detection_note", "")])

    fp = res["fe_psnr"]
    with open(out / "fe_psnr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        for k in ["n_images", "n_members", "pooled_spearman", "within_image_spearman_mean",
                  "within_image_spearman_median", "n_images_negative_within_rho",
                  "top1_agreement", "top1_rate", "top2_agreement", "top2_rate"]:
            v = fp[k]
            w.writerow([k, _fmt(v) if isinstance(v, float) else v])
        w.writerow([])
        w.writerow(["image_id", "within_rho", "top1", "top2"])
        for e in fp["per_image"]:
            w.writerow([e["image_id"], _fmt(e["within_rho"]), e["top1"], e["top2"]])

    with open(out / "effective_size.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        for k, v in res["effective_size"].items():
            w.writerow([k, _fmt(v) if isinstance(v, float) else v])

    if primary:
        with open(out / "primary_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "n", "psnr_mean_capped", "psnr_std", "psnr_median", "psnr_ci95_low",
                        "psnr_ci95_high", "ssim_mean", "ssim_std", "fe_mean", "fe_std",
                        "runtime_s_mean", "failure_rate", "n_exact_ge_cap", "unc_pearson_mean",
                        "unc_spearman_mean", "unc_ece_mean", "uncertainty_source"])
            for m in methods_all:
                d = res["methods"][m]; fr = res["failure"][m]
                u = res["uncertainty"].get(m, {})
                w.writerow([label(m), d["psnr"]["n"], _fmt(d["psnr"]["mean"]), _fmt(d["psnr"]["std"]),
                            _fmt(d["psnr"]["median"]), _fmt(d["psnr"]["ci95_low"]),
                            _fmt(d["psnr"]["ci95_high"]), _fmt(d["ssim"]["mean"]),
                            _fmt(d["ssim"]["std"]), _fmt(d["fe"]["mean"]), _fmt(d["fe"]["std"]),
                            _fmt(d["runtime_s"]["mean"], 2), _fmt(fr["failure_rate"]), fr["n_exact"],
                            _fmt(u.get("pearson", {}).get("mean")), _fmt(u.get("spearman", {}).get("mean")),
                            _fmt(u.get("ece", {}).get("mean")), unc_source(m)])
        conf = res.get("confirmatory")
        if conf is not None:
            with open(out / "confirmatory_image_level.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ranks", "status", "method", "role", "n", "rho", "p_one_sided", "ci95_low", "ci95_high"])
                for m, v in (conf.get("results") or {}).items():
                    w.writerow([conf["ranks"], conf["status"], label(m), v["role"], conf["n_available"],
                                _fmt(v["rho"]), _fmt(v["p_one_sided"], 6), _fmt(v["ci95_low"]), _fmt(v["ci95_high"])])
                if not conf.get("results"):
                    w.writerow([conf["ranks"], conf["status"], label(CONFIRM_PRIMARY), "primary",
                                conf["n_available"], "", "", "", ""])


def section_markdown(title, rows, res, note) -> str:
    methods_all = res["methods_order"]
    L = [f"## {title}", "", note, "", f"Images: {res['n_images']}. Devices: {res.get('devices')}.", "",
         CAP_NOTE, ""]
    L += ["| method | n | PSNR mean ± std | PSNR median | PSNR 95% CI | SSIM mean ± std | FE mean | runtime (s) | failures | exact |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for m in methods_all:
        d = res["methods"][m]; fr = res["failure"][m]
        if d["psnr"].get("n", 0) == 0:
            continue
        L.append(f"| {label(m)} | {d['psnr']['n']} | {d['psnr']['mean']:.2f} ± {d['psnr']['std']:.2f} | "
                 f"{d['psnr']['median']:.2f} | [{_fmt(d['psnr']['ci95_low'], 2)}, {_fmt(d['psnr']['ci95_high'], 2)}] | "
                 f"{d['ssim']['mean']:.3f} ± {d['ssim']['std']:.3f} | {d['fe']['mean']:.3f} | "
                 f"{d['runtime_s']['mean']:.1f} | {fr['n_failures']}/{fr['n']} | {fr['n_exact']}/{fr['n']} |")
    L += ["", "Paired PSNR differences on identical images (method minus baseline, capped PSNR). "
              "Holm adjustment over all comparisons in this table:", "",
          "| method vs baseline | n | mean Δ | median Δ | 95% CI | better / worse / tie | Holm p |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for e in res["paired"]:
        if e["metric"] != "psnr":
            continue
        L.append(f"| {label(e['method'])} vs {label(e['baseline'])} | {e['n']} | {e['mean']:+.2f} | {e['median']:+.2f} | "
                 f"[{_fmt(e.get('ci95_low'), 2)}, {_fmt(e.get('ci95_high'), 2)}] | "
                 f"{e['n_better']} / {e['n_worse']} / {e['n_equal']} | {_fmt(e.get('wilcoxon_p_holm'), 4) or 'n/a'} |")
    fp = res["fe_psnr"]
    L += ["", f"FE as a reliability proxy: pooled Spearman {_fmt(fp['pooled_spearman'], 3)}, "
              f"within-image mean {_fmt(fp['within_image_spearman_mean'], 3)} / median "
              f"{_fmt(fp['within_image_spearman_median'], 3)}, negative within-image correlation in "
              f"{fp['n_images_negative_within_rho']} images, Top-1 {fp['top1_agreement']}/{fp['n_images']}, "
              f"Top-2 {fp['top2_agreement']}/{fp['n_images']}.", ""]
    L += ["| method | uncertainty source | Pearson | Spearman | ECE | failure AUROC |", "|---|---|--:|--:|--:|--:|"]
    for m, u in res["uncertainty"].items():
        fr = res["failure"][m]
        auc = _fmt(fr.get("detection_auroc"), 3) or "not computed"
        L.append(f"| {label(m)} | {u['source']} | {_fmt(u['pearson'].get('mean'), 3)} | "
                 f"{_fmt(u['spearman'].get('mean'), 3)} | {_fmt(u['ece'].get('mean'), 3)} | {auc} |")
    es = res["effective_size"]
    L += ["", f"Effective ensemble size (FE-softmax): mean {_fmt(es.get('mean'), 2)}, "
              f"median {_fmt(es.get('median'), 2)}, range [{_fmt(es.get('min'), 2)}, {_fmt(es.get('max'), 2)}].",
          "", f"Note. {BEST_SINGLE_NOTE}", ""]
    conf = res.get("confirmatory")
    if conf is not None:
        L += ["### Pre-registered confirmatory test (image-level reliability)", "",
              f"Held-out ranks {conf['ranks']} only; hypothesis: {conf['hypothesis']}; primary method "
              f"{label(conf['primary_method'])}; alpha {conf['alpha']}. Status: {conf['status']} "
              f"({conf['n_available']}/{conf['n_required']} images).", ""]
        if conf.get("results"):
            L += ["| method | role | rho | one-sided p | bootstrap 95% CI |", "|---|---|--:|--:|--:|"]
            for m, v in conf["results"].items():
                L.append(f"| {label(m)} | {v['role']} | {v['rho']:.3f} | {v['p_one_sided']:.2e} | "
                         f"[{v['ci95_low']:.3f}, {v['ci95_high']:.3f}] |")
            L += ["", f"Primary hypothesis {'SUPPORTED' if conf['primary_pass'] else 'NOT SUPPORTED'} "
                      f"at alpha {conf['alpha']}.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _aggregates(r):
    """Arrays for figures: the frame-corrected file when it exists."""
    v2 = r["_dir"] / "aggregates_v2.npz"
    return np.load(v2 if v2.exists() else r["_dir"] / "aggregates.npz")


def _recon(r, m):
    ag = _aggregates(r)
    return np.abs(ag[f"recon_{m}__real"] + 1j * ag[f"recon_{m}__imag"]).squeeze()


def _gt(r):
    return np.load(r["_dir"] / "members.npz")["gt"].squeeze()


def _boxplot(plt, rows, methods, path, title):
    data, labels = [], []
    for m in methods:
        v = [cap(r["methods"][m]["psnr"]) for r in rows if _has(r, m)]
        if v:
            data.append(v); labels.append(label(m))
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(data) + 3), 4.8))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel(f"PSNR (dB, capped at {PSNR_CAP_DB:g})"); ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_figures(rows, figdir: Path, title: str, methods_all):
    plt = _plt()
    figdir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    main_methods = METHODS + [m for m in ["k1_long"] if m in methods_all]
    _boxplot(plt, rows, main_methods, figdir / "method_boxplots.png", f"{title}: PSNR by method (n={n})")
    if len(methods_all) > len(main_methods):
        _boxplot(plt, rows, methods_all, figdir / "method_boxplots_all.png",
                 f"{title}: PSNR by method, including classical baselines (n={n})")

    k1 = np.array([r["methods"]["k1"]["psnr"] for r in rows]); order = k1.argsort()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(range(n), k1[order], "k--", lw=1, label=label("k1"))
    lines = [("uniform", "#c8553d"), ("median", "#8c8c8c"), ("best_single_by_fe", "#4c72b0"), ("fe_softmax", "#2f7d4f")]
    if "k1_long" in methods_all and all(_has(r, "k1_long") for r in rows):
        lines.append(("k1_long", "#b07aa1"))
    for m, c in lines:
        ax.plot(range(n), np.array([cap(r["methods"][m]["psnr"]) for r in rows])[order], "o-", ms=3, lw=1, color=c, label=label(m))
    ax.set_xlabel("image (sorted by K=1 PSNR)"); ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{title}: paired per-image PSNR"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figdir / "paired_psnr.png", dpi=150); plt.close(fig)

    fe = sum((r.get("member_fe", []) for r in rows), []); ps = sum((r.get("member_psnr", []) for r in rows), [])
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.scatter(fe, ps, s=10, alpha=0.5)
    ax.set_xlabel("member Fourier error (FE)"); ax.set_ylabel("member PSNR (dB)")
    ax.set_title(f"{title}: FE vs PSNR, {len(fe)} members"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figdir / "fe_psnr_scatter.png", dpi=150); plt.close(fig)

    # qualitative grid: worst (top), quartiles, best (bottom) by FE-softmax PSNR
    fs = np.array([r["methods"]["fe_softmax"]["psnr"] for r in rows]); o = fs.argsort()
    picks = list(dict.fromkeys(int(o[int(round(q * (n - 1)))]) for q in [0, 0.25, 0.5, 0.75, 1.0]))
    fig, axes = plt.subplots(len(picks), 6, figsize=(11, 1.9 * len(picks)), squeeze=False)
    for i, k in enumerate(picks):
        r = rows[k]
        axes[i, 0].imshow(_gt(r), cmap="gray"); axes[i, 0].set_ylabel(r["image_id"].replace("image_", "#"), fontsize=7)
        for j, m in enumerate(METHODS):
            axes[i, j + 1].imshow(_recon(r, m), cmap="gray")
            axes[i, j + 1].set_xlabel(f"{r['methods'][m]['psnr']:.1f} dB", fontsize=7)
            if i == 0: axes[i, j + 1].set_title(label(m), fontsize=7)
        if i == 0: axes[i, 0].set_title("ground truth", fontsize=7)
        for a in axes[i]: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{title}: worst (top), quartiles and best (bottom) by FE-softmax PSNR", fontsize=9)
    fig.tight_layout(); fig.savefig(figdir / "qualitative_grid.png", dpi=150); plt.close(fig)

    for fname, sel, ttl in [("uncertainty_vs_error.png", picks, "uncertainty vs error, worst (top) to best"),
                            ("success_failure_examples.png", [int(o[-1]), int(o[0])], "best (top) and worst (bottom)")]:
        fig, axes = plt.subplots(len(sel), 5, figsize=(9, 1.9 * len(sel)), squeeze=False)
        for i, k in enumerate(sel):
            r = rows[k]; ag = _aggregates(r)
            rec = _recon(r, "fe_softmax"); gt = _gt(r)
            panels = [(gt, "ground truth", "gray"), (rec, f"FE-softmax {r['methods']['fe_softmax']['psnr']:.1f} dB", "gray"),
                      (np.abs(rec - gt), "|error|", "magma"),
                      (ag["unc_fe_softmax"].squeeze() if "unc_fe_softmax" in ag else np.zeros_like(gt), "fused uncertainty", "magma"),
                      (ag["rphys_fe_softmax"].squeeze() if "rphys_fe_softmax" in ag else np.zeros_like(gt), "physics residual", "magma")]
            for j, (img, t, cm) in enumerate(panels):
                axes[i, j].imshow(img, cmap=cm); axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                if i == 0 or fname.startswith("success"): axes[i, j].set_title(t, fontsize=7)
            axes[i, 0].set_ylabel(r["image_id"].replace("image_", "#"), fontsize=7)
        fig.suptitle(f"{title}: {ttl}", fontsize=9)
        fig.tight_layout(); fig.savefig(figdir / fname, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _safe_console():
    """Never crash on a character the console code page cannot show (e.g. cp1256)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def main() -> int:
    _safe_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-heldout", type=int, default=None,
                    help="restrict the primary table to held-out ranks 0..N-1 (e.g. 50 after extending to 100)")
    ap.add_argument("--dev-run-dir", default=None, help="optional separate run with the development images")
    ap.add_argument("--extra-run", action="append", default=[],
                    help="run directory with extra comparison methods on the same images (repeatable)")
    ap.add_argument("--splits-dir", default=str(REPO / "splits"))
    ap.add_argument("--report-dir", default=None,
                    help="output folder; default <run-dir>/report_n<N> with --n-heldout, else <run-dir>")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if args.report_dir:
        out_root = Path(args.report_dir)
    elif args.n_heldout is not None:
        out_root = run_dir / f"report_n{args.n_heldout}"
    else:
        out_root = run_dir
    out_root.mkdir(parents=True, exist_ok=True)
    lists = split_lists(Path(args.splits_dir))
    rows, failed = load_run(run_dir)
    if args.dev_run_dir:
        dev_rows, dev_failed = load_run(Path(args.dev_run_dir))
        rows += dev_rows; failed += dev_failed
    missing_core = sorted({m for r in rows for m in METHODS if m not in r["methods"]})
    if missing_core:
        raise SystemExit(f"{run_dir} is not a K=5 run (missing methods {missing_core}). "
                         "Pass the K=5 run as --run-dir and this one as --extra-run.")
    extras, extra_records = merge_extra_runs(rows, args.extra_run)
    methods_all = METHODS + extras
    pairs = PAIRS + [(t, x) for x in extras for t in EXTRA_PAIR_TARGETS]
    sec = partition(rows, lists, args.n_heldout)

    cfg = {}
    if (run_dir / "config.yaml").exists():
        import yaml
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    ev = cfg.get("evaluation", {})
    failure_db = float(ev.get("failure_psnr_db", 15.0))
    auroc_min = int(ev.get("auroc_min_per_class", 5))

    notes = {
        "heldout": ("PRIMARY RESULTS. Held-out MNIST test images only; none of them was used to select "
                    "the aggregation method or T = 0.05. These numbers form the paper table."),
        "development": ("DEVELOPMENT / VALIDATION SUBSET. These 8 images were used to select the aggregation "
                        "method and T = 0.05. Not part of the paper table; reported only for transparency."),
        "smoke": "SOFTWARE VERIFICATION ONLY. Reduced settings; no scientific interpretation.",
    }
    summary = {"run_dir": str(run_dir.resolve()), "n_heldout_requested": args.n_heldout,
               "failure_definition": {"psnr_below_db": failure_db, "auroc_min_per_class": auroc_min},
               "psnr_cap_db": PSNR_CAP_DB, "extra_runs": extra_records,
               "sections": {}, "notes": {"best_single_by_fe_uncertainty": BEST_SINGLE_NOTE,
                                         "wording": WORDING_NOTE, "psnr_cap": CAP_NOTE}}
    for f in ["split_manifest.json", "protocol_lock.json", "environment.json"]:
        p = run_dir / f
        if p.exists():
            summary[f.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
    if "protocol_lock" in summary:
        summary["protocol_lock"].pop("frozen_config", None)
        summary["frozen_config"] = cfg

    md = [f"# Report: {run_dir.name}" + (f" (first {args.n_heldout} held-out images)" if args.n_heldout else ""), "",
          "Sections are separated by MNIST test index. Only the held-out section is used for the paper table.", "",
          f"Failure is defined in advance as PSNR < {failure_db:g} dB.", "",
          f"Wording. {WORDING_NOTE}", ""]
    if extra_records:
        md += ["Extra comparison runs merged by MNIST test index:", ""]
        for rec in extra_records:
            md.append(f"- `{rec['run_dir']}`: methods {', '.join(label(m) for m in rec['methods'])}; "
                      f"{rec['merged']} images merged, {rec['n_failed']} failed, devices {rec['devices']}")
        md.append("")
    integrity = {
        "heldout_images_with_dev_index": 0,
        "development_images_found": [r["mnist_test_index"] for r in sec["development"]],
        "other_images_found": [r["mnist_test_index"] for r in sec["other"]],
        "heldout_beyond_n": [r["mnist_test_index"] for r in sec["heldout_beyond_n"]],
        "runner_failures": [{"image_id": r.get("image_id"), "error": r.get("error")} for r in failed],
    }
    assert not (set(r["mnist_test_index"] for r in sec["heldout"]) & set(lists["dev"]))
    if sec["heldout"]:
        ranks = [r["_heldout_rank"] for r in sec["heldout"]]
        expected = args.n_heldout if args.n_heldout is not None else max(ranks) + 1
        integrity["heldout_missing_ranks"] = sorted(set(range(expected)) - set(ranks))
        integrity["heldout_missing_extra_methods"] = {
            m: [r["mnist_test_index"] for r in sec["heldout"] if m not in r["methods"]] for m in extras}
    versions = sorted({r.get("_unc_eval_version") for r in sec["heldout"] if r.get("_unc_eval_version")})
    integrity["uncertainty_eval_versions_heldout"] = versions
    summary["integrity"] = integrity
    if versions:
        md += [f"Pixel-wise uncertainty metrics: evaluation version {', '.join(map(str, versions))} "
               "(2 = u moved into the ground-truth frame before comparison with the error; "
               "1 = misregistered, superseded).", ""]
        if 1 in versions:
            print("WARNING: some held-out images still use uncertainty evaluation version 1; "
                  "run scripts/rescore_uncertainty.py on the run directory first.")

    print(f"Loaded {len(rows)} completed images ({len(failed)} failed): "
          + ", ".join(f"{k}={len(v)}" for k, v in sec.items() if v))
    for name, title in [("heldout", "Held-out evaluation (primary)"),
                        ("development", "Development / validation subset"),
                        ("smoke", "Smoke test")]:
        rs = sec[name]
        if not rs:
            continue
        present = [m for m in methods_all if any(_has(r, m) for r in rs)]
        sec_pairs = [(a, b) for a, b in pairs if a in present and b in present]
        res = analyse(rs, failure_db, auroc_min, present, sec_pairs)
        if name == "heldout":
            res["confirmatory"] = confirmatory_test(rs)
        write_tables(rs, res, out_root / "tables" / name, primary=(name == "heldout"))
        if not args.no_figures:
            try:
                make_figures(rs, out_root / "figures" / name, title, present)
            except Exception as e:
                print(f"  figures for {name} skipped: {e}")
        md.append(section_markdown(title, rs, res, notes[name]))
        res_json = {k: v for k, v in res.items() if k != "fe_psnr"}
        res_json["fe_psnr"] = {k: v for k, v in res["fe_psnr"].items() if k != "per_image"}
        res_json["mnist_test_indices"] = [r["mnist_test_index"] for r in rs]
        summary["sections"][name] = res_json
        p = res["methods"]
        print(f"\n[{name}] n={len(rs)}")
        for m in present:
            print(f"  {label(m):<40} PSNR {p[m]['psnr']['mean']:6.2f} ± {p[m]['psnr']['std']:.2f}  "
                  f"SSIM {p[m]['ssim']['mean']:.3f}  FE {p[m]['fe']['mean']:.3f}  "
                  f"fail {res['failure'][m]['n_failures']}/{res['failure'][m]['n']}")
        if name == "heldout":
            c = res["confirmatory"]
            print(f"  confirmatory test (ranks {c['ranks']}): {c['status']}"
                  + (f", primary {'SUPPORTED' if c['primary_pass'] else 'NOT SUPPORTED'}" if c.get("results") else ""))
    if not sec["development"]:
        md.append("## Development / validation subset\n\nNot included in this run directory. The development-stage "
                  "analysis (CPU, z_channels = 64) is in AGGREGATION_FINALIZATION_REPORT.md and is not directly "
                  "comparable to Stage-2 settings. Optionally run configs/stage2_mnist_dev8.yaml and pass "
                  "--dev-run-dir to report it here under identical settings.\n")
        summary["sections"]["development"] = {"note": "not in this run; see AGGREGATION_FINALIZATION_REPORT.md "
                                                      "(CPU, z_channels=64, not comparable)"}
    md.append(f"## Integrity checks\n\n```\n{json.dumps(integrity, indent=2)}\n```\n")
    (out_root / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out_root / "report_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote report.md, report_summary.json, tables/<section>/, figures/<section>/ in {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
