"""Pre-specified analysis of the stage-3 experiment (complex-valued objects).

Implements the analysis fixed in STAGE3_COMPLEX_PLAN.md. Nothing here may be changed after the
plan is tagged, except to fix a software error, which must be listed as a deviation.

    python scripts/stage3_report.py --main results/stage3_complex_heldout \
        --k1long results/stage3_complex_heldout_k1long \
        --baselines results/stage3_complex_heldout_baselines \
        --out results/stage3_complex_heldout/report

For the development check, omit --k1long; the hypothesis tests are then skipped.

Outputs: summary.json, methods.csv, hypotheses.csv and report.md in --out.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

CAP = 60.0          # PSNR cap in summaries (dB); values at or above it count as exact
FAIL = 15.0         # failure threshold (dB)
N_BOOT = 10_000
BOOT_SEED = 20261111

# The Holm family: (id, first method, second method). Fixed in the plan.
FAMILY = [
    ("C1", "fe_softmax", "k1"),
    ("C2", "best_single_by_fe", "k1"),
    ("C3", "fe_softmax", "k1_long"),
    ("C4", "best_single_by_fe", "k1_long"),
    ("C5", "fe_softmax", "hio_supp_best5"),
    ("C6", "fe_softmax", "raar_supp_best5"),
    ("C7", "fe_softmax", "hio_sw_best5"),
    ("C8", "fe_softmax", "raar_sw_best5"),
]
EXTRA_FIELDS = ("psnr_amp", "phase_err_rad", "nrmse_c", "psnr_real_part")


def load_run(run_dir: Path) -> dict:
    out = {}
    for mj in sorted((run_dir / "per_image").glob("image_*/metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        if m.get("status") == "ok":
            out[int(m["mnist_test_index"])] = m
    return out


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    adj, run = np.empty(len(p)), 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[idx]))
        adj[idx] = run
    return adj


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    nz = d[np.abs(d) > 1e-9]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    return {"n": n, "mean": float(d.mean()), "ci": [float(d.mean() - t * se), float(d.mean() + t * se)],
            "median": float(np.median(d)), "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()),
            "equal": int((np.abs(d) <= 1e-9).sum()), "p_uncorrected": p}


def boot_mean_ci(x: np.ndarray, rng) -> list[float]:
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--k1long")
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)

    main_run, bl_run = load_run(Path(a.main)), load_run(Path(a.baselines))
    k1l_run = load_run(Path(a.k1long)) if a.k1long else {}
    ids = sorted(set(main_run) & set(bl_run) & (set(k1l_run) if a.k1long else set(main_run)))
    n = len(ids)
    if n == 0:
        raise SystemExit("no image is complete in all runs")

    # ---- per-method values -------------------------------------------------------------
    rows = {}
    def put(name, rec_list, runtime):
        ps = np.array([min(r["psnr"], 100.0) for r in rec_list])
        rows[name] = {"psnr": np.minimum(ps, CAP), "psnr_raw": ps, "ssim": np.array([r["ssim"] for r in rec_list]),
                      "fe": np.array([r.get("fe", np.nan) for r in rec_list]), "runtime": runtime}
        for f in EXTRA_FIELDS:
            rows[name][f] = np.array([r.get(f, np.nan) for r in rec_list], dtype=float)
    for k in ["k1", "uniform", "median", "best_single_by_fe", "fe_softmax"]:
        put(k, [main_run[i]["methods"][k] for i in ids],
            float(np.mean([main_run[i].get("runtime_total_s", np.nan) for i in ids])) if k != "k1" else
            float(np.mean([main_run[i]["runtime_per_member_s"][0] for i in ids])))
    if a.k1long:
        put("k1_long", [k1l_run[i]["methods"]["k1_long"] for i in ids],
            float(np.mean([k1l_run[i].get("runtime_total_s", np.nan) for i in ids])))
    for k in bl_run[ids[0]]["methods"]:
        put(k, [bl_run[i]["methods"][k] for i in ids],
            float(np.mean([bl_run[i]["methods"][k].get("runtime_s", np.nan) for i in ids])))

    table = []
    for k, v in rows.items():
        table.append({"method": k, "n": n, "psnr_mean": float(v["psnr"].mean()), "psnr_std": float(v["psnr"].std(ddof=1)),
                      "psnr_median": float(np.median(v["psnr"])), "psnr_ci": boot_mean_ci(v["psnr"], rng),
                      "psnr_amp_mean": float(np.nanmean(np.minimum(v["psnr_amp"], CAP))),
                      "phase_err_mean": float(np.nanmean(v["phase_err_rad"])), "nrmse_c_mean": float(np.nanmean(v["nrmse_c"])),
                      "ssim_amp_mean": float(v["ssim"].mean()), "fe_mean": float(np.nanmean(v["fe"])),
                      "failures": int((v["psnr"] < FAIL).sum()), "exact": int((v["psnr_raw"] >= CAP).sum()),
                      "runtime_s": v["runtime"]})

    # ---- hypotheses ----------------------------------------------------------------------
    hyp = []
    if a.k1long:
        res = [paired(rows[m1]["psnr"], rows[m2]["psnr"]) for _, m1, m2 in FAMILY]
        adj = holm(np.array([r["p_uncorrected"] for r in res]))
        for (hid, m1, m2), r, pa in zip(FAMILY, res, adj):
            r.update({"id": hid, "first": m1, "second": m2, "p_holm": float(pa),
                      "supported": bool(r["mean"] > 0 and r["ci"][0] > 0 and pa < 0.05)})
            hyp.append(r)
    # C9: image-level reliability of FE, one-sided Spearman(FE of the fused reconstruction, -PSNR) > 0
    fe_f, ps_f = rows["fe_softmax"]["fe"], rows["fe_softmax"]["psnr"]
    rho, p2 = stats.spearmanr(fe_f, -ps_f)
    p1 = p2 / 2 if rho > 0 else 1 - p2 / 2
    bs = []
    for _ in range(N_BOOT):
        j = rng.integers(0, n, n)
        bs.append(stats.spearmanr(fe_f[j], -ps_f[j])[0])
    c9 = {"id": "C9", "rho": float(rho), "p_one_sided": float(p1),
          "boot_ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
          "supported": bool(rho > 0 and p1 < 0.05)}

    # ---- descriptive: FE at the start level ------------------------------------------------
    within, top1, top2 = [], 0, 0
    for i in ids:
        fe = np.array(main_run[i]["member_fe"], float)
        ps = np.array(main_run[i]["member_psnr"], float)
        if np.ptp(ps) > 0 and np.ptp(fe) > 0:
            within.append(stats.spearmanr(-fe, ps)[0])
        order = np.argsort(-ps)
        top1 += int(np.argmin(fe) == order[0])
        top2 += int(np.argmin(fe) in order[:2])
    start = {"within_image_spearman_mean": float(np.nanmean(within)), "within_image_spearman_median": float(np.nanmedian(within)),
             "top1": top1, "top2": top2, "n": n}
    unc = {k: float(np.nanmean([main_run[i]["methods"][k].get("unc_spearman", np.nan) for i in ids]))
           for k in ["uniform", "median", "fe_softmax"]}

    summary = {"n": n, "images": ids, "methods": table, "hypotheses": hyp, "C9": c9, "fe_start_level": start,
               "pixelwise_unc_spearman_mean": unc, "cap_db": CAP, "failure_db": FAIL}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / "methods.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n", "psnr_mean", "psnr_std", "psnr_median", "ci_low", "ci_high", "psnr_amp", "phase_err_rad",
                    "nrmse_c", "ssim_amp", "fe", "failures", "exact", "runtime_s"])
        for r in table:
            w.writerow([r["method"], r["n"], f"{r['psnr_mean']:.2f}", f"{r['psnr_std']:.2f}", f"{r['psnr_median']:.2f}",
                        f"{r['psnr_ci'][0]:.2f}", f"{r['psnr_ci'][1]:.2f}", f"{r['psnr_amp_mean']:.2f}", f"{r['phase_err_mean']:.3f}",
                        f"{r['nrmse_c_mean']:.3f}", f"{r['ssim_amp_mean']:.3f}", f"{r['fe_mean']:.4f}", r["failures"], r["exact"],
                        f"{r['runtime_s']:.1f}"])
    if hyp:
        with open(out / "hypotheses.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "first", "second", "n", "mean_dB", "ci_low", "ci_high", "better", "worse", "p", "p_holm", "supported"])
            for r in hyp:
                w.writerow([r["id"], r["first"], r["second"], r["n"], f"{r['mean']:.2f}", f"{r['ci'][0]:.2f}", f"{r['ci'][1]:.2f}",
                            r["better"], r["worse"], f"{r['p_uncorrected']:.2g}", f"{r['p_holm']:.2g}", r["supported"]])
    lines = [f"# Stage 3 report ({n} images)", "", "| method | PSNR (complex) | median | 95% CI | amp. PSNR | phase err. | failures | exact |",
             "|---|--:|--:|---|--:|--:|--:|--:|"]
    for r in sorted(table, key=lambda r: -r["psnr_mean"]):
        lines.append(f"| {r['method']} | {r['psnr_mean']:.2f} ± {r['psnr_std']:.2f} | {r['psnr_median']:.2f} | "
                     f"[{r['psnr_ci'][0]:.2f}, {r['psnr_ci'][1]:.2f}] | {r['psnr_amp_mean']:.2f} | {r['phase_err_mean']:.3f} | "
                     f"{r['failures']} | {r['exact']} |")
    if hyp:
        lines += ["", "| id | comparison | mean (dB) | 95% CI | better/worse | p (Holm) | supported |", "|---|---|--:|---|--:|--:|---|"]
        for r in hyp:
            lines.append(f"| {r['id']} | {r['first']} vs {r['second']} | {r['mean']:+.2f} | [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] | "
                         f"{r['better']}/{r['worse']} | {r['p_holm']:.3g} | {r['supported']} |")
    lines += ["", f"C9: Spearman(FE of fused, -PSNR) = {c9['rho']:.3f}, one-sided p = {c9['p_one_sided']:.2g}, "
              f"bootstrap [{c9['boot_ci'][0]:.2f}, {c9['boot_ci'][1]:.2f}], supported = {c9['supported']}",
              f"FE at the start level: within-image Spearman mean {start['within_image_spearman_mean']:.2f}, "
              f"Top-1 {top1}/{n}, Top-2 {top2}/{n}"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
