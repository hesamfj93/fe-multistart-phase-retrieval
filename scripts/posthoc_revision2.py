"""Post-hoc analyses added in the second pre-submission revision (2026-11-11).

All analyses use the stored starts and aggregates of the held-out run; nothing is retrained,
and no frozen decision depends on them.

1. Complex-field accuracy of the five aggregation rules. The primary metrics compare the real
   part of the ground-truth-aligned reconstruction with the real object. Here the aligned
   complex reconstruction is compared with the object:
       NRMSE_complex = ||x_hat - x|| / ||x||,  NRMSE_real = ||Re x_hat - x|| / ||x||,
       imaginary energy fraction = ||Im x_hat||^2 / ||x_hat||^2,
       negative real-part energy fraction = ||min(Re x_hat, 0)||^2 / ||x_hat||^2.
2. A median-absolute-deviation (MAD) rejection baseline in the spirit of Hong et al. (2026),
   adapted to aligned phase-retrieval starts (see `mad_reject`). Its rule and threshold are
   fixed in this file; they were not tuned on held-out data.
3. Runtime summary from the stored metrics.
4. The full Holm family of the frozen analysis plan, exported for the supplement.

    python scripts/posthoc_revision2.py --run-dir results/stage2_mnist_heldout \
        --k1long results/stage2_mnist_heldout_k1long --baselines results/stage2_mnist_heldout_baselines
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.metrics.alignment import align_ensemble_to_anchor  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors, select_anchor  # noqa: E402

METHODS = ["k1", "uniform", "median", "best_single_by_fe", "fe_softmax"]
LABEL = {"k1": "K=1", "uniform": "uniform", "median": "median", "best_single_by_fe": "best-single-by-FE",
         "fe_softmax": "FE-softmax", "mad": "MAD image-level", "madpix": "MAD pixel-wise"}

# ---- MAD rule, fixed before it was run on held-out data -------------------------------------
MAD_C = 3.0          # rejection threshold in robust standard deviations
MAD_SCALE = 1.4826   # MAD -> standard deviation for Gaussian data


def mad_reject(aligned: torch.Tensor, fe: torch.Tensor, c: float = MAD_C) -> tuple[torch.Tensor, np.ndarray]:
    """Image-level MAD rejection on the aligned stack, then a uniform mean of the kept starts.

    Each start k gets the distance d_k = ||x_k - med||_2 to the pixel-wise median `med` of the
    aligned stack (real and imaginary parts separately). A start is rejected when
    d_k > median(d) + c * 1.4826 * MAD(d). If MAD(d) = 0, no start is rejected. The kept starts
    are averaged with equal weights. FE is not used by the rule.
    """
    re = aligned.real.median(dim=0, keepdim=True).values
    im = aligned.imag.median(dim=0, keepdim=True).values
    med = torch.complex(re, im)
    d = (aligned - med).abs().pow(2).sum(dim=(1, 2, 3)).sqrt().cpu().numpy()
    md = np.median(d)
    mad = np.median(np.abs(d - md))
    keep = d <= md + c * MAD_SCALE * mad if mad > 0 else np.ones_like(d, dtype=bool)
    kept = aligned[torch.from_numpy(keep)]
    return kept.mean(dim=0, keepdim=True), keep


def mad_reject_pixel(aligned: torch.Tensor, c: float = MAD_C) -> tuple[torch.Tensor, float]:
    """Pixel-wise MAD rejection: at each pixel, starts whose complex value lies farther than
    c * 1.4826 * MAD from the pixel-wise median are excluded, and the rest are averaged.
    Where MAD = 0, all starts are kept. Returns the estimate and the fraction of rejected values."""
    med = torch.complex(aligned.real.median(dim=0, keepdim=True).values, aligned.imag.median(dim=0, keepdim=True).values)
    dev = (aligned - med).abs()
    mad = dev.median(dim=0, keepdim=True).values
    keep = (dev <= c * MAD_SCALE * mad) | (mad <= 0)
    kf = keep.to(aligned.real.dtype)
    est = (aligned * kf).sum(dim=0, keepdim=True) / kf.sum(dim=0, keepdim=True).clamp_min(1)
    return est, float(1 - kf.mean())


def _runner():
    spec = importlib.util.spec_from_file_location("runner_ph2", REPO / "scripts" / "run_phase_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _complex(npz, key):
    return npz[f"{key}__real"].astype(np.float64) + 1j * npz[f"{key}__imag"].astype(np.float64)


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    nz = d[np.abs(d) > 1e-9]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    return {"mean": float(d.mean()), "ci": [float(d.mean() - t * se), float(d.mean() + t * se)],
            "median": float(np.median(d)), "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()),
            "equal": int((np.abs(d) <= 1e-9).sum()), "wilcoxon_p_uncorrected": p}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--k1long", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--mad", action="store_true", help="also run the MAD rejection baseline")
    args = ap.parse_args()
    run = Path(args.run_dir)
    import yaml
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    T = float(cfg["uncertainty"].get("fidelity_temperature", 0.05))
    ov = int(cfg["measurement"].get("oversample", 2))
    nb = int(cfg["uncertainty"].get("n_bins_ece", 15))
    R = _runner()

    dirs = []
    for mj in sorted((run / "per_image").glob("image_*/metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        if m.get("status") == "ok" and int(m.get("split_rank", 10**9)) < 100:
            dirs.append((int(m["split_rank"]), mj.parent, m))
    dirs.sort()
    n = len(dirs)

    cx = {k: {"nrmse_complex": [], "nrmse_real": [], "imag_frac": [], "negreal_frac": []} for k in METHODS + ["mad", "madpix"]}
    psnr = {k: [] for k in METHODS + ["mad", "madpix"]}
    ssim = {k: [] for k in METHODS + ["mad", "madpix"]}
    rej_pix = []
    mad_kept, post_times = [], []
    for _, d, m in dirs:
        a = np.load(d / "aggregates_v2.npz")
        b = np.load(d / "members.npz")
        x = b["gt"].astype(np.float64)
        nx = np.linalg.norm(x)
        for k in METHODS:
            xh = _complex(a, f"recon_{k}")
            cx[k]["nrmse_complex"].append(np.linalg.norm(xh - x) / nx)
            cx[k]["nrmse_real"].append(np.linalg.norm(xh.real - x) / nx)
            e = np.sum(np.abs(xh) ** 2)
            cx[k]["imag_frac"].append(np.sum(xh.imag ** 2) / e)
            cx[k]["negreal_frac"].append(np.sum(np.minimum(xh.real, 0) ** 2) / e)
            psnr[k].append(min(float(m["methods"][k]["psnr"]), 60.0))
            ssim[k].append(float(m["methods"][k]["ssim"]))

        # MAD baseline and post-processing time of the frozen pipeline (CPU)
        stack = torch.from_numpy(_complex(b, "stack_raw")).to(torch.complex64)
        y = torch.from_numpy(b["measurement"]).float()
        gt = torch.from_numpy(b["gt"]).float().to(torch.complex64)
        t0 = time.perf_counter()
        fe = member_fourier_errors(stack, y, oversample=ov)
        anchor = select_anchor(stack, method="best_fe", y_intensity=y, oversample=ov)
        aligned = align_ensemble_to_anchor(stack, anchor_index=anchor)
        R.compute_strategies(stack, y, fe, anchor, aligned, T, ov)
        post_times.append(time.perf_counter() - t0)
        if not args.mad:
            continue
        mean_c, keep = mad_reject(aligned, fe)
        mad_kept.append(int(keep.sum()))
        pix_c, rj = mad_reject_pixel(aligned)
        rej_pix.append(rj)
        for name, est in (("mad", mean_c), ("madpix", pix_c)):
            std = aligned.abs().std(dim=0, keepdim=True, unbiased=False)
            rec, al, _, _, _ = R.score_reconstruction(est, std, gt, y, ov, nb)
            psnr[name].append(min(float(rec["psnr"]), 60.0))
            ssim[name].append(float(rec["ssim"]))
            xh = al.detach().cpu().numpy().astype(np.complex128)
            cx[name]["nrmse_complex"].append(np.linalg.norm(xh - x) / nx)
            cx[name]["nrmse_real"].append(np.linalg.norm(xh.real - x) / nx)
            e = np.sum(np.abs(xh) ** 2)
            cx[name]["imag_frac"].append(np.sum(xh.imag ** 2) / e)
            cx[name]["negreal_frac"].append(np.sum(np.minimum(xh.real, 0) ** 2) / e)

    P = {k: np.array(v) for k, v in psnr.items()}
    S = {k: np.array(v) for k, v in ssim.items()}
    out = {"n": n, "mad_rule": {"statistic": "L2 distance of each aligned start to the pixel-wise median of the stack",
                                "threshold": f"median(d) + {MAD_C} * {MAD_SCALE} * MAD(d)", "combination": "uniform mean of kept starts"}}
    if not args.mad:
        for k in ("mad", "madpix"):
            cx.pop(k); P.pop(k); S.pop(k)
    out["complex_metrics"] = {k: {kk: {"mean": float(np.mean(v)), "median": float(np.median(v))} for kk, v in cx[k].items()}
                              for k in cx}
    if args.mad:
      out["mad_summary"] = {
        "psnr_mean": float(P["mad"].mean()), "psnr_median": float(np.median(P["mad"])), "ssim_mean": float(S["mad"].mean()),
        "failures": int((P["mad"] < 15).sum()), "kept_histogram": {str(i): int((np.array(mad_kept) == i).sum()) for i in range(1, 6)},
        "images_with_rejection": int((np.array(mad_kept) < 5).sum())}
      out["madpix_summary"] = {"psnr_mean": float(P["madpix"].mean()), "psnr_median": float(np.median(P["madpix"])),
                               "ssim_mean": float(S["madpix"].mean()), "failures": int((P["madpix"] < 15).sum()),
                               "rejected_value_fraction_mean": float(np.mean(rej_pix))}
      out["mad_paired_psnr"] = {}
      for nm in ("mad", "madpix"):
          for k in METHODS:
              out["mad_paired_psnr"][f"{nm}_vs_{k}"] = paired(P[nm], P[k])
          out["mad_paired_psnr"][f"fe_softmax_vs_{nm}"] = paired(P["fe_softmax"], P[nm])
          out["mad_paired_psnr"][f"best_single_vs_{nm}"] = paired(P["best_single_by_fe"], P[nm])
    out["paired_nrmse_complex"] = {
        "fe_softmax_minus_k1": paired(np.array(cx["fe_softmax"]["nrmse_complex"]), np.array(cx["k1"]["nrmse_complex"])),
        "best_single_minus_k1": paired(np.array(cx["best_single_by_fe"]["nrmse_complex"]), np.array(cx["k1"]["nrmse_complex"])),
        "fe_softmax_minus_uniform": paired(np.array(cx["fe_softmax"]["nrmse_complex"]), np.array(cx["uniform"]["nrmse_complex"]))}
    out["postprocess_time_cpu_s"] = {"mean": float(np.mean(post_times)), "max": float(np.max(post_times))}

    # runtimes from stored metrics
    rt = {"k5_per_start_mean_s": float(np.mean([np.mean(m["runtime_per_member_s"]) for _, _, m in dirs])),
          "k5_total_mean_s": float(np.mean([m["runtime_total_s"] for _, _, m in dirs])),
          "k5_postprocess_mean_s": float(np.mean([m["runtime_postprocess_s"] for _, _, m in dirs]))}
    k1l = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.k1long).glob("per_image/image_*/metrics.json"))]
    rt["k1long_total_mean_s"] = float(np.mean([m["runtime_total_s"] for m in k1l if m.get("status") == "ok"]))
    out["runtime"] = rt

    # Holm family of the frozen plan, as reported
    rows = [r for r in csv.DictReader(open(run / "report_n100/tables/heldout/paired_comparison.csv", encoding="utf-8"))
            if r["metric"] == "psnr"]
    p = np.array([float(r["wilcoxon_p_uncorrected"]) for r in rows])
    order = np.argsort(p, kind="stable")
    adj = np.empty(len(p))
    run_max = 0.0
    for i, idx in enumerate(order):
        run_max = max(run_max, min(1.0, (len(p) - i) * p[idx]))
        adj[idx] = run_max
    fam = []
    for r, a_ in zip(rows, adj):
        fam.append({"method": r["method"], "baseline": r["baseline"], "mean": float(r["mean_delta"]),
                    "ci": [float(r["ci95_low"]), float(r["ci95_high"])], "better": int(r["n_better"]), "worse": int(r["n_worse"]),
                    "p_uncorrected": float(r["wilcoxon_p_uncorrected"]), "p_holm_reported": float(r["wilcoxon_p_holm"]),
                    "p_holm_recomputed": float(a_)})
    out["holm_family"] = fam
    out["holm_max_abs_diff"] = float(np.max(np.abs(adj - np.array([float(r["wilcoxon_p_holm"]) for r in rows]))))

    dst = run / "report_n100/posthoc/posthoc_revision2.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    with open(run / "report_n100/posthoc/holm_family.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "baseline", "mean_dB", "ci_low", "ci_high", "better", "worse", "p_uncorrected", "p_holm"])
        for r in fam:
            w.writerow([r["method"], r["baseline"], f"{r['mean']:.2f}", f"{r['ci'][0]:.2f}", f"{r['ci'][1]:.2f}",
                        r["better"], r["worse"], f"{r['p_uncorrected']:.2g}", f"{r['p_holm_recomputed']:.2g}"])
    print(json.dumps({k: out[k] for k in ("n", "mad_summary", "madpix_summary", "runtime", "postprocess_time_cpu_s", "holm_max_abs_diff") if k in out}, indent=1))
    for k, v in out["complex_metrics"].items():
        print(f"{LABEL[k]:22s} " + "  ".join(f"{kk} {vv['mean']:.4f}/{vv['median']:.4f}" for kk, vv in v.items()))
    for k, v in out.get("mad_paired_psnr", {}).items():
        print(f"{k:24s} {v['mean']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] {v['better']}/{v['worse']}/{v['equal']} p={v['wilcoxon_p_uncorrected']:.2g}")
    for k, v in out["paired_nrmse_complex"].items():
        print(f"{k:24s} {v['mean']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['wilcoxon_p_uncorrected']:.2g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
