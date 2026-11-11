#!/usr/bin/env python
"""Aggregation finalization analysis (no retraining — operates on saved stacks).

Task 1: temperature sweep for FE-softmax.
Task 2: aggregation baseline comparison table.
Task 3: FE-PSNR relationship (pooled, within-image, agreement, plots).

Writes:
  results/temperature_sweep.csv
  results/aggregation_baselines.csv
  results/fe_psnr_analysis.csv
  figures/temperature_sweep.png
  figures/fe_psnr_scatter.png
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.uncertainty.ensemble_uq import (
    ensemble_aggregate, member_fourier_errors, fidelity_weights, select_anchor,
    effective_sample_size, pixelwise_median_complex, trimmed_mean_complex,
)
from pr_uq.uncertainty.fusion import fuse_uncertainty
from pr_uq.uncertainty.calibration import calibrate
from pr_uq.physics.projection import physics_residual
from pr_uq.metrics.alignment import align_ensemble_to_anchor, align_to_gt
from pr_uq.metrics.reconstruction import psnr, ssim, fourier_magnitude_error

DIR = REPO / "results" / "mnist_k5_3000iter" / "arrays"
K1_CSV = REPO / "results" / "mnist_k1_3000iter" / "metrics" / "per_image.csv"
NAMES = sorted(p.stem for p in DIR.glob("*.npz"))
OVERSAMPLE = 2


def load(name):
    b = np.load(DIR / f"{name}.npz")
    stack = torch.from_numpy(b["stack__real"] + 1j * b["stack__imag"]).to(torch.complex64)
    gt = torch.from_numpy(b["gt"]).to(torch.complex64)
    y = torch.from_numpy(b["measurement"]).float()
    return stack, gt, y


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2: return 0.0
    rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d > 1e-12 else 0.0


def uq_metrics(mean_c, std, gt, y):
    """PSNR/SSIM/FE + fused-uncertainty calibration for a given mean & std."""
    aligned = align_to_gt(mean_c, gt).aligned_recon
    p = psnr(aligned, gt); s = ssim(aligned, gt)
    fe = fourier_magnitude_error(mean_c, gt, oversample=OVERSAMPLE)
    r_phys = physics_residual(mean_c, y, oversample=OVERSAMPLE, return_image_domain=True)
    fused = fuse_uncertainty(std, r_phys, alpha=0.5, normalization="robust", summary="mean")
    err = (aligned.real - gt.real).abs()
    cal = calibrate(fused.u, err, n_bins=15)
    return dict(psnr=p, ssim=s, fe=fe, pearson=cal.pearson_r,
                spearman=cal.spearman_r, ece=cal.ece)


def load_k1():
    k1 = {}
    with open(K1_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k1[r["name"]] = dict(psnr=float(r["psnr"]), ssim=float(r["ssim"]),
                                 fe=float(r["fourier_mag_err"]))
    return k1


# ===================================================================
# TASK 1 — Temperature sweep
# ===================================================================
def task1_temperature_sweep():
    Ts = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    print("=" * 100)
    print("TASK 1 — FE-softmax temperature sweep (best-FE anchor, shared weights)")
    print("=" * 100)
    header = f"{'T':>6} | {'PSNR':>7} {'SSIM':>6} {'FE':>6} | {'Pearson':>8} {'Spearman':>8} {'ECE':>6} | {'K_eff':>6}"
    print(header); print("-" * len(header))
    rows = []
    for T in Ts:
        agg = {m: [] for m in ["psnr", "ssim", "fe", "pearson", "spearman", "ece", "keff"]}
        for name in NAMES:
            stack, gt, y = load(name)
            st = ensemble_aggregate(stack, y_intensity=y, anchor_method="best_fe",
                                    weight_method="softmax", temperature=T, oversample=OVERSAMPLE)
            m = uq_metrics(st.mean_complex, st.std_pixelwise, gt, y)
            for k in ["psnr", "ssim", "fe", "pearson", "spearman", "ece"]:
                agg[k].append(m[k])
            agg["keff"].append(effective_sample_size(st.weights))
        row = {k: float(np.mean(v)) for k, v in agg.items()}
        row["T"] = T
        rows.append(row)
        print(f"{T:>6.2f} | {row['psnr']:>7.2f} {row['ssim']:>6.3f} {row['fe']:>6.3f} | "
              f"{row['pearson']:>8.3f} {row['spearman']:>8.3f} {row['ece']:>6.3f} | {row['keff']:>6.2f}")

    with open(REPO / "results" / "temperature_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["T", "psnr", "ssim", "fe", "pearson", "spearman", "ece", "keff"])
        w.writeheader()
        for r in rows: w.writerow(r)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    Ts_ = [r["T"] for r in rows]
    ax1.plot(Ts_, [r["psnr"] for r in rows], "o-", color="#2f7d4f", label="PSNR (dB)")
    ax1.set_xlabel("softmax temperature T"); ax1.set_ylabel("PSNR (dB)", color="#2f7d4f")
    ax1.tick_params(axis="y", labelcolor="#2f7d4f")
    ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot(Ts_, [r["spearman"] for r in rows], "s--", color="#c8553d", label="Spearman(U,err)")
    ax2.plot(Ts_, [r["pearson"] for r in rows], "^--", color="#dd8452", label="Pearson(U,err)")
    ax2.plot(Ts_, [r["keff"]/5 for r in rows], "d:", color="#4c72b0", label="K_eff / 5")
    ax2.set_ylabel("correlation / normalized K_eff", color="#333")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=8)
    ax1.set_title("FE-softmax temperature sweep (8 MNIST images, 3000 iter)")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    out = REPO / "figures" / "temperature_sweep.png"
    fig.savefig(out, dpi=145, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nWrote {out}")
    return rows


# ===================================================================
# TASK 2 — Aggregation baseline comparison
# ===================================================================
def task2_baselines():
    print("\n" + "=" * 110)
    print("TASK 2 — Aggregation baseline comparison")
    print("=" * 110)
    k1 = load_k1()

    def method_k1(stack, gt, y):
        return None  # handled separately

    def m_uniform(stack, gt, y):
        st = ensemble_aggregate(stack, y_intensity=y, anchor_method="first",
                                weight_method="uniform", oversample=OVERSAMPLE)
        return st.mean_complex, st.std_pixelwise

    def m_median(stack, gt, y):
        anchor = select_anchor(stack, "best_fe", y, OVERSAMPLE)
        aligned = align_ensemble_to_anchor(stack, anchor_index=anchor)
        mean_c = pixelwise_median_complex(aligned)
        std = aligned.abs().std(dim=0, keepdim=True, unbiased=False)
        return mean_c, std

    def m_trimmed(stack, gt, y):
        anchor = select_anchor(stack, "best_fe", y, OVERSAMPLE)
        aligned = align_ensemble_to_anchor(stack, anchor_index=anchor)
        mean_c = trimmed_mean_complex(aligned, trim=1)
        std = aligned.abs().std(dim=0, keepdim=True, unbiased=False)
        return mean_c, std

    def m_best_single(stack, gt, y):
        fe = member_fourier_errors(stack, y, oversample=OVERSAMPLE)
        best = int(fe.argmin())
        mean_c = stack[best:best+1]
        # std of full aligned stack (uncertainty still from ensemble spread)
        anchor = select_anchor(stack, "best_fe", y, OVERSAMPLE)
        aligned = align_ensemble_to_anchor(stack, anchor_index=anchor)
        std = aligned.abs().std(dim=0, keepdim=True, unbiased=False)
        return mean_c, std

    def m_bestfe_uniform(stack, gt, y):
        st = ensemble_aggregate(stack, y_intensity=y, anchor_method="best_fe",
                                weight_method="uniform", oversample=OVERSAMPLE)
        return st.mean_complex, st.std_pixelwise

    def m_inverse(stack, gt, y):
        st = ensemble_aggregate(stack, y_intensity=y, anchor_method="best_fe",
                                weight_method="inverse", oversample=OVERSAMPLE)
        return st.mean_complex, st.std_pixelwise

    def m_softmax(stack, gt, y):
        st = ensemble_aggregate(stack, y_intensity=y, anchor_method="best_fe",
                                weight_method="softmax", temperature=0.1, oversample=OVERSAMPLE)
        return st.mean_complex, st.std_pixelwise

    def m_topk(stack, gt, y):
        st = ensemble_aggregate(stack, y_intensity=y, anchor_method="best_fe",
                                weight_method="top_k", top_k=3, oversample=OVERSAMPLE)
        return st.mean_complex, st.std_pixelwise

    def m_medoid(stack, gt, y):
        st = ensemble_aggregate(stack, y_intensity=y, anchor_method="medoid",
                                weight_method="uniform", oversample=OVERSAMPLE)
        # medoid reconstruction = the medoid member itself
        med = select_anchor(stack, "medoid", y, OVERSAMPLE)
        aligned = align_ensemble_to_anchor(stack, anchor_index=med)
        return aligned[med:med+1], aligned.abs().std(dim=0, keepdim=True, unbiased=False)

    methods = [
        ("K=1", None),
        ("Uniform mean", m_uniform),
        ("Pixel-wise median", m_median),
        ("Trimmed mean (trim=1)", m_trimmed),
        ("Best-single-by-FE", m_best_single),
        ("Best-FE anchor + uniform", m_bestfe_uniform),
        ("Inverse-FE weighting", m_inverse),
        ("FE-softmax (T=0.1)", m_softmax),
        ("Top-3 FE", m_topk),
        ("Medoid", m_medoid),
    ]

    header = f"{'method':<28} | {'PSNR':>7} {'medPSNR':>7} {'SSIM':>6} {'FE':>6} | {'Pearson':>8} {'Spearman':>8} {'ECE':>6}"
    print(header); print("-" * len(header))
    rows = []
    for mname, fn in methods:
        if mname == "K=1":
            P = [k1[n]["psnr"] for n in NAMES]; S = [k1[n]["ssim"] for n in NAMES]; FE = [k1[n]["fe"] for n in NAMES]
            row = dict(method=mname, psnr=np.mean(P), medpsnr=np.median(P), ssim=np.mean(S),
                       fe=np.mean(FE), pearson=float("nan"), spearman=float("nan"), ece=float("nan"))
        else:
            agg = {m: [] for m in ["psnr", "ssim", "fe", "pearson", "spearman", "ece"]}
            for name in NAMES:
                stack, gt, y = load(name)
                mean_c, std = fn(stack, gt, y)
                m = uq_metrics(mean_c, std, gt, y)
                for k in agg: agg[k].append(m[k])
            row = dict(method=mname, psnr=np.mean(agg["psnr"]), medpsnr=np.median(agg["psnr"]),
                       ssim=np.mean(agg["ssim"]), fe=np.mean(agg["fe"]),
                       pearson=np.mean(agg["pearson"]), spearman=np.mean(agg["spearman"]),
                       ece=np.mean(agg["ece"]))
        rows.append(row)
        pe = "  n/a  " if np.isnan(row["pearson"]) else f"{row['pearson']:>8.3f}"
        sp = "  n/a  " if np.isnan(row["spearman"]) else f"{row['spearman']:>8.3f}"
        ec = " n/a " if np.isnan(row["ece"]) else f"{row['ece']:>6.3f}"
        print(f"{row['method']:<28} | {row['psnr']:>7.2f} {row['medpsnr']:>7.2f} {row['ssim']:>6.3f} "
              f"{row['fe']:>6.3f} | {pe} {sp} {ec}")

    with open(REPO / "results" / "aggregation_baselines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "psnr", "medpsnr", "ssim", "fe", "pearson", "spearman", "ece"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nWrote {REPO/'results'/'aggregation_baselines.csv'}")
    return rows


# ===================================================================
# TASK 3 — FE-PSNR relationship
# ===================================================================
def task3_fe_psnr():
    print("\n" + "=" * 100)
    print("TASK 3 — FE vs PSNR relationship")
    print("=" * 100)
    all_fe, all_psnr, all_img = [], [], []
    within_rho = []
    top1_hits, top2_hits = 0, 0
    per_image = []

    for name in NAMES:
        stack, gt, y = load(name)
        fe = member_fourier_errors(stack, y, oversample=OVERSAMPLE).numpy()
        ps = np.array([psnr(align_to_gt(stack[k:k+1], gt).aligned_recon, gt) for k in range(stack.shape[0])])
        rho = spearman(-fe, ps)
        within_rho.append(rho)
        all_fe += fe.tolist(); all_psnr += ps.tolist(); all_img += [name]*len(fe)

        fe_order = fe.argsort()       # ascending FE (best first)
        psnr_order = (-ps).argsort()  # descending PSNR (best first)
        top1 = fe_order[0] == psnr_order[0]
        # Top-2: is the best-FE member among the top-2 PSNR members?
        top2 = fe_order[0] in psnr_order[:2]
        top1_hits += top1; top2_hits += top2
        per_image.append(dict(image=name, within_rho=rho, top1=int(top1), top2=int(top2),
                              best_fe_member=int(fe_order[0]), best_psnr_member=int(psnr_order[0])))

    pooled = spearman(-np.array(all_fe), np.array(all_psnr))
    print(f"Pooled member-level Spearman rho(-FE, PSNR) over {len(all_fe)} members: {pooled:+.3f}")
    print(f"Mean within-image Spearman rho: {np.mean(within_rho):+.3f}  "
          f"(range {min(within_rho):+.2f} to {max(within_rho):+.2f})")
    print(f"Within-image rho per image: {[f'{r:+.2f}' for r in within_rho]}")
    print(f"Images with negative within-image rho: {sum(1 for r in within_rho if r < 0)}/{len(within_rho)}")
    print(f"\nBest-FE == Best-PSNR (Top-1 agreement): {top1_hits}/{len(NAMES)} = {top1_hits/len(NAMES)*100:.0f}%")
    print(f"Best-FE within Top-2 PSNR (Top-2 agreement): {top2_hits}/{len(NAMES)} = {top2_hits/len(NAMES)*100:.0f}%")

    with open(REPO / "results" / "fe_psnr_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "within_rho", "top1", "top2", "best_fe_member", "best_psnr_member"])
        w.writeheader()
        for r in per_image: w.writerow(r)
        f.write(f"# pooled_spearman={pooled:.4f}\n")
        f.write(f"# mean_within_rho={np.mean(within_rho):.4f}\n")

    # Scatter: per-image panels
    n = len(NAMES); cols = 4; rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(3.2*cols, 3.0*rows_))
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(NAMES):
        stack, gt, y = load(name)
        fe = member_fourier_errors(stack, y, oversample=OVERSAMPLE).numpy()
        ps = np.array([psnr(align_to_gt(stack[k:k+1], gt).aligned_recon, gt) for k in range(stack.shape[0])])
        ax = axes[i]
        ax.scatter(fe, ps, c=range(len(fe)), cmap="viridis", s=60, zorder=3)
        for k in range(len(fe)):
            ax.annotate(str(k), (fe[k], ps[k]), fontsize=7, ha="center", va="center", color="white")
        ax.set_title(f"{name.replace('mnist_','')}  ρ={within_rho[i]:+.2f}", fontsize=9)
        ax.set_xlabel("FE", fontsize=8); ax.set_ylabel("PSNR (dB)", fontsize=8)
        ax.grid(alpha=0.3)
    for j in range(n, len(axes)): axes[j].axis("off")
    fig.suptitle(f"Per-image FE vs PSNR (member index labelled). Pooled ρ={pooled:+.2f}, mean within-image ρ={np.mean(within_rho):+.2f}",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    out = REPO / "figures" / "fe_psnr_scatter.png"
    fig.savefig(out, dpi=145, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nWrote {out}")
    return dict(pooled=pooled, mean_within=float(np.mean(within_rho)), top1=top1_hits, top2=top2_hits, n=len(NAMES))


if __name__ == "__main__":
    t1 = task1_temperature_sweep()
    t2 = task2_baselines()
    t3 = task3_fe_psnr()
    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)
