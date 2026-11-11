#!/usr/bin/env python
"""Figures for the manuscript, re-plotted from stored results. Nothing is retrained.

Development stage (8 development images, z_channels = 64, CPU):
    fig_dev_example.pdf       members, aggregates and uncertainty for development image 2890
    fig_dev_temperature.pdf   temperature sweep (PSNR and effective ensemble size)

Held-out evaluation (ranks 0-99), after scripts/rescore_uncertainty.py has been run:
    fig_heldout_paired.pdf    mean paired PSNR differences with 95% CI
    fig_heldout_reliability.pdf  FE vs PSNR over all starts; failure score vs PSNR
    fig_heldout_examples.pdf  worst, quartile and best images by FE-softmax PSNR

Run from the repository root:
    python paper/make_paper_figures.py
    python paper/make_paper_figures.py --heldout results/stage2_mnist_heldout \
        --k1long results/stage2_mnist_heldout_k1long --baselines results/stage2_mnist_heldout_baselines
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.metrics.alignment import align_ensemble_to_anchor, align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import psnr  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors  # noqa: E402

OUT = REPO / "paper" / "figures"
INK = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.labelcolor": INK,
    "pdf.fonttype": 42,
})


def _runner():
    spec = importlib.util.spec_from_file_location("runner_fig", REPO / "scripts" / "run_phase_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fig_dev_example(name: str = "mnist_002890") -> None:
    R = _runner()
    b = np.load(REPO / "results" / "mnist_k5_3000iter" / "arrays" / f"{name}.npz")
    stack = torch.from_numpy(b["stack__real"] + 1j * b["stack__imag"]).to(torch.complex64)
    gt = torch.from_numpy(b["gt"]).to(torch.complex64)
    y = torch.from_numpy(b["measurement"]).float()
    K = stack.shape[0]

    fe = member_fourier_errors(stack, y, oversample=2)
    anchor = int(fe.argmin())
    aligned = align_ensemble_to_anchor(stack, anchor_index=anchor)
    strategies, best = R.compute_strategies(stack, y, fe, anchor, aligned, 0.05, 2)

    member_psnr = [psnr(align_to_gt(stack[k:k + 1], gt).aligned_recon, gt) for k in range(K)]

    show = [("uniform", "uniform mean"), ("best_single_by_fe", "best-single-by-FE"),
            ("fe_softmax", "FE-softmax")]
    agg = []
    for key, label in show:
        mean_c, std = strategies[key]
        # score_reconstruction returns u already in the ground-truth frame
        rec, aligned_gt, u, _, _ = R.score_reconstruction(mean_c, std, gt, y, 2)
        agg.append((label, aligned_gt.real.squeeze().numpy(), rec["psnr"]))
        if key == "fe_softmax":
            u_map = u.squeeze().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(7.16, 3.35))
    for k in range(K):
        ax = axes[0, k]
        ax.imshow(np.abs(stack[k].squeeze().numpy()), cmap="gray", vmin=0, vmax=1)
        tag = " (lowest FE)" if k == anchor else ""
        ax.set_title(f"member {k}{tag}\nFE {float(fe[k]):.3f}, {member_psnr[k]:.1f} dB", fontsize=7.5)
    for j, (label, img, p) in enumerate(agg):
        ax = axes[1, j]
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{label}\n{p:.1f} dB", fontsize=7.5)
    ax = axes[1, 3]
    im = ax.imshow(u_map, cmap="magma", vmin=0, vmax=1)
    ax.set_title("uncertainty $u$\n(FE-softmax)", fontsize=7.5)
    cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    ax = axes[1, 4]
    ax.imshow(gt.real.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title("ground truth\n", fontsize=7.5)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    axes[0, 0].set_ylabel("raw members", fontsize=8)
    axes[1, 0].set_ylabel("aggregates", fontsize=8)
    fig.subplots_adjust(left=0.04, right=0.975, top=0.86, bottom=0.02, wspace=0.28, hspace=0.42)
    fig.savefig(OUT / "fig_dev_example.pdf")
    fig.savefig(OUT / "fig_dev_example.png", dpi=200)
    plt.close(fig)
    print("member FE", [round(float(v), 3) for v in fe], "PSNR", [round(v, 1) for v in member_psnr])
    print("aggregates", [(a, round(p, 2)) for a, _, p in agg], "anchor", anchor, "best", best)


def fig_dev_temperature() -> None:
    rows = list(csv.DictReader(open(REPO / "results" / "temperature_sweep.csv", encoding="utf-8")))
    T = np.array([float(r["T"]) for r in rows])
    p = np.array([float(r["psnr"]) for r in rows])
    keff = np.array([float(r["keff"]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.55))
    for ax, vals, lab in [(axes[0], p, "mean PSNR (dB)"), (axes[1], keff, r"mean $K_{\mathrm{eff}}$")]:
        ax.axvline(0.05, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.plot(T, vals, color=INK, lw=1.2, marker="o", ms=3.2, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([0.03, 0.1, 0.3])
        ax.set_xticklabels(["0.03", "0.1", "0.3"])
        ax.minorticks_off()
        ax.set_xlabel("temperature $T$")
        ax.set_ylabel(lab)
        ax.text(0.05, 1.0, " $T$ = 0.05", transform=ax.get_xaxis_transform(), color=MUTED,
                fontsize=6.5, ha="left", va="top")
        ax.grid(True, axis="y", color=GRID, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].set_ylim(2.5, 5.2)
    axes[1].axhline(5, color=GRID, lw=0.6)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(OUT / "fig_dev_temperature.pdf")
    fig.savefig(OUT / "fig_dev_temperature.png", dpi=200)
    plt.close(fig)


# =============================================================================
# Held-out figures
# =============================================================================

def _splits_order(n: int) -> list[int]:
    lines = [l for l in (REPO / "splits" / "mnist_test_heldout_indices.csv").read_text(encoding="utf-8").splitlines()
             if l and not l.startswith("#")][1:]
    return [int(l.split(",")[1]) for l in lines][:n]


def _load_rows(run: Path) -> dict[int, dict]:
    import json
    rows = {}
    for mj in sorted((run / "per_image").glob("image_*/metrics.json")):
        r = json.loads(mj.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        v2 = mj.parent / "uncertainty_v2.json"
        if v2.exists():
            for m, vals in json.loads(v2.read_text(encoding="utf-8"))["methods"].items():
                r["methods"][m].update(vals)
        r["_dir"] = mj.parent
        rows[int(r["mnist_test_index"])] = r
    return rows


def _cap(v: float) -> float:
    return min(float(v), 60.0)


def fig_heldout_paired(main: dict, k1long: dict, base: dict, order: list[int]) -> None:
    from scipy import stats
    comps = [
        ("FE-softmax vs K=1", "fe_softmax", "k1", main),
        ("best-single vs K=1", "best_single_by_fe", "k1", main),
        ("FE-softmax vs uniform", "fe_softmax", "uniform", main),
        ("FE-softmax vs median", "fe_softmax", "median", main),
        ("best-single vs FE-softmax", "best_single_by_fe", "fe_softmax", main),
        ("FE-softmax vs one long start", "fe_softmax", "k1_long", k1long),
        ("best-single vs one long start", "best_single_by_fe", "k1_long", k1long),
        ("FE-softmax vs HIO supp. best5", "fe_softmax", "hio_supp_best5", base),
        ("FE-softmax vs RAAR supp. best5", "fe_softmax", "raar_supp_best5", base),
    ]
    labels, means, lo, hi, notes = [], [], [], [], []
    for lab, a, b, other in comps:
        d = np.array([_cap(main[i]["methods"][a]["psnr"]) - _cap(other[i]["methods"][b]["psnr"]) for i in order])
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = stats.t.ppf(0.975, len(d) - 1)
        labels.append(lab); means.append(m); lo.append(m - t * se); hi.append(m + t * se)
        notes.append(f"{int((d > 0).sum())}/{int((d < 0).sum())}")
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axvline(0, color=MUTED, lw=0.8)
    for k in (4, 6):                       # group boundaries: aggregation | compute-matched | classical
        ax.axhline(y[k] - 0.5, color=MUTED, lw=0.5, ls=(0, (1, 2)))
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)],
                fmt="o", color=INK, ms=3.5, lw=1.1, capsize=2.2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("paired PSNR difference (dB)")
    xmax = max(hi) + 1.6
    ax.set_xlim(min(-0.6, min(lo) - 0.3), xmax)
    for yy, nn in zip(y, notes):
        ax.text(xmax - 0.05, yy, nn, ha="right", va="center", fontsize=6.5, color=MUTED)
    ax.text(xmax - 0.05, y[0] + 0.75, "better/worse", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_heldout_paired.pdf"); fig.savefig(OUT / "fig_heldout_paired.png", dpi=200)
    plt.close(fig)


def fig_heldout_reliability(main: dict, order: list[int]) -> None:
    from scipy import stats
    fe = np.concatenate([main[i]["member_fe"] for i in order])
    ps = np.concatenate([main[i]["member_psnr"] for i in order])
    best = np.concatenate([np.arange(len(main[i]["member_fe"])) == int(np.argmin(main[i]["member_fe"])) for i in order])
    s = np.array([main[i]["methods"]["fe_softmax"]["failure_score"] for i in order])
    f = np.array([main[i]["methods"]["fe_softmax"]["fe"] for i in order])
    p = np.array([main[i]["methods"]["fe_softmax"]["psnr"] for i in order])
    confirm = np.arange(len(order)) >= 50

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    ax = axes[0]
    ax.scatter(fe[~best], ps[~best], s=6, color="#9a9a9a", alpha=0.55, lw=0, label="other starts")
    ax.scatter(fe[best], ps[best], s=9, color=INK, alpha=0.9, lw=0, label="lowest FE in its image")
    ax.set_xlabel("FE of the start"); ax.set_ylabel("PSNR of the start (dB)")
    ax.set_title(f"(a) 500 starts, Spearman {stats.spearmanr(-fe, ps).correlation:.2f}", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False, loc="upper right", handletextpad=0.1)
    for ax, x, lab, ttl in [(axes[1], s, "failure score $s$", "(b) $s$, pre-specified"),
                            (axes[2], f, "FE of the fused reconstruction", "(c) FE, post hoc")]:
        ax.scatter(x[~confirm], p[~confirm], s=9, facecolors="none", edgecolors="#8a8a8a", lw=0.6, label="ranks 0-49")
        ax.scatter(x[confirm], p[confirm], s=9, color=INK, lw=0, label="ranks 50-99")
        rho = stats.spearmanr(x[confirm], -p[confirm]).correlation
        ax.set_xlabel(lab); ax.set_ylabel("FE-softmax PSNR (dB)")
        ax.set_title(f"{ttl}: $\\rho$ = {rho:.2f} (ranks 50-99)", fontsize=7.5)
        ax.legend(fontsize=6, frameon=False, loc="upper right", handletextpad=0.1)
    for ax in axes:
        ax.axhline(15, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=GRID, lw=0.4)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(OUT / "fig_heldout_reliability.pdf"); fig.savefig(OUT / "fig_heldout_reliability.png", dpi=200)
    plt.close(fig)


def fig_heldout_examples(main: dict, base_dir: Path, order: list[int]) -> None:
    fs = np.array([main[i]["methods"]["fe_softmax"]["psnr"] for i in order]); o = fs.argsort(); n = len(order)
    picks = [order[int(o[int(round(q * (n - 1)))])] for q in (0, 0.25, 0.5, 0.75, 1.0)]
    cols = [("ground truth", None), ("K=1", "k1"), ("uniform", "uniform"), ("best-single", "best_single_by_fe"),
            ("FE-softmax", "fe_softmax"), ("HIO supp. best5", "hio_supp_best5"), ("$u$ (FE-softmax)", "u"),
            ("|error| (FE-softmax)", "err")]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(7.16, 0.98 * len(picks) + 0.35), squeeze=False)
    for i, idx in enumerate(picks):
        r = main[idx]
        ag = np.load(r["_dir"] / "aggregates_v2.npz")
        gt = np.load(r["_dir"] / "members.npz")["gt"].squeeze()
        bl = np.load(base_dir / "per_image" / f"image_{idx:05d}" / "baselines.npz")
        import json
        bm = json.loads((base_dir / "per_image" / f"image_{idx:05d}" / "metrics.json").read_text(encoding="utf-8"))
        for j, (title, key) in enumerate(cols):
            ax = axes[i, j]; lab = ""
            if key is None:
                img, cm, vmax = gt, "gray", 1
            elif key == "u":
                img, cm, vmax = ag["unc_fe_softmax"].squeeze(), "magma", 1
            elif key == "err":
                rec = ag["recon_fe_softmax__real"].squeeze()
                img, cm, vmax = np.abs(rec - gt), "magma", 1
            elif key.startswith("hio"):
                img = np.abs(bl[f"recon_{key}__real"] + 1j * bl[f"recon_{key}__imag"]).squeeze()
                cm, vmax = "gray", 1; lab = f"{bm['methods'][key]['psnr']:.1f} dB"
            else:
                img = ag[f"recon_{key}__real"].squeeze(); cm, vmax = "gray", 1
                lab = f"{r['methods'][key]['psnr']:.1f} dB"
            ax.imshow(np.clip(img, 0, vmax), cmap=cm, vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if lab:
                ax.set_xlabel(lab, fontsize=6.5, labelpad=1.5)
            if i == 0:
                ax.set_title(title, fontsize=7)
        axes[i, 0].set_ylabel(f"#{idx}\n$s$ = {r['methods']['fe_softmax']['failure_score']:.2f}", fontsize=6.5)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.03, wspace=0.06, hspace=0.32)
    fig.savefig(OUT / "fig_heldout_examples.pdf"); fig.savefig(OUT / "fig_heldout_examples.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=None, help="K=5 held-out run directory (rescored)")
    ap.add_argument("--k1long", default=None)
    ap.add_argument("--baselines", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--skip-dev", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.skip_dev:
        fig_dev_temperature()
        fig_dev_example()
    if a.heldout:
        order = _splits_order(a.n)
        main = _load_rows(Path(a.heldout))
        missing = [i for i in order if i not in main or not (main[i]["_dir"] / "uncertainty_v2.json").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} held-out images missing or not rescored; run scripts/rescore_uncertainty.py")
        fig_heldout_reliability(main, order)
        if a.k1long and a.baselines:
            fig_heldout_paired(main, _load_rows(Path(a.k1long)), _load_rows(Path(a.baselines)), order)
            fig_heldout_examples(main, Path(a.baselines), order)
