#!/usr/bin/env python
"""Supplementary figures for the stage-3 experiment (complex-valued objects), from stored results.

    fig_s3_scatter.pdf    per-image PSNR of FE-softmax against RAAR with shrinkwrap (best of five),
                          and FE against PSNR for that RAAR configuration
    fig_s3_examples.pdf   amplitude and phase of three images chosen by rule

Nothing is retrained. Reconstructions in aggregates.npz and baselines.npz are stored already aligned
to the ground truth (flip variant, cyclic shift, global phase).

    python paper/make_stage3_figures.py --main results/stage3_complex_heldout \
        --k1long results/stage3_complex_heldout_k1long --baselines results/stage3_complex_heldout_baselines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "figures"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#d9d9d9"
DOT, ACCENT = "#2a6f97", "#b5523b"
CAP, FAIL, AMP_T = 60.0, 15.0, 0.05

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "pdf.fonttype": 42,
})


def load(run: Path) -> dict:
    out = {}
    for mj in sorted((run / "per_image").glob("image_*/metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        if m.get("status") == "ok":
            m["_dir"] = mj.parent
            out[int(m["mnist_test_index"])] = m
    return out


def cplx(npz, key: str) -> np.ndarray:
    return (npz[f"{key}__real"] + 1j * npz[f"{key}__imag"]).squeeze()


def fig_scatter(main: dict, bl: dict, ids: list[int]) -> None:
    pn = np.array([min(main[i]["methods"]["fe_softmax"]["psnr"], CAP) for i in ids])
    pr = np.array([min(bl[i]["methods"]["raar_sw_best5"]["psnr"], CAP) for i in ids])
    fr = np.array([bl[i]["methods"]["raar_sw_best5"]["fe"] for i in ids])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.7))
    a.plot([8, 61], [8, 61], color=GRID, lw=0.8, zorder=0)
    a.axhline(FAIL, color=MUTED, lw=0.6, ls="--"); a.axvline(FAIL, color=MUTED, lw=0.6, ls="--")
    jit = np.where(pr >= CAP, np.random.default_rng(0).uniform(-0.6, 0.6, len(pr)), 0.0)
    a.scatter(pr + jit, pn, s=9, color=DOT, edgecolor="white", linewidth=0.3)
    a.set_xlabel("RAAR shrinkwrap best5, complex PSNR (dB, capped at 60)")
    a.set_ylabel("FE-softmax, complex PSNR (dB)")
    a.set_xlim(8, 62); a.set_ylim(8, 30)
    a.text(0.98, 0.04, f"FE-softmax better on {(pn > pr).sum()} of {len(ids)}", transform=a.transAxes,
           ha="right", va="bottom", fontsize=7, color=INK)
    a.set_title("(a) per-image PSNR", loc="left")
    fe_plot = np.maximum(fr, 1e-6)
    b.scatter(fe_plot, pr, s=9, color=DOT, edgecolor="white", linewidth=0.3)
    b.set_xscale("log"); b.axhline(FAIL, color=MUTED, lw=0.6, ls="--")
    b.set_xlabel("FE of RAAR shrinkwrap best5 (values below $10^{-6}$ shown at $10^{-6}$)")
    b.set_ylabel("complex PSNR (dB, capped at 60)")
    fn = np.array([main[i]["methods"]["fe_softmax"]["fe"] for i in ids])
    b.axvspan(fn.min(), fn.max(), color=ACCENT, alpha=0.12, lw=0)
    b.text(np.sqrt(fn.min() * fn.max()), 61, "FE range of\nFE-softmax", ha="center", va="top", fontsize=6.5, color=ACCENT)
    b.set_title("(b) FE within RAAR shrinkwrap", loc="left")
    for ax in (a, b):
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / "fig_s3_scatter.pdf"); fig.savefig(OUT / "fig_s3_scatter.png", dpi=200)
    plt.close(fig)


def fig_examples(main: dict, k1l: dict, bl: dict, ids: list[int]) -> list[int]:
    # rule: images at the 10th, 50th and 90th percentile of PSNR(FE-softmax) - PSNR(RAAR sw best5)
    d = np.array([min(main[i]["methods"]["fe_softmax"]["psnr"], CAP) - min(bl[i]["methods"]["raar_sw_best5"]["psnr"], CAP)
                  for i in ids])
    o = np.argsort(d, kind="stable"); n = len(ids)
    picks = [ids[int(o[int(round(q * (n - 1)))])] for q in (0.1, 0.5, 0.9)]
    cols = [("ground truth\n", None), ("K=1\n", "k1"), ("selection\n", "best_single_by_fe"), ("FE-softmax\n", "fe_softmax"),
            ("k1_long\n", "k1_long"), ("HIO shrinkwrap\nbest5", "hio_sw_best5"), ("RAAR shrinkwrap\nbest5", "raar_sw_best5")]
    fig, axes = plt.subplots(2 * len(picks), len(cols), figsize=(7.16, 1.0 * 2 * len(picks) + 0.5), squeeze=False)
    for r, idx in enumerate(picks):
        gt = cplx(np.load(main[idx]["_dir"] / "members.npz"), "gt")
        ag = np.load(main[idx]["_dir"] / "aggregates.npz")
        b = np.load(bl[idx]["_dir"] / "baselines.npz")
        lg = np.load(k1l[idx]["_dir"] / "aggregates.npz")
        for j, (title, key) in enumerate(cols):
            if key is None:
                z, lab = gt, ""
            elif key == "k1_long":
                z, lab = cplx(lg, "recon_k1_long"), f"{k1l[idx]['methods']['k1_long']['psnr']:.1f} dB"
            elif key.startswith(("hio", "raar")):
                z = cplx(b, f"recon_{key}")
                p = bl[idx]["methods"][key]["psnr"]
                lab = "exact" if p >= CAP else f"{p:.1f} dB"
            else:
                z, lab = cplx(ag, f"recon_{key}"), f"{main[idx]['methods'][key]['psnr']:.1f} dB"
            amp = np.abs(z)
            ph = np.where(amp > AMP_T, np.angle(z), np.nan)
            ax0, ax1 = axes[2 * r, j], axes[2 * r + 1, j]
            ax0.imshow(np.clip(amp, 0, 1), cmap="gray", vmin=0, vmax=1)
            im = ax1.imshow(ph, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
            ax1.set_facecolor("#ffffff")
            for ax in (ax0, ax1):
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.4)
            if lab:
                ax1.set_xlabel(lab, fontsize=6.5, labelpad=1.5)
            if r == 0:
                ax0.set_title(title, fontsize=7)
        axes[2 * r, 0].set_ylabel(f"#{idx}\namplitude", fontsize=6.5)
        axes[2 * r + 1, 0].set_ylabel("phase", fontsize=6.5)
    cax = fig.add_axes([0.93, 0.04, 0.012, 0.22])
    cb = fig.colorbar(im, cax=cax, ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"]); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(left=0.06, right=0.92, top=0.96, bottom=0.03, wspace=0.05, hspace=0.12)
    fig.savefig(OUT / "fig_s3_examples.pdf"); fig.savefig(OUT / "fig_s3_examples.png", dpi=200)
    plt.close(fig)
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--k1long", required=True)
    ap.add_argument("--baselines", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m, k, b = load(Path(a.main)), load(Path(a.k1long)), load(Path(a.baselines))
    ids = sorted(set(m) & set(k) & set(b))
    fig_scatter(m, b, ids)
    picks = fig_examples(m, k, b, ids)
    print("examples:", picks)


if __name__ == "__main__":
    main()
