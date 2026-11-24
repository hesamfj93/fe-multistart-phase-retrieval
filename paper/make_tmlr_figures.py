#!/usr/bin/env python
"""Figures of the TMLR manuscript that are new relative to the IEEE version.

fig_forest, fig_bestofk, fig_fe_reliability, fig_realdata; written to paper/figures_tmlr/.
Numbers come from the frozen reports and the stored per-image results.

    python paper/make_tmlr_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pr_uq.metrics.alignment import align_to_gt  # noqa: E402
sys.path.insert(0, str(REPO / "paper"))
from tmlr_fe import fe_softmax_measured  # noqa: E402

OUT = REPO / "paper" / "figures_tmlr"
C_OURS, C_NETADM, C_CLASSICAL = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

# Pre-specified comparisons: label, mean, ci, better/n, supported, solver.
# Study 1: supp table of the Holm family (report_n100); Study 2: stage3_complex_heldout/report; Study 3: study3/report.
VS_K1 = [
    ("S1 real, none, sel.", 1.76, (1.21, 2.31), "61/100", True, "ours"),
    ("S1 real, none, FE-sm.", 1.82, (1.29, 2.36), "76/100", True, "ours"),
    ("S2 complex, none, sel.", 3.09, (2.38, 3.80), "71/100", True, "ours"),
    ("S2 complex, none, FE-sm.", 3.36, (2.74, 3.98), "89/100", True, "ours"),
]
VS_LONG = [
    ("S1 real, sel.", 0.73, (0.12, 1.34), "59/100", False, "ours"),
    ("S1 real, FE-sm.", 0.80, (0.21, 1.38), "61/100", False, "ours"),
    ("S2 complex, sel.", 1.10, (0.37, 1.83), "63/100", True, "ours"),
    ("S2 complex, FE-sm.", 1.37, (0.73, 2.01), "67/100", True, "ours"),
]
S3_LABEL = {
    "N1": ("S3 real, moderate", "ours"), "N2": ("S3 real, strong", "ours"),
    "N3": ("S3 complex, moderate", "ours"), "N4": ("S3 complex, strong", "ours"),
    "N5": ("S3 real, moderate", "netadm"), "N6": ("S3 real, strong", "netadm"),
    "N7": ("real, mod., HIO", "ours"), "N8": ("real, strong, HIO", "ours"),
    "N9": ("complex, mod., RAAR", "ours"), "N10": ("complex, strong, RAAR", "ours"),
    "N11": ("real, mod., HIO", "netadm"), "N12": ("real, strong, HIO", "netadm"),
}


def s3_rows(ids):
    s = json.loads((REPO / "results/study3/report/summary.json").read_text())
    t = {x["id"]: x for x in s["tests"]}
    rows = []
    for i in ids:
        x = t[i]
        lab, solver = S3_LABEL[i]
        name = "Net-ADM" if solver == "netadm" else "ours"
        rows.append((f"{i} {lab} ({name})", x["mean"], tuple(x["ci"]), f"{x['first_better']}/{x['n']}",
                     x["decision"] != "no claim", solver))
    return rows


LONG_LABEL = {"real_ours_moderate": ("real, mod.", "ours"), "real_ours_strong": ("real, strong", "ours"),
              "complex_ours_moderate": ("complex, mod.", "ours"), "complex_ours_strong": ("complex, strong", "ours"),
              "real_netadm_moderate": ("real, mod.", "netadm"), "real_netadm_strong": ("real, strong", "netadm")}


def long_rows():
    """Compute-matched control of Study 3 (planned after Study 3, tag study3-long-plan-v1): L1 to L12."""
    s = json.loads((REPO / "results/study3_long/report/summary.json").read_text())
    rows = []
    for x in s["tests"]:
        lab, solver = LONG_LABEL[x["run"]]
        rule = "FE-sm." if x["comparison"].startswith("FE-softmax") else "sel."
        name = "Net-ADM" if solver == "netadm" else "ours"
        rows.append((f"{x['id']} S3 {lab}, {rule} ({name})", x["mean"], tuple(x["ci"]),
                     f"{x['first_better']}/{x['n']}", x["decision"] != "no claim", solver))
    return rows


def forest_panel(ax, rows, title, xlim=None):
    y = np.arange(len(rows))[::-1]
    for yi, (lab, m, ci, b, sup, solver) in zip(y, rows):
        c = C_NETADM if solver == "netadm" else C_OURS
        ax.plot(ci, [yi, yi], color=c, lw=1.6, solid_capstyle="round")
        ax.plot([m], [yi], "o", ms=5, mfc=c if sup else "white", mec=c, mew=1.4)
        ax.annotate(b, (ci[1], yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("PSNR difference (dB)")
    ax.set_title(title, fontsize=8, loc="left", color=INK)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", color="#e5e4e0", lw=0.6)
    ax.set_axisbelow(True)


def fig_forest():
    rows_a = VS_K1 + s3_rows(["N1", "N2", "N3", "N4", "N5", "N6"])
    rows_c = s3_rows(["N7", "N8", "N9", "N10", "N11", "N12"])
    rows_b = VS_LONG + long_rows()
    fig = plt.figure(figsize=(7.2, 4.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 6], width_ratios=[1, 1], hspace=0.45, wspace=1.15)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    forest_panel(ax_a, rows_a, "(a) five FE-guided starts vs one start", (-0.5, 4.8))
    forest_panel(ax_b, rows_b, "(b) vs one start of equal compute", (-0.5, 3.9))
    forest_panel(ax_c, rows_c, "(c) vs classical, equal wall time", (-28, 2))
    h = [plt.Line2D([], [], color=C_OURS, marker="o", lw=1.6, label="our solver"),
         plt.Line2D([], [], color=C_NETADM, marker="o", lw=1.6, label="Net-ADM"),
         plt.Line2D([], [], color=INK2, marker="o", mfc="white", lw=0, label="not supported")]
    ax_b.legend(handles=h, fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.035), ncol=3)
    fig.savefig(OUT / "fig_forest.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_forest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_bestofk():
    s = json.loads((REPO / "results/study3/report/summary.json").read_text())
    cl, de = s["classical"], s["descriptive"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.45))
    panels = [("(a) real objects, HIO real and nonnegative", "real_hio_realpos", [("ours", "real_ours", 81), ("netadm", "real_netadm", 22)]),
              ("(b) complex objects, RAAR with shrinkwrap", "complex_raar_sw", [("ours", "complex_ours", 50)])]
    for ax, (title, ckey, uns) in zip(axs, panels):
        for lvl, ls in (("moderate", "-"), ("strong", "--")):
            d = cl[f"{ckey}_{lvl}"]
            ks = [int(k.replace("best of ", "")) for k in d]
            ms = [v["mean"] for v in d.values()]
            ax.plot(ks, ms, ls, color=C_CLASSICAL, lw=1.6, marker="o", ms=3.5)
            for solver, run, keq in uns:
                c = C_NETADM if solver == "netadm" else C_OURS
                ax.axhline(de[f"{run}_{lvl}"]["FE-softmax"]["mean"], color=c, ls=ls, lw=1.2)
        for solver, run, keq in uns:
            c = C_NETADM if solver == "netadm" else C_OURS
            ax.axvline(keq, color=c, lw=0.8, alpha=0.7, ls=":")
            ax.annotate(f"$K$={keq}", (keq, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=c)
        ax.set_xscale("log")
        ax.set_xlabel("classical starts $K$ (lowest FE of the first $K$)")
        ax.set_ylabel("mean PSNR (dB)")
        ax.set_title(title, fontsize=8, loc="left", pad=12)
        ax.grid(color="#e5e4e0", lw=0.6)
        ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=C_CLASSICAL, lw=1.6, marker="o", ms=3.5, label="classical, lowest FE of $K$"),
         plt.Line2D([], [], color=C_OURS, lw=1.2, label="ours, FE-softmax of 5"),
         plt.Line2D([], [], color=C_NETADM, lw=1.2, label="Net-ADM, FE-softmax of 5"),
         plt.Line2D([], [], color=INK2, lw=1.2, ls="-", label="moderate noise"),
         plt.Line2D([], [], color=INK2, lw=1.2, ls="--", label="strong noise")]
    fig.legend(handles=h, fontsize=7, frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_bestofk.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_bestofk.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_fe_reliability():
    s = json.loads((REPO / "paper/tmlr_posthoc_review.json").read_text())["runs"]
    names = {"real_ours_moderate": "Study 3, real, moderate, ours", "real_ours_strong": "Study 3, real, strong, ours",
             "complex_ours_moderate": "Study 3, complex, moderate, ours", "complex_ours_strong": "Study 3, complex, strong, ours",
             "real_netadm_moderate": "Study 3, real, moderate, Net-ADM", "real_netadm_strong": "Study 3, real, strong, Net-ADM"}
    runs = [("real_ours_moderate", "ours, real, moderate"), ("real_ours_strong", "ours, real, strong"),
            ("complex_ours_moderate", "ours, complex, moderate"), ("complex_ours_strong", "ours, complex, strong"),
            ("real_netadm_moderate", "Net-ADM, real, moderate"), ("real_netadm_strong", "Net-ADM, real, strong")]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), sharey=False)
    for ax, (run, title) in zip(axs.ravel(), runs):
        fe, ps = [], []
        for p in sorted((REPO / "results/study3" / run / "per_image").glob("*/metrics.json")):
            m = json.loads(p.read_text())
            fe.append(fe_softmax_measured(p.parent))  # FE against the measurement
            ps.append(min(m["methods"]["fe_softmax"]["psnr"], 60))
        c = C_NETADM if "netadm" in run else C_OURS
        ax.scatter(fe, ps, s=10, color=c, alpha=0.8, linewidths=0)
        ax.axhline(15, color=INK2, ls="--", lw=0.8)
        rho = s[names[run]]["N13_measured_fe"]["rho"]
        ax.set_title(f"{title}\nSpearman(FE, $-$PSNR) = {rho:.2f}", fontsize=7.5, loc="left")
        ax.set_xlabel("FE of FE-softmax")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(color="#e5e4e0", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fe_reliability.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_fe_reliability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _cx(re, im):
    return torch.from_numpy(re + 1j * im).to(torch.complex64).view(1, 1, *re.shape)


def _starts(path):
    d = np.load(path)
    return d["real"], d["imag"], d["fe"]


def _amp(x):
    a = x.abs()[0, 0].numpy()
    return a / max(a.max(), 1e-12)


def _gap_share(a2):
    p = a2.sum(-1)
    lo, hi = np.searchsorted(np.cumsum(p) / p.sum(), [0.005, 0.995])
    seg = p[lo:hi + 1]
    return float((seg < 0.01 * seg.max()).mean())


def fig_realdata():
    meas = np.load(REPO / "results/realdata/cxidb28_classical/measurement.npz")
    y, mask = meas["y"], meas["mask"].astype(bool)
    tw = np.load(REPO / "results/realdata_dev/twin0_classical/measurement.npz")
    gt = _cx(tw["gt_real"], tw["gt_imag"])
    r_re, r_im, r_fe = _starts(REPO / "results/realdata_dev/twin0_classical/raar_shrinkwrap_starts.npz")
    i = int(np.argmin(r_fe))
    tw_raar = align_to_gt(_cx(r_re[i], r_im[i]), gt).aligned_recon
    oa = np.load(REPO / "results/realdata_dev/twin0_oursA/ours_aggregates.npz")
    tw_ours = align_to_gt(_cx(oa["fe_softmax__real"], oa["fe_softmax__imag"]), gt).aligned_recon
    R_re, R_im, R_fe = _starts(REPO / "results/realdata/cxidb28_classical/raar_shrinkwrap_starts.npz")
    H_re, H_im, H_fe = _starts(REPO / "results/realdata/cxidb28_classical/hio_shrinkwrap_starts.npz")
    h = int(np.argmin(H_fe))
    hio = _cx(H_re[h], H_im[h])
    r1 = int(np.argmin(R_fe))
    gaps = np.array([_gap_share(R_re[k] ** 2 + R_im[k] ** 2) for k in range(len(R_fe))])
    two = np.where(gaps >= 0.2)[0]
    r2 = int(two[np.argmin(R_fe[two])])
    rank_r2 = int(np.where(np.argsort(R_fe) == r2)[0][0]) + 1
    om = np.load(REPO / "results/realdata/cxidb28_oursA/ours_aggregates.npz")
    ours = _cx(om["fe_softmax__real"], om["fe_softmax__imag"])
    om_fe = json.loads((REPO / "results/realdata/summary.json").read_text())["oursA"]["aggregates_fe"]["fe_softmax"]
    al = lambda x: align_to_gt(x, hio).aligned_recon  # noqa: E731

    fig = plt.figure(figsize=(7.2, 4.1))
    gs = fig.add_gridspec(2, 4, wspace=0.08, hspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    img = np.log10(np.where(mask, np.maximum(y, y[mask].min()), np.nan))
    ax.imshow(img, cmap="magma")
    ax.set_title("(a) measured pattern\n(log intensity)", fontsize=7.5)
    ax.axis("off")
    for j, (x, t) in enumerate([(gt, "(b) twin 0: ground truth"), (tw_raar, f"twin 0: RAAR, lowest FE\nof 50 (FE {r_fe[i]:.3f})"),
                                 (tw_ours, "twin 0: ours, FE-softmax")]):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.imshow(_amp(x), cmap="gray")
        ax.set_title(t, fontsize=7.5)
        ax.axis("off")
    panels = [(al(_cx(R_re[r1], R_im[r1])), f"(c) RAAR, lowest FE\n(FE {R_fe[r1]:.3f})"),
              (al(_cx(R_re[r2], R_im[r2])), f"RAAR, lowest FE with two\nclusters ({rank_r2}th, FE {R_fe[r2]:.3f})"),
              (hio, f"HIO, lowest FE\n(FE {H_fe[h]:.3f})"),
              (al(ours), f"ours, FE-softmax\n(FE {om_fe:.2f})")]
    for j, (x, t) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(_amp(x), cmap="gray")
        ax.set_title(t, fontsize=7.5)
        ax.axis("off")
    fig.savefig(OUT / "fig_realdata.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_realdata.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("realdata: RAAR lowest", r1, R_fe[r1], "two-cluster", r2, R_fe[r2], "rank", rank_r2, "HIO", h, H_fe[h], "ours", om_fe)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    fig_forest()
    fig_bestofk()
    fig_fe_reliability()
    fig_realdata()
    print("written:", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
