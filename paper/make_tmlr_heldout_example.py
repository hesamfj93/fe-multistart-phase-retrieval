#!/usr/bin/env python
"""Figure 1 of the TMLR manuscript from a held-out image (replaces the development example).

The image is chosen by a fixed rule, not by appearance: among the 100 held-out images of Study 1,
the one whose FE-softmax minus K = 1 PSNR difference is closest to the median difference of the
study (ties: lowest rank). Uses the stored starts (members.npz) and the runner's own aggregation and
scoring, so the printed values equal those in metrics.json.

    python paper/make_tmlr_heldout_example.py
"""
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
src = (HERE / "make_paper_figures.py").read_text(encoding="utf-8")
ns = {"__name__": "make_paper_figures_tmlr", "__file__": str(HERE / "make_paper_figures.py")}
exec(compile(src, "make_paper_figures.py", "exec"), ns)
REPO, plt, _runner = ns["REPO"], ns["plt"], ns["_runner"]
OUT = HERE / "figures_tmlr"
RUN = REPO / "results" / "stage2_mnist_heldout" / "per_image"


def pick() -> Path:
    rows = []
    for f in sorted(RUN.glob("image_*/metrics.json")):
        m = json.loads(f.read_text())
        d = min(m["methods"]["fe_softmax"]["psnr"], 60) - min(m["methods"]["k1"]["psnr"], 60)
        rows.append((m["split_rank"], d, f.parent))
    med = float(np.median([d for _, d, _ in rows]))
    rank, d, folder = min(rows, key=lambda r: (abs(r[1] - med), r[0]))
    print(f"median FE-softmax - K=1: {med:.2f} dB; chosen rank {rank}, {folder.name}, {d:.2f} dB")
    return folder


def main() -> None:
    R = _runner()
    folder = pick()
    z = np.load(folder / "members.npz")
    m = json.loads((folder / "metrics.json").read_text())
    stack = torch.from_numpy(z["stack_raw__real"] + 1j * z["stack_raw__imag"]).to(torch.complex64)
    gt = torch.from_numpy(z["gt"]).to(torch.complex64)
    y = torch.from_numpy(z["measurement"]).float()
    fe = torch.from_numpy(z["member_fe"].astype(np.float32))
    K = stack.shape[0]
    anchor = int(fe.argmin())
    aligned = R.align_ensemble_to_anchor(stack, anchor_index=anchor)
    strategies, best = R.compute_strategies(stack, y, fe, anchor, aligned, 0.05, 2)
    member_psnr = [float(v) for v in z["member_psnr"]]

    show = [("uniform", "uniform mean"), ("best_single_by_fe", "selection"), ("fe_softmax", "FE-softmax")]
    agg = []
    for key, label in show:
        mean_c, std = strategies[key]
        rec, aligned_gt, u, _, _ = R.score_reconstruction(mean_c, std, gt, y, 2)
        assert abs(rec["psnr"] - m["methods"][key]["psnr"]) < 1e-3, (key, rec["psnr"], m["methods"][key]["psnr"])
        agg.append((label, aligned_gt.real.squeeze().numpy(), rec["psnr"]))
        if key == "fe_softmax":
            u_map = u.squeeze().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(7.16, 3.35))
    for k in range(K):
        ax = axes[0, k]
        ax.imshow(np.abs(stack[k].squeeze().numpy()), cmap="gray", vmin=0, vmax=1)
        tag = " (lowest FE)" if k == anchor else ""
        ax.set_title(f"start {k}{tag}\nFE {float(fe[k]):.3f}, {member_psnr[k]:.1f} dB", fontsize=7.5)
    for j, (label, img, p) in enumerate(agg):
        ax = axes[1, j]
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{label}\n{p:.1f} dB", fontsize=7.5)
    ax = axes[1, 3]
    im = ax.imshow(u_map, cmap="magma", vmin=0, vmax=1)
    ax.set_title("map $u$\n(FE-softmax)", fontsize=7.5)
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
    axes[0, 0].set_ylabel("raw starts", fontsize=8)
    axes[1, 0].set_ylabel("aggregates", fontsize=8)
    fig.subplots_adjust(left=0.04, right=0.975, top=0.86, bottom=0.02, wspace=0.28, hspace=0.42)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig_heldout_example.pdf")
    fig.savefig(OUT / "fig_heldout_example.png", dpi=200)
    plt.close(fig)
    print("start FE", [round(float(v), 3) for v in fe], "PSNR", [round(v, 1) for v in member_psnr])
    print("K=1", round(m["methods"]["k1"]["psnr"], 2), "aggregates", [(a, round(p, 2)) for a, _, p in agg],
          "anchor", anchor, "selected", best, "MNIST index", m["mnist_test_index"])


if __name__ == "__main__":
    main()
