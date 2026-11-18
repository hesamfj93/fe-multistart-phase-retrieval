#!/usr/bin/env python
"""Summary of the runs on the measured CXIDB 28 pattern (REALDATA_PLAN.md).

Reads results/realdata/cxidb28_classical and results/realdata/cxidb28_oursA and writes
results/realdata/report.md, summary.json and figure.png.

    python scripts/realdata_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pr_uq.metrics.alignment import align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import nrmse_complex  # noqa: E402

ROOT = REPO / "results" / "realdata"
CLASSICAL = ("raar_shrinkwrap", "hio_shrinkwrap")


def load_stack(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    s = np.load(path)
    return torch.from_numpy(s["real"] + 1j * s["imag"]).unsqueeze(1).to(torch.complex64), s["fe"]


def main() -> int:
    lines = ["# Measured CXIDB 28 pattern: results", "",
             "No ground truth. FE is the Fourier error over measured pixels.", ""]
    summary, best = {}, {}
    mc = ROOT / "cxidb28_classical" / "metrics.json"
    if mc.exists():
        m = json.loads(mc.read_text(encoding="utf-8"))
        summary["intensity_scale"] = m["data"]["intensity_scale"]
        ks = list(m[CLASSICAL[0]]["best_of_first_k"].keys())
        lines += ["## Classical methods with shrinkwrap", "",
                  "| method | " + " | ".join(f"FE, best of {k}" for k in ks) + " | median start FE | s/start |",
                  "|---|" + "--:|" * (len(ks) + 2)]
        for nm in CLASSICAL:
            c = m[nm]
            lines.append(f"| {nm} | " + " | ".join(f"{c['best_of_first_k'][k]['fe']:.4f}" for k in ks)
                         + f" | {float(np.median(c['start_fe'])):.4f} | {float(np.mean(c['start_runtime_s'])):.0f} |")
            stack, _ = load_stack(ROOT / "cxidb28_classical" / f"{nm}_starts.npz")
            i = c["best_of_first_k"][ks[-1]]["index"]
            best[nm] = stack[i:i + 1]
            summary[nm] = {"best_of_first_k": c["best_of_first_k"], "median_start_fe": float(np.median(c["start_fe"])),
                           "mean_runtime_s": float(np.mean(c["start_runtime_s"]))}
        lines.append("")
        if len(best) == 2:
            a = align_to_gt(best[CLASSICAL[1]], best[CLASSICAL[0]]).aligned_recon
            d = float(nrmse_complex(a, best[CLASSICAL[0]]))
            summary["nrmse_hio_vs_raar"] = d
            lines += ["Agreement of the two lowest-FE reconstructions: complex NRMSE of the HIO result, "
                      f"aligned to the RAAR result, relative to the RAAR result: {d:.3f}. "
                      "This measures reproducibility across methods, not accuracy.", ""]
    mo = ROOT / "cxidb28_oursA" / "metrics.json"
    if mo.exists():
        o = json.loads(mo.read_text(encoding="utf-8"))["ours"]
        summary["oursA"] = {"start_fe": o["start_fe"], "aggregates_fe": {a: v["fe"] for a, v in o["aggregates"].items()},
                            "mean_runtime_s": float(np.mean(o["start_runtime_s"]))}
        lines += ["## Our solver, run A, K = 5 (negative result of the development phase)", "",
                  "| start FE | " + " | ".join(f"FE {a}" for a in o["aggregates"]) + " | s/start |",
                  "|---|" + "--:|" * (len(o["aggregates"]) + 1),
                  "| " + ", ".join(f"{v:.3f}" for v in o["start_fe"]) + " | "
                  + " | ".join(f"{v['fe']:.3f}" for v in o["aggregates"].values())
                  + f" | {float(np.mean(o['start_runtime_s'])):.0f} |", ""]
        agg = np.load(ROOT / "cxidb28_oursA" / "ours_aggregates.npz")
        best["ours fe_softmax"] = torch.from_numpy(agg["fe_softmax__real"] + 1j * agg["fe_softmax__imag"]).view(1, 1, 128, 128)
    if best:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, len(best), figsize=(3 * len(best), 3.2))
            for ax, (t, x) in zip(np.atleast_1d(axs), best.items()):
                ax.imshow(x.abs().numpy()[0, 0], cmap="gray")
                ax.set_title(t, fontsize=8)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(ROOT / "figure.png", dpi=150)
            plt.close(fig)
        except ImportError:
            pass
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
