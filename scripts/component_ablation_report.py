#!/usr/bin/env python
"""Report and decision rule of the component ablation (STUDY3_DEV_PLAN.md, Section 3).

Same measures and decision rule as scripts/spectral_ablation_report.py. The frozen design is the
stored development run results/spectral_ablation/{real,complex}_frozen; the variants are
results/component_ablation/{real,complex}_{noproj,noreg,noclip,alloff}.

    python scripts/component_ablation_report.py --out results/component_ablation/report
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("sar", REPO / "scripts" / "spectral_ablation_report.py")
SAR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SAR)

VARIANTS = ("frozen", "noproj", "noreg", "noclip", "alloff")
TYPES = SAR.TYPES


def run_dir(root: Path, frozen_root: Path, t: str, v: str) -> Path:
    return frozen_root / f"{t}_frozen" if v == "frozen" else root / f"{t}_{v}"


def decide(s: dict) -> dict:
    """The rule of the spectral ablation, over the variants of this ablation."""
    rows = {}
    for v in VARIANTS[1:]:
        better, not_worse = {}, {}
        for t in TYPES:
            f, x = s[t]["frozen"], s[t][v]
            better[t] = (x["median_start_fe"] <= SAR.FE_RATIO * f["median_start_fe"]
                         and x["mean_start_psnr"] >= f["mean_start_psnr"] + SAR.PSNR_GAIN)
            not_worse[t] = x["mean_start_psnr"] >= f["mean_start_psnr"] - SAR.PSNR_TOL
        qualifies = any(better[t] and not_worse[o] for t, o in (("real", "complex"), ("complex", "real")))
        rows[v] = {"materially_better": better, "not_worse": not_worse, "qualifies": qualifies,
                   "mean_start_psnr_avg": float(np.mean([s[t][v]["mean_start_psnr"] for t in TYPES]))}
    qual = [v for v in rows if rows[v]["qualifies"]]
    selected = max(qual, key=lambda v: rows[v]["mean_start_psnr_avg"]) if qual else None
    return {"variants": rows, "selected": selected or "frozen"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="results/component_ablation")
    ap.add_argument("--frozen-root", default="results/spectral_ablation")
    ap.add_argument("--out", default="results/component_ablation/report")
    a = ap.parse_args()
    root, froot, out = Path(a.root), Path(a.frozen_root), Path(a.out)
    s, missing = {t: {} for t in TYPES}, []
    for t in TYPES:
        for v in VARIANTS:
            ms = SAR.load_run(run_dir(root, froot, t, v))
            if len(ms) != SAR.N_IMAGES:
                missing.append(f"{t}_{v}: {len(ms)} of {SAR.N_IMAGES} images")
                continue
            s[t][v] = SAR.summarize(ms)
    if missing:
        print("Incomplete runs, no decision:\n  " + "\n  ".join(missing))
        return 1
    d = decide(s)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"summaries": s, "decision": d}, indent=2), encoding="utf-8")
    L = ["# Component ablation, 8 development images", ""]
    for t in TYPES:
        L += [f"## {t} objects", "",
              "| variant | median start FE | mean start PSNR | start failures | within-image SD | K=1 | selection | FE-softmax | best start |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for v in VARIANTS:
            x = s[t][v]
            L.append(f"| {v} | {x['median_start_fe']:.4f} | {x['mean_start_psnr']:.2f} | {x['start_failures']}/{x['n_starts']} | "
                     f"{x['mean_within_image_sd_db']:.2f} | {x['mean_k1_psnr']:.2f} | {x['mean_selection_psnr']:.2f} | "
                     f"{x['mean_fe_softmax_psnr']:.2f} | {x['mean_best_start_psnr']:.2f} |")
        L.append("")
    L += ["## Decision", "", "| variant | materially better (real, complex) | not worse (real, complex) | qualifies |", "|---|---|---|---|"]
    for v, r in d["variants"].items():
        L.append(f"| {v} | {r['materially_better']['real']}, {r['materially_better']['complex']} | "
                 f"{r['not_worse']['real']}, {r['not_worse']['complex']} | {r['qualifies']} |")
    L += ["", f"Solver of the third study: `{d['selected']}`.", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
