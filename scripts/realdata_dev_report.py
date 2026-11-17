#!/usr/bin/env python
"""Summary of the twin development phase (REALDATA_DEV_PLAN.md) and its decision rule.

Reads results/realdata_dev/twin{0,1,2}_{oursA,oursB,classical}/metrics.json and writes
results/realdata_dev/report.md and summary.json.

    python scripts/realdata_dev_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "results" / "realdata_dev"
TWINS = (0, 1, 2)
AGG = ("k1", "uniform", "median", "best_single_by_fe", "fe_softmax")


def load(name):
    p = ROOT / name / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    lines = ["# Twin development phase: results", "",
             "Complex NRMSE after alignment (lower is better; 1 is the error of a zero image) and FE over measured pixels.", ""]
    summary = {}
    missing = []
    for var in ("A", "B"):
        rows = []
        for s in TWINS:
            m = load(f"twin{s}_ours{var}")
            if m is None:
                missing.append(f"twin{s}_ours{var}")
                continue
            o = m["ours"]
            st = [d["nrmse_c"] for d in o["start_scores"]]
            rows.append({"twin": s, "fe_gt": m["data"]["fe_ground_truth"],
                         **{a: o["aggregates"][a]["nrmse_c"] for a in AGG},
                         **{f"fe_{a}": o["aggregates"][a]["fe"] for a in AGG},
                         "best_start": min(st), "worst_start": max(st),
                         "lowest_fe_is_best": int(np.argmin(o["start_fe"])) == int(np.argmin(st)),
                         "median_start_fe": float(np.median(o["start_fe"])),
                         "mean_runtime_s": float(np.mean(o["start_runtime_s"]))})
        summary[f"ours{var}"] = rows
        lines += [f"## Our solver, run {var}", "",
                  "| twin | FE of truth | k1 | uniform | median | lowest FE | FE-softmax | best start | worst start | FE of FE-softmax | median start FE | lowest FE is best | s/start |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|"]
        for r in rows:
            lines.append(f"| {r['twin']} | {r['fe_gt']:.3f} | {r['k1']:.3f} | {r['uniform']:.3f} | {r['median']:.3f} | "
                         f"{r['best_single_by_fe']:.3f} | {r['fe_softmax']:.3f} | {r['best_start']:.3f} | {r['worst_start']:.3f} | "
                         f"{r['fe_fe_softmax']:.3f} | {r['median_start_fe']:.3f} | {r['lowest_fe_is_best']} | {r['mean_runtime_s']:.0f} |")
        if rows:
            lines.append(f"| mean | | {np.mean([r['k1'] for r in rows]):.3f} | {np.mean([r['uniform'] for r in rows]):.3f} | "
                         f"{np.mean([r['median'] for r in rows]):.3f} | {np.mean([r['best_single_by_fe'] for r in rows]):.3f} | "
                         f"{np.mean([r['fe_softmax'] for r in rows]):.3f} | | | | | | |")
        lines.append("")
    crows = []
    for s in TWINS:
        m = load(f"twin{s}_classical")
        if m is None:
            missing.append(f"twin{s}_classical")
            continue
        for nm in ("raar_shrinkwrap", "hio_shrinkwrap"):
            if nm not in m:
                continue
            c = m[nm]
            st = [d["nrmse_c"] for d in c["start_scores"]]
            crows.append({"twin": s, "method": nm, **{f"best_of_{k}": v["nrmse_c"] for k, v in c["best_of_first_k"].items()},
                          **{f"fe_best_of_{k}": v["fe"] for k, v in c["best_of_first_k"].items()},
                          "best_start": min(st), "median_start": float(np.median(st)),
                          "mean_runtime_s": float(np.mean(c["start_runtime_s"]))})
    summary["classical"] = crows
    if crows:
        ks = [k for k in ("1", "5", "10", "20", "50") if f"best_of_{k}" in crows[0]]
        lines += ["## Classical methods with shrinkwrap", "",
                  "| twin | method | " + " | ".join(f"best of {k}" for k in ks) + " | FE, best of 50 | best start | median start | s/start |",
                  "|---|---|" + "--:|" * len(ks) + "--:|--:|--:|--:|"]
        for r in crows:
            lines.append(f"| {r['twin']} | {r['method']} | " + " | ".join(f"{r['best_of_' + k]:.3f}" for k in ks)
                         + f" | {r.get('fe_best_of_50', float('nan')):.3f} | {r['best_start']:.3f} | {r['median_start']:.3f} | {r['mean_runtime_s']:.0f} |")
        lines.append("")
    # decision rule of REALDATA_DEV_PLAN.md
    if len(summary.get("oursA", [])) == 3 and len(summary.get("oursB", [])) == 3:
        mA = float(np.mean([r["fe_softmax"] for r in summary["oursA"]]))
        mB = float(np.mean([r["fe_softmax"] for r in summary["oursB"]]))
        choice = "A" if mA <= mB else "B"
        neg = min(mA, mB) > 1.0
        summary["decision"] = {"mean_nrmse_fe_softmax_A": mA, "mean_nrmse_fe_softmax_B": mB, "variant": choice,
                               "negative_result": neg}
        lines += ["## Decision", "",
                  f"Mean complex NRMSE of FE-softmax over the three twins: A {mA:.3f}, B {mB:.3f}. "
                  f"Variant for the measured pattern: {choice}."
                  + (" Both are above 1, so our solver is reported as a negative result." if neg else ""), ""]
    if missing:
        lines += ["Missing runs: " + ", ".join(missing), ""]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
