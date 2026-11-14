#!/usr/bin/env python
"""Report and decision rule of the spectral-weight ablation (SPECTRAL_ABLATION_PLAN.md).

Reads the eight development runs results/spectral_ablation/{real,complex}_{frozen,uniform,
widening,dcfinal}, summarizes the five starts of each image, and applies the decision rule
fixed in the plan before the runs:

    a variant V is "materially better" on an object type t if
        median start FE(V, t)   <= 0.5 * median start FE(frozen, t)    and
        mean start PSNR(V, t)   >= mean start PSNR(frozen, t) + 1.0 dB;
    V qualifies if it is materially better on at least one object type and its mean start
    PSNR on the other type is at least that of the frozen design minus 0.5 dB.

    Outcome A: no variant qualifies. The frozen design stays; the ablation is reported as a
               development-image analysis.
    Outcome B: the qualifying variant with the highest mean start PSNR, averaged over the two
               object types, becomes the design of a new held-out study.

PSNR values are capped at 60 dB; a start or reconstruction below 15 dB is a failure.

    python scripts/spectral_ablation_report.py --root results/spectral_ablation \
        --out results/spectral_ablation/report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = ("frozen", "uniform", "widening", "dcfinal")
TYPES = ("real", "complex")
N_IMAGES = 8
CAP, FAIL = 60.0, 15.0
FE_RATIO, PSNR_GAIN, PSNR_TOL = 0.5, 1.0, 0.5


def load_run(run_dir: Path) -> list[dict]:
    """Per-image metrics.json of a run, completed images only, ordered by image id."""
    out = []
    for mj in sorted((run_dir / "per_image").glob("image_*/metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        if m.get("status") == "ok":
            out.append(m)
    return out


def summarize(ms: list[dict]) -> dict:
    """Start-level and aggregate statistics of one run (five starts per image)."""
    cap = lambda v: np.minimum(np.asarray(v, dtype=float), CAP)  # noqa: E731
    fe = np.array([f for m in ms for f in m["member_fe"]], dtype=float)
    ps = cap([p for m in ms for p in m["member_psnr"]])
    k1 = cap([m["methods"]["k1"]["psnr"] for m in ms])
    fused = cap([m["methods"]["fe_softmax"]["psnr"] for m in ms])
    sel = cap([m["methods"]["best_single_by_fe"]["psnr"] for m in ms])
    oracle = cap([max(m["member_psnr"]) for m in ms])
    sd = [float(np.std(cap(m["member_psnr"]), ddof=1)) for m in ms]
    return {
        "n_images": len(ms),
        "n_starts": int(fe.size),
        "median_start_fe": float(np.median(fe)),
        "mean_start_psnr": float(ps.mean()),
        "start_failures": int((ps < FAIL).sum()),
        "mean_within_image_sd_db": float(np.mean(sd)),
        "mean_k1_psnr": float(k1.mean()),
        "mean_selection_psnr": float(sel.mean()),
        "mean_fe_softmax_psnr": float(fused.mean()),
        "mean_best_start_psnr": float(oracle.mean()),
        "fe_softmax_failures": int((fused < FAIL).sum()),
        "mean_gain_fe_softmax_vs_k1_db": float((fused - k1).mean()),
        "median_fe_softmax_fe": float(np.median([m["methods"]["fe_softmax"]["fe"] for m in ms])),
    }


def decide(s: dict) -> dict:
    """Apply the decision rule of the plan to summaries s[type][variant]."""
    rows = {}
    for v in VARIANTS[1:]:
        better, not_worse = {}, {}
        for t in TYPES:
            f, x = s[t]["frozen"], s[t][v]
            better[t] = (x["median_start_fe"] <= FE_RATIO * f["median_start_fe"]
                         and x["mean_start_psnr"] >= f["mean_start_psnr"] + PSNR_GAIN)
            not_worse[t] = x["mean_start_psnr"] >= f["mean_start_psnr"] - PSNR_TOL
        qualifies = any(better[t] and not_worse[o] for t, o in (("real", "complex"), ("complex", "real")))
        rows[v] = {"materially_better": better, "not_worse": not_worse, "qualifies": qualifies,
                   "mean_start_psnr_avg": float(np.mean([s[t][v]["mean_start_psnr"] for t in TYPES]))}
    qual = [v for v in rows if rows[v]["qualifies"]]
    selected = max(qual, key=lambda v: rows[v]["mean_start_psnr_avg"]) if qual else None
    return {"variants": rows, "outcome": "B" if selected else "A", "selected": selected}


def main() -> int:
    """Summarize the eight runs, apply the decision rule, write summary.json and report.md."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="results/spectral_ablation")
    ap.add_argument("--out", default="results/spectral_ablation/report")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)

    s, missing = {t: {} for t in TYPES}, []
    for t in TYPES:
        for v in VARIANTS:
            ms = load_run(root / f"{t}_{v}")
            if len(ms) != N_IMAGES:
                missing.append(f"{t}_{v}: {len(ms)} of {N_IMAGES} images")
                continue
            s[t][v] = summarize(ms)
    if missing:
        print("Incomplete runs, no decision:\n  " + "\n  ".join(missing))
        return 1

    d = decide(s)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"summaries": s, "decision": d,
                                                  "rule": {"fe_ratio": FE_RATIO, "psnr_gain_db": PSNR_GAIN,
                                                           "psnr_tolerance_db": PSNR_TOL}}, indent=2),
                                      encoding="utf-8")
    L = ["# Spectral-weight ablation, 8 development images", ""]
    for t in TYPES:
        L += [f"## {t} objects", "",
              "| variant | median start FE | mean start PSNR | start failures | within-image SD | K=1 | selection | FE-softmax | best start | fused failures |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for v in VARIANTS:
            x = s[t][v]
            L.append(f"| {v} | {x['median_start_fe']:.4f} | {x['mean_start_psnr']:.2f} | "
                     f"{x['start_failures']}/{x['n_starts']} | {x['mean_within_image_sd_db']:.2f} | "
                     f"{x['mean_k1_psnr']:.2f} | {x['mean_selection_psnr']:.2f} | {x['mean_fe_softmax_psnr']:.2f} | "
                     f"{x['mean_best_start_psnr']:.2f} | {x['fe_softmax_failures']}/{x['n_images']} |")
        L.append("")
    L += ["## Decision", "", "| variant | materially better (real, complex) | not worse (real, complex) | qualifies |",
          "|---|---|---|---|"]
    for v, r in d["variants"].items():
        L.append(f"| {v} | {r['materially_better']['real']}, {r['materially_better']['complex']} | "
                 f"{r['not_worse']['real']}, {r['not_worse']['complex']} | {r['qualifies']} |")
    L += ["", f"Outcome {d['outcome']}" + (f": selected variant `{d['selected']}`." if d["selected"]
                                         else ": the frozen design stays."), ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
