#!/usr/bin/env python
"""Post hoc analyses requested in the fifth review, from the stored starts only (no new runs).

Not part of any analysis plan. For the held-out runs of our solver and Net-ADM in Studies 1 to 3,
from members.npz (five raw starts, their FE against the measurement, the ground truth):

1. Temperature of FE-softmax: PSNR of FE-softmax for T in {0.01, 0.02, 0.05, 0.1, 0.2, 0.5} and of
   the uniform mean on the same alignment (T -> infinity). T = 0.05 must reproduce the stored value.
2. Number of starts: selection and FE-softmax (T = 0.05) over the first k starts, k = 1 to 5, with
   the anchor re-chosen as the lowest-FE start among the first k. k = 5 must reproduce the stored values.
3. PSNR inside the object: PSNR of K = 1, selection and FE-softmax over the pixels where the ground
   truth is nonzero, against the PSNR over the whole 32 x 32 window as stored.

PSNR as in the runner (real objects: real part of the aligned image; complex objects: complex PSNR),
capped at 60 dB. Differences are paired over images with a t-based 95% CI.

    python scripts/posthoc_review5.py --out results/posthoc_review5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pr_uq.metrics.alignment import align_ensemble_to_anchor, align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import psnr, psnr_complex  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import fidelity_weights  # noqa: E402

CAP = 60.0
TEMPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
RUNS = {  # name: (folder, complex object)
    "S1 real, no noise": ("results/stage2_mnist_heldout", False),
    "S2 complex, no noise": ("results/stage3_complex_heldout", True),
    "S3 ours real, moderate": ("results/study3/real_ours_moderate", False),
    "S3 ours real, strong": ("results/study3/real_ours_strong", False),
    "S3 ours complex, moderate": ("results/study3/complex_ours_moderate", True),
    "S3 ours complex, strong": ("results/study3/complex_ours_strong", True),
    "S3 Net-ADM real, moderate": ("results/study3/real_netadm_moderate", False),
    "S3 Net-ADM real, strong": ("results/study3/real_netadm_strong", False),
}


def load(f: Path):
    z = np.load(f)
    stack = torch.from_numpy(z["stack_raw__real"] + 1j * z["stack_raw__imag"]).to(torch.complex64)
    g = z["gt"] if "gt" in z.files else z["gt__real"] + 1j * z["gt__imag"]
    gt = torch.from_numpy(g).to(torch.complex64)
    fe = torch.from_numpy(z["member_fe"].astype(np.float32))
    return stack, gt, fe, z["member_psnr"].astype(float)


def score(x, gt, cplx, mask=None):
    a = align_to_gt(x, gt).aligned_recon
    if mask is None:
        v = psnr_complex(a, gt) if cplx else psnr(a, gt)
    else:
        d = (a - gt) if cplx else (a.real - gt.real)
        m = float(d.abs().pow(2)[mask].mean())
        v = 100.0 if m <= 1e-20 else 10 * math.log10(1.0 / m)
    return min(v, CAP)


def fused(stack, fe, T):
    anchor = int(fe.argmin())
    al = align_ensemble_to_anchor(stack, anchor_index=anchor)
    if T == math.inf:
        w = torch.full((stack.shape[0],), 1.0 / stack.shape[0])
    else:
        w = fidelity_weights(fe, method="softmax", temperature=T)
    w = w.to(device=al.device, dtype=al.dtype)
    return (al * w.view(-1, 1, 1, 1)).sum(dim=0, keepdim=True)


def ci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return {"mean": float(d.mean()), "ci": [float(d.mean() - h), float(d.mean() + h)]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (folder, cplx) in RUNS.items():
        files = sorted((REPO / folder / "per_image").glob("image_*/members.npz"))
        metr = [json.loads((f.parent / "metrics.json").read_text()) for f in files]
        temp = {T: [] for T in TEMPS + (math.inf,)}
        ksel = {k: [] for k in range(1, 6)}
        kfs = {k: [] for k in range(1, 6)}
        inside = {"K=1": [], "selection": [], "FE-softmax": []}
        stored_fs, stored_sel = [], []
        for f, m in zip(files, metr):
            stack, gt, fe, mpsnr = load(f)
            mask = gt.abs() > 0
            for T in temp:
                temp[T].append(score(fused(stack, fe, T), gt, cplx))
            for k in range(1, 6):
                b = int(fe[:k].argmin())
                ksel[k].append(min(float(mpsnr[b]), CAP))
                kfs[k].append(score(fused(stack[:k], fe[:k], 0.05), gt, cplx) if k > 1 else min(float(mpsnr[0]), CAP))
            best = int(fe.argmin())
            inside["K=1"].append(score(stack[0:1], gt, cplx, mask))
            inside["selection"].append(score(stack[best:best + 1], gt, cplx, mask))
            inside["FE-softmax"].append(score(fused(stack, fe, 0.05), gt, cplx, mask))
            stored_fs.append(min(m["methods"]["fe_softmax"]["psnr"], CAP))
            stored_sel.append(min(m["methods"]["best_single_by_fe"]["psnr"], CAP))
        k1 = np.array(ksel[1])
        s = {"images": len(files),
             "check_T005_max_abs_diff_db": float(np.abs(np.array(temp[0.05]) - stored_fs).max()),
             "check_k5_selection_max_abs_diff_db": float(np.abs(np.array(ksel[5]) - stored_sel).max()),
             "temperature": {("uniform" if T == math.inf else str(T)): {"mean": float(np.mean(v)),
                                                                        "minus_K1": ci(np.array(v) - k1)}
                             for T, v in temp.items()},
             "k_curve": {str(k): {"selection": float(np.mean(ksel[k])), "fe_softmax": float(np.mean(kfs[k]))}
                         for k in range(1, 6)},
             "inside_object": {m: float(np.mean(v)) for m, v in inside.items()},
             "inside_object_fe_softmax_minus_K1": ci(np.array(inside["FE-softmax"]) - np.array(inside["K=1"])),
             "inside_object_selection_minus_K1": ci(np.array(inside["selection"]) - np.array(inside["K=1"])),
             "window_fe_softmax_minus_K1": ci(np.array(temp[0.05]) - k1)}
        summary[name] = s
        print(name, json.dumps({k: s[k] for k in ("images", "check_T005_max_abs_diff_db", "check_k5_selection_max_abs_diff_db")}))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Post hoc analyses of the fifth review (stored starts, no new runs)", "",
             "## FE-softmax temperature: mean PSNR (dB)", "",
             "| run | " + " | ".join(f"T={T}" for T in TEMPS) + " | uniform | K=1 |", "|---|" + "--:|" * (len(TEMPS) + 2)]
    for n, s in summary.items():
        lines.append(f"| {n} | " + " | ".join(f"{s['temperature'][str(T)]['mean']:.2f}" for T in TEMPS)
                     + f" | {s['temperature']['uniform']['mean']:.2f} | {s['k_curve']['1']['selection']:.2f} |")
    lines += ["", "## Number of starts: mean PSNR (dB), selection / FE-softmax over the first k starts", "",
              "| run | k=1 | k=2 | k=3 | k=4 | k=5 |", "|---|--:|--:|--:|--:|--:|"]
    for n, s in summary.items():
        lines.append(f"| {n} | " + " | ".join(f"{s['k_curve'][str(k)]['selection']:.2f} / {s['k_curve'][str(k)]['fe_softmax']:.2f}"
                                               for k in range(1, 6)) + " |")
    lines += ["", "## PSNR inside the object (pixels where the ground truth is nonzero)", "",
              "| run | K=1 | selection | FE-softmax | FE-softmax - K=1 inside [95% CI] | same, whole window |",
              "|---|--:|--:|--:|---|---|"]
    for n, s in summary.items():
        d, w = s["inside_object_fe_softmax_minus_K1"], s["window_fe_softmax_minus_K1"]
        io = s["inside_object"]
        lines.append(f"| {n} | {io['K=1']:.2f} | {io['selection']:.2f} | {io['FE-softmax']:.2f} | "
                     f"{d['mean']:+.2f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}] | {w['mean']:+.2f} [{w['ci'][0]:.2f}, {w['ci'][1]:.2f}] |")
    lines += ["", "Checks: T = 0.05 and k = 5 reproduce the stored FE-softmax and selection PSNR to within the values below.", ""]
    for n, s in summary.items():
        lines.append(f"- {n}: {s['check_T005_max_abs_diff_db']:.2g} dB, {s['check_k5_selection_max_abs_diff_db']:.2g} dB")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
