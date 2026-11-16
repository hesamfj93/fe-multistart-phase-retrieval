#!/usr/bin/env python
"""Post hoc: the ensemble outlier rule of Hong et al. (2026) on the stored starts of both studies.

Not part of any analysis plan. No new reconstruction: the rule is applied to the five final
starts stored in members.npz of the two held-out runs.

Rule (Hong et al., Ain Shams Eng. J. 17(11):104411, Sec. 3.1.3, Eq. 14). Each network i has a
data distance d_i to the observation; with d~ = median(d) and MAD = median(|d_i - d~|), network i
is an outlier if |d_i - d~| / MAD > threshold. The outliers are removed and the remaining
reconstructions are averaged. Here d_i is the squared magnitude residual
||F P x_i| - sqrt(y)||^2, which is proportional to FE_i^2 for a given image; the test is
two-sided, as written. The paper does not give the threshold. Primary value: 3 * 1.4826 = 4.4478,
which equals the "three scaled MADs" of the MAD rules of this study; 2 and 3 are reported as
sensitivity, and a one-sided variant (only large residuals are outliers) at the primary value.

Differences from Hong et al.: they apply the rule inside every outer iteration and use the mean
as the prior of the next update; here it is applied once to the final starts. Their tasks have
no shift, flip or global-phase ambiguity; here the kept starts are averaged after the alignment to
the lowest-FE start that FE-softmax also uses (stack_aligned).

Reference "mean_aligned5" is the plain mean of the same five aligned starts, so the effect of the
rule itself is EOR minus mean_aligned5. (The stored "uniform" aggregate aligns to start 0 instead.)

Scoring is that of the studies: alignment to the ground truth, real-part PSNR for real objects,
complex PSNR for complex objects, capped at 60 dB; failure < 15 dB. Statistics are descriptive
and not adjusted for multiplicity.

    python scripts/posthoc_eor_rule.py --out results/posthoc_eor/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.metrics.alignment import align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import psnr, psnr_complex  # noqa: E402

CAP, FAIL = 60.0, 15.0
PRIMARY = 3 * 1.4826
VARIANTS = (("two_sided_4.4478", PRIMARY, False), ("two_sided_3", 3.0, False),
            ("two_sided_2", 2.0, False), ("one_sided_4.4478", PRIMARY, True))
RUNS = {"real": "results/stage2_mnist_heldout", "complex": "results/stage3_complex_heldout"}


def eor_keep(fe: np.ndarray, threshold: float, one_sided: bool) -> np.ndarray:
    """Boolean mask of the starts kept by the MAD outlier rule on d = FE^2."""
    d = fe.astype(float) ** 2
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if mad == 0:
        return np.ones_like(d, dtype=bool)
    z = (d - med) / mad
    out = z > threshold if one_sided else np.abs(z) > threshold
    return ~out


def score(x: torch.Tensor, gt: torch.Tensor, complex_obj: bool) -> float:
    """PSNR after alignment to the ground truth, as in the studies."""
    a = align_to_gt(x, gt).aligned_recon
    return float(psnr_complex(a, gt) if complex_obj else psnr(a, gt))


def load_image(d: Path, complex_obj: bool):
    z = np.load(d / "members.npz")
    al = torch.from_numpy(z["stack_aligned__real"] + 1j * z["stack_aligned__imag"]).to(torch.complex64)
    if complex_obj:
        gt = torch.from_numpy(z["gt__real"] + 1j * z["gt__imag"]).to(torch.complex64)
    else:
        gt = torch.from_numpy(z["gt"]).to(torch.complex64)
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    return al, gt, z["member_fe"].astype(float), z["softmax_weights"].astype(float), m


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison a - b: mean, t-interval, median, wins, Wilcoxon (unadjusted)."""
    diff = a - b
    n = diff.size
    se = diff.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    nz = diff[diff != 0]
    p = float(stats.wilcoxon(nz).pvalue) if nz.size else 1.0
    return {"mean_difference_db": float(diff.mean()), "t_ci95": [float(diff.mean() - t * se), float(diff.mean() + t * se)],
            "median_difference_db": float(np.median(diff)), "a_better": int((diff > 0).sum()),
            "b_better": int((diff < 0).sum()), "wilcoxon_p_two_sided_unadjusted": p}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"note": "post hoc; not part of any analysis plan; no new reconstruction; descriptive, unadjusted",
           "rule": "Hong et al. 2026, Eq. 14, d_i = FE_i^2, two-sided unless stated", "primary_threshold": PRIMARY,
           "studies": {}}
    for otype, run in RUNS.items():
        complex_obj = otype == "complex"
        dirs = sorted(p for p in (REPO / run / "per_image").glob("image_*") if (p / "members.npz").exists())
        rec = {v[0]: [] for v in VARIANTS}
        nkept = {v[0]: [] for v in VARIANTS}
        best_removed = {v[0]: 0 for v in VARIANTS}
        stored = {k: [] for k in ("k1", "uniform", "median", "best_single_by_fe", "fe_softmax")}
        check, mean5 = [], []
        for d in dirs:
            al, gt, fe, w, m = load_image(d, complex_obj)
            for k in stored:
                stored[k].append(m["methods"][k]["psnr"])
            wt = torch.from_numpy(w).to(al.dtype).view(-1, 1, 1, 1)
            mean5.append(score(al.mean(0, keepdim=True), gt, complex_obj))
            check.append(abs(score((al * wt).sum(0, keepdim=True), gt, complex_obj) - m["methods"]["fe_softmax"]["psnr"]))
            for name, thr, one in VARIANTS:
                keep = eor_keep(fe, thr, one)
                nkept[name].append(int(keep.sum()))
                best_removed[name] += int(not keep[int(np.argmin(fe))])
                rec[name].append(score(al[torch.from_numpy(keep)].mean(0, keepdim=True), gt, complex_obj))
        ref = {k: np.minimum(np.array(v, dtype=float), CAP) for k, v in stored.items()}
        ref["mean_aligned5"] = np.minimum(np.array(mean5), CAP)
        res = {"n": len(dirs), "check_fe_softmax_recomputed_max_abs_diff_db": float(max(check)),
               "stored": {k: {"mean": float(v.mean()), "median": float(np.median(v)), "failures": int((v < FAIL).sum())}
                          for k, v in ref.items()},
               "eor": {}}
        for name, _, _ in VARIANTS:
            p = np.minimum(np.array(rec[name]), CAP)
            k = np.array(nkept[name])
            res["eor"][name] = {
                "mean": float(p.mean()), "median": float(np.median(p)), "failures": int((p < FAIL).sum()),
                "images_with_rejection": int((k < 5).sum()), "starts_removed": int((5 - k).sum()),
                "lowest_fe_start_removed": best_removed[name],
                "vs_fe_softmax": compare(p, ref["fe_softmax"]),
                "vs_selection": compare(p, ref["best_single_by_fe"]),
                "vs_mean_aligned5": compare(p, ref["mean_aligned5"]),
            }
        out["studies"][otype] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
