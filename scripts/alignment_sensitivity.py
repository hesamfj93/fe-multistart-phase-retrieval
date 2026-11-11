#!/usr/bin/env python
"""Sensitivity of the reported metrics to the alignment search (evaluation only, no retraining).

The frozen pipeline aligns over four flip variants (identity, horizontal, vertical, both;
non-identity variants conjugated) and cyclic shifts. Only the identity and the conjugated
180-degree rotation (twin) are ambiguities of the Fourier magnitude, and a cyclic shift on the
32x32 window can wrap content across the border. This script recomputes, from the stored starts:

  A. how often each variant is chosen, for the ground-truth alignment of every method and for
     the alignment of every start to the anchor;
  B. PSNR/SSIM of all five methods when only the true ambiguities are searched
       B1: ground-truth alignment restricted (methods unchanged);
       B2: start-to-anchor and ground-truth alignment restricted (aggregation recomputed);
  C. as B2, with non-wrapping (zero-filled) shifts instead of cyclic shifts.

    python scripts/alignment_sensitivity.py --run-dir results/stage2_mnist_heldout --n 100
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.metrics.alignment import align_to_gt, apply_variant, best_global_phase  # noqa: E402
from pr_uq.metrics.reconstruction import psnr, ssim  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors, select_anchor  # noqa: E402

METHODS = ["k1", "uniform", "median", "best_single_by_fe", "fe_softmax"]
TRUE_VARIANTS = (0, 3)


def _runner():
    spec = importlib.util.spec_from_file_location("runner_sens", REPO / "scripts" / "run_phase_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _linear_shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift with zero fill (no wrap-around)."""
    out = torch.zeros_like(x)
    H, W = x.shape[-2:]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[..., yd, xd] = x[..., ys, xs]
    return out


def align_linear(recon, ref, variants=TRUE_VARIANTS, real=True):
    """Best variant x non-wrapping shift x global phase (exhaustive over shifts)."""
    H, W = recon.shape[-2:]
    best = (np.inf, None, None)
    for v in variants:
        rv = apply_variant(recon, v)
        # linear cross-correlation via zero padding to 2H x 2W
        A = torch.fft.fft2(torch.nn.functional.pad(rv, (0, W, 0, H)))
        B = torch.fft.fft2(torch.nn.functional.pad(ref, (0, W, 0, H)))
        xc = torch.fft.ifft2(B * A.conj()).abs().squeeze()
        for _ in range(3):                                  # try the top peaks
            idx = int(torch.argmax(xc)); dy, dx = divmod(idx, 2 * W)
            xc.view(-1)[idx] = -1
            dy = dy - 2 * H if dy >= H else dy
            dx = dx - 2 * W if dx >= W else dx
            if abs(dy) >= H or abs(dx) >= W:
                continue
            rs = _linear_shift(rv, dy, dx)
            rs = rs * best_global_phase(rs, ref)
            a = rs.real if real else rs
            b = ref.real if real else ref
            mse = float((a - b).abs().pow(2).mean())
            if mse < best[0]:
                best = (mse, rs, (v, dy, dx))
    return best[1], best[2]


def aggregate(R, stack, y, fe, anchor, aligned, T, ov):
    strategies, _ = R.compute_strategies(stack, y, fe, anchor, aligned, T, ov)
    return {m: strategies[m][0] for m in METHODS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    R = _runner()
    lines = [l for l in (REPO / "splits" / "mnist_test_heldout_indices.csv").read_text(encoding="utf-8").splitlines()
             if l and not l.startswith("#")][1:]
    order = [int(l.split(",")[1]) for l in lines][: a.n]

    var_gt = {m: Counter() for m in METHODS}
    var_start = Counter()
    res = {k: {m: [] for m in METHODS} for k in ("frozen", "B1", "B2", "C")}
    for idx in order:
        d = run / "per_image" / f"image_{idx:05d}"
        b = np.load(d / "members.npz")
        stack = torch.from_numpy(b["stack_raw__real"] + 1j * b["stack_raw__imag"]).to(torch.complex64)
        y = torch.from_numpy(b["measurement"]).float()
        gt = torch.from_numpy(b["gt"]).to(torch.complex64)
        fe = member_fourier_errors(stack, y, oversample=2)
        anchor = select_anchor(stack, method="best_fe", y_intensity=y, oversample=2)

        # frozen alignment (4 variants, cyclic)
        al4, al2, alL = [], [], []
        for k in range(stack.shape[0]):
            if k == anchor:
                al4.append(stack[k:k + 1]); al2.append(stack[k:k + 1]); alL.append(stack[k:k + 1]); continue
            r4 = align_to_gt(stack[k:k + 1], stack[anchor:anchor + 1], return_real=False)
            var_start[r4.variant] += 1
            al4.append(r4.aligned_recon)
            al2.append(align_to_gt(stack[k:k + 1], stack[anchor:anchor + 1], return_real=False,
                                   try_variants=TRUE_VARIANTS).aligned_recon)
            alL.append(align_linear(stack[k:k + 1], stack[anchor:anchor + 1], real=False)[0])
        recs4 = aggregate(R, stack, y, fe, anchor, torch.cat(al4), 0.05, 2)
        recs2 = aggregate(R, stack, y, fe, anchor, torch.cat(al2), 0.05, 2)
        recsL = aggregate(R, stack, y, fe, anchor, torch.cat(alL), 0.05, 2)
        for m in METHODS:
            r = align_to_gt(recs4[m], gt); var_gt[m][r.variant] += 1
            res["frozen"][m].append((psnr(r.aligned_recon, gt), ssim(r.aligned_recon, gt)))
            r = align_to_gt(recs4[m], gt, try_variants=TRUE_VARIANTS)
            res["B1"][m].append((psnr(r.aligned_recon, gt), ssim(r.aligned_recon, gt)))
            r = align_to_gt(recs2[m], gt, try_variants=TRUE_VARIANTS)
            res["B2"][m].append((psnr(r.aligned_recon, gt), ssim(r.aligned_recon, gt)))
            rl, _ = align_linear(recsL[m], gt, real=False)
            res["C"][m].append((psnr(rl, gt), ssim(rl, gt)))

    summary = {"n": len(order), "variant_counts_gt_alignment": {m: dict(var_gt[m]) for m in METHODS},
               "variant_counts_start_to_anchor": dict(var_start), "settings": {}}
    for k, per in res.items():
        summary["settings"][k] = {}
        for m in METHODS:
            arr = np.array(per[m])
            summary["settings"][k][m] = {"psnr_mean": float(arr[:, 0].mean()), "ssim_mean": float(arr[:, 1].mean()),
                                         "failures": int((arr[:, 0] < 15).sum())}
        base = np.array(per["k1"])[:, 0]
        for m in ("best_single_by_fe", "fe_softmax"):
            dlt = np.array(per[m])[:, 0] - base
            summary["settings"][k][f"{m}_minus_k1"] = {"mean": float(dlt.mean()), "n_better": int((dlt > 0).sum())}
    out = Path(a.out) if a.out else run / "alignment_sensitivity.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
