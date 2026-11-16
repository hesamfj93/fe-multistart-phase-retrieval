#!/usr/bin/env python
"""Pilot: RAAR with shrinkwrap started from the untrained-network output (docs/PILOT_RAAR_REFINEMENT.md).

Development images only. Uses the five stored starts of results/spectral_ablation/complex_frozen
and the RAAR settings of configs/stage3_complex_heldout_raar45.yaml.

    python scripts/pilot_raar_refinement.py --out results/pilot_raar_refinement
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pr_uq.baselines.oversampled as ov  # noqa: E402
from pr_uq.metrics.alignment import align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import fourier_magnitude_error, psnr_complex  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors  # noqa: E402
from pr_uq.utils.config import load_config  # noqa: E402

CAP, FAIL = 60.0, 15.0
N_RANDOM, K_R1, K_R5 = 50, 46, 50
RUN = REPO / "results" / "spectral_ablation" / "complex_frozen"
CFG = REPO / "configs" / "stage3_complex_heldout_raar45.yaml"


def raar_from(z0: torch.Tensor, y: torch.Tensor, image_hw, iters: int, beta: float, er_iters: int, oversample: int):
    """RAAR with shrinkwrap from a given full-grid start; otherwise identical to ov.run_raar_full."""
    y_mag = y.clamp_min(0).sqrt()
    window = ov.object_window(image_hw, oversample, y.device)
    z = z0.to(torch.complex64)
    support, sigma = window.clone(), ov.SW_SIGMA0
    for it in range(iters):
        pm = ov._proj_magnitude(z, y_mag)
        rm = 2 * pm - z
        rs_rm = 2 * ov._proj_object(rm, support, "shrinkwrap") - rm
        z = 0.5 * beta * (rs_rm + z) + (1.0 - beta) * pm
        if (it + 1) % ov.SW_EVERY == 0:
            support = ov._shrinkwrap_support(ov._proj_magnitude(z, y_mag), window, sigma)
            sigma = max(ov.SW_SIGMA_MIN, sigma * ov.SW_SIGMA_DECAY)
    z = ov._error_reduction(ov._proj_object(z, support, "shrinkwrap"), y_mag, support, er_iters, "shrinkwrap")
    return ov._crop(z, image_hw).to(torch.complex64)


def embed(x: torch.Tensor, oversample: int) -> torch.Tensor:
    """Place a (1,1,H,W) object in the central window of the (1,1,oH,oW) grid, as the forward model does."""
    H, W = x.shape[-2:]
    z = torch.zeros(1, 1, H * oversample, W * oversample, dtype=torch.complex64)
    h0, w0 = (H * oversample - H) // 2, (W * oversample - W) // 2
    z[..., h0:h0 + H, w0:w0 + W] = x
    return z


def score(x, gt, y, ov_):
    p = min(float(psnr_complex(align_to_gt(x, gt).aligned_recon, gt)), CAP)
    return p, float(member_fourier_errors(x, y, oversample=ov_)[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = load_config(str(CFG))["baselines"]
    iters, beta, er = int(b["iterations"]), float(b["raar_beta"]), int(b["er_iterations"])
    ovs = int(load_config(str(CFG))["measurement"]["oversample"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images = []
    for d in sorted((RUN / "per_image").glob("image_*")):
        t0 = time.perf_counter()
        z = np.load(d / "members.npz")
        y = torch.from_numpy(z["measurement"]).float()
        gt = torch.from_numpy(z["gt__real"] + 1j * z["gt__imag"]).to(torch.complex64)
        raw = torch.from_numpy(z["stack_raw__real"] + 1j * z["stack_raw__imag"]).to(torch.complex64)
        al = torch.from_numpy(z["stack_aligned__real"] + 1j * z["stack_aligned__imag"]).to(torch.complex64)
        w = torch.from_numpy(z["softmax_weights"].astype(np.float32)).to(torch.complex64).view(-1, 1, 1, 1)
        fe_check = float(np.max(np.abs(member_fourier_errors(raw, y, oversample=ovs).numpy() - z["member_fe"])))
        hw = tuple(gt.shape[-2:])
        fused = (al * w).sum(0, keepdim=True)
        rec = {"mnist_test_index": int(json.loads((d / "metrics.json").read_text())["mnist_test_index"]),
               "member_fe_check_max_abs": fe_check}
        rec["fe_softmax"] = dict(zip(("psnr", "fe"), score(fused, gt, y, ovs)))
        r1 = raar_from(embed(fused, ovs), y, hw, iters, beta, er, ovs)
        rec["R1"] = dict(zip(("psnr", "fe"), score(r1, gt, y, ovs)))
        r5 = [score(raar_from(embed(raw[k:k + 1], ovs), y, hw, iters, beta, er, ovs), gt, y, ovs) for k in range(raw.shape[0])]
        rec["R5_starts"] = [{"psnr": p, "fe": f} for p, f in r5]
        k5 = int(np.argmin([f for _, f in r5]))
        rec["R5"] = {"psnr": r5[k5][0], "fe": r5[k5][1], "index": k5}
        rnd = []
        for s in range(N_RANDOM):
            res = ov.run_raar_full(y, hw, iters=iters, beta=beta, er_iters=er, seed=int(b["base_seed"]) + s,
                                   oversample=ovs, device="cpu", constraint="shrinkwrap")
            rnd.append(score(res.recon, gt, y, ovs))
        rec["random_starts"] = [{"psnr": p, "fe": f} for p, f in rnd]
        for K in (5, K_R1, K_R5):
            k = int(np.argmin([f for _, f in rnd[:K]]))
            rec[f"random_best{K}"] = {"psnr": rnd[k][0], "fe": rnd[k][1], "index": k}
        rec["runtime_s"] = time.perf_counter() - t0
        images.append(rec)
        print(f"image {rec['mnist_test_index']}: FE-softmax {rec['fe_softmax']['psnr']:.1f}, R1 {rec['R1']['psnr']:.1f}, "
              f"R5 {rec['R5']['psnr']:.1f}, random best46 {rec['random_best46']['psnr']:.1f}, "
              f"best50 {rec['random_best50']['psnr']:.1f} ({rec['runtime_s']:.0f} s)", flush=True)

    def agg(key):
        p = np.array([r[key]["psnr"] for r in images])
        return {"mean": float(p.mean()), "exact": int((p >= CAP).sum()), "failures": int((p < FAIL).sum())}

    s = {k: agg(k) for k in ("fe_softmax", "R1", "R5", "random_best5", f"random_best{K_R1}", f"random_best{K_R5}")}
    r5_all = np.array([x["psnr"] for r in images for x in r["R5_starts"]])
    rnd_all = np.array([x["psnr"] for r in images for x in r["random_starts"]])
    s["share_exact_refined_network_starts"] = [int((r5_all >= CAP).sum()), int(r5_all.size)]
    s["share_exact_random_starts"] = [int((rnd_all >= CAP).sum()), int(rnd_all.size)]
    s["share_failed_refined_network_starts"] = [int((r5_all < FAIL).sum()), int(r5_all.size)]
    s["share_failed_random_starts"] = [int((rnd_all < FAIL).sum()), int(rnd_all.size)]
    r1_ok = s["R1"]["exact"] > s[f"random_best{K_R1}"]["exact"] and s["R1"]["failures"] <= s[f"random_best{K_R1}"]["failures"]
    r5_ok = s["R5"]["exact"] > s[f"random_best{K_R5}"]["exact"] and s["R5"]["failures"] <= s[f"random_best{K_R5}"]["failures"]
    decision = {"R1_promising": bool(r1_ok), "R5_promising": bool(r5_ok), "promising": bool(r1_ok or r5_ok)}
    (out / "summary.json").write_text(json.dumps({"note": "pilot on development images; docs/PILOT_RAAR_REFINEMENT.md",
                                                  "device": "cpu", "summary": s, "decision": decision,
                                                  "images": images}, indent=2), encoding="utf-8")
    print(json.dumps({"summary": s, "decision": decision}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
