#!/usr/bin/env python
"""Capacity test of the frozen generator on the development images. Diagnostic, not part of any plan.

The untrained reconstructions of both held-out studies keep a median Fourier error (FE) of about
0.2 on noiseless data. This test asks whether the generator can represent the objects at all:
the same member (two decoders, latents, seed 0, Adam with the frozen learning rates, gradient
clipping and number of iterations) is fitted directly to the ground truth with the loss
mean |x_net - x|^2. No measurement, regularization, projection, spectral weight or support is
used. If the fitted output reaches a high PSNR and a small FE, the generator can represent the
objects, and the FE floor of the reconstructions comes from fitting the Fourier magnitude
(the optimization landscape and the loss design), not from capacity.

PSNR is the real-part PSNR for real objects and the complex PSNR for complex objects, after the
same alignment as in the studies; FE is the Fourier error of the fitted output against the
noiseless magnitude of the object.

    python scripts/capacity_test.py --config configs/stage2_mnist_dev8.yaml --out results/capacity_test/real
    python scripts/capacity_test.py --config configs/stage3_complex_dev.yaml --out results/capacity_test/complex
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

from pr_uq.data.complex_objects import object_class, object_from_cfg  # noqa: E402
from pr_uq.data.datasets import build_dataset  # noqa: E402
from pr_uq.metrics.alignment import align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import fourier_magnitude_error, psnr, psnr_complex  # noqa: E402
from pr_uq.models.ensemble import EnsembleMember, MemberSpec  # noqa: E402
from pr_uq.optim.trainer import TrainerConfig, _make_optimizer  # noqa: E402
from pr_uq.utils.config import load_config  # noqa: E402

CHECKPOINTS = (500, 1000, 2000, 3000)


def dataset_index(sample) -> int:
    """MNIST test index of a sample (the name ends with it)."""
    return int(str(sample.name).split("_")[-1])


def main() -> int:
    """Fit the generator to each development image and write per-image and summary JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    cfg = load_config(a.config)
    tcfg = TrainerConfig(**{k: v for k, v in cfg["trainer"].items() if k in TrainerConfig.__dataclass_fields__})
    iters, ov = int(tcfg.iterations), int(cfg["measurement"]["oversample"])
    complex_obj = object_class(cfg["dataset"]) == "complex"
    score = psnr_complex if complex_obj else psnr
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for smp in build_dataset(cfg["dataset"]):
        idx = dataset_index(smp)
        x = object_from_cfg(smp.image.to(a.device), cfg["dataset"], idx).to(torch.complex64)
        H, W = x.shape[-2:]
        spec = MemberSpec(out_h=H, out_w=W, z_channels=int(cfg["model"]["z_channels"]),
                          z_hw=int(cfg["model"]["z_hw"]), base_ch=int(cfg["model"]["base_ch"]), seed=a.seed)
        member = EnsembleMember(spec, device=a.device)
        opt_theta, opt_z = _make_optimizer(member, tcfg)
        curve, t0 = {}, time.perf_counter()
        for step in range(1, iters + 1):
            opt_theta.zero_grad(set_to_none=True)
            if opt_z is not None:
                opt_z.zero_grad(set_to_none=True)
            loss = (member.complex_image() - x).abs().pow(2).mean()
            loss.backward()
            if tcfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(list(member.parameters()), max_norm=tcfg.grad_clip)
            opt_theta.step()
            if opt_z is not None:
                opt_z.step()
            if step in CHECKPOINTS or step == iters:
                with torch.no_grad():
                    xf = member.complex_image()
                    curve[step] = score(align_to_gt(xf, x).aligned_recon, x)
        with torch.no_grad():
            xf = member.complex_image()
            row = {"mnist_test_index": idx, "psnr": score(align_to_gt(xf, x).aligned_recon, x),
                   "fe": fourier_magnitude_error(xf, x, oversample=ov),
                   "min_amplitude_in_background": float(member.amplitude()[x.abs() == 0].min()),
                   "psnr_curve": {str(k): v for k, v in curve.items()},
                   "runtime_s": time.perf_counter() - t0}
        rows.append(row)
        print(f"image {idx}: PSNR {row['psnr']:.2f} dB, FE {row['fe']:.2e}, {row['runtime_s']:.0f} s", flush=True)

    ps, fe = np.array([r["psnr"] for r in rows]), np.array([r["fe"] for r in rows])
    summary = {"note": "diagnostic; generator fitted directly to the ground truth; development images only",
               "config": a.config, "object": "complex" if complex_obj else "real", "seed": a.seed,
               "iterations": iters, "lr_theta": tcfg.lr_theta, "lr_z": tcfg.lr_z, "grad_clip": tcfg.grad_clip,
               "n_images": len(rows), "psnr_mean": float(ps.mean()), "psnr_min": float(ps.min()),
               "fe_median": float(np.median(fe)), "fe_max": float(fe.max()), "images": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"mean PSNR {ps.mean():.2f} dB (min {ps.min():.2f}), median FE {np.median(fe):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
