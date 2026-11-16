#!/usr/bin/env python
"""Choose the two Poisson levels of the third study (STUDY3_DEV_PLAN.md, Section 2).

For each photon budget N_p of the grid, the FE of the ground truth against the noisy measurement
is computed for the 8 development images, as real objects and as complex objects with the
stage-3 object model. The moderate level is the N_p whose median FE over the 16 objects is
closest to 0.03, the strong level the one closest to 0.10. No reconstruction is involved.

    python scripts/study3_noise_levels.py --out results/study3_dev/noise_levels.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.data.complex_objects import object_from_cfg  # noqa: E402
from pr_uq.data.datasets import build_dataset, generate_measurement  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors  # noqa: E402
from pr_uq.utils.config import load_config  # noqa: E402

GRID = (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7)
TARGETS = {"moderate": 0.03, "strong": 0.10}
CONFIGS = {"real": "configs/stage2_mnist_dev8.yaml", "complex": "configs/stage3_complex_dev.yaml"}
NOISE_SEED = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fe = {str(n): {} for n in GRID}
    for otype, cpath in CONFIGS.items():
        cfg = load_config(str(REPO / cpath))
        ov = int(cfg["measurement"]["oversample"])
        for smp in build_dataset(cfg["dataset"]):
            idx = int(str(smp.name).split("_")[-1])
            x = object_from_cfg(smp.image, cfg["dataset"], idx).to(torch.complex64)
            for n in GRID:
                y, _ = generate_measurement(image=x, oversample=ov, noise_model="poisson", noise_level=n,
                                            seed=NOISE_SEED + idx)
                fe[str(n)][f"{otype}_{idx}"] = float(member_fourier_errors(x, y, oversample=ov)[0])
    med = {n: float(np.median(list(v.values()))) for n, v in fe.items()}
    chosen = {k: min(GRID, key=lambda n: abs(med[str(n)] - t)) for k, t in TARGETS.items()}
    out = {"rule": "STUDY3_DEV_PLAN.md Section 2", "targets": TARGETS, "noise_seed": NOISE_SEED,
           "median_fe_of_ground_truth": med, "chosen_photon_budget": chosen, "fe_of_ground_truth": fe}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for n in GRID:
        print(f"N_p = {n:9.0f}: median FE of the ground truth {med[str(n)]:.4f}")
    print("chosen:", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
