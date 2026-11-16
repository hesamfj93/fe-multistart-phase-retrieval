#!/usr/bin/env python
"""Diagnostic: sensitivity of RAAR with shrinkwrap to a rounding-size change of its start.

Reruns the first 10 starts of the equal-wall-time configuration (configs/stage3_complex_heldout_raar45.yaml)
on CPU for held-out ranks 100, 120, 140, 160 and 180, with the random initial phase of each start
perturbed by a uniform value in [-1e-6, 1e-6] rad per pixel, and compares each start with the
unperturbed CPU run (results/stage3_complex_heldout_raar45). It counts starts whose PSNR (capped at
60 dB) moved by less than 0.1 dB and starts whose failure status (< 15 dB) changed. Not a test of any
hypothesis; it explains why the CPU and GPU runs of the same configuration differ on single images.

    python scripts/raar_rounding_sensitivity.py --out results/raar_rounding_sensitivity
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pr_uq.baselines.oversampled as ov  # noqa: E402
from pr_uq.physics.forward import inverse_field  # noqa: E402

EPS = 1e-6
RANKS = (100, 120, 140, 160, 180)
N_STARTS = 10


def init_perturbed(y_mag, seed):
    """The initializer of pr_uq.baselines.oversampled with a perturbation of size EPS added to the phase."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    phase = torch.rand(tuple(y_mag.shape), generator=g) * 2 * torch.pi
    g2 = torch.Generator().manual_seed(10_000 + int(seed))
    phase = phase + EPS * (2 * torch.rand(tuple(y_mag.shape), generator=g2) - 1)
    z0 = y_mag.to(torch.complex64) * torch.exp(1j * phase.to(y_mag.device))
    return inverse_field(z0, signal_hw=tuple(y_mag.shape[-2:])).real.to(torch.complex64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/raar_rounding_sensitivity")
    ap.add_argument("--reference", default="results/stage3_complex_heldout_raar45")
    a = ap.parse_args()
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("rb", REPO / "scripts" / "run_baselines.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    cfg = rb.load_config(str(REPO / "configs" / "stage3_complex_heldout_raar45.yaml"))
    cfg["experiment"]["device"] = "cpu"
    cfg["baselines"]["starts"] = N_STARTS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ov._init = init_perturbed
    logger = logging.getLogger("raar_rounding_sensitivity")
    rows = []
    for rank, smp in enumerate(rb.build_dataset(cfg["dataset"]), start=100):
        if rank not in RANKS:
            continue
        rb.process_image(smp, rank, cfg, "cpu", out, logger)
        idx = rb.RUNNER.dataset_index(smp)
        new = json.loads(next(out.glob(f"per_image/*{idx:05d}*/metrics.json")).read_text(encoding="utf-8"))
        old = json.loads(next(Path(a.reference).glob(f"per_image/*{idx:05d}*/metrics.json")).read_text(encoding="utf-8"))
        p0 = np.minimum(old["starts"]["raar_sw"]["start_psnr"][:N_STARTS], 60)
        p1 = np.minimum(new["starts"]["raar_sw"]["start_psnr"][:N_STARTS], 60)
        rows.append({"rank": rank, "mnist_test_index": idx, "unperturbed_psnr": p0.tolist(), "perturbed_psnr": p1.tolist(),
                     "within_0p1_db": int((np.abs(p0 - p1) < 0.1).sum()),
                     "failure_status_changed": int(((p0 < 15) != (p1 < 15)).sum())})
        print(idx, rows[-1]["within_0p1_db"], rows[-1]["failure_status_changed"], flush=True)
    summary = {"eps_rad": EPS, "ranks": list(RANKS), "starts_per_image": N_STARTS, "n_starts": N_STARTS * len(rows),
               "within_0p1_db": sum(r["within_0p1_db"] for r in rows),
               "failure_status_changed": sum(r["failure_status_changed"] for r in rows), "images": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"within 0.1 dB: {summary['within_0p1_db']} of {summary['n_starts']}; "
          f"failure status changed: {summary['failure_status_changed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
