"""FE of the FE-softmax reconstruction against the measurement, rebuilt from the stored arrays.

The runner stores, under methods/<rule>/fe in metrics.json, the FE of each aggregate against the
Fourier magnitudes of the ground truth. Without noise these equal the square roots of the measured
intensities, so the stored value is the FE of the paper. Under Poisson noise (Study 3) they differ,
and the stored value is computed against the noise-free magnitudes. The FE of the paper, against
the noisy measurement, is rebuilt here from members.npz: the FE-softmax aggregate is the
softmax-weighted sum of the aligned starts (stack_aligned, softmax_weights), as in the runner.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors  # noqa: E402


def fe_softmax_measured(image_dir: Path) -> float:
    z = np.load(image_dir / "members.npz")
    st = torch.from_numpy(z["stack_aligned__real"] + 1j * z["stack_aligned__imag"]).to(torch.complex64)
    w = torch.from_numpy(np.asarray(z["softmax_weights"], dtype=np.float32)).view(-1, 1, 1, 1).to(torch.complex64)
    y = torch.from_numpy(np.asarray(z["measurement"], dtype=np.float32))
    return float(member_fourier_errors((w * st).sum(0, keepdim=True), y, oversample=2)[0])


def load_with_measured_fe(run_dir: Path) -> list[dict]:
    """Per-image metrics of an untrained run, each with an added key fe_softmax_measured."""
    out = []
    for p in sorted((run_dir / "per_image").glob("image_*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("status", "ok") != "ok":
            continue
        m["fe_softmax_measured"] = fe_softmax_measured(p.parent)
        m["measurement_path"] = str(p.parent / "members.npz")
        out.append(m)
    return out
