"""Exploratory, development data only: the FE aggregates of the held-out protocol applied to the
50 starts of each classical method on the twins (cached stacks, no new reconstruction).
Usage: python fe_on_classical.py twin0 twin1 ...
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pr_uq.metrics.alignment import align_ensemble_to_anchor, align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import nrmse_complex  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors  # noqa: E402

spec = importlib.util.spec_from_file_location("rpr", REPO / "scripts" / "run_phase_retrieval.py")
rpr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rpr)
torch.set_num_threads(1)
res = {}
for tw in sys.argv[1:]:
    d = REPO / "results" / "realdata_dev" / f"{tw}_classical"
    m = np.load(d / "measurement.npz")
    y = torch.from_numpy(m["y"]).view(1, 1, 256, 256).float()
    mask = torch.from_numpy(m["mask"].astype(bool)).view(1, 1, 256, 256)
    gt = torch.from_numpy(m["gt_real"] + 1j * m["gt_imag"]).view(1, 1, 128, 128).to(torch.complex64)
    for nm in ("raar_shrinkwrap", "hio_shrinkwrap"):
        s = np.load(d / f"{nm}_starts.npz")
        stack = torch.from_numpy(s["real"] + 1j * s["imag"]).unsqueeze(1).to(torch.complex64)
        row = {}
        for K in (5, 10, 20, 50):
            st = stack[:K]
            fe = member_fourier_errors(st, y, 2, mask=mask)
            anchor = int(fe.argmin())
            aligned = align_ensemble_to_anchor(st, anchor_index=anchor)
            strat, _ = rpr.compute_strategies(st, y, fe, anchor, aligned, 0.05, 2)
            row[K] = {a: round(float(nrmse_complex(align_to_gt(x, gt).aligned_recon, gt)), 3)
                      for a, (x, _) in strat.items()}
        res[f"{tw}/{nm}"] = row
        print(tw, nm, json.dumps(row), flush=True)
Path(sys.argv[0]).with_name("fe_on_classical_" + "_".join(sys.argv[1:]) + ".json").write_text(json.dumps(res, indent=2))
