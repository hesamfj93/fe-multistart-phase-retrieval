"""Exploratory diagnostic, outside REALDATA_DEV_PLAN.md: is the failure of run A an
implementation error or a limitation of the solver on this object?

One start of run A settings (seed 0) on twin 0 without noise and without a mask (D1),
and on twin 0 blurred by a Gaussian of sigma 2 px, also noiseless and unmasked (D2).
Usage: python diag.py D1|D2 out_dir
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pr_uq.data import cxidb28_twin as twin  # noqa: E402
from pr_uq.metrics.alignment import align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import nrmse_complex  # noqa: E402
from pr_uq.models.ensemble import EnsembleMember, MemberSpec  # noqa: E402
from pr_uq.optim.trainer import TrainerConfig, train_member  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors  # noqa: E402

which, out = sys.argv[1], Path(sys.argv[2])
IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 429
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
cfg = yaml.safe_load((REPO / "configs/realdata_dev/twin0_oursA.yaml").read_text())
gt = twin.exit_wave(0)
N = 32 if which == "D4" else 128
if which in ("D3", "D4"):
    from pr_uq.data.datasets import make_mnist_offline
    from pr_uq.data.complex_objects import to_complex_object
    smp = make_mnist_offline(size=N, root=str(REPO / "data"), split="test", indices=[IDX])[0]
    gt = to_complex_object(smp.image.view(1, 1, N, N), seed=IDX)
if which == "D2":
    n = gt.shape[-1]
    f = torch.fft.fftfreq(n)
    g = torch.exp(-2 * (np.pi * 2.0) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    gt = torch.fft.ifft2(torch.fft.fft2(gt) * g)
# the measurement function of the runner, with an all-True mask and no noise
mask_all = torch.ones(1, 1, 2 * N, 2 * N, dtype=torch.bool)
y = twin.measurement(gt, mask_all, "none").float()
tdict = dict(cfg["methods"]["ours"]["trainer"])
tdict["data_loss_scale"] = 1.0 if which == "D4" else 0.00706 / float((y ** 2).mean())
tcfg = TrainerConfig(**{k: v for k, v in tdict.items() if k in TrainerConfig.__dataclass_fields__})
mc = cfg["methods"]["ours"]["model"]
spec = MemberSpec(out_h=N, out_w=N, z_channels=int(mc["z_channels"]), z_hw=int(mc["z_hw"]),
                  base_ch=int(mc["base_ch"]), seed=0)
t0 = time.time()
r, lg = train_member(EnsembleMember(spec, device="cpu"), y, tcfg, device="cpu", mask=None)
fe = float(member_fourier_errors(r, y, 2)[0])
a = align_to_gt(r, gt).aligned_recon
res = {"which": which, "idx": IDX, "fe": fe, "fe_truth": float(member_fourier_errors(gt, y, 2)[0]),
       "nrmse_c": float(nrmse_complex(a, gt)), "runtime_s": time.time() - t0}
print(json.dumps(res))
(out / f"{which}_{IDX}.json").write_text(json.dumps(res, indent=2))
torch.save({"recon": r.detach(), "gt": gt}, out / f"{which}_{IDX}.pt")
