#!/usr/bin/env python
"""Real-data demonstration: one diffraction pattern with unmeasured pixels.

The pattern is the CXIDB entry 28 measurement after scripts/cxidb28_preprocess.py (256 x 256,
zero frequency at (128, 128), boolean mask of measured pixels), or a simulated twin with the same
mask (pr_uq/data/cxidb28_twin.py). The object field is 128 x 128 with oversampling 2.

Methods
- "ours": K starts of the untrained generator (128 x 128 output), then the protocol of the
  held-out studies with the masked Fourier error: best-FE anchor, alignment, and the
  reconstructions k1, uniform, median, best_single_by_fe and fe_softmax (temperature T).
- classical entries: N starts each of HIO or RAAR with a constraint set; reported as the
  lowest-FE start among the first k starts for each k in `report_k`.

Scale. The real intensities are in detector units. They are multiplied by the factor that makes
the sum over measured pixels equal to that of the noiseless twin with `scale_reference_twin_seed`,
so that the object amplitude is of the order of the twin's (about 0.5), inside the bound of 1 of
the generator's sigmoid and of the shrinkwrap constraint. The data loss of our solver is multiplied
by REF_MEAN_Y2 / mean(y^2 over measured pixels), the scale it has for the 32 x 32 objects of the
held-out studies, so that the frozen regularization weights keep their relative size.

Scores. FE over measured pixels for every start and reconstruction. For the twin also the complex
PSNR and NRMSE after the alignment of the held-out studies (translation, conjugate flip, global phase).

Every start is cached under <output>/cache; an interrupted run resumes with the same command.

    python scripts/run_real_data.py --config configs/realdata_dev/<name>.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.baselines.oversampled import run_hio_full, run_raar_full  # noqa: E402
from pr_uq.data import cxidb28_twin as twin  # noqa: E402
from pr_uq.metrics.alignment import align_ensemble_to_anchor, align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import nrmse_complex, psnr_complex  # noqa: E402
from pr_uq.models.ensemble import EnsembleMember, MemberSpec  # noqa: E402
from pr_uq.optim.trainer import TrainerConfig, train_member  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors  # noqa: E402
from pr_uq.utils.config import resolve_device  # noqa: E402

# median of mean(y^2) over the 8 development complex objects of the held-out studies (32 x 32,
# oversampling 2, noiseless); real objects give 0.0116
REF_MEAN_Y2 = 0.00706
FIELD = 128
OVERSAMPLE = 2


def _rpr():
    """compute_strategies and the environment record of the held-out runner, unchanged."""
    spec = importlib.util.spec_from_file_location("rpr", REPO / "scripts" / "run_phase_retrieval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_measurement(dcfg: dict):
    """Return y (1,1,256,256), mask (1,1,256,256) bool, gt or None, info dict."""
    d = np.load(REPO / dcfg["pattern_npz"])
    mask = torch.from_numpy(d["mask"].astype(bool)).view(1, 1, 2 * FIELD, 2 * FIELD)
    info = {"pattern_npz": dcfg["pattern_npz"], "n_measured": int(mask.sum()),
            "n_unmeasured": int((~mask).sum())}
    if dcfg["source"] == "twin":
        gt = twin.exit_wave(int(dcfg["twin_seed"]))
        y = twin.measurement(gt, mask, dcfg.get("noise", "none"), dcfg.get("noise_level"),
                             int(dcfg.get("noise_seed", 0)))
        info.update(source="twin", twin_seed=int(dcfg["twin_seed"]), noise=dcfg.get("noise", "none"),
                    noise_level=dcfg.get("noise_level"), noise_seed=int(dcfg.get("noise_seed", 0)),
                    intensity_scale=1.0)
        return y, mask, gt, info
    if dcfg["source"] == "cxidb28":
        raw = torch.from_numpy(d["magnitude"].astype(np.float64) ** 2).view(1, 1, 2 * FIELD, 2 * FIELD)
        raw = torch.where(mask, raw, torch.zeros_like(raw))
        ref = twin.measurement(twin.exit_wave(int(dcfg["scale_reference_twin_seed"])), mask, "none")
        s = float(ref.double().sum() / raw.sum())
        info.update(source="cxidb28", intensity_scale=s,
                    scale_reference_twin_seed=int(dcfg["scale_reference_twin_seed"]))
        return (raw * s).float(), mask, None, info
    raise ValueError(f"unknown source {dcfg['source']!r}")


def score(recon: torch.Tensor, gt: torch.Tensor | None) -> dict:
    if gt is None:
        return {}
    a = align_to_gt(recon, gt).aligned_recon
    return {"psnr_c": psnr_complex(a, gt), "nrmse_c": nrmse_complex(a, gt)}


def run_starts(name, n, fn, cache: Path, device, logger):
    """Run or resume n starts; fn(seed) -> (recon (1,1,H,W) complex, runtime_s)."""
    cache.mkdir(parents=True, exist_ok=True)
    recons, runtimes = [], []
    for k in range(n):
        f = cache / f"start_{k:03d}.pt"
        if f.exists():
            b = torch.load(f, map_location=device)
            recons.append(b["recon"].to(device)); runtimes.append(float(b["runtime_s"]))
            continue
        r, t = fn(k)
        torch.save({"recon": r.detach().cpu(), "seed": k, "runtime_s": t}, f)
        recons.append(r.detach()); runtimes.append(t)
        logger(f"{name} start {k + 1}/{n} | {t:.1f}s")
    return torch.cat(recons, dim=0), runtimes


def save_stack(path: Path, stack: torch.Tensor, **extra):
    a = stack.detach().cpu().numpy()[:, 0]
    np.savez_compressed(path, real=a.real.astype(np.float32), imag=a.imag.astype(np.float32), **extra)


def preview(path: Path, panels: list[tuple[str, torch.Tensor]]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axs = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    for ax, (title, x) in zip(np.atleast_1d(axs), panels):
        ax.imshow(x.detach().cpu().abs().numpy()[0, 0], cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="auto | cuda | cpu")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = resolve_device(args.device or cfg["experiment"].get("device", "auto"))
    out = REPO / cfg["experiment"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    rpr = _rpr()
    (out / "environment.json").write_text(json.dumps(rpr._environment_record(device), indent=2), encoding="utf-8")
    logf = open(out / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    y, mask, gt, info = load_measurement(cfg["data"])
    y, mask = y.to(device), mask.to(device)
    if gt is not None:
        gt = gt.to(device)
    np.savez_compressed(out / "measurement.npz", y=y.cpu().numpy()[0, 0], mask=mask.cpu().numpy()[0, 0],
                        **({} if gt is None else {"gt_real": gt.cpu().numpy()[0, 0].real,
                                                  "gt_imag": gt.cpu().numpy()[0, 0].imag}))
    log(f"device {device} | {info}")
    metrics = {"data": info, "device": device}
    if gt is not None:
        metrics["data"]["fe_ground_truth"] = float(member_fourier_errors(gt, y, OVERSAMPLE, mask=mask)[0])
    panels = [] if gt is None else [("twin", gt)]

    # ---- our solver ----------------------------------------------------------------------------
    oc = cfg["methods"].get("ours")
    if oc:
        tdict = dict(oc["trainer"])
        mean_y2 = float((y[mask] ** 2).mean())
        tdict["data_loss_scale"] = REF_MEAN_Y2 / mean_y2
        tcfg = TrainerConfig(**{k: v for k, v in tdict.items() if k in TrainerConfig.__dataclass_fields__})
        K, base = int(oc["K"]), int(oc["base_seed"])
        mc = oc["model"]

        def fit(k):
            spec = MemberSpec(out_h=FIELD, out_w=FIELD, z_channels=int(mc["z_channels"]), z_hw=int(mc["z_hw"]),
                              base_ch=int(mc["base_ch"]), seed=base + k)
            r, lg = train_member(EnsembleMember(spec, device=device), y, tcfg, device=device, mask=mask)
            return r, lg.runtime_s

        stack, rt = run_starts("ours", K, fit, out / "cache" / "ours", device, log)
        fe = member_fourier_errors(stack, y, OVERSAMPLE, mask=mask)
        anchor = int(fe.argmin())
        aligned = align_ensemble_to_anchor(stack, anchor_index=anchor)
        strat, best = rpr.compute_strategies(stack, y, fe, anchor, aligned,
                                             float(cfg.get("temperature", 0.05)), OVERSAMPLE)
        agg = {}
        for nm, (xc, _) in strat.items():
            agg[nm] = {"fe": float(member_fourier_errors(xc, y, OVERSAMPLE, mask=mask)[0]), **score(xc, gt)}
        metrics["ours"] = {"data_loss_scale": tdict["data_loss_scale"], "start_fe": fe.tolist(),
                           "start_runtime_s": rt, "best_index": best,
                           "start_scores": [score(stack[k:k + 1], gt) for k in range(K)],
                           "aggregates": agg}
        save_stack(out / "ours_starts.npz", stack, fe=fe.numpy())
        np.savez_compressed(out / "ours_aggregates.npz", **{f"{nm}__{p}": getattr(xc.cpu().numpy()[0, 0], p)
                                                            for nm, (xc, _) in strat.items() for p in ("real", "imag")})
        panels += [(f"ours fe_softmax FE={agg['fe_softmax']['fe']:.3f}", strat["fe_softmax"][0]),
                   (f"ours k1 FE={agg['k1']['fe']:.3f}", strat["k1"][0])]
        log(f"ours | start FE {np.round(fe.numpy(), 4).tolist()} | fe_softmax FE {agg['fe_softmax']['fe']:.4f} "
            + ("" if gt is None else f"| NRMSE k1 {agg['k1']['nrmse_c']:.3f} fe_softmax {agg['fe_softmax']['nrmse_c']:.3f}"))

    # ---- classical methods ---------------------------------------------------------------------
    for c in cfg["methods"].get("classical", []):
        nm = f"{c['algorithm']}_{c['constraint']}"
        run = {"hio": run_hio_full, "raar": run_raar_full}[c["algorithm"]]
        kw = dict(iters=int(c["iters"]), beta=float(c["beta"]), er_iters=int(c["er_iters"]),
                  oversample=OVERSAMPLE, device=device, constraint=c["constraint"], mask=mask)

        def solve(k, run=run, kw=kw):
            r = run(y, (FIELD, FIELD), seed=k, **kw)
            return r.recon, r.runtime_s

        n = int(c["starts"])
        stack, rt = run_starts(nm, n, solve, out / "cache" / nm, device, log)
        fe = member_fourier_errors(stack, y, OVERSAMPLE, mask=mask)
        best_of = {}
        for k in [k for k in c.get("report_k", [1, n]) if k <= n]:
            i = int(fe[:k].argmin())
            best_of[str(k)] = {"index": i, "fe": float(fe[i]), **score(stack[i:i + 1], gt)}
        metrics[nm] = {"start_fe": fe.tolist(), "start_runtime_s": rt, "best_of_first_k": best_of,
                       "start_scores": [score(stack[k:k + 1], gt) for k in range(n)]}
        save_stack(out / f"{nm}_starts.npz", stack, fe=fe.numpy())
        i = best_of[str(n)]["index"]
        panels.append((f"{nm} best of {n} FE={float(fe[i]):.3f}", stack[i:i + 1]))
        log(f"{nm} | best of {n}: FE {float(fe[i]):.4f}"
            + ("" if gt is None else f" NRMSE {best_of[str(n)]['nrmse_c']:.3f}"))

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    preview(out / "preview.png", panels)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
