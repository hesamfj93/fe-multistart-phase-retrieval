#!/usr/bin/env python
"""Consolidated phase-retrieval experiment runner.

One runner, one config system, five reconstruction strategies, full resume.

Strategies evaluated per image (all from a single trained K-member stack):
  1. K=1                       — single-start (member 0 only)
  2. uniform                   — K=5 uniform mean (legacy anchor)
  3. median                    — K=5 pixel-wise median (best-FE anchor)
  4. best_single_by_fe         — the single lowest-FE member
  5. fe_softmax                — best-FE anchor + FE-softmax weights (T from config)

Scientific wording (kept conservative in all outputs):
  * the plain-mean degradation is a "uniform aggregation failure", not a bug;
  * FE is a practical, empirically evaluated, ground-truth-free reliability
    proxy, not a guaranteed best-member estimator;
  * reconstruction selection and fidelity-weighted fusion are two complementary
    operating modes.

Output layout:
  results/<name>/
    config.yaml
    environment.json
    per_image/image_xxxxx/{members.npz, aggregates.npz, metrics.json, preview.png}
    logs/run.log

Resume: an image whose per_image/<id>/metrics.json exists and parses is skipped.
Member-level resume is handled below the image level: each trained member is
cached under per_image/<id>/_members/, so an interrupted image continues from
the last completed member without recomputing valid ones.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.data.datasets import build_dataset, generate_measurement
from pr_uq.metrics.alignment import align_ensemble_to_anchor, align_to_gt, map_to_gt_frame
from pr_uq.metrics.reconstruction import psnr, ssim, fourier_magnitude_error
from pr_uq.models.ensemble import EnsembleMember, MemberSpec
from pr_uq.optim.trainer import TrainerConfig, train_member
from pr_uq.physics.projection import physics_residual
from pr_uq.uncertainty.ensemble_uq import (
    member_fourier_errors, select_anchor, fidelity_weights, effective_sample_size,
    pixelwise_median_complex,
)
from pr_uq.uncertainty.fusion import fuse_uncertainty
from pr_uq.uncertainty.calibration import calibrate
from pr_uq.utils.config import load_config, resolve_device, save_config
from pr_uq.utils.logging import get_logger, write_json
from pr_uq.utils.seed import seed_everything


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _c2ri(a: np.ndarray) -> dict:
    """Split a complex array into real/imag for npz round-tripping."""
    return {"real": a.real.astype(np.float32), "imag": a.imag.astype(np.float32)}


def _save_npz(path: Path, **arrays) -> None:
    payload = {}
    for k, v in arrays.items():
        if v is None:
            continue
        a = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        if np.iscomplexobj(a):
            payload[f"{k}__real"] = a.real.astype(np.float32)
            payload[f"{k}__imag"] = a.imag.astype(np.float32)
        else:
            payload[k] = a
    np.savez_compressed(path, **payload)


def _environment_record(device: str) -> dict:
    import platform
    rec = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": device,
        "numpy": np.__version__,
    }
    if torch.cuda.is_available():
        rec["cuda_version"] = torch.version.cuda
        rec["gpu_name"] = torch.cuda.get_device_name(0)
        rec["gpu_count"] = torch.cuda.device_count()
    # git commit if available
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), stderr=subprocess.DEVNULL
        ).decode().strip()
        rec["git_commit"] = sha
    except Exception:
        rec["git_commit"] = None
    return rec


def _trainer_from_cfg(d: dict) -> TrainerConfig:
    fields = TrainerConfig.__dataclass_fields__.keys()
    return TrainerConfig(**{k: v for k, v in d.items() if k in fields})


# --------------------------------------------------------------------------
# Reconstruction strategies (all operate on a saved, aligned stack)
# --------------------------------------------------------------------------

def compute_strategies(
    stack: torch.Tensor,           # (K,1,H,W) complex, RAW (unaligned)
    y: torch.Tensor,               # (1,1,oH,oW) measured intensity
    fe: torch.Tensor,              # (K,) per-member Fourier error
    anchor: int,
    aligned: torch.Tensor,         # (K,1,H,W) complex aligned to best-FE anchor
    temperature: float,
    oversample: int,
) -> dict:
    """Return {name: (mean_complex, std_or_None)} for all five strategies."""
    K = stack.shape[0]
    out = {}

    # 1. K=1 — member 0, no uncertainty
    out["k1"] = (stack[0:1], None)

    # 2. uniform — legacy anchor (member 0) + uniform mean
    aligned_legacy = align_ensemble_to_anchor(stack, anchor_index=0)
    umean = aligned_legacy.mean(dim=0, keepdim=True)
    ustd = aligned_legacy.abs().std(dim=0, keepdim=True, unbiased=False)
    out["uniform"] = (umean, ustd)

    # std of the best-FE-aligned stack (shared uncertainty basis for 3-5)
    aligned_std = aligned.abs().std(dim=0, keepdim=True, unbiased=False)

    # 3. median — best-FE anchor + per-pixel median
    out["median"] = (pixelwise_median_complex(aligned), aligned_std)

    # 4. best_single_by_fe — the single lowest-FE member (uncertainty from full stack)
    best = int(fe.argmin())
    out["best_single_by_fe"] = (stack[best:best+1], aligned_std)

    # 5. fe_softmax — best-FE anchor + softmax weights
    # fe (and therefore w) lives on the CPU; move it to the reconstructions' device.
    w = fidelity_weights(fe, method="softmax", temperature=temperature).to(
        device=aligned.device, dtype=aligned.dtype)
    wmean = (aligned * w.view(-1, 1, 1, 1)).sum(dim=0, keepdim=True)
    # softmax-weighted std around the weighted mean
    mags = aligned.abs()
    wr = w.real.view(-1, 1, 1, 1)
    wmag = (mags * wr).sum(dim=0, keepdim=True)
    wstd = (wr * (mags - wmag).pow(2)).sum(dim=0, keepdim=True).clamp_min(0).sqrt()
    out["fe_softmax"] = (wmean, wstd)

    return out, best


# Version of the pixel-wise uncertainty evaluation written into metrics.json.
#   1: u compared with the error map without moving u into the ground-truth frame
#      (misregistered whenever the reconstruction was shifted or flipped).
#   2: u and the physics residual are moved into the ground-truth frame first.
UNCERTAINTY_EVAL_VERSION = 2


def score_reconstruction(mean_c, std, gt, y, oversample, n_bins_ece=15):
    """PSNR/SSIM/FE (+ fused-uncertainty calibration if std given).

    Returned maps (aligned reconstruction, u, physics residual, error) are all in the
    ground-truth frame.
    """
    gt_res = align_to_gt(mean_c, gt)
    aligned = gt_res.aligned_recon
    rec = {
        "psnr": psnr(aligned, gt),
        "ssim": ssim(aligned, gt),
        "fe": fourier_magnitude_error(mean_c, gt, oversample=oversample),
    }
    if std is not None:
        r_phys = physics_residual(mean_c, y, oversample=oversample, return_image_domain=True)
        fused = fuse_uncertainty(std, r_phys, alpha=0.5, normalization="robust", summary="mean")
        # u and r_phys live in the frame of mean_c; the error lives in the frame of gt.
        u_gt = map_to_gt_frame(fused.u, gt_res)
        r_gt = map_to_gt_frame(r_phys, gt_res)
        err = (aligned.real - gt.real).abs()
        cal = calibrate(u_gt, err, n_bins=n_bins_ece)
        rec.update({
            "unc_pearson": cal.pearson_r,
            "unc_spearman": cal.spearman_r,
            "unc_ece": cal.ece,
            "failure_score": fused.failure_score,   # spatial mean, frame-independent
        })
        return rec, aligned, u_gt, r_gt, err
    return rec, aligned, None, None, None


# --------------------------------------------------------------------------
# Per-image processing
# --------------------------------------------------------------------------

def dataset_index(sample) -> int:
    """MNIST test-set index encoded in the sample name (e.g. mnist_006051 -> 6051)."""
    return int(sample.name.rsplit("_", 1)[1])


def process_image(sample, rank, cfg, device, out_dir, logger) -> dict:
    ds_idx = dataset_index(sample)
    img_id = f"image_{ds_idx:05d}"
    img_dir = out_dir / "per_image" / img_id
    metrics_path = img_dir / "metrics.json"

    # Image-level resume
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("status") == "ok" and prev.get("name") == sample.name:
                logger.info(f"[{img_id}] SKIP, already complete")
                return {**prev, "_skipped": True}
        except Exception:
            pass  # corrupt -> recompute

    img_dir.mkdir(parents=True, exist_ok=True)
    members_cache = img_dir / "_members"
    members_cache.mkdir(exist_ok=True)

    t_img = time.perf_counter()
    try:
        img = sample.image.to(device)
        H, W = img.shape[-2:]
        oversample = cfg["measurement"]["oversample"]
        K = int(cfg["ensemble"]["K"])
        base_seed = int(cfg["ensemble"]["base_seed"])
        temperature = float(cfg["uncertainty"].get("fidelity_temperature", 0.05))

        # Measurement (deterministic per image via fixed seed)
        y, meas_meta = generate_measurement(
            image=img, oversample=oversample,
            noise_model=cfg["measurement"]["noise_model"],
            noise_level=cfg["measurement"]["noise_level"],
            seed=int(cfg["measurement"].get("noise_seed", 0)) + ds_idx,
            device=device,
        )

        trainer_cfg = _trainer_from_cfg(cfg["trainer"])

        # Train / resume each member
        recons, seeds, runtimes = [], [], []
        n_resumed = 0
        member_psnr, member_ssim = [], []
        for k in range(K):
            seed_k = base_seed + k
            mc = members_cache / f"member_{k:02d}.pt"
            if mc.exists():
                blob = torch.load(mc, map_location=device)
                recons.append(blob["recon"].to(device))
                seeds.append(int(blob["seed"]))
                runtimes.append(float(blob["runtime_s"]))
                member_psnr.append(float(blob["psnr"]))
                member_ssim.append(float(blob["ssim"]))
                n_resumed += 1
                logger.info(f"[{img_id}] member {k+1}/{K} [resumed]")
                continue
            spec = MemberSpec(out_h=H, out_w=W,
                              z_channels=int(cfg["model"]["z_channels"]),
                              z_hw=int(cfg["model"]["z_hw"]),
                              base_ch=int(cfg["model"]["base_ch"]),
                              seed=seed_k)
            member = EnsembleMember(spec, device=device)
            recon, log = train_member(member, y, trainer_cfg, device=device, gt=img)
            # per-member eval-only metrics (aligned to GT)
            aln = align_to_gt(recon, img.to(torch.complex64)).aligned_recon
            p_k = psnr(aln, img.to(torch.complex64))
            s_k = ssim(aln, img.to(torch.complex64))
            torch.save({"recon": recon.detach().cpu(), "seed": seed_k,
                        "runtime_s": log.runtime_s, "psnr": p_k, "ssim": s_k}, mc)
            recons.append(recon); seeds.append(seed_k); runtimes.append(log.runtime_s)
            member_psnr.append(p_k); member_ssim.append(s_k)
            logger.info(f"[{img_id}] member {k+1}/{K} done | seed={seed_k} "
                        f"| PSNR={p_k:.2f} | {log.runtime_s:.1f}s")

        t_post = time.perf_counter()
        stack = torch.cat(recons, dim=0)
        gt_c = img.to(torch.complex64)

        # FE, anchor, alignment (only meaningful for K>1)
        fe = member_fourier_errors(stack, y, oversample=oversample)
        if K > 1:
            anchor = select_anchor(stack, method="best_fe", y_intensity=y, oversample=oversample)
            aligned = align_ensemble_to_anchor(stack, anchor_index=anchor)
        else:
            anchor = 0
            aligned = stack

        if K > 1:
            strategies, best_idx = compute_strategies(
                stack, y, fe, anchor, aligned, temperature, oversample)
        else:
            # Single-start run (e.g. the compute-matched control): one reconstruction, no
            # ensemble methods, stored under its own name so it is never confused with the
            # K=1 member of a K=5 run.
            single = str(cfg["experiment"].get("single_method_name", "k1"))
            strategies, best_idx = {single: (stack[0:1], None)}, 0

        # softmax weights + K_eff for the record
        w_soft = fidelity_weights(fe, method="softmax", temperature=temperature)
        k_eff = effective_sample_size(w_soft)

        # Score every strategy
        method_metrics = {}
        recon_arrays = {}
        unc_arrays = {}
        for mname, (mean_c, std) in strategies.items():
            rec, aligned_recon, u, r_phys, err = score_reconstruction(
                mean_c, std, gt_c, y, oversample, cfg["uncertainty"].get("n_bins_ece", 15))
            method_metrics[mname] = rec
            recon_arrays[f"recon_{mname}"] = aligned_recon
            if u is not None:
                unc_arrays[f"unc_{mname}"] = u
                if f"rphys_{mname}" not in unc_arrays:
                    unc_arrays[f"rphys_{mname}"] = r_phys

        # ---- Save members.npz ----
        _save_npz(img_dir / "members.npz",
                  gt=img, measurement=y,
                  stack_raw=stack, stack_aligned=aligned,
                  member_fe=fe, member_seeds=np.array(seeds),
                  member_psnr=np.array(member_psnr), member_ssim=np.array(member_ssim),
                  softmax_weights=w_soft)

        # ---- Save aggregates.npz ----
        agg_payload = {**recon_arrays, **unc_arrays}
        # the shared ensemble std map (from best-FE aligned stack)
        if K > 1:
            agg_payload["ensemble_std"] = aligned.abs().std(dim=0, keepdim=True, unbiased=False)
        _save_npz(img_dir / "aggregates.npz", **agg_payload)

        # ---- preview.png ----
        _save_preview(img_dir / "preview.png", img, strategies, gt_c, unc_arrays, oversample)

        post_rt = time.perf_counter() - t_post
        wall_rt = time.perf_counter() - t_img
        # Training time comes from the member records, so it stays correct when
        # members were loaded from the cache instead of trained in this session.
        total_rt = float(sum(runtimes)) + post_rt
        result = {
            "status": "ok",
            "image_id": img_id,
            "name": sample.name,
            "mnist_test_index": ds_idx,
            "protocol_split": cfg["dataset"].get("protocol_split"),
            "split_rank": rank,
            "device_used": device,
            "label": sample.label if sample.label is not None else -1,
            "K": K,
            "anchor_index": int(anchor),
            "best_single_index": int(best_idx),
            "effective_sample_size": float(k_eff),
            "member_seeds": [int(s) for s in seeds],
            "member_fe": [float(x) for x in fe.tolist()],
            "member_psnr": [float(x) for x in member_psnr],
            "member_ssim": [float(x) for x in member_ssim],
            "softmax_weights": [float(x) for x in w_soft.tolist()],
            "methods": method_metrics,
            "runtime_per_member_s": [float(x) for x in runtimes],
            "runtime_total_s": float(total_rt),
            "runtime_postprocess_s": float(post_rt),
            "runtime_wall_this_session_s": float(wall_rt),
            "n_members_resumed": int(n_resumed),
            "noise_model": meas_meta["noise_model"],
            "noise_level": meas_meta["noise_level"],
            "uncertainty_eval_version": UNCERTAINTY_EVAL_VERSION if K > 1 else None,
            "error": None,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # clean member cache once the image is complete (keep disk small)
        if cfg["experiment"].get("cleanup_member_cache", True):
            for mc in members_cache.glob("member_*.pt"):
                mc.unlink()
            try:
                members_cache.rmdir()
            except OSError:
                pass

        shown = [m for m in ("k1", "fe_softmax", "best_single_by_fe") if m in method_metrics] or list(method_metrics)
        logger.info(f"[{img_id}] OK | best-FE member={best_idx} | "
                    + " ".join(f"{m}={method_metrics[m]['psnr']:.2f}" for m in shown)
                    + f" | {total_rt:.1f}s")
        return result

    except Exception as e:
        err = {"status": "failed", "image_id": img_id, "name": sample.name,
               "mnist_test_index": ds_idx, "protocol_split": cfg["dataset"].get("protocol_split"),
               "split_rank": rank, "device_used": device,
               "error": str(e), "traceback": traceback.format_exc(),
               "runtime_total_s": time.perf_counter() - t_img}
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(err, f, indent=2)
        logger.error(f"[{img_id}] FAILED: {e}")
        return err


def _save_preview(path, gt, strategies, gt_c, unc_arrays, oversample):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [n for n in ["k1", "uniform", "median", "best_single_by_fe", "fe_softmax"] if n in strategies] or list(strategies)
    fig, axes = plt.subplots(2, len(names) + 1, figsize=(2.0 * (len(names) + 1), 4.2))
    axes[0, 0].imshow(gt.real.squeeze().cpu().numpy(), cmap="gray")
    axes[0, 0].set_title("GT", fontsize=8); axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    for j, nm in enumerate(names):
        mean_c, std = strategies[nm]
        aln = align_to_gt(mean_c, gt_c).aligned_recon
        p = psnr(aln, gt_c)
        axes[0, j + 1].imshow(np.abs(aln.squeeze().cpu().numpy()), cmap="gray")
        axes[0, j + 1].set_title(f"{nm}\n{p:.1f}dB", fontsize=7); axes[0, j + 1].axis("off")
        key = f"unc_{nm}"
        if key in unc_arrays:
            axes[1, j + 1].imshow(unc_arrays[key].squeeze().cpu().numpy(), cmap="magma")
            axes[1, j + 1].set_title("uncertainty", fontsize=7)
        axes[1, j + 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

PROTOCOL_ROLES = {"dev", "heldout"}          # runs whose design must stay frozen
FINGERPRINT_EXCLUDE = {                     # keys that may differ between sessions
    ("dataset", "n_images"), ("dataset", "root"), ("dataset", "splits_dir"),
    ("experiment", "output_dir"), ("experiment", "device"),
    ("experiment", "cleanup_member_cache"),
}
FROZEN_METHODS = ["k1", "uniform", "median", "best_single_by_fe", "fe_softmax"]


def _frozen_view(cfg: dict) -> dict:
    """Config with session-specific keys removed; this is what gets fingerprinted."""
    import copy
    c = copy.deepcopy(cfg)
    for sec, key in FINGERPRINT_EXCLUDE:
        c.get(sec, {}).pop(key, None)
    return c


def _fingerprint(cfg: dict) -> str:
    import hashlib
    blob = json.dumps(_frozen_view(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _diff_keys(a, b, prefix=""):
    keys = set(a) | set(b) if isinstance(a, dict) and isinstance(b, dict) else set()
    out = []
    for k in sorted(keys):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _diff_keys(va, vb, f"{prefix}{k}.")
        elif va != vb:
            out.append(f"{prefix}{k}: locked={va!r} now={vb!r}")
    return out


def _protocol_record(cfg: dict) -> dict:
    """Machine-readable record of the split and the frozen design."""
    from pr_uq.data.splits import DEV_INDICES, split_path, file_sha256, read_split_csv
    ds = cfg["dataset"]
    role = ds.get("protocol_split")
    rec = {"protocol_split": role, "dev_indices": list(DEV_INDICES)}
    if role:
        sp = split_path(ds.get("splits_dir", "splits"), role)
        idx = read_split_csv(sp)[: int(ds["n_images"])]
        rec.update({
            "split_file": str(sp),
            "split_file_sha256": file_sha256(sp),
            "n_images_selected": len(idx),
            "selected_indices": idx,
            "overlap_with_dev": sorted(set(idx) & set(DEV_INDICES)) if role != "dev" else None,
        })
    rec["frozen_design"] = {
        "methods": FROZEN_METHODS,
        "anchor": "best_fe (lowest per-member Fourier error)",
        "alignment": "circular shift, flip, conjugate (twin), global phase",
        "fe_softmax_temperature": float(cfg["uncertainty"].get("fidelity_temperature", 0.05)),
        "K": int(cfg["ensemble"]["K"]),
        "base_seed": int(cfg["ensemble"]["base_seed"]),
        "iterations": int(cfg["trainer"]["iterations"]),
        "z_channels": int(cfg["model"]["z_channels"]),
        "base_ch": int(cfg["model"]["base_ch"]),
        "selected_on": "development images only (AGGREGATION_FINALIZATION_REPORT.md)",
    }
    return rec


def _check_or_create_lock(cfg: dict, out_dir: Path, create: bool) -> dict:
    """Create protocol_lock.json on first run; abort on any later design change."""
    from pr_uq.data.splits import split_path, file_sha256
    lock_path = out_dir / "protocol_lock.json"
    ds = cfg["dataset"]
    role = ds.get("protocol_split")
    sha = file_sha256(split_path(ds.get("splits_dir", "splits"), role)) if role else None
    now = {"fingerprint": _fingerprint(cfg), "frozen_config": _frozen_view(cfg),
           "protocol_split": role, "split_file_sha256": sha}
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        problems = []
        if lock["fingerprint"] != now["fingerprint"]:
            problems += _diff_keys(lock["frozen_config"], now["frozen_config"])
        if lock.get("split_file_sha256") != sha:
            problems.append("split index file changed since the run started")
        if lock.get("protocol_split") != role:
            problems.append(f"protocol_split locked={lock.get('protocol_split')} now={role}")
        if problems:
            raise SystemExit(
                "PROTOCOL LOCK MISMATCH: this output directory was created with a different "
                "design. Refusing to mix results.\n  " + "\n  ".join(problems) +
                "\nUse a new --output-dir, or restore the original configuration.")
        return lock
    if create:
        now["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        lock_path.write_text(json.dumps(now, indent=2), encoding="utf-8")
    return now


def _safe_console():
    """Never crash on a character the console code page cannot show (e.g. cp1256)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def main():
    _safe_console()
    ap = argparse.ArgumentParser(description="Consolidated phase-retrieval runner.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-images", type=int, default=None,
                    help="number of images (protocol runs: first N rows of the split file)")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto | cuda | cpu")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print which images would run or be skipped; train nothing, write nothing")
    ap.add_argument("--allow-protocol-override", action="store_true",
                    help="permit --iters/--K on dev/heldout runs (breaks the frozen protocol)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    role = cfg["dataset"].get("protocol_split")
    if role in PROTOCOL_ROLES and (args.iters is not None or args.K is not None) \
            and not args.allow_protocol_override:
        raise SystemExit(f"Refusing --iters/--K on a '{role}' protocol run: the Stage-2 design "
                         "is frozen. Edit nothing; use the config as provided.")
    if args.n_images is not None: cfg["dataset"]["n_images"] = args.n_images
    if args.iters is not None: cfg["trainer"]["iterations"] = args.iters
    if args.K is not None: cfg["ensemble"]["K"] = args.K
    if args.device is not None: cfg["experiment"]["device"] = args.device
    if args.output_dir is not None: cfg["experiment"]["output_dir"] = args.output_dir

    device = resolve_device(cfg["experiment"]["device"])
    if role in PROTOCOL_ROLES and device == "cpu" and cfg["experiment"]["device"] == "auto":
        raise SystemExit("CUDA is not available, and device=auto would silently fall back to CPU "
                         f"for a '{role}' run. Fix the CUDA install, or pass --device cpu to force CPU.")

    out_dir = Path(cfg["experiment"]["output_dir"])
    samples = build_dataset(cfg["dataset"])   # raises on any dev overlap for heldout/smoke

    if args.dry_run:
        if out_dir.exists():
            _check_or_create_lock(cfg, out_dir, create=False)
        done = 0
        for rank, smp in enumerate(samples):
            mj = out_dir / "per_image" / f"image_{dataset_index(smp):05d}" / "metrics.json"
            st = "skip (complete)" if mj.exists() and json.loads(mj.read_text(encoding="utf-8")).get("status") == "ok" else "run"
            done += st.startswith("skip")
            print(f"  rank {rank:>3}  {smp.name}  label={smp.label}  -> {st}")
        rec = _protocol_record(cfg)
        print(f"\nsplit={role} n={len(samples)} complete={done} to_run={len(samples) - done} "
              f"device={device} overlap_with_dev={rec.get('overlap_with_dev')}")
        return 0

    for sub in ["per_image", "logs", "tables", "figures"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    _check_or_create_lock(cfg, out_dir, create=True)

    seed_everything(int(cfg["experiment"]["seed"]), deterministic=False)
    logger = get_logger("pr_run", output_dir=out_dir / "logs")
    save_config(cfg, out_dir / "config.yaml")
    env = _environment_record(device)
    write_json(env, out_dir / "environment.json")
    with open(out_dir / "logs" / "environment_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **env}) + "\n")
    rec = _protocol_record(cfg)
    write_json(rec, out_dir / "split_manifest.json")
    with open(out_dir / "split_indices_used.csv", "w", encoding="utf-8") as f:
        f.write("rank,mnist_test_index,label\n")
        for rank, smp in enumerate(samples):
            f.write(f"{rank},{dataset_index(smp)},{smp.label}\n")
    logger.info(f"Device: {device} | split: {role} | output: {out_dir.resolve()}")
    logger.info(f"{len(samples)} images, K={cfg['ensemble']['K']}, "
                f"T={rec['frozen_design']['fe_softmax_temperature']}, "
                f"overlap_with_dev={rec.get('overlap_with_dev')}")

    n_ok = n_fail = n_skip = 0
    for rank, sample in enumerate(samples):
        res = process_image(sample, rank, cfg, device, out_dir, logger)
        if res.get("status") == "ok":
            n_ok += 1
            n_skip += int(bool(res.get("_skipped")))
        else:
            n_fail += 1
    logger.info(f"Done. ok={n_ok} (skipped {n_skip}) failed={n_fail} of {len(samples)}")
    print(f"\nRun complete: {n_ok} ok ({n_skip} already complete), {n_fail} failed. "
          f"Results in {out_dir.resolve()}")
    print(f"Next: python scripts/generate_report.py --run-dir {out_dir}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
