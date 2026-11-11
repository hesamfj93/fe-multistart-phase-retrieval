"""LEGACY MODULE, used only by scripts/_archive/. Not protocol-compliant.

The Stage-2 runner is scripts/run_phase_retrieval.py, which selects images from the
split files in splits/. Do not use run_experiment() for evaluation.

End-to-end pipeline used by every script.

`run_experiment(config)` does:
  1. seeds everything,
  2. loads the dataset subset,
  3. for each image: generates measurement, trains K-ensemble, evaluates,
  4. writes per-image CSV row + saves arrays + saves images,
  5. computes aggregate metrics and writes summary CSV/JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .data.datasets import build_dataset, generate_measurement, ImageSample
from .metrics.alignment import align_ensemble_to_anchor, align_to_gt
from .metrics.reconstruction import score_against_gt
from .models.ensemble import Ensemble, EnsembleMember, MemberSpec
from .optim.trainer import TrainerConfig, train_member
from .physics.projection import physics_residual
from .uncertainty.ensemble_uq import ensemble_aggregate, ensemble_mean_and_std, normalized_uncertainty
from .uncertainty.fusion import fuse_uncertainty
from .uncertainty.calibration import calibrate
from .uncertainty.failure_detection import evaluate_failure_detection
from .utils.config import resolve_device, save_config
from .utils.logging import MetricsWriter, get_logger, write_json
from .utils.seed import seed_everything


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _trainer_config_from(d: dict) -> TrainerConfig:
    """Cast a dict to TrainerConfig, dropping unknown keys with a warning."""
    fields = TrainerConfig.__dataclass_fields__.keys()
    known = {k: v for k, v in d.items() if k in fields}
    unknown = sorted(set(d.keys()) - set(fields))
    if unknown:
        # Log via print here — logger isn't constructed yet at first call
        print(f"[pipeline] WARN: ignoring unknown trainer keys: {unknown}")
    return TrainerConfig(**known)


def _save_arrays(out_dir: Path, name: str, **arrays: torch.Tensor) -> None:
    """Save a bundle of named tensors as a single .npz file."""
    payload = {}
    for k, v in arrays.items():
        if v is None:
            continue
        a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        # Split complex into real/imag for safe round-tripping
        if np.iscomplexobj(a):
            payload[f"{k}__real"] = a.real
            payload[f"{k}__imag"] = a.imag
        else:
            payload[k] = a
    np.savez_compressed(out_dir / f"{name}.npz", **payload)


def _save_image(t: torch.Tensor, path: Path) -> None:
    """Save a (1,1,H,W) real or complex tensor as a PNG (magnitude for complex)."""
    arr = t.detach().cpu()
    if torch.is_complex(arr):
        arr = arr.abs()
    arr = arr.squeeze().numpy()
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-12)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    # Write a minimal PNG via numpy + PIL fallback
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        # Save as .npy instead
        np.save(path.with_suffix(".npy"), arr)


# -----------------------------------------------------------------------------
# Per-image processing
# -----------------------------------------------------------------------------

def _process_image(
    sample: ImageSample,
    config: dict,
    device: str,
    out_dirs: dict[str, Path],
    logger,
) -> dict:
    """Run the full per-image pipeline; return a flat dict of metrics."""
    t_start = time.perf_counter()
    img = sample.image.to(device)
    H, W = img.shape[-2:]

    # ---- Forward measurement -----------------------------------------------
    y, meas_meta = generate_measurement(
        image=img,
        oversample=config["measurement"]["oversample"],
        noise_model=config["measurement"]["noise_model"],
        noise_level=config["measurement"]["noise_level"],
        seed=int(config["measurement"]["noise_seed"]) + hash(sample.name) % 100000,
        device=device,
    )

    # ---- Ensemble training -------------------------------------------------
    K = int(config["ensemble"]["K"])
    base_seed = int(config["ensemble"]["base_seed"])
    spec = MemberSpec(
        out_h=H, out_w=W,
        z_channels=int(config["model"]["z_channels"]),
        z_hw=int(config["model"]["z_hw"]),
        base_ch=int(config["model"]["base_ch"]),
    )
    trainer_cfg = _trainer_config_from(config["trainer"])

    recons: list[torch.Tensor] = []
    losses: list[list[float]] = []
    runtimes: list[float] = []
    psnr_curves: list[list[float]] = []
    fe_curves: list[list[float]] = []
    curve_steps: list[list[int]] = []
    members_dir = out_dirs["arrays"].parent / "members"
    members_dir.mkdir(parents=True, exist_ok=True)
    for k in range(K):
        member_pt = members_dir / f"{sample.name}__k{k:02d}.pt"
        if member_pt.exists():
            # Resume: load already-trained member
            blob = torch.load(member_pt, map_location=device)
            recons.append(blob["recon"].to(device))
            losses.append(blob.get("loss_curve", []))
            runtimes.append(float(blob.get("runtime_s", 0.0)))
            psnr_curves.append(blob.get("psnr_curve", []))
            fe_curves.append(blob.get("fe_curve", []))
            curve_steps.append(blob.get("steps", []))
            logger.info(f"[{sample.name}] member {k+1}/{K} [resumed from disk]")
            continue
        member_spec = MemberSpec(
            out_h=spec.out_h, out_w=spec.out_w,
            z_channels=spec.z_channels, z_hw=spec.z_hw, base_ch=spec.base_ch,
            seed=base_seed + k,
        )
        member = EnsembleMember(member_spec, device=device)
        recon, log = train_member(member, y, trainer_cfg, device=device, gt=img)
        # Persist this member immediately
        torch.save(
            {
                "recon": recon.detach().cpu(),
                "loss_curve": log.loss_total,
                "psnr_curve": log.psnr_curve,
                "fe_curve": log.fourier_err_curve,
                "steps": log.steps,
                "runtime_s": log.runtime_s,
                "seed": base_seed + k,
            },
            member_pt,
        )
        recons.append(recon)
        losses.append(log.loss_total)
        runtimes.append(log.runtime_s)
        psnr_curves.append(log.psnr_curve)
        fe_curves.append(log.fourier_err_curve)
        curve_steps.append(log.steps)
        logger.info(
            f"[{sample.name}] member {k+1}/{K} done | "
            f"final_loss={log.loss_total[-1]:.4e} | "
            f"final_PSNR={log.psnr_curve[-1]:.2f}dB | "
            f"runtime={log.runtime_s:.1f}s"
        )

    stack = torch.cat(recons, dim=0)  # (K, 1, H, W) complex

    # ---- Ensemble statistics ----------------------------------------------
    # Aggregation strategy is configurable. Defaults (best_fe anchor + softmax
    # fidelity weighting) fix the K=5 regression where under-converged members
    # drag down a plain uniform mean; see FINAL_ALIGNMENT_REPORT.md. Set
    # anchor_method=first, weight_method=uniform to reproduce the legacy path.
    unc_cfg = config.get("uncertainty", {})
    stats = ensemble_aggregate(
        stack,
        y_intensity=y,
        anchor_method=unc_cfg.get("anchor_method", "best_fe"),
        weight_method=unc_cfg.get("weight_method", "softmax"),
        temperature=float(unc_cfg.get("fidelity_temperature", 0.1)),
        top_k=unc_cfg.get("top_k", None),
        oversample=config["measurement"]["oversample"],
        align=(K > 1),
    )
    mean_recon = stats.mean_complex

    # ---- Physics residual (image domain) ----------------------------------
    r_phys = physics_residual(
        mean_recon,
        y,
        oversample=config["measurement"]["oversample"],
        return_image_domain=True,
    )

    # ---- Uncertainty fusion ----------------------------------------------
    fused = fuse_uncertainty(
        sigma_pred=stats.std_pixelwise,
        r_phys=r_phys,
        alpha=float(config["uncertainty"]["alpha"]),
        normalization=config["uncertainty"]["normalization"],
        summary=config["uncertainty"]["summary"],
    )

    # ---- Reconstruction quality (with alignment) -------------------------
    align = align_to_gt(mean_recon, img.to(torch.complex64))
    aligned = align.aligned_recon
    scores = score_against_gt(
        mean_recon, img.to(torch.complex64),
        oversample=config["measurement"]["oversample"],
        runtime_s=sum(runtimes),
    )

    # ---- Calibration ------------------------------------------------------
    abs_err = (aligned.real - img.real).abs()
    cal = calibrate(
        uncertainty=fused.u,
        abs_error=abs_err,
        n_bins=int(config["uncertainty"]["n_bins_ece"]),
    )

    # ---- Persist ----------------------------------------------------------
    if config["experiment"].get("save_arrays", True):
        _save_arrays(
            out_dirs["arrays"],
            sample.name,
            gt=img,
            measurement=y,
            recon_mean=mean_recon,
            recon_aligned=aligned,
            stack=stack,
            std_pixelwise=stats.std_pixelwise,
            r_phys=r_phys,
            uncertainty=fused.u,
            abs_error=abs_err,
        )
    # Save per-iteration curves (training diagnostic)
    try:
        import numpy as np
        np.savez_compressed(
            out_dirs["curves"] / f"{sample.name}.npz",
            steps=np.array(curve_steps, dtype=np.int32),
            loss=np.array(losses, dtype=np.float32),
            psnr=np.array(psnr_curves, dtype=np.float32),
            fourier_err=np.array(fe_curves, dtype=np.float32),
            runtimes=np.array(runtimes, dtype=np.float32),
        )
    except Exception as e:
        logger.warning(f"curve save failed for {sample.name}: {e}")
    if config["experiment"].get("save_images", True):
        try:
            _save_image(img, out_dirs["images"] / f"{sample.name}_gt.png")
            _save_image(aligned, out_dirs["images"] / f"{sample.name}_recon.png")
            _save_image(fused.u, out_dirs["images"] / f"{sample.name}_uncertainty.png")
            if K > 1:
                _save_image(stats.std_pixelwise, out_dirs["images"] / f"{sample.name}_std.png")
        except Exception as e:
            logger.warning(f"image save failed for {sample.name}: {e}")

    elapsed = time.perf_counter() - t_start
    return {
        "name": sample.name,
        "dataset": sample.dataset or "unknown",
        "label": sample.label if sample.label is not None else -1,
        "K": K,
        "psnr": scores.psnr,
        "ssim": scores.ssim,
        "mse": scores.mse,
        "nmse": scores.nmse,
        "raw_psnr": scores.raw_psnr,
        "fourier_mag_err": scores.fourier_mag_error,
        "variant": scores.variant,
        "shift_dy": scores.shift[0],
        "shift_dx": scores.shift[1],
        "ece": cal.ece,
        "pearson_r": cal.pearson_r,
        "spearman_r": cal.spearman_r,
        "nll": cal.nll,
        "failure_score": fused.failure_score,
        "alpha": fused.alpha,
        "runtime_total_s": elapsed,
        "runtime_train_s": float(sum(runtimes)),
        "noise_model": meas_meta["noise_model"],
        "noise_level": meas_meta["noise_level"],
    }


# -----------------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------------

def run_experiment(config: dict) -> dict:
    """Run the full pipeline for one config. Returns a summary dict.

    Resume behavior: if `<output_dir>/arrays/<name>.npz` already exists for
    an image, that image is skipped and its previously-computed metrics row
    is re-read from `per_image.csv`. This lets you split a long run across
    several short invocations: just re-run the same command, and only the
    unfinished images get processed. To force a clean run, delete the
    output directory first.
    """
    seed_everything(int(config["experiment"]["seed"]), deterministic=False)
    device = resolve_device(config["experiment"]["device"])
    out_dir = Path(config["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = {
        "configs": out_dir / "configs",
        "images": out_dir / "images",
        "curves": out_dir / "curves",
        "checkpoints": out_dir / "checkpoints",
        "metrics": out_dir / "metrics",
        "arrays": out_dir / "arrays",
    }
    for p in out_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    logger = get_logger("pr_uq", output_dir=out_dir)
    logger.info(f"Output directory: {out_dir.resolve()}")
    logger.info(f"Device: {device}")

    # Persist effective config + a couple of provenance things
    save_config(config, out_dirs["configs"] / "effective.yaml")
    try:
        import torch as _torch
        manifest = {
            "torch_version": _torch.__version__,
            "cuda_available": _torch.cuda.is_available(),
            "cuda_device_count": _torch.cuda.device_count(),
            "config_path": str((out_dirs["configs"] / "effective.yaml").resolve()),
        }
    except Exception as e:
        manifest = {"warning": str(e)}
    write_json(manifest, out_dir / "manifest.json")

    # Build dataset
    samples = build_dataset(config["dataset"])
    logger.info(f"Dataset {config['dataset']['name']}: {len(samples)} images")

    # Resume: load existing per_image.csv rows, skip images whose .npz already exists
    csv_path = out_dirs["metrics"] / "per_image.csv"
    existing_rows: dict[str, dict] = {}
    if csv_path.exists():
        import csv as _csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                existing_rows[r["name"]] = r
        logger.info(f"Resume: found {len(existing_rows)} pre-existing rows in CSV")

    # Reopen MetricsWriter in append mode if rows already exist
    writer = MetricsWriter(csv_path)
    if existing_rows:
        # Tell the writer its schema so it appends rather than rewriting header
        writer._fieldnames = list(next(iter(existing_rows.values())).keys())

    rows: list[dict] = list(existing_rows.values())
    n_skipped = 0
    for i, sample in enumerate(samples):
        arr_file = out_dirs["arrays"] / f"{sample.name}.npz"
        if sample.name in existing_rows and arr_file.exists():
            logger.info(f"--- ({i+1}/{len(samples)}) {sample.name} [SKIP - already done] ---")
            n_skipped += 1
            continue
        logger.info(f"--- ({i+1}/{len(samples)}) {sample.name} ---")
        try:
            row = _process_image(sample, config, device, out_dirs, logger)
        except Exception as e:
            logger.error(f"Image {sample.name} failed: {e}")
            raise
        writer.write_row(row)
        rows.append(row)
        logger.info(
            f"[{sample.name}] PSNR={row['psnr']:.2f}dB SSIM={row['ssim']:.3f} "
            f"ECE={row['ece']:.4f} fail_score={row['failure_score']:.3f}"
        )
    if n_skipped:
        logger.info(f"Resumed run: {n_skipped} pre-existing images skipped, "
                    f"{len(rows) - n_skipped} processed this call.")

    # Aggregate
    if rows:
        # Coerce CSV-loaded string values back to numeric for aggregation.
        # Rows produced this session are already numeric; resumed rows are str.
        def _coerce(v):
            if not isinstance(v, str):
                return v
            try:
                return int(v) if v.isdigit() or (v.startswith("-") and v[1:].isdigit()) else float(v)
            except ValueError:
                return v
        norm_rows = [{k: _coerce(v) for k, v in r.items()} for r in rows]

        psnrs = [r["psnr"] for r in norm_rows]
        failure_scores = [r["failure_score"] for r in norm_rows]
        fdet = evaluate_failure_detection(
            failure_scores=failure_scores,
            psnr_list=psnrs,
            threshold_kind=config["failure_detection"]["threshold_kind"],
            threshold_value_db=float(config["failure_detection"]["threshold_value_db"]),
            margin_db=float(config["failure_detection"]["margin_db"]),
            summary_name=config["uncertainty"]["summary"],
        )

        def _mean(key: str) -> float:
            xs = [r[key] for r in norm_rows]
            return float(sum(xs) / len(xs))

        def _med(key: str) -> float:
            xs = sorted([r[key] for r in norm_rows])
            return float(xs[len(xs) // 2])

        summary = {
            "n_images": len(norm_rows),
            "K": norm_rows[0]["K"],
            "mean_psnr": _mean("psnr"),
            "median_psnr": _med("psnr"),
            "mean_ssim": _mean("ssim"),
            "mean_fourier_mag_err": _mean("fourier_mag_err"),
            "mean_ece": _mean("ece"),
            "mean_pearson": _mean("pearson_r"),
            "mean_spearman": _mean("spearman_r"),
            "auroc_failure_detection": fdet.auroc,
            "failure_threshold_db": fdet.threshold_db,
            "n_failures": fdet.n_failures,
            "n_successes": fdet.n_successes,
            "mean_runtime_train_s": _mean("runtime_train_s"),
        }
    else:
        summary = {"n_images": 0}

    write_json(summary, out_dir / "summary.json")
    # Plaintext one-liner for quick scanning
    if rows:
        with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

    logger.info("Done.")
    logger.info(json.dumps(summary, indent=2))
    return summary
