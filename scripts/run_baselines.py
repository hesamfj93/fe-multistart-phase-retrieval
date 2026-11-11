#!/usr/bin/env python
"""Classical baselines (textbook HIO and RAAR on the full oversampled grid) on protocol images.

For every image, each algorithm (HIO, RAAR) runs under each object constraint
(support_only, real_nonneg) from `starts` random-phase seeds. Two results are reported per
algorithm and constraint:
  <alg>_<prior>_s1      the first start only (seed base_seed)
  <alg>_<prior>_best<N> the start with the lowest Fourier error against the measurement,
                        the same ground-truth-free selection rule as best-single-by-FE
where <prior> is "supp" (support only, the untrained method's prior) or "realpos"
(real and non-negative, the classical CDI prior).

The measurement is generated exactly as in run_phase_retrieval.py (same function, same
seed rule), and the metrics use the same alignment, PSNR, SSIM and FE code.

  python scripts/run_baselines.py --config configs/stage2_mnist_heldout_baselines.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.baselines.oversampled import run_hio_full, run_raar_full  # noqa: E402
from pr_uq.data.datasets import build_dataset, generate_measurement  # noqa: E402
from pr_uq.metrics.alignment import align_to_gt  # noqa: E402
from pr_uq.metrics.reconstruction import fourier_magnitude_error, psnr, ssim  # noqa: E402
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors  # noqa: E402
from pr_uq.utils.config import load_config, resolve_device, save_config  # noqa: E402
from pr_uq.utils.logging import get_logger, write_json  # noqa: E402

PRIOR_TAG = {"support_only": "supp", "real_nonneg": "realpos"}
ALGORITHMS = {"hio": run_hio_full, "raar": run_raar_full}


def _runner_module():
    spec = importlib.util.spec_from_file_location("rpr_shared", REPO / "scripts" / "run_phase_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNNER = _runner_module()


def method_names(cfg) -> list[str]:
    b = cfg["baselines"]
    n = int(b["starts"])
    out = []
    for alg in b["algorithms"]:
        for c in b["constraints"]:
            out += [f"{alg}_{PRIOR_TAG[c]}_s1", f"{alg}_{PRIOR_TAG[c]}_best{n}"]
    return out


def process_image(sample, rank, cfg, device, out_dir, logger) -> dict:
    ds_idx = RUNNER.dataset_index(sample)
    img_id = f"image_{ds_idx:05d}"
    img_dir = out_dir / "per_image" / img_id
    mpath = img_dir / "metrics.json"
    if mpath.exists():
        try:
            prev = json.loads(mpath.read_text(encoding="utf-8"))
            if prev.get("status") == "ok" and prev.get("name") == sample.name:
                logger.info(f"[{img_id}] SKIP, already complete")
                return {**prev, "_skipped": True}
        except Exception:
            pass
    img_dir.mkdir(parents=True, exist_ok=True)
    b = cfg["baselines"]
    t_img = time.perf_counter()
    try:
        img = sample.image.to(device)
        gt = img.to(torch.complex64)
        H, W = img.shape[-2:]
        ov = int(cfg["measurement"]["oversample"])
        y, meta = generate_measurement(
            image=img, oversample=ov, noise_model=cfg["measurement"]["noise_model"],
            noise_level=cfg["measurement"]["noise_level"],
            seed=int(cfg["measurement"].get("noise_seed", 0)) + ds_idx, device=device)
        methods, starts_info, arrays = {}, {}, {"gt": img, "measurement": y}
        n = int(b["starts"])
        for alg in b["algorithms"]:
            fn = ALGORITHMS[alg]
            beta = float(b[f"{alg}_beta"])
            for c in b["constraints"]:
                tag = f"{alg}_{PRIOR_TAG[c]}"
                res = [fn(y, (H, W), iters=int(b["iterations"]), beta=beta,
                          er_iters=int(b["er_iterations"]), seed=int(b["base_seed"]) + k,
                          oversample=ov, device=device, constraint=c) for k in range(n)]
                stack = torch.cat([r.recon for r in res], dim=0)
                fe_meas = member_fourier_errors(stack, y, oversample=ov)
                best = int(fe_meas.argmin())
                start_psnr = []
                for k in range(n):
                    start_psnr.append(psnr(align_to_gt(stack[k:k + 1], gt).aligned_recon, gt))
                for label, k, rt in [("s1", 0, res[0].runtime_s),
                                     (f"best{n}", best, sum(r.runtime_s for r in res))]:
                    rec = stack[k:k + 1]
                    aligned = align_to_gt(rec, gt).aligned_recon
                    methods[f"{tag}_{label}"] = {
                        "psnr": psnr(aligned, gt), "ssim": ssim(aligned, gt),
                        "fe": fourier_magnitude_error(rec, gt, oversample=ov), "runtime_s": float(rt)}
                    arrays[f"recon_{tag}_{label}"] = aligned
                starts_info[tag] = {"start_fe": [float(x) for x in fe_meas.tolist()],
                                    "start_psnr": [float(x) for x in start_psnr],
                                    "best_index": best, "start_seeds": [int(b["base_seed"]) + k for k in range(n)]}
                arrays[f"starts_{tag}"] = stack
        RUNNER._save_npz(img_dir / "baselines.npz", **arrays)
        _preview(img_dir / "preview.png", img, arrays, methods, cfg)
        result = {
            "status": "ok", "image_id": img_id, "name": sample.name, "label": sample.label,
            "mnist_test_index": ds_idx, "protocol_split": cfg["dataset"].get("protocol_split"),
            "split_rank": rank, "device_used": device, "kind": "classical_baselines",
            "methods": methods, "starts": starts_info,
            "runtime_total_s": float(time.perf_counter() - t_img),
            "noise_model": meta["noise_model"], "noise_level": meta["noise_level"], "error": None,
        }
        mpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
        short = "  ".join(f"{k}={v['psnr']:.1f}" for k, v in methods.items() if "best" in k)
        logger.info(f"[{img_id}] OK | {short} | {result['runtime_total_s']:.1f}s")
        return result
    except Exception as e:
        err = {"status": "failed", "image_id": img_id, "name": sample.name, "mnist_test_index": ds_idx,
               "protocol_split": cfg["dataset"].get("protocol_split"), "split_rank": rank,
               "device_used": device, "error": str(e), "traceback": traceback.format_exc(),
               "runtime_total_s": time.perf_counter() - t_img}
        mpath.write_text(json.dumps(err, indent=2), encoding="utf-8")
        logger.error(f"[{img_id}] FAILED: {e}")
        return err


def _preview(path, img, arrays, methods, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    keys = [k for k in methods if "best" in k]
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(2.0 * (len(keys) + 1), 2.3))
    axes[0].imshow(img.squeeze().cpu().numpy(), cmap="gray"); axes[0].set_title("GT", fontsize=8)
    for a, k in zip(axes[1:], keys):
        a.imshow(np.abs(arrays[f"recon_{k}"].squeeze().cpu().numpy()), cmap="gray")
        a.set_title(f"{k}\n{methods[k]['psnr']:.1f} dB", fontsize=7)
    for a in axes:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main() -> int:
    RUNNER._safe_console()
    ap = argparse.ArgumentParser(description="Classical HIO/RAAR baselines on protocol images.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-images", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto | cuda | cpu")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if "baselines" not in cfg:
        raise SystemExit("config has no 'baselines' section")
    if args.n_images is not None: cfg["dataset"]["n_images"] = args.n_images
    if args.device is not None: cfg["experiment"]["device"] = args.device
    if args.output_dir is not None: cfg["experiment"]["output_dir"] = args.output_dir
    role = cfg["dataset"].get("protocol_split")
    device = resolve_device(cfg["experiment"]["device"])
    if role in RUNNER.PROTOCOL_ROLES and device == "cpu" and cfg["experiment"]["device"] == "auto":
        raise SystemExit("CUDA is not available and device=auto would fall back to CPU silently. "
                         "Pass --device cpu to run the baselines on the CPU on purpose.")
    out_dir = Path(cfg["experiment"]["output_dir"])
    samples = build_dataset(cfg["dataset"])   # raises on any development overlap

    if args.dry_run:
        if out_dir.exists():
            RUNNER._check_or_create_lock(cfg, out_dir, create=False)
        done = 0
        for rank, smp in enumerate(samples):
            mj = out_dir / "per_image" / f"image_{RUNNER.dataset_index(smp):05d}" / "metrics.json"
            ok = mj.exists() and json.loads(mj.read_text(encoding="utf-8")).get("status") == "ok"
            done += ok
        from pr_uq.data.splits import DEV_INDICES
        overlap = sorted({RUNNER.dataset_index(s) for s in samples} & set(DEV_INDICES))
        print(f"split={role} n={len(samples)} complete={done} to_run={len(samples) - done} "
              f"device={device} methods={method_names(cfg)} overlap_with_dev={overlap}")
        return 0

    for sub in ["per_image", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    RUNNER._check_or_create_lock(cfg, out_dir, create=True)
    logger = get_logger("pr_baselines", output_dir=out_dir / "logs")
    save_config(cfg, out_dir / "config.yaml")
    env = RUNNER._environment_record(device)
    write_json(env, out_dir / "environment.json")
    with open(out_dir / "logs" / "environment_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **env}) + "\n")
    from pr_uq.data.splits import DEV_INDICES, file_sha256, split_path
    sp = split_path(cfg["dataset"].get("splits_dir", "splits"), role) if role else None
    idx = [RUNNER.dataset_index(s) for s in samples]
    write_json({"protocol_split": role, "dev_indices": list(DEV_INDICES), "selected_indices": idx,
                "overlap_with_dev": sorted(set(idx) & set(DEV_INDICES)),
                "split_file": str(sp) if sp else None, "split_file_sha256": file_sha256(sp) if sp else None,
                "frozen_baselines": cfg["baselines"], "methods": method_names(cfg)},
               out_dir / "split_manifest.json")
    logger.info(f"Device: {device} | split: {role} | {len(samples)} images | methods: {method_names(cfg)}")
    n_ok = n_fail = n_skip = 0
    for rank, smp in enumerate(samples):
        r = process_image(smp, rank, cfg, device, out_dir, logger)
        if r.get("status") == "ok":
            n_ok += 1; n_skip += int(bool(r.get("_skipped")))
        else:
            n_fail += 1
    logger.info(f"Done. ok={n_ok} (skipped {n_skip}) failed={n_fail} of {len(samples)}")
    print(f"\nRun complete: {n_ok} ok ({n_skip} already complete), {n_fail} failed. Results in {out_dir.resolve()}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
