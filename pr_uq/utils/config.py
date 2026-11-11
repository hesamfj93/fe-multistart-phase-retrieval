"""YAML config loading + simple merging with defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "evaluation": {
        "failure_psnr_db": 15.0,
        "auroc_min_per_class": 5,
    },
    "experiment": {
        "name": "untitled",
        "output_dir": "results/untitled",
        "seed": 0,
        "device": "auto",                # 'auto' | 'cpu' | 'cuda' | 'cuda:0'
        "save_arrays": True,
        "save_images": True,
        "save_checkpoints": False,
    },
    "dataset": {
        "name": "mnist",
        "n_images": 20,
        "size": 32,
        "root": "./data",
        "split": "test",
        "download": True,
        "seed": 0,                      # legacy selection only (no protocol_split)
        "protocol_split": None,         # 'dev' | 'heldout' | 'smoke' (MNIST test set)
        "splits_dir": "splits",         # location of the split index CSV files
    },
    "measurement": {
        "oversample": 2,
        "noise_model": "none",           # 'none' | 'gaussian' | 'poisson'
        "noise_level": 0.0,
        "noise_seed": 0,
    },
    "model": {
        "z_channels": 128,
        "z_hw": 4,
        "base_ch": 128,
    },
    "ensemble": {
        "K": 1,
        "base_seed": 0,
    },
    "trainer": {
        "iterations": 3000,
        "lr_theta": 1e-3,
        "lr_z": 1e-3,
        "optimize_z": True,
        "grad_clip": 1.0,
        "data_loss": "magnitude_l2",
        "lambda_tv_A": 1e-3,
        "lambda_tv_phi": 1e-4,
        "lambda_phi_l2": 0.0,
        "lambda_support": 1e-2,
        "use_spectral_annealing": True,
        "sigma_start_frac": 0.5,
        "sigma_end_frac": 0.05,
        "annealing_schedule": "log",
        "annealing_fraction": 0.6,
        "use_physics_projection": True,
        "beta_schedule": {"type": "linear", "start": 0.7, "end": 0.1},
        "project_in_loss": True,
        "project_detached": True,
        "use_adaptive_support": True,
        "support_initial_fill": "full",
        "support_update_every": 300,
        "support_smooth_sigma": 1.0,
        "support_threshold_frac": 0.08,
        "support_gamma": 10.0,
        "log_every": 100,
    },
    "uncertainty": {
        "alpha": 0.5,
        "alphas_sweep": [0.0, 0.25, 0.5, 0.75, 1.0],
        "normalization": "robust",
        "summary": "mean",
        "n_bins_ece": 15,
    },
    "failure_detection": {
        "threshold_kind": "relative",
        "threshold_value_db": 15.0,
        "margin_db": 5.0,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. Returns a new dict."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and merge with defaults."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    if not isinstance(user, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(user)}")
    return deep_merge(DEFAULT_CONFIG, user)


def resolve_device(spec: str) -> str:
    """Resolve 'auto' → 'cuda' if available else 'cpu'; pass-through otherwise."""
    import torch
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def save_config(config: dict, path: str | Path) -> None:
    """Persist a (possibly merged) config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
