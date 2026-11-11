"""Adapter interface for external baselines.

We provide a uniform API:

    result = run_baseline(name, y_intensity, image_hw, config) -> BaselineResult

For HIO and RAAR (implemented in-tree) this dispatches directly. For
external repositories (Zhuang Double-DIP, Net-ADM, etc.) the adapter
either calls the external code (if installed) or raises `NotImplementedError`
with a clear pointer to `README_external.md`.

We **never** silently return fake numbers — if an external method isn't
available, the wrapper errors out loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .hio import run_hio
from .raar import run_raar


@dataclass
class BaselineResult:
    name: str
    recon: torch.Tensor
    runtime_s: float
    final_data_loss: float
    extra: dict


def _bool_or(d: dict, key: str, default: bool) -> bool:
    return bool(d.get(key, default))


def run_baseline(
    name: str,
    y_intensity: torch.Tensor,
    image_hw: tuple[int, int],
    config: dict[str, Any],
) -> BaselineResult:
    """Dispatch to the named baseline."""
    name = name.lower()
    if name == "hio":
        r = run_hio(
            y_intensity=y_intensity,
            image_hw=image_hw,
            iters=int(config.get("iters", 2000)),
            beta=float(config.get("beta", 0.9)),
            restarts=int(config.get("restarts", 3)),
            support_area_frac=float(config.get("support_area_frac", 0.5)),
            oversample=int(config.get("oversample", 2)),
            seed=int(config.get("seed", 0)),
            device=config.get("device", "cpu"),
        )
        return BaselineResult(
            name="HIO",
            recon=r.recon,
            runtime_s=r.runtime_s,
            final_data_loss=r.final_data_loss,
            extra={"iters": r.iters, "restarts": r.restarts},
        )
    if name == "raar":
        r = run_raar(
            y_intensity=y_intensity,
            image_hw=image_hw,
            iters=int(config.get("iters", 2000)),
            beta=float(config.get("beta", 0.87)),
            restarts=int(config.get("restarts", 3)),
            support_area_frac=float(config.get("support_area_frac", 0.5)),
            oversample=int(config.get("oversample", 2)),
            seed=int(config.get("seed", 0)),
            device=config.get("device", "cpu"),
        )
        return BaselineResult(
            name="RAAR",
            recon=r.recon,
            runtime_s=r.runtime_s,
            final_data_loss=r.final_data_loss,
            extra={"iters": r.iters, "restarts": r.restarts},
        )
    if name in {"zhuang", "double_dip", "doubledip"}:
        raise NotImplementedError(
            "Zhuang Double-DIP requires the external repo. See "
            "pr_uq/baselines/README_external.md for installation."
        )
    if name in {"net_adm", "netadm"}:
        raise NotImplementedError(
            "Net-ADM requires the external repo. See README_external.md."
        )
    if name in {"accelerated_trad", "atrad"}:
        raise NotImplementedError(
            "Accelerated-TRAD requires the external repo. See README_external.md."
        )
    if name in {"deepmmse"}:
        raise NotImplementedError(
            "DeepMMSE reimplementation pending. See README_external.md."
        )
    if name in {"lodip"}:
        raise NotImplementedError(
            "LoDIP requires the external repo. See README_external.md."
        )
    raise ValueError(f"Unknown baseline {name!r}")
