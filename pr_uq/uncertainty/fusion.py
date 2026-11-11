"""Fuse ensemble-variance and physics-residual into a single uncertainty map.

The fused uncertainty is

    u = α · normalize(σ_pred) + (1 − α) · normalize(r_phys)

with α ∈ [0, 1]. α is selected on a validation split by maximizing
failure-detection AUROC. We provide:

  * `fuse_uncertainty` — single α fusion for one image.
  * `fuse_uncertainty_batch` — batched version.
  * `failure_score` — scalar per-image summary (mean / quantile / max).
  * `alpha_sweep` — utility to evaluate AUROC across α grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .ensemble_uq import normalized_uncertainty


@dataclass
class FusedUncertainty:
    """Container for a fused uncertainty map and its components."""

    u: torch.Tensor                # (1, 1, H, W) ∈ [0, 1] fused uncertainty
    sigma_norm: torch.Tensor       # normalized ensemble std
    r_norm: torch.Tensor           # normalized physics residual (image domain)
    alpha: float                   # mixing coefficient used
    failure_score: float           # scalar summary


def fuse_uncertainty(
    sigma_pred: torch.Tensor,
    r_phys: torch.Tensor,
    alpha: float = 0.5,
    normalization: str = "robust",
    summary: str = "mean",
) -> FusedUncertainty:
    """Fuse a single image's σ_pred and r_phys into one uncertainty map.

    Parameters
    ----------
    sigma_pred : (1, 1, H, W) ensemble std (non-negative).
    r_phys : (1, 1, H, W) physics residual in image domain (non-negative).
    alpha : weight on σ_pred; (1 − α) on r_phys.
    normalization : 'robust' (default) or 'minmax'.
    summary : 'mean' | 'p95' | 'max' — how to summarize into a scalar
        failure_score.
    """
    sigma_n = normalized_uncertainty(sigma_pred, method=normalization)
    r_n = normalized_uncertainty(r_phys, method=normalization)
    u = alpha * sigma_n + (1.0 - alpha) * r_n
    u = u.clamp(0, 1)
    score = failure_score(u, summary=summary)
    return FusedUncertainty(
        u=u, sigma_norm=sigma_n, r_norm=r_n, alpha=alpha, failure_score=score
    )


def failure_score(u: torch.Tensor, summary: str = "mean") -> float:
    """Reduce an uncertainty map to a single per-image scalar."""
    if summary == "mean":
        return float(u.mean().item())
    if summary == "p95":
        return float(torch.quantile(u.flatten(), 0.95).item())
    if summary == "max":
        return float(u.max().item())
    raise ValueError(f"Unknown summary {summary!r}")


def alpha_sweep(
    sigma_list: list[torch.Tensor],
    r_phys_list: list[torch.Tensor],
    psnr_list: list[float],
    failure_threshold_db: float,
    alphas: list[float] | None = None,
    normalization: str = "robust",
    summary: str = "mean",
) -> dict[float, float]:
    """Compute failure-detection AUROC across a grid of α values.

    Parameters
    ----------
    sigma_list, r_phys_list : aligned lists of per-image tensors.
    psnr_list : aligned-PSNR per image.
    failure_threshold_db : reconstructions with PSNR < threshold are 'failures'.
    alphas : grid of α values. Defaults to {0, 0.1, …, 1.0}.

    Returns
    -------
    dict mapping α -> AUROC.
    """
    if alphas is None:
        alphas = [i / 10.0 for i in range(11)]
    from .failure_detection import compute_auroc

    labels = [1 if p < failure_threshold_db else 0 for p in psnr_list]
    out: dict[float, float] = {}
    for a in alphas:
        scores = []
        for sig, rp in zip(sigma_list, r_phys_list):
            fu = fuse_uncertainty(sig, rp, alpha=a, normalization=normalization, summary=summary)
            scores.append(fu.failure_score)
        out[a] = compute_auroc(scores, labels)
    return out
