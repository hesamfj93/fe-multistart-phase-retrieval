"""Pixel-wise statistics over an ensemble of reconstructions.

We provide:
  * `ensemble_mean_and_std` — aligned mean and std of (K, 1, H, W) stack.
  * `normalized_uncertainty` — robust per-image normalization to [0, 1].

The std map is the basic prior-induced uncertainty signal. It must be
fused with the physics residual (see `fusion.py`) for calibration.

K=1 case
--------
For a single member the ensemble std is zero everywhere. The functions
here handle that gracefully and return a flat-zero array.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..metrics.alignment import align_ensemble_to_anchor, align_to_gt
from ..physics.forward import forward_magnitude


@dataclass
class EnsembleStats:
    """Container for ensemble statistics on real-valued output."""

    mean_real: torch.Tensor          # (1, 1, H, W) — aligned real reconstruction
    mean_complex: torch.Tensor       # (1, 1, H, W) — aligned complex reconstruction
    std_pixelwise: torch.Tensor      # (1, 1, H, W) — per-pixel std across members
    n_members: int = 1
    anchor_index: int = 0            # which member was used as alignment anchor
    weights: torch.Tensor | None = None      # (K,) aggregation weights actually used
    member_fe: torch.Tensor | None = None    # (K,) per-member Fourier-magnitude error


def member_fourier_errors(
    stack: torch.Tensor,
    y_intensity: torch.Tensor,
    oversample: int = 2,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-member normalized Fourier-magnitude error against the measurement.

    FE_k = || |F(x_k)| - sqrt(Y) || / || sqrt(Y) ||

    This is a *ground-truth-free* quality proxy — it only uses the measured
    intensity Y, so it is available at inference time. Lower is better.
    Returns a (K,) tensor.
    """
    y_mag = y_intensity.clamp_min(0).sqrt()
    denom = y_mag.pow(2).sum().clamp_min(eps).sqrt()
    fes = []
    for k in range(stack.shape[0]):
        fp = forward_magnitude(stack[k : k + 1], oversample=oversample)
        fes.append((((fp - y_mag).pow(2).sum().sqrt()) / denom).item())
    # Returned on the CPU (a K-vector used for indexing and logging). Callers that
    # combine it with reconstructions must move it to the stack's device.
    return torch.tensor(fes, dtype=torch.float32)


def select_anchor(
    stack: torch.Tensor,
    method: str = "best_fe",
    y_intensity: torch.Tensor | None = None,
    oversample: int = 2,
) -> int:
    """Choose the alignment anchor member.

    'first'    — member 0 (legacy behaviour).
    'best_fe'  — member with the lowest Fourier-magnitude error (needs
                 `y_intensity`). This is the most reliable anchor: aligning
                 to the member that best explains the measurement avoids
                 anchoring on an outlier twin/shift variant.
    'medoid'   — member with the smallest mean magnitude distance to all
                 others (no measurement needed, but sensitive to a bad
                 majority).
    """
    K = stack.shape[0]
    if K == 1 or method == "first":
        return 0
    if method == "best_fe":
        if y_intensity is None:
            raise ValueError("select_anchor(method='best_fe') requires y_intensity")
        fes = member_fourier_errors(stack, y_intensity, oversample=oversample)
        return int(torch.argmin(fes).item())
    if method == "medoid":
        mags = stack.abs() if torch.is_complex(stack) else stack
        dists = torch.zeros(K)
        for i in range(K):
            for j in range(K):
                if i != j:
                    dists[i] += float((mags[i] - mags[j]).pow(2).mean().item())
        return int(torch.argmin(dists).item())
    raise ValueError(f"Unknown anchor method: {method!r}")


def fidelity_weights(
    member_fe: torch.Tensor,
    method: str = "softmax",
    temperature: float = 0.1,
    top_k: int | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Turn per-member Fourier errors into aggregation weights (sum to 1).

    'uniform'  — equal weights (standard deep-ensemble averaging).
    'softmax'  — w_k ∝ exp(-FE_k / temperature). Smoothly downweights
                 high-error members; temperature controls sharpness
                 (smaller = more aggressive downweighting).
    'inverse'  — w_k ∝ 1 / (FE_k + eps)^2.
    'top_k'    — hard selection: uniform over the `top_k` lowest-FE members,
                 zero for the rest.

    Downweighting under-converged members is what recovers the reconstruction
    quality lost when a plain mean averages good and bad members together.
    """
    fe = member_fe.float()
    K = fe.numel()
    if method == "uniform":
        return torch.ones(K, device=fe.device) / K
    if method == "top_k":
        k = top_k or K
        order = torch.argsort(fe)
        w = torch.zeros(K, device=fe.device)
        w[order[:k]] = 1.0 / k
        return w
    if method == "softmax":
        logits = -fe / max(temperature, eps)
        logits = logits - logits.max()
        w = torch.exp(logits)
        return w / w.sum().clamp_min(eps)
    if method == "inverse":
        w = 1.0 / (fe + eps).pow(2)
        return w / w.sum().clamp_min(eps)
    raise ValueError(f"Unknown weight method: {method!r}")


def _weighted_mean_std(
    stack: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted complex mean and weighted per-pixel magnitude std.

    stack   : (K, 1, H, W) complex (already aligned).
    weights : (K,) non-negative, summing to 1.
    """
    w = weights.to(device=stack.device, dtype=stack.real.dtype).view(-1, 1, 1, 1)
    mean_c = (stack * w.to(stack.dtype)).sum(dim=0, keepdim=True)
    mags = stack.abs() if torch.is_complex(stack) else stack
    mag_mean = (mags * w).sum(dim=0, keepdim=True)
    # Weighted variance (reliability-weighted); reduces to unbiased-ish for uniform w
    var = (w * (mags - mag_mean).pow(2)).sum(dim=0, keepdim=True)
    return mean_c, var.clamp_min(0).sqrt()


def ensemble_aggregate(
    stack: torch.Tensor,
    y_intensity: torch.Tensor | None = None,
    anchor_method: str = "best_fe",
    weight_method: str = "uniform",
    temperature: float = 0.1,
    top_k: int | None = None,
    oversample: int = 2,
    align: bool = True,
) -> EnsembleStats:
    """Full ensemble aggregation: pick anchor, align, weighted mean + std.

    This supersedes `ensemble_mean_and_std` and exposes the two knobs that
    matter for the K=5 regression diagnosis:
      * `anchor_method` — how the alignment reference is chosen.
      * `weight_method` — how members are combined ('uniform' = the old
        behaviour; 'softmax'/'inverse'/'top_k' downweight bad members).

    With `anchor_method='first'`, `weight_method='uniform'` this reproduces
    the legacy pipeline exactly.
    """
    if stack.dim() != 4:
        raise ValueError(f"Expected (K, 1, H, W), got shape {tuple(stack.shape)}")
    K = stack.shape[0]

    if K == 1:
        x = stack
        return EnsembleStats(
            mean_real=x.real if torch.is_complex(x) else x,
            mean_complex=x,
            std_pixelwise=torch.zeros_like(x.real if torch.is_complex(x) else x),
            n_members=1,
            anchor_index=0,
            weights=torch.ones(1, device=stack.device),
            member_fe=None,
        )

    fe = None
    if y_intensity is not None:
        fe = member_fourier_errors(stack, y_intensity, oversample=oversample)

    anchor = select_anchor(stack, method=anchor_method, y_intensity=y_intensity, oversample=oversample)
    aligned = align_ensemble_to_anchor(stack, anchor_index=anchor) if align else stack

    if weight_method == "uniform":
        weights = torch.ones(K, device=stack.device) / K
    else:
        if fe is None:
            raise ValueError(f"weight_method={weight_method!r} requires y_intensity")
        weights = fidelity_weights(fe, method=weight_method, temperature=temperature, top_k=top_k)

    mean_c, std = _weighted_mean_std(aligned, weights)
    return EnsembleStats(
        mean_real=mean_c.real if torch.is_complex(mean_c) else mean_c,
        mean_complex=mean_c,
        std_pixelwise=std,
        n_members=K,
        anchor_index=anchor,
        weights=weights,
        member_fe=fe,
    )


def effective_sample_size(weights: torch.Tensor) -> float:
    """Kish effective sample size: (Σw)^2 / Σ(w^2).

    For uniform weights over K members this equals K; for a single dominant
    weight it approaches 1. Reports how many members effectively contribute
    to a weighted aggregate.
    """
    w = weights.float()
    s1 = w.sum()
    s2 = w.pow(2).sum().clamp_min(1e-12)
    return float((s1 * s1 / s2).item())


def pixelwise_median_complex(stack: torch.Tensor) -> torch.Tensor:
    """Per-pixel median of an aligned complex stack.

    Median is taken separately on the real and imaginary parts (a common,
    cheap complex-median surrogate). Robust to a minority of outlier members
    without needing any fidelity score.
    """
    real_med = stack.real.median(dim=0, keepdim=True).values
    imag_med = stack.imag.median(dim=0, keepdim=True).values
    return (real_med + 1j * imag_med).to(torch.complex64)


def trimmed_mean_complex(stack: torch.Tensor, trim: int = 1) -> torch.Tensor:
    """Per-pixel trimmed mean of the magnitude-ranked members.

    At each pixel, drop the `trim` highest and `trim` lowest magnitude values
    and average the rest. Falls back to the plain mean if trimming would
    remove every member. Complex phase is taken from the plain mean to keep
    a well-defined complex output.
    """
    K = stack.shape[0]
    if 2 * trim >= K:
        return stack.mean(dim=0, keepdim=True)
    mags = stack.abs()
    order = mags.argsort(dim=0)  # ascending per pixel
    keep = order[trim : K - trim]  # (K-2trim, 1, H, W) indices
    gathered = torch.gather(stack, 0, keep)
    return gathered.mean(dim=0, keepdim=True)


def ensemble_mean_and_std(
    stack: torch.Tensor,
    align: bool = True,
) -> EnsembleStats:
    """Compute mean and per-pixel std of a complex reconstruction stack.

    Parameters
    ----------
    stack : (K, 1, H, W) complex tensor.
    align : if True, align all members to member 0 before averaging
        (recommended — see `align_ensemble_to_anchor`).

    Returns
    -------
    EnsembleStats
    """
    if stack.dim() != 4:
        raise ValueError(f"Expected (K, 1, H, W), got shape {tuple(stack.shape)}")
    K = stack.shape[0]

    if K == 1:
        x = stack
        return EnsembleStats(
            mean_real=x.real if torch.is_complex(x) else x,
            mean_complex=x,
            std_pixelwise=torch.zeros_like(x.real if torch.is_complex(x) else x),
            n_members=1,
        )

    if align:
        stack = align_ensemble_to_anchor(stack, anchor_index=0)

    mean_c = stack.mean(dim=0, keepdim=True)
    # Per-pixel std of the magnitude — robust to residual phase mis-alignment
    mags = stack.abs() if torch.is_complex(stack) else stack
    std = mags.std(dim=0, keepdim=True, unbiased=False)
    return EnsembleStats(
        mean_real=mean_c.real if torch.is_complex(mean_c) else mean_c,
        mean_complex=mean_c,
        std_pixelwise=std,
        n_members=K,
    )


def normalized_uncertainty(
    u: torch.Tensor,
    method: str = "robust",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Map a non-negative uncertainty tensor to [0, 1].

    'minmax':  (u - u.min()) / (u.max() - u.min())
    'robust':  clip at the 1st and 99th percentile, then min-max.
              More stable when there are a few outlier pixels.
    'l2':     u / ||u||
    """
    u = u.float()
    if method == "minmax":
        u_min, u_max = u.min(), u.max()
        return ((u - u_min) / (u_max - u_min + eps)).clamp(0, 1)
    if method == "robust":
        flat = u.flatten()
        if flat.numel() == 0:
            return u
        lo = torch.quantile(flat, 0.01)
        hi = torch.quantile(flat, 0.99)
        if (hi - lo).abs() < eps:
            return torch.zeros_like(u)
        return ((u - lo) / (hi - lo + eps)).clamp(0, 1)
    if method == "l2":
        n = u.pow(2).sum().sqrt().clamp_min(eps)
        return u / n
    raise ValueError(f"Unknown normalization method: {method!r}")
