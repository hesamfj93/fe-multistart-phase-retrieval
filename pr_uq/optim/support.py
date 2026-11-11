"""Adaptive support mask refinement.

We start with a permissive support (the full image or a coarse rectangle)
and refine it every Δ iterations based on the current amplitude estimate
and — when an uncertainty signal is available — the per-pixel confidence.

The classical Shrinkwrap rule (Marchesini 2003) is:

    M(i,j) = 1 if  A_smoothed(i,j) > τ_A * max(A_smoothed)
             0 otherwise

We extend this to a *soft* sigmoid mask AND a fused score that down-weights
high-uncertainty pixels:

    score(i,j) = A_smoothed(i,j) * (1 - u_norm(i,j))^η
    M(i,j)     = sigmoid(γ * (score(i,j) - τ * max(score)))

This way pixels with low amplitude AND low uncertainty are excluded as
background. Pixels with low amplitude but high uncertainty stay in the
candidate pool — the prior is unsure, so we don't aggressively prune.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_smooth(x: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5) -> torch.Tensor:
    """Separable 2D Gaussian smoothing for (B, C, H, W) tensors."""
    if sigma <= 0:
        return x
    half = kernel_size // 2
    ks = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - half
    g = torch.exp(-0.5 * (ks / sigma) ** 2)
    g = g / g.sum()
    kx = g.view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1)
    ky = g.view(1, 1, -1, 1).expand(x.shape[1], 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, half), groups=x.shape[1])
    x = F.conv2d(x, ky, padding=(half, 0), groups=x.shape[1])
    return x


def initial_support(h: int, w: int, fill: str = "full", device: torch.device | str = "cpu") -> torch.Tensor:
    """Initial soft support — defaults to all ones."""
    if fill == "full":
        return torch.ones(1, 1, h, w, device=device)
    if fill == "centered":
        # 60% centered rectangle, common heuristic
        mask = torch.zeros(1, 1, h, w, device=device)
        h0, w0 = int(0.2 * h), int(0.2 * w)
        mask[..., h0 : h - h0, w0 : w - w0] = 1.0
        return mask
    raise ValueError(f"Unknown initial support fill: {fill!r}")


def update_support(
    amplitude: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    smooth_sigma: float = 1.0,
    threshold_frac: float = 0.1,
    gamma: float = 10.0,
    eta: float = 1.0,
    use_uncertainty: bool = True,
) -> torch.Tensor:
    """One adaptive support refinement step.

    Parameters
    ----------
    amplitude : (1, 1, H, W) current amplitude in [0, 1].
    uncertainty : (1, 1, H, W) normalized uncertainty in [0, 1], or None
        for the classical (amplitude-only) Shrinkwrap update.
    smooth_sigma : Gaussian smoothing applied before thresholding.
    threshold_frac : τ — fraction of max(score) defining the threshold.
    gamma : sharpness of the sigmoid (large → harder mask).
    eta : power on (1 − u) — higher → uncertainty matters more.
    use_uncertainty : if False, ignores `uncertainty` and uses Shrinkwrap.

    Returns
    -------
    M : (1, 1, H, W) soft mask in [0, 1].
    """
    a_s = gaussian_smooth(amplitude, sigma=smooth_sigma)
    if use_uncertainty and uncertainty is not None:
        u = uncertainty.clamp(0, 1)
        score = a_s * (1.0 - u).clamp_min(1e-6).pow(eta)
    else:
        score = a_s
    peak = score.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    thr = threshold_frac * peak
    return torch.sigmoid(gamma * (score - thr))
