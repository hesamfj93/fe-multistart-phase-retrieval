"""Regularization terms used in the loss.

All functions return a scalar tensor. Combine them with `lambda_*` weights
in the trainer.
"""

from __future__ import annotations

import torch


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV on a (B, 1, H, W) tensor (real)."""
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs()
    return dh.mean() + dw.mean()


def phase_l2(phi: torch.Tensor) -> torch.Tensor:
    """L2 regularization on the phase magnitude.

    Encourages |Φ| → 0 in the absence of evidence. For purely real images
    this should drive Φ to 0 (modulo the trivial sign ambiguity). For
    genuinely complex images this term should have a small weight.
    """
    return phi.pow(2).mean()


def soft_support_penalty(amplitude: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
    """Penalize amplitude where the (soft) support mask is small.

    Parameters
    ----------
    amplitude : (B, 1, H, W) in [0, 1]
    support_mask : (B, 1, H, W) in [0, 1], 1 = inside support, 0 = outside

    The penalty is the mean of A · (1 − M).
    """
    return (amplitude * (1.0 - support_mask)).mean()


def regularization_total(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    support_mask: torch.Tensor | None,
    lambda_tv_A: float = 1e-3,
    lambda_tv_phi: float = 1e-4,
    lambda_phi_l2: float = 0.0,
    lambda_support: float = 1e-2,
) -> torch.Tensor:
    """Convenience: weighted sum of all enabled regularizers.

    Components with weight 0 are skipped at zero cost.
    """
    loss = amplitude.new_tensor(0.0)
    if lambda_tv_A > 0:
        loss = loss + lambda_tv_A * total_variation(amplitude)
    if lambda_tv_phi > 0:
        loss = loss + lambda_tv_phi * total_variation(phase)
    if lambda_phi_l2 > 0:
        loss = loss + lambda_phi_l2 * phase_l2(phase)
    if lambda_support > 0 and support_mask is not None:
        loss = loss + lambda_support * soft_support_penalty(amplitude, support_mask)
    return loss
