"""Spectral weighting / annealing.

We multiply the data loss by a centered Gaussian weight in the Fourier
domain, with σ decaying from σ_start to σ_end across iterations. A smaller σ
gives a narrower weight, so early iterations weight all detector frequencies
almost equally and late iterations emphasize the lowest frequencies. The
schedule was introduced with a coarse-to-fine intent; as implemented
(σ_start > σ_end) it narrows over time. It is part of the frozen design of the
held-out runs and is documented, not changed.

This is functionally similar to the multi-scale loss in PPRNet, but cast as
a continuous schedule for our single-stage optimization.
"""

from __future__ import annotations

import math

import torch


def gaussian_weight(
    h: int, w: int, sigma: float, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Centered isotropic Gaussian weight of shape (1, 1, h, w).

    `sigma` is in pixels (the standard deviation of the Gaussian in the
    Fourier domain). A typical sweep is sigma ∈ [0.05 · h, 0.5 · h].
    """
    cy, cx = h / 2.0, w / 2.0
    ys = torch.arange(h, device=device, dtype=torch.float32) - cy
    xs = torch.arange(w, device=device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r2 = yy.pow(2) + xx.pow(2)
    w_tensor = torch.exp(-r2 / (2.0 * sigma * sigma + 1e-12))
    return w_tensor.unsqueeze(0).unsqueeze(0)


def annealing_sigma(
    step: int,
    total_steps: int,
    sigma_start: float,
    sigma_end: float,
    schedule: str = "log",
    anneal_fraction: float = 0.6,
) -> float:
    """Return σ at iteration `step`.

    The schedule reaches `sigma_end` at `anneal_fraction · total_steps` and
    then stays constant. This lets the model spend the last portion of
    optimization at full bandwidth.
    """
    anneal_steps = max(1, int(anneal_fraction * total_steps))
    if step >= anneal_steps:
        return sigma_end
    frac = step / anneal_steps
    if schedule == "linear":
        return sigma_start + (sigma_end - sigma_start) * frac
    if schedule == "log":
        log_s = math.log(sigma_start) + (math.log(sigma_end) - math.log(sigma_start)) * frac
        return math.exp(log_s)
    raise ValueError(f"Unknown schedule: {schedule!r}")


class SpectralAnnealer:
    """Caches the current Gaussian weight tensor to avoid recomputation each step."""

    def __init__(
        self,
        h: int,
        w: int,
        sigma_start: float,
        sigma_end: float,
        total_steps: int,
        schedule: str = "log",
        anneal_fraction: float = 0.6,
        device: torch.device | str = "cpu",
        update_every: int = 50,
    ):
        self.h, self.w = h, w
        self.sigma_start, self.sigma_end = sigma_start, sigma_end
        self.total_steps = total_steps
        self.schedule = schedule
        self.anneal_fraction = anneal_fraction
        self.device = device
        self.update_every = update_every
        self._cached_sigma: float | None = None
        self._cached_weight: torch.Tensor | None = None

    def weight_at(self, step: int) -> torch.Tensor:
        if step % self.update_every == 0 or self._cached_weight is None:
            sigma = annealing_sigma(
                step,
                self.total_steps,
                self.sigma_start,
                self.sigma_end,
                self.schedule,
                self.anneal_fraction,
            )
            self._cached_sigma = sigma
            self._cached_weight = gaussian_weight(self.h, self.w, sigma, self.device)
        return self._cached_weight
