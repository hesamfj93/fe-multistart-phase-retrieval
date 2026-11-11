"""Annealing schedules for hyperparameters that vary across iterations.

Currently we anneal:

  * β(t) — the mixing coefficient between physics projection and network
    output. We start at β_start ≈ 0.7 (trust the data) and decrease to
    β_end ≈ 0.1 (trust the prior) so that the network drives the result by
    the end of optimization.

  * λ_support(t) — optional support weight ramp.
"""

from __future__ import annotations

import math


def linear_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    """Linear interpolation start → end over total_steps, clipped after."""
    if step >= total_steps:
        return end
    frac = step / max(1, total_steps)
    return start + (end - start) * frac


def cosine_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    """Cosine decay from start to end."""
    if step >= total_steps:
        return end
    frac = step / max(1, total_steps)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * frac))


def step_schedule(step: int, milestones: list[int], values: list[float]) -> float:
    """Piecewise constant: returns values[i] for step in [milestones[i-1], milestones[i])."""
    assert len(values) == len(milestones) + 1, "values must have one more entry than milestones"
    for i, m in enumerate(milestones):
        if step < m:
            return values[i]
    return values[-1]


def build_schedule(spec: dict):
    """Return a callable schedule(step, total_steps) given a spec dict.

    Spec examples:
      {'type': 'linear', 'start': 0.7, 'end': 0.1}
      {'type': 'cosine', 'start': 0.7, 'end': 0.1}
      {'type': 'constant', 'value': 0.5}
      {'type': 'step', 'milestones': [1000, 3000], 'values': [0.7, 0.3, 0.1]}
    """
    kind = spec.get("type", "constant")
    if kind == "constant":
        value = float(spec["value"])
        return lambda step, total: value
    if kind == "linear":
        s, e = float(spec["start"]), float(spec["end"])
        return lambda step, total: linear_schedule(step, total, s, e)
    if kind == "cosine":
        s, e = float(spec["start"]), float(spec["end"])
        return lambda step, total: cosine_schedule(step, total, s, e)
    if kind == "step":
        return lambda step, total: step_schedule(step, list(spec["milestones"]), list(spec["values"]))
    raise ValueError(f"Unknown schedule type: {kind!r}")
