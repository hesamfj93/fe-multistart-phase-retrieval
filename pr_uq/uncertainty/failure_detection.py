"""Failure detection via the uncertainty channel.

A reconstruction is labeled a "failure" when its aligned PSNR is below a
threshold (per-dataset or fixed). The uncertainty's per-image summary
(`failure_score` from `fusion.py`) is the prediction. We then ask: how
well does that score discriminate failures from successes?

Standard metric: AUROC of failure_score vs binary failure label.

We implement AUROC from scratch (Mann-Whitney U statistic over score
ranks) to avoid a hard scikit-learn dependency for this one metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FailureDetectionResult:
    auroc: float
    n_failures: int
    n_successes: int
    threshold_db: float
    summary: str


def compute_auroc(scores: list[float], labels: list[int]) -> float:
    """AUROC = P(score(positive) > score(negative)).

    Implemented via the Mann-Whitney U statistic. Ties are split.
    Returns 0.5 for the degenerate case of all-one-class.
    """
    n = len(scores)
    if n != len(labels):
        raise ValueError("scores and labels must have the same length")
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    nP, nN = len(pos), len(neg)
    if nP == 0 or nN == 0:
        return 0.5
    # Rank all scores
    paired = sorted(enumerate(scores), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and paired[j + 1][1] == paired[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed
        for k in range(i, j + 1):
            ranks[paired[k][0]] = avg_rank
        i = j + 1
    sum_ranks_pos = sum(ranks[k] for k, l in enumerate(labels) if l == 1)
    U = sum_ranks_pos - nP * (nP + 1) / 2.0
    return float(U / (nP * nN))


def failure_label(
    psnr_db: float,
    threshold_kind: str = "fixed",
    threshold_value_db: float = 15.0,
    dataset_median_db: float | None = None,
    margin_db: float = 5.0,
) -> int:
    """Return 1 if this reconstruction is a failure, else 0.

    threshold_kind:
      'fixed'    — failure when psnr < threshold_value_db
      'relative' — failure when psnr < (dataset_median_db − margin_db)
    """
    if threshold_kind == "fixed":
        return int(psnr_db < threshold_value_db)
    if threshold_kind == "relative":
        if dataset_median_db is None:
            raise ValueError("dataset_median_db is required for 'relative' threshold")
        return int(psnr_db < dataset_median_db - margin_db)
    raise ValueError(f"Unknown threshold_kind {threshold_kind!r}")


def evaluate_failure_detection(
    failure_scores: list[float],
    psnr_list: list[float],
    threshold_kind: str = "fixed",
    threshold_value_db: float = 15.0,
    margin_db: float = 5.0,
    summary_name: str = "mean",
) -> FailureDetectionResult:
    """Top-level evaluator: turn (scores, psnrs) into AUROC + bookkeeping."""
    if threshold_kind == "relative":
        # Compute median from data
        sorted_p = sorted(psnr_list)
        med = sorted_p[len(sorted_p) // 2] if sorted_p else 0.0
        threshold = med - margin_db
        labels = [failure_label(p, "fixed", threshold_value_db=threshold) for p in psnr_list]
    else:
        threshold = threshold_value_db
        labels = [
            failure_label(p, threshold_kind, threshold_value_db=threshold_value_db)
            for p in psnr_list
        ]
    auroc = compute_auroc(failure_scores, labels)
    return FailureDetectionResult(
        auroc=auroc,
        n_failures=sum(labels),
        n_successes=len(labels) - sum(labels),
        threshold_db=threshold,
        summary=summary_name,
    )
