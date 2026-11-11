"""Tests for calibration and failure-detection.

Run with:  pytest tests/test_calibration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pr_uq.uncertainty.calibration import (
    calibrate, expected_calibration_error, _pearson
)
from pr_uq.uncertainty.failure_detection import compute_auroc, failure_label


def test_perfect_calibration_low_ece():
    """If u = |err| exactly, ECE should be ~0."""
    torch.manual_seed(0)
    err = torch.rand(1, 1, 32, 32)
    u = err.clone()
    ece = expected_calibration_error(u, err, n_bins=10)
    assert ece < 1e-6, f"perfect calibration ECE should be 0, got {ece}"


def test_pearson_perfectly_correlated():
    x = torch.linspace(0, 1, 100)
    assert _pearson(x, 2 * x + 1.0) > 0.999


def test_pearson_anticorrelated():
    x = torch.linspace(0, 1, 100)
    assert _pearson(x, -x) < -0.999


def test_auroc_perfect_separator():
    """AUROC = 1.0 when positives score strictly above negatives."""
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    assert compute_auroc(scores, labels) == 1.0


def test_auroc_random_classifier():
    """AUROC ≈ 0.5 for randomly mixed scores."""
    torch.manual_seed(0)
    scores = torch.rand(200).tolist()
    labels = torch.randint(0, 2, (200,)).tolist()
    a = compute_auroc(scores, labels)
    assert 0.4 < a < 0.6


def test_auroc_inverted():
    """AUROC for the inverse ordering is 1 - AUROC."""
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [1, 1, 1, 0, 0, 0]  # inverted
    assert compute_auroc(scores, labels) == 0.0


def test_failure_label_fixed():
    assert failure_label(10.0, "fixed", threshold_value_db=15.0) == 1
    assert failure_label(20.0, "fixed", threshold_value_db=15.0) == 0
