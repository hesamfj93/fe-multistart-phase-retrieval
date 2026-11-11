"""Uncertainty quantification, calibration, and failure detection."""

from .ensemble_uq import (
    ensemble_aggregate,
    select_anchor,
    fidelity_weights,
    member_fourier_errors,
)
from .ensemble_uq import EnsembleStats, ensemble_mean_and_std, normalized_uncertainty
from .fusion import (
    FusedUncertainty,
    fuse_uncertainty,
    failure_score,
    alpha_sweep,
)
from .calibration import (
    CalibrationResult,
    calibrate,
    expected_calibration_error,
    reliability_diagram,
    gaussian_nll,
)
from .failure_detection import (
    FailureDetectionResult,
    compute_auroc,
    evaluate_failure_detection,
    failure_label,
)

__all__ = [
    "EnsembleStats",
    "ensemble_mean_and_std",
    "normalized_uncertainty",
    "FusedUncertainty",
    "fuse_uncertainty",
    "failure_score",
    "alpha_sweep",
    "CalibrationResult",
    "calibrate",
    "expected_calibration_error",
    "reliability_diagram",
    "gaussian_nll",
    "FailureDetectionResult",
    "compute_auroc",
    "evaluate_failure_detection",
    "failure_label",
]
