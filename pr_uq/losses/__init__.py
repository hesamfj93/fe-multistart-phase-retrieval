"""Loss functions and regularizers for untrained FPR."""

from .data_loss import (
    magnitude_l2,
    log_magnitude_l2,
    poisson_nll,
    build_data_loss,
)
from .regularization import (
    total_variation,
    phase_l2,
    soft_support_penalty,
    regularization_total,
)
from .spectral_weight import (
    gaussian_weight,
    annealing_sigma,
    SpectralAnnealer,
)

__all__ = [
    "magnitude_l2",
    "log_magnitude_l2",
    "poisson_nll",
    "build_data_loss",
    "total_variation",
    "phase_l2",
    "soft_support_penalty",
    "regularization_total",
    "gaussian_weight",
    "annealing_sigma",
    "SpectralAnnealer",
]
