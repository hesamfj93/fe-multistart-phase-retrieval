"""Physics submodule: forward model, projection, noise."""

from .forward import forward_field, forward_intensity, forward_magnitude, inverse_field
from .projection import beta_mix, physics_residual, project_magnitude
from .noise import add_gaussian_noise, add_poisson_noise, apply_noise

__all__ = [
    "forward_field",
    "forward_intensity",
    "forward_magnitude",
    "inverse_field",
    "project_magnitude",
    "beta_mix",
    "physics_residual",
    "add_gaussian_noise",
    "add_poisson_noise",
    "apply_noise",
]
