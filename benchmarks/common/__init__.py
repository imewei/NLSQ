"""Common utilities for NLSQ benchmarks.

This module provides shared components to reduce code duplication:
- models: Standard model functions for benchmarking
- data: Data generation utilities
- constants: Shared configuration constants
"""

from benchmarks.common.models import (
    exponential_model,
    gaussian_model,
    polynomial_model,
    sinusoidal_model,
)
from benchmarks.common.constants import (
    DEFAULT_DATA_SIZES,
    DEFAULT_N_REPEATS,
    DEFAULT_WARMUP_RUNS,
    DEFAULT_METHODS,
    DEFAULT_SEED,
    DEFAULT_NOISE_LEVEL,
)

__all__ = [
    # Models
    "exponential_model",
    "gaussian_model",
    "polynomial_model",
    "sinusoidal_model",
    # Constants
    "DEFAULT_DATA_SIZES",
    "DEFAULT_N_REPEATS",
    "DEFAULT_WARMUP_RUNS",
    "DEFAULT_METHODS",
    "DEFAULT_SEED",
    "DEFAULT_NOISE_LEVEL",
]
