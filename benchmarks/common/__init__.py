"""Common utilities for NLSQ benchmarks.

This module provides shared components to reduce code duplication:
- models: Standard model functions for benchmarking
- data: Data generation utilities
- constants: Shared configuration constants
"""

from benchmarks.common.constants import (
    DEFAULT_DATA_SIZES,
    DEFAULT_METHODS,
    DEFAULT_N_REPEATS,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_SEED,
    DEFAULT_WARMUP_RUNS,
)
from benchmarks.common.models import (
    exponential_model,
    gaussian_model,
    polynomial_model,
    sinusoidal_model,
)

__all__ = [
    "DEFAULT_DATA_SIZES",
    "DEFAULT_METHODS",
    "DEFAULT_NOISE_LEVEL",
    "DEFAULT_N_REPEATS",
    "DEFAULT_SEED",
    "DEFAULT_WARMUP_RUNS",
    "exponential_model",
    "gaussian_model",
    "polynomial_model",
    "sinusoidal_model",
]
