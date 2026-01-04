"""Shared constants for NLSQ benchmarks.

These constants provide consistent default values across all benchmark scripts.
"""

from __future__ import annotations

# Default problem sizes for benchmarking
DEFAULT_DATA_SIZES: tuple[int, ...] = (100, 1000, 10000)
EXTENDED_DATA_SIZES: tuple[int, ...] = (100, 1000, 10000, 100000)

# Default benchmark configuration
DEFAULT_N_REPEATS: int = 5
DEFAULT_WARMUP_RUNS: int = 1
DEFAULT_METHODS: tuple[str, ...] = ("trf", "lm")
DEFAULT_BACKENDS: tuple[str, ...] = ("cpu",)

# Data generation defaults
DEFAULT_SEED: int = 42
DEFAULT_NOISE_LEVEL: float = 0.1

# Jacobian benchmark test cases: (n_params, n_data, expected_speedup)
JACOBIAN_TEST_CASES: tuple[tuple[int, int, str], ...] = (
    (1000, 100, "10-100x"),
    (500, 100, "5-50x"),
    (200, 100, "2-10x"),
    (100, 100, "~1x"),
)

# Sparse Jacobian benchmark sizes
LARGE_JACOBIAN_SHAPE: tuple[int, int] = (100_000, 50)
MEDIUM_JACOBIAN_SHAPE: tuple[int, int] = (10_000, 50)

# Cache benchmark targets
TARGET_CACHE_HIT_RATE: float = 0.80
TARGET_SPEEDUP_FACTOR: float = 2.0
TARGET_WARM_JIT_TIME_MS: float = 2.0

# Stability benchmark thresholds
SVD_SKIP_THRESHOLD: int = 10_000_000  # Elements in Jacobian

# CI regression thresholds
COLD_JIT_SLOWDOWN_THRESHOLD_PCT: float = 10.0
HOT_PATH_SLOWDOWN_THRESHOLD_PCT: float = 5.0
MEMORY_REGRESSION_THRESHOLD_PCT: float = 10.0
