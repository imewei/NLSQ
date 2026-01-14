"""CMA-ES configuration for global optimization.

This module provides configuration dataclasses and utilities for CMA-ES
optimization using the evosax library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CMAES_PRESETS",
    "CMAESConfig",
    "auto_configure_cmaes_memory",
    "estimate_cmaes_memory_gb",
    "get_evosax_import_error",
    "is_evosax_available",
]

logger = logging.getLogger(__name__)

# Global cache for evosax availability check
_EVOSAX_AVAILABLE: bool | None = None
_EVOSAX_IMPORT_ERROR: str | None = None


def is_evosax_available() -> bool:
    """Check if evosax is available for import.

    Uses lazy import pattern to avoid import overhead until needed.
    Caches the result for subsequent calls.

    Returns
    -------
    bool
        True if evosax can be imported successfully, False otherwise.
    """
    global _EVOSAX_AVAILABLE, _EVOSAX_IMPORT_ERROR  # noqa: PLW0603

    if _EVOSAX_AVAILABLE is None:
        try:
            # Import to verify package is available and functional
            # We need to test actual imports, not just find_spec, because
            # evosax may be installed but have missing dependencies
            import evosax.algorithms  # type: ignore[import-not-found,import-untyped]
            import evosax.core.restart  # type: ignore[import-not-found,import-untyped]

            _EVOSAX_AVAILABLE = True
            _EVOSAX_IMPORT_ERROR = None
        except ImportError as e:
            _EVOSAX_AVAILABLE = False
            _EVOSAX_IMPORT_ERROR = str(e)
            logger.info(
                "evosax not available - CMA-ES will fall back to multi-start. "
                "Install with: pip install 'nlsq[global]'"
            )

    return _EVOSAX_AVAILABLE


def get_evosax_import_error() -> str | None:
    """Get the import error message if evosax is not available.

    Returns
    -------
    str | None
        The import error message, or None if evosax is available.
    """
    is_evosax_available()  # Ensure availability check has been performed
    return _EVOSAX_IMPORT_ERROR


@dataclass(slots=True)
class CMAESConfig:
    """Configuration for CMA-ES global optimization.

    Attributes
    ----------
    popsize : int | None
        Population size. If None, uses CMA-ES default: int(4 + 3 * log(n)).
    max_generations : int
        Maximum number of generations before stopping. Default: 100.
    sigma : float
        Initial step size (standard deviation). Default: 0.5.
    tol_fun : float
        Function value tolerance for convergence. Default: 1e-8.
    tol_x : float
        Parameter tolerance for convergence. Default: 1e-8.
    restart_strategy : Literal['none', 'bipop']
        Restart strategy. Default: 'bipop'.
    max_restarts : int
        Maximum restart attempts for BIPOP. Default: 9.
    population_batch_size : int | None
        Batch size for population evaluation. If None, evaluates all candidates
        in parallel. Set to smaller values (e.g., 4) to reduce memory usage.
    data_chunk_size : int | None
        Chunk size for data streaming. If None, processes full dataset at once.
        Set to smaller values (e.g., 50000) for datasets >10M points to avoid OOM.
        Must be >= 1024 for numerical stability.
    refine_with_nlsq : bool
        Whether to refine best solution with NLSQ TRF. Default: True.
    seed : int | None
        Random seed for reproducibility. If None, uses random seed.

    Examples
    --------
    >>> config = CMAESConfig(popsize=32, max_generations=200)
    >>> config = CMAESConfig.from_preset('cmaes-global')

    Memory-efficient configuration for large datasets:

    >>> config = CMAESConfig(
    ...     population_batch_size=4,  # Evaluate 4 candidates at a time
    ...     data_chunk_size=50000,    # Process 50K points per chunk
    ... )
    """

    # Population and generations
    popsize: int | None = None  # None = auto: int(4 + 3 * log(n))
    max_generations: int = 100

    # Step size and tolerances
    sigma: float = 0.5
    tol_fun: float = 1e-8
    tol_x: float = 1e-8

    # Restart strategy (BIPOP enabled by default per spec)
    restart_strategy: Literal["none", "bipop"] = "bipop"
    max_restarts: int = 9

    # Memory management
    population_batch_size: int | None = None
    data_chunk_size: int | None = None

    # NLSQ refinement
    refine_with_nlsq: bool = True

    # Reproducibility
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        if self.popsize is not None and self.popsize < 4:
            raise ValueError(f"popsize must be >= 4, got {self.popsize}")

        if self.max_generations < 1:
            raise ValueError(
                f"max_generations must be >= 1, got {self.max_generations}"
            )

        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")

        if self.tol_fun <= 0:
            raise ValueError(f"tol_fun must be > 0, got {self.tol_fun}")

        if self.tol_x <= 0:
            raise ValueError(f"tol_x must be > 0, got {self.tol_x}")

        if self.max_restarts < 0:
            raise ValueError(f"max_restarts must be >= 0, got {self.max_restarts}")

        if self.restart_strategy not in ("none", "bipop"):
            raise ValueError(
                f"restart_strategy must be 'none' or 'bipop', "
                f"got '{self.restart_strategy}'"
            )

        if self.population_batch_size is not None and self.population_batch_size < 1:
            raise ValueError(
                f"population_batch_size must be >= 1, got {self.population_batch_size}"
            )

        if self.data_chunk_size is not None and self.data_chunk_size < 1024:
            raise ValueError(
                f"data_chunk_size must be >= 1024 for numerical stability, "
                f"got {self.data_chunk_size}"
            )

    @classmethod
    def from_preset(cls, preset_name: str) -> CMAESConfig:
        """Create a CMAESConfig from a named preset.

        Parameters
        ----------
        preset_name : str
            Name of the preset. One of 'cmaes-fast', 'cmaes', 'cmaes-global'.

        Returns
        -------
        CMAESConfig
            Configuration for the specified preset.

        Raises
        ------
        ValueError
            If preset_name is not recognized.

        Examples
        --------
        >>> config = CMAESConfig.from_preset('cmaes-fast')
        >>> config.max_generations
        50
        """
        if preset_name not in CMAES_PRESETS:
            available = ", ".join(sorted(CMAES_PRESETS.keys()))
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available presets: {available}"
            )

        preset_config = CMAES_PRESETS[preset_name]
        return cls(**preset_config)


# CMA-ES presets - source of truth for CMA-ES preset configurations
CMAES_PRESETS: dict[str, dict] = {
    "cmaes-fast": {
        "popsize": None,  # auto
        "max_generations": 50,
        "restart_strategy": "none",
        "max_restarts": 0,
    },
    "cmaes": {
        "popsize": None,  # auto
        "max_generations": 100,
        "restart_strategy": "bipop",
        "max_restarts": 9,
    },
    "cmaes-global": {
        "popsize": None,  # Will be doubled in optimizer (2x auto)
        "max_generations": 200,
        "restart_strategy": "bipop",
        "max_restarts": 9,
    },
}


def estimate_cmaes_memory_gb(
    n_data: int,
    popsize: int,
    population_batch_size: int | None = None,
    data_chunk_size: int | None = None,
) -> float:
    """Estimate peak GPU memory usage for CMA-ES in GB.

    Parameters
    ----------
    n_data : int
        Number of data points.
    popsize : int
        Population size.
    population_batch_size : int | None, optional
        Batch size for population evaluation. If None, uses full popsize.
    data_chunk_size : int | None, optional
        Chunk size for data streaming. If None, uses full dataset.

    Returns
    -------
    float
        Estimated peak memory usage in GB.

    Examples
    --------
    >>> estimate_cmaes_memory_gb(10_000_000, popsize=16)
    1.1920928955078125

    >>> estimate_cmaes_memory_gb(10_000_000, popsize=16, population_batch_size=4)
    0.298023223876953

    >>> estimate_cmaes_memory_gb(10_000_000, popsize=16, data_chunk_size=50000)
    0.005960464477539062
    """
    eff_pop = population_batch_size if population_batch_size else popsize
    eff_data = data_chunk_size if data_chunk_size else n_data

    # Memory per candidate during fitness evaluation:
    # - predictions array: eff_data * 8 bytes (float64)
    # - residuals array: eff_data * 8 bytes (float64)
    # - intermediate computations: ~eff_data * 8 bytes
    bytes_per_candidate = eff_data * 8 * 3  # 3 arrays
    peak_bytes = eff_pop * bytes_per_candidate

    # Add overhead for input data (xdata, ydata)
    # When data_chunk_size is set, we use dynamic_slice which doesn't copy
    if data_chunk_size is None:
        peak_bytes += n_data * 8 * 2  # xdata + ydata

    return peak_bytes / (1024**3)


def auto_configure_cmaes_memory(
    n_data: int,
    popsize: int,
    available_memory_gb: float = 8.0,
    safety_factor: float = 0.7,
) -> tuple[int | None, int | None]:
    """Auto-configure batch sizes to fit in available memory.

    Parameters
    ----------
    n_data : int
        Number of data points.
    popsize : int
        Population size.
    available_memory_gb : float, optional
        Available GPU/CPU memory in GB. Default: 8.0.
    safety_factor : float, optional
        Safety factor for memory allocation (0-1). Default: 0.7.

    Returns
    -------
    tuple[int | None, int | None]
        (population_batch_size, data_chunk_size) configuration.
        None means no batching/chunking needed for that dimension.

    Examples
    --------
    >>> auto_configure_cmaes_memory(1_000_000, popsize=16, available_memory_gb=8.0)
    (None, None)  # No batching needed

    >>> auto_configure_cmaes_memory(100_000_000, popsize=16, available_memory_gb=8.0)
    (4, None)  # Population batching only

    >>> auto_configure_cmaes_memory(100_000_000, popsize=16, available_memory_gb=1.0)
    (1, 65536)  # Both population batching and data chunking
    """
    target_memory_bytes = available_memory_gb * safety_factor * (1024**3)
    bytes_per_point_per_candidate = 8 * 3  # float64, 3 arrays

    # Try full data, full population first
    full_memory = n_data * popsize * bytes_per_point_per_candidate
    if full_memory <= target_memory_bytes:
        return None, None  # No batching needed

    # Try population batching only (halving until it fits)
    for pop_batch in [popsize // 2, popsize // 4, popsize // 8, 1]:
        if pop_batch < 1:
            continue
        memory = n_data * pop_batch * bytes_per_point_per_candidate
        if memory <= target_memory_bytes:
            return pop_batch, None

    # Need data chunking - calculate optimal chunk size
    # With pop_batch=1, find largest chunk that fits
    max_chunk = int(target_memory_bytes / bytes_per_point_per_candidate)

    # Use power-of-2 bucket sizes for JIT cache efficiency
    # These match CHUNK_BUCKETS from nlsq/streaming/large_dataset.py
    chunk_buckets = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)

    # Find largest bucket that fits
    data_chunk = 1024  # Minimum
    for bucket in chunk_buckets:
        if bucket <= max_chunk:
            data_chunk = bucket

    return 1, data_chunk
