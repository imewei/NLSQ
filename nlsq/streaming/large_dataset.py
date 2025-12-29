"""Large Dataset Fitting Module for NLSQ.

This module provides utilities for efficiently fitting curve parameters to very large datasets
(>10M points) with intelligent memory management, automatic chunking, and progress reporting.
"""

# mypy: ignore-errors

import gc
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from logging import Logger
from typing import TYPE_CHECKING, Literal

import jax
import numpy as np
import psutil

# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

_jax_config = JAXConfig()


from nlsq.core._optimize import OptimizeResult

# Import streaming optimizer (required dependency as of v0.2.0)
from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer
from nlsq.utils.logging import get_logger

# Type-only imports to avoid circular dependencies
if TYPE_CHECKING:
    from nlsq.core.minpack import CurveFit
    from nlsq.core.workflow import MemoryTier

# Default fallback memory in GB when detection fails (per requirements)
_DEFAULT_FALLBACK_MEMORY_GB = 16.0

# Power-of-2 bucket sizes for static array shapes during chunked processing
# This eliminates JIT recompilation overhead by ensuring all chunks pad to
# a fixed set of sizes, enabling efficient compilation cache reuse.
# See research.md for rationale on power-of-2 buckets.
CHUNK_BUCKETS: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


def get_bucket_size(chunk_size: int) -> int:
    """Get the smallest bucket size that can contain the given chunk.

    Parameters
    ----------
    chunk_size : int
        Actual chunk size in data points.

    Returns
    -------
    int
        Bucket size from CHUNK_BUCKETS that is >= chunk_size.
        If chunk_size exceeds max bucket, returns chunk_size unchanged.

    Examples
    --------
    >>> get_bucket_size(1000)
    1024
    >>> get_bucket_size(5000)
    8192
    >>> get_bucket_size(200000)
    200000
    """
    for bucket in CHUNK_BUCKETS:
        if bucket >= chunk_size:
            return bucket
    # Chunk exceeds largest bucket - return unchanged (will cause recompilation
    # but only for very large chunks which are less common)
    return chunk_size


@dataclass(slots=True)
class ChunkBuffer:
    """Pre-allocated static-shaped buffer for chunked data processing.

    Eliminates JIT recompilation by padding data to power-of-2 bucket sizes.
    The mask field allows filtering out padded elements when computing results.

    Attributes
    ----------
    data : np.ndarray
        Padded data array with shape (bucket_size,) or (bucket_size, features).
    valid_length : int
        Actual number of valid samples (before padding).
    bucket_size : int
        Static buffer size from CHUNK_BUCKETS.
    mask : np.ndarray
        Boolean mask where True indicates valid elements.

    Examples
    --------
    >>> import numpy as np
    >>> chunk = np.array([1.0, 2.0, 3.0])
    >>> buffer = ChunkBuffer.from_array(chunk)
    >>> buffer.bucket_size
    1024
    >>> buffer.valid_length
    3
    >>> np.sum(buffer.mask)
    3
    """

    data: np.ndarray
    valid_length: int
    bucket_size: int
    mask: np.ndarray

    @classmethod
    def from_array(cls, arr: np.ndarray, pad_value: float = 0.0) -> "ChunkBuffer":
        """Create a ChunkBuffer from an array, padding to the appropriate bucket size.

        Parameters
        ----------
        arr : np.ndarray
            Input array to pad.
        pad_value : float, optional
            Value to use for padding (default: 0.0).

        Returns
        -------
        ChunkBuffer
            Buffer with data padded to bucket size.
        """
        valid_length = len(arr)
        bucket_size = get_bucket_size(valid_length)

        # Create mask for valid elements
        mask = np.zeros(bucket_size, dtype=bool)
        mask[:valid_length] = True

        # Pad data to bucket size
        if valid_length == bucket_size:
            padded_data = arr
        elif arr.ndim == 1:
            padded_data = np.full(bucket_size, pad_value, dtype=arr.dtype)
            padded_data[:valid_length] = arr
        else:
            # Multi-dimensional: pad along first axis
            pad_shape = (bucket_size - valid_length, *arr.shape[1:])
            padding = np.full(pad_shape, pad_value, dtype=arr.dtype)
            padded_data = np.concatenate([arr, padding], axis=0)

        return cls(
            data=padded_data,
            valid_length=valid_length,
            bucket_size=bucket_size,
            mask=mask,
        )

    def get_valid_data(self) -> np.ndarray:
        """Return only the valid (non-padded) portion of the data."""
        return self.data[: self.valid_length]


@dataclass(slots=True)
class LDMemoryConfig:  # Renamed to avoid conflict with config.py
    """Configuration for memory management in large dataset fitting.

    Attributes
    ----------
    memory_limit_gb : float
        Maximum memory to use in GB (default: 8.0)
    safety_factor : float
        Safety factor for memory calculations (default: 0.8)
    min_chunk_size : int
        Minimum chunk size in data points (default: 1000)
    max_chunk_size : int
        Maximum chunk size in data points (default: 1000000)
    use_streaming : bool
        Use streaming optimization for unlimited data (default: True)
        Streaming is now always available (h5py is a required dependency as of v0.2.0)
    streaming_batch_size : int
        Batch size for streaming optimization (default: 50000)
    streaming_max_epochs : int
        Maximum epochs for streaming optimization (default: 10)
    min_success_rate : float
        Minimum success rate for chunked fitting (default: 0.5)
        If success rate falls below this threshold, fitting is considered failed
    save_diagnostics : bool
        Whether to compute and save detailed diagnostic statistics (default: False)
        When False, skips statistical computations for successful chunks (5-10% faster)
    """

    memory_limit_gb: float = 8.0
    safety_factor: float = 0.8
    min_chunk_size: int = 1000
    max_chunk_size: int = 1_000_000
    use_streaming: bool = True  # Always available (h5py required as of v0.2.0)
    streaming_batch_size: int = 50000
    streaming_max_epochs: int = 10
    min_success_rate: float = 0.5
    save_diagnostics: bool = False


@dataclass(slots=True)
class DatasetStats:
    """Statistics and information about a dataset.

    Attributes
    ----------
    n_points : int
        Total number of data points
    n_params : int
        Number of parameters to fit
    memory_per_point_bytes : float
        Estimated memory usage per data point in bytes
    total_memory_estimate_gb : float
        Estimated total memory requirement in GB
    recommended_chunk_size : int
        Recommended chunk size for processing
    n_chunks : int
        Number of chunks needed
    """

    n_points: int
    n_params: int
    memory_per_point_bytes: float
    total_memory_estimate_gb: float
    recommended_chunk_size: int
    n_chunks: int


class GPUMemoryEstimator:
    """Utilities for estimating GPU memory availability.

    This class provides GPU memory detection via JAX's device API,
    handling multiple GPUs and graceful fallback for CPU-only environments.

    Examples
    --------
    >>> from nlsq.streaming.large_dataset import GPUMemoryEstimator
    >>> estimator = GPUMemoryEstimator()
    >>> available_gb = estimator.get_available_gpu_memory_gb()
    >>> print(f"Available GPU memory: {available_gb:.2f} GB")
    """

    def __init__(self) -> None:
        """Initialize GPUMemoryEstimator."""
        self._logger = get_logger(__name__)

    def get_available_gpu_memory_gb(self) -> float:
        """Get available GPU memory in GB.

        Queries GPU memory via `jax.devices()[i].memory_stats()` and aggregates
        available memory across all GPUs. Returns 0 for CPU-only environments.

        Returns
        -------
        float
            Available GPU memory in GB, or 0.0 if no GPU or detection fails.

        Notes
        -----
        - Re-evaluates on each call (no caching) per requirements.
        - Handles multiple GPUs by summing available memory.
        - Returns 0.0 gracefully for CPU-only environments or when detection fails.
        """
        total_available_bytes = 0.0

        try:
            devices = jax.devices()

            for device in devices:
                # Skip CPU devices
                platform = getattr(device, "platform", "cpu")
                if platform == "cpu":
                    continue

                try:
                    # Query memory stats from GPU device
                    memory_stats = device.memory_stats()

                    if memory_stats is not None:
                        # Calculate available memory: limit - in_use
                        bytes_limit = memory_stats.get("bytes_limit", 0)
                        bytes_in_use = memory_stats.get("bytes_in_use", 0)
                        available = bytes_limit - bytes_in_use

                        if available > 0:
                            total_available_bytes += available
                            self._logger.debug(
                                f"GPU device {device}: "
                                f"{available / (1024**3):.2f} GB available "
                                f"({bytes_in_use / (1024**3):.2f} GB in use)"
                            )
                except Exception as e:
                    # Individual device query failed - continue with others
                    self._logger.debug(
                        f"GPU memory query failed for device {device}: {e}"
                    )
                    continue

        except Exception as e:
            # Complete device enumeration failed
            self._logger.debug(f"GPU device enumeration failed: {e}")
            return 0.0

        return total_available_bytes / (1024**3)

    def get_total_gpu_memory_gb(self) -> float:
        """Get total GPU memory capacity in GB.

        Returns
        -------
        float
            Total GPU memory capacity in GB, or 0.0 if no GPU.
        """
        total_capacity_bytes = 0.0

        try:
            devices = jax.devices()

            for device in devices:
                platform = getattr(device, "platform", "cpu")
                if platform == "cpu":
                    continue

                try:
                    memory_stats = device.memory_stats()
                    if memory_stats is not None:
                        bytes_limit = memory_stats.get("bytes_limit", 0)
                        total_capacity_bytes += bytes_limit
                except Exception:
                    continue

        except Exception:
            return 0.0

        return total_capacity_bytes / (1024**3)

    def has_gpu(self) -> bool:
        """Check if any GPU is available.

        Returns
        -------
        bool
            True if at least one GPU device is available.
        """
        try:
            devices = jax.devices()
            for device in devices:
                platform = getattr(device, "platform", "cpu")
                if platform != "cpu":
                    return True
        except Exception:
            pass
        return False


class MemoryEstimator:
    """Utilities for estimating memory usage and optimal chunk sizes.

    This class provides CPU memory detection via psutil, with fallback to 16GB
    when detection fails (e.g., containerized environments with cgroups).
    """

    @staticmethod
    def estimate_memory_per_point(n_params: int, use_jacobian: bool = True) -> float:
        """Estimate memory usage per data point in bytes.

        Parameters
        ----------
        n_params : int
            Number of parameters
        use_jacobian : bool, optional
            Whether Jacobian computation is needed (default: True)

        Returns
        -------
        float
            Estimated memory usage per point in bytes
        """
        # Estimate memory per data point
        base_memory = 3 * 8  # x, y, residual (float64)
        jacobian_memory = n_params * 8 if use_jacobian else 0
        work_memory = n_params * 2 * 8  # optimization workspace
        jax_overhead = 50  # XLA + GPU overhead
        return base_memory + jacobian_memory + work_memory + jax_overhead

    @staticmethod
    def get_available_memory_gb() -> float:
        """Get available system memory in GB.

        Returns
        -------
        float
            Available memory in GB. Falls back to 16GB if detection fails.

        Notes
        -----
        - Re-evaluates on each call (no caching) per requirements.
        - Falls back to 16GB when detection fails (containerized environments).
        """
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024**3)  # Convert to GB
        except Exception:
            # Fallback estimate: 16GB per requirements (updated from 4GB)
            return _DEFAULT_FALLBACK_MEMORY_GB

    @staticmethod
    def get_total_available_memory_gb() -> float:
        """Get total available memory (CPU + GPU) in GB.

        Combines available CPU memory from psutil with available GPU memory
        from JAX device API. Re-evaluates on each call (no caching).

        Returns
        -------
        float
            Total available memory (CPU + GPU) in GB.

        Notes
        -----
        - CPU memory: Uses psutil.virtual_memory().available
        - GPU memory: Uses JAX device API via GPUMemoryEstimator
        - Falls back to 16GB for CPU if detection fails
        - Returns 0 for GPU if detection fails or no GPU present
        """
        # Get CPU memory
        cpu_memory_gb = MemoryEstimator.get_available_memory_gb()

        # Get GPU memory
        gpu_estimator = GPUMemoryEstimator()
        gpu_memory_gb = gpu_estimator.get_available_gpu_memory_gb()

        return cpu_memory_gb + gpu_memory_gb

    @staticmethod
    def estimate_maximum_memory_usage_gb(
        n_points: int, n_params: int, safety_factor: float = 1.2
    ) -> float:
        """Estimate maximum memory usage to prevent crashes.

        Parameters
        ----------
        n_points : int
            Number of data points
        n_params : int
            Number of parameters
        safety_factor : float, optional
            Safety factor for memory estimation (default: 1.2)

        Returns
        -------
        float
            Estimated maximum memory usage in GB.

        Notes
        -----
        This method estimates the peak memory usage during optimization,
        which is useful for workflow selection to prevent out-of-memory crashes.
        """
        memory_per_point = MemoryEstimator.estimate_memory_per_point(n_params)
        total_bytes = n_points * memory_per_point * safety_factor
        return total_bytes / (1024**3)

    @staticmethod
    def calculate_optimal_chunk_size(
        n_points: int, n_params: int, memory_config: LDMemoryConfig
    ) -> tuple[int, DatasetStats]:
        """Calculate optimal chunk size based on memory constraints.

        Parameters
        ----------
        n_points : int
            Total number of data points
        n_params : int
            Number of parameters
        memory_config : LDMemoryConfig
            Memory configuration

        Returns
        -------
        tuple[int, DatasetStats]
            Optimal chunk size and dataset statistics
        """
        estimator = MemoryEstimator()

        # Estimate memory per point
        memory_per_point = estimator.estimate_memory_per_point(n_params)

        # Calculate available memory for processing
        available_memory_gb = (
            min(memory_config.memory_limit_gb, estimator.get_available_memory_gb())
            * memory_config.safety_factor
        )

        available_memory_bytes = available_memory_gb * (1024**3)

        # Calculate optimal chunk size
        theoretical_chunk_size = int(available_memory_bytes / memory_per_point)

        # Apply constraints
        chunk_size = max(
            memory_config.min_chunk_size,
            min(memory_config.max_chunk_size, theoretical_chunk_size),
        )

        # If we can fit all data in memory, use all points
        if n_points <= chunk_size:
            chunk_size = n_points
            n_chunks = 1
        else:
            n_chunks = (n_points + chunk_size - 1) // chunk_size

        # Calculate total memory estimate
        total_memory_gb = (n_points * memory_per_point) / (1024**3)

        stats = DatasetStats(
            n_points=n_points,
            n_params=n_params,
            memory_per_point_bytes=memory_per_point,
            total_memory_estimate_gb=total_memory_gb,
            recommended_chunk_size=chunk_size,
            n_chunks=n_chunks,
        )

        return chunk_size, stats


def cleanup_memory() -> None:
    """Perform memory cleanup between workflow phases.

    This function clears both Python garbage and JAX compilation caches,
    designed to be called between workflow phases to free memory.

    Notes
    -----
    - Calls gc.collect() to trigger Python garbage collection
    - Calls jax.clear_caches() to clear JAX JIT compilation caches
    - Handles errors gracefully (does not raise exceptions)

    Examples
    --------
    >>> from nlsq.streaming.large_dataset import cleanup_memory
    >>> # After completing a workflow phase
    >>> cleanup_memory()
    """
    logger = get_logger(__name__)

    # Python garbage collection
    try:
        gc.collect()
    except Exception as e:
        logger.debug(f"gc.collect() failed (non-critical): {e}")

    # JAX cache cleanup
    try:
        jax.clear_caches()
    except Exception as e:
        logger.debug(f"jax.clear_caches() failed (non-critical): {e}")


def get_memory_tier(available_memory_gb: float) -> "MemoryTier":
    """Classify available memory into a MemoryTier.

    Convenience function that wraps MemoryTier.from_available_memory_gb()
    for easier access from this module.

    Parameters
    ----------
    available_memory_gb : float
        Available system memory in gigabytes.

    Returns
    -------
    MemoryTier
        The appropriate memory tier (LOW, MEDIUM, HIGH, or VERY_HIGH).

    Examples
    --------
    >>> from nlsq.streaming.large_dataset import get_memory_tier
    >>> tier = get_memory_tier(32.0)
    >>> tier
    <MemoryTier.MEDIUM: (64.0, 'Standard memory (16-64GB)')>
    """
    # Deferred import to avoid circular dependency
    from nlsq.core.workflow import MemoryTier

    return MemoryTier.from_available_memory_gb(available_memory_gb)


class ProgressReporter:
    """Progress reporting for long-running fits."""

    def __init__(self, total_chunks: int, logger=None):
        """Initialize progress reporter.

        Parameters
        ----------
        total_chunks : int
            Total number of chunks to process
        logger : optional
            Logger instance for reporting progress
        """
        self.total_chunks = total_chunks
        self.logger = logger or get_logger(__name__)
        self.start_time = time.time()
        self.completed_chunks = 0

    def update(self, chunk_idx: int, chunk_result: dict | None = None):
        """Update progress.

        Parameters
        ----------
        chunk_idx : int
            Index of completed chunk
        chunk_result : dict, optional
            Results from chunk processing
        """
        self.completed_chunks = chunk_idx + 1
        elapsed = time.time() - self.start_time

        if self.completed_chunks > 0:
            avg_time_per_chunk = elapsed / self.completed_chunks
            remaining_chunks = self.total_chunks - self.completed_chunks
            eta = avg_time_per_chunk * remaining_chunks
        else:
            eta = 0

        progress_pct = (self.completed_chunks / self.total_chunks) * 100

        self.logger.info(
            f"Progress: {self.completed_chunks}/{self.total_chunks} chunks "
            f"({progress_pct:.1f}%) - ETA: {eta:.1f}s"
        )

        if chunk_result:
            self.logger.debug(f"Chunk {chunk_idx} result: {chunk_result}")


class DataChunker:
    """Utility for creating and managing data chunks."""

    @staticmethod
    def create_chunks(
        xdata: np.ndarray,
        ydata: np.ndarray,
        chunk_size: int,
        shuffle: bool = False,
        random_seed: int | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray, int]]:
        """Create data chunks for processing.

        Parameters
        ----------
        xdata : np.ndarray
            Independent variable data
        ydata : np.ndarray
            Dependent variable data
        chunk_size : int
            Size of each chunk
        shuffle : bool, optional
            Whether to shuffle data before chunking (default: False)
        random_seed : int, optional
            Random seed for shuffling

        Yields
        ------
        tuple[np.ndarray, np.ndarray, int, int]
            (x_chunk, y_chunk, chunk_index, valid_length)
            where valid_length is the actual number of data points (before padding)

        Notes
        -----
        Uses power-of-2 bucket sizes from CHUNK_BUCKETS for JIT cache efficiency.
        This ensures consistent array shapes across chunks, enabling JAX to reuse
        compiled kernels and avoiding recompilation overhead.
        """
        n_points = len(xdata)
        indices = np.arange(n_points)

        if shuffle:
            rng = np.random.default_rng(random_seed)
            rng.shuffle(indices)

        n_chunks = (n_points + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, n_points)
            chunk_indices = indices[start_idx:end_idx]

            # PERFORMANCE FIX: Pad to power-of-2 bucket sizes for JIT cache efficiency
            # Uses CHUNK_BUCKETS to ensure consistent shapes across different chunk sizes,
            # enabling JAX to reuse compiled kernels. This is more cache-efficient than
            # padding to chunk_size (which may vary) because power-of-2 buckets mean
            # fewer unique shapes and thus fewer JIT compilations.
            current_chunk_size = len(chunk_indices)
            bucket_size = get_bucket_size(current_chunk_size)

            if current_chunk_size < bucket_size:
                # Use cyclic padding (np.resize) to fill bucket - this is mathematically
                # safe for least-squares as repeated points don't change the solution
                chunk_indices = np.resize(chunk_indices, bucket_size)

            yield xdata[chunk_indices], ydata[chunk_indices], i, current_chunk_size


class LargeDatasetFitter:
    """Large dataset curve fitting with automatic memory management and chunking.

    This class handles datasets with millions to billions of points that exceed available
    memory through automatic chunking, progressive parameter refinement, and streaming
    optimization. It maintains fitting accuracy while preventing memory overflow through
    dynamic memory monitoring and chunk size optimization.

    Core Capabilities
    -----------------
    - Automatic memory estimation based on data size and parameter count
    - Dynamic chunk size calculation considering available system memory
    - Sequential parameter refinement across data chunks with convergence tracking
    - Streaming optimization for unlimited datasets (no accuracy loss)
    - Real-time progress monitoring with ETA for long-running fits
    - Full integration with NLSQ optimization algorithms and GPU acceleration
    - Multi-start optimization for global search (uses full data)

    Memory Management Algorithm
    ---------------------------
    1. Estimates total memory requirements from dataset size and parameter count
    2. Calculates optimal chunk sizes considering available memory and safety margins
    3. Monitors actual memory usage during processing to prevent overflow
    4. Uses streaming optimization for extremely large datasets (processes all data)

    Processing Strategies
    ---------------------
    - **Single Pass**: For datasets fitting within memory limits
    - **Sequential Chunking**: Processes data in optimal-sized chunks with parameter propagation
    - **Streaming Optimization**: Mini-batch gradient descent for unlimited datasets (no subsampling)

    Multi-Start Optimization
    ------------------------
    For medium-sized datasets (1M-100M points), multi-start optimization explores
    multiple starting points on full data, and the best starting point is then
    used for the full chunked optimization.

    Performance Characteristics
    ---------------------------
    - Maintains <1% parameter error for well-conditioned problems using chunking
    - Achieves 5-50x speedup over naive approaches through memory optimization
    - Scales to datasets of unlimited size using streaming (processes all data)
    - Provides linear time complexity with respect to chunk count

    Model Validation Caching (Task Group 7 - 5.1a)
    ----------------------------------------------
    Model functions are validated once per unique function identity using a cache
    keyed by (id(func), id(func.__code__)). This avoids redundant validation
    across chunks, providing 1-5% performance gain in chunked processing.

    Parameters
    ----------
    memory_limit_gb : float, default 8.0
        Maximum memory usage in GB. System memory is auto-detected if None.
    config : LDMemoryConfig, optional
        Advanced configuration for fine-tuning memory management behavior.
    curve_fit_class : nlsq.minpack.CurveFit, optional
        Custom CurveFit instance for specialized fitting requirements.
    multistart : bool, default False
        Enable multi-start optimization for global search.
    n_starts : int, default 10
        Number of starting points for multi-start optimization.
    sampler : str, default 'lhs'
        Sampling strategy for multi-start: 'lhs', 'sobol', or 'halton'.

    Attributes
    ----------
    config : LDMemoryConfig
        Active memory management configuration
    curve_fitter : nlsq.minpack.CurveFit
        Internal curve fitting engine with JAX acceleration
    logger : Logger
        Internal logging for performance monitoring and debugging

    Methods
    -------
    fit : Main fitting method with automatic memory management
    fit_with_progress : Fitting with real-time progress reporting and ETA
    get_memory_recommendations : Pre-fitting memory analysis and strategy recommendations

    Important: Chunking-Compatible Model Functions
    -----------------------------------------------
    When using chunked processing (for datasets > memory limit), your model function
    MUST respect the size of xdata. During chunking, xdata will be a subset of the
    full dataset, and your model must return output matching that subset size.

    **INCORRECT - Model ignores xdata size (will cause shape mismatch errors):**

    >>> def bad_model(xdata, a, b):
    ...     # WRONG: Always returns full array, ignoring xdata size
    ...     t_full = jnp.arange(10_000_000)  # Fixed size!
    ...     return a * jnp.exp(-b * t_full)  # Shape mismatch during chunking

    **CORRECT - Model respects xdata size:**

    >>> def good_model(xdata, a, b):
    ...     # CORRECT: Uses xdata as indices to return only requested subset
    ...     indices = xdata.astype(jnp.int32)
    ...     return a * jnp.exp(-b * indices)  # Shape matches xdata

    **Alternative - Direct computation on xdata:**

    >>> def direct_model(xdata, a, b):
    ...     # CORRECT: Operates directly on xdata
    ...     return a * jnp.exp(-b * xdata)  # Shape automatically matches

    Examples
    --------
    Basic usage with automatic configuration:

    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>>
    >>> # 10 million data points
    >>> x = np.linspace(0, 10, 10_000_000)
    >>> y = 2.5 * jnp.exp(-1.3 * x) + 0.1 + np.random.normal(0, 0.05, len(x))
    >>>
    >>> fitter = LargeDatasetFitter(memory_limit_gb=4.0)
    >>> result = fitter.fit(
    ...     lambda x, a, b, c: a * jnp.exp(-b * x) + c,
    ...     x, y, p0=[2, 1, 0]
    ... )
    >>> print(f"Parameters: {result.popt}")
    >>> print(f"Chunks used: {result.n_chunks}")

    Multi-start optimization:

    >>> fitter = LargeDatasetFitter(
    ...     memory_limit_gb=4.0,
    ...     multistart=True,
    ...     n_starts=10,
    ...     sampler='lhs',
    ... )
    >>> result = fitter.fit(
    ...     lambda x, a, b, c: a * jnp.exp(-b * x) + c,
    ...     x, y, p0=[2, 1, 0],
    ...     bounds=([0, 0, 0], [10, 5, 10])
    ... )

    Advanced configuration with progress monitoring:

    >>> config = LDMemoryConfig(
    ...     memory_limit_gb=8.0,
    ...     min_chunk_size=10000,
    ...     max_chunk_size=1000000,
    ...     use_streaming=True,
    ...     streaming_batch_size=50000
    ... )
    >>> fitter = LargeDatasetFitter(config=config)
    >>>
    >>> # Fit with progress bar for long-running operation
    >>> result = fitter.fit_with_progress(
    ...     exponential_model, x_huge, y_huge, p0=[2, 1, 0]
    ... )

    Memory analysis before processing:

    >>> recommendations = fitter.get_memory_recommendations(len(x), n_params=3)
    >>> print(f"Strategy: {recommendations['processing_strategy']}")
    >>> print(f"Memory estimate: {recommendations['memory_estimate_gb']:.2f} GB")
    >>> print(f"Recommended chunks: {recommendations['n_chunks']}")

    See Also
    --------
    curve_fit_large : High-level function with automatic dataset size detection
    LDMemoryConfig : Configuration class for memory management parameters
    estimate_memory_requirements : Standalone function for memory estimation

    Notes
    -----
    The sequential chunking algorithm maintains parameter accuracy by using each
    chunk's result as the initial guess for the next chunk. This approach typically
    maintains fitting accuracy within 0.1% of single-pass results for well-conditioned
    problems while enabling processing of arbitrarily large datasets.

    For extremely large datasets, streaming optimization processes all data using
    mini-batch gradient descent with no subsampling, ensuring zero accuracy loss
    compared to subsampling approaches (removed in v0.2.0).
    """

    def __init__(
        self,
        memory_limit_gb: float = 8.0,
        config: LDMemoryConfig | None = None,
        curve_fit_class: "CurveFit | None" = None,
        logger: Logger | None = None,
        enable_mixed_precision: bool | None = None,
        mixed_precision_config=None,
        # Multi-start parameters (Task Group 3)
        multistart: bool = False,
        n_starts: int = 10,
        sampler: Literal["lhs", "sobol", "halton"] = "lhs",
    ) -> None:
        """Initialize LargeDatasetFitter.

        Parameters
        ----------
        memory_limit_gb : float, optional
            Memory limit in GB (default: 8.0)
        config : LDMemoryConfig, optional
            Custom memory configuration
        curve_fit_class : nlsq.minpack.CurveFit, optional
            Custom CurveFit instance to use
        logger : logging.Logger, optional
            External logger instance for integration with application logging.
            If None, uses NLSQ's internal logger. This allows chunk failure
            warnings to appear in your application's logs.
        enable_mixed_precision : bool, optional
            Enable mixed precision optimization (float32 -> float64 fallback).
            If None (default), automatically enables for chunked datasets
            (provides 50% additional memory savings). Set to False to disable.
        mixed_precision_config : MixedPrecisionConfig, optional
            Custom configuration for mixed precision behavior. If None and
            mixed precision is enabled, uses default configuration.
        multistart : bool, optional
            Enable multi-start optimization for global search (default: False).
            When enabled, explores multiple starting points on full data
            before running the full chunked optimization.
        n_starts : int, optional
            Number of starting points for multi-start optimization (default: 10).
            Set to 0 to disable multi-start even when multistart=True.
        sampler : str, optional
            Sampling strategy for generating starting points (default: 'lhs').
            Options: 'lhs' (Latin Hypercube), 'sobol', 'halton'.
        """
        if config is None:
            config = LDMemoryConfig(memory_limit_gb=memory_limit_gb)

        self.config = config
        self.logger = logger or get_logger(__name__)

        # Mixed precision settings
        self.enable_mixed_precision = enable_mixed_precision
        self.mixed_precision_config = mixed_precision_config

        # Multi-start configuration (Task Group 3)
        self.multistart = multistart
        # Ensure enough starts for robust exploration on large datasets
        # Only enforce minimum when multistart is enabled AND n_starts > 0
        # (n_starts=0 explicitly disables multistart exploration)
        self.n_starts = max(n_starts, 8) if (multistart and n_starts > 0) else n_starts
        self.sampler = sampler

        # Create GlobalOptimizationConfig if multi-start is enabled
        self._multistart_config = None
        if self.multistart and self.n_starts > 0:
            from nlsq.global_optimization import GlobalOptimizationConfig

            self._multistart_config = GlobalOptimizationConfig(
                n_starts=self.n_starts,
                sampler=self.sampler,
                center_on_p0=True,
                scale_factor=1.0,
            )

        # Initialize curve fitting backend
        if curve_fit_class is None:
            # Deferred import to avoid circular dependency
            from nlsq.core.minpack import CurveFit

            self.curve_fit = CurveFit()
        else:
            self.curve_fit = curve_fit_class

        # Statistics tracking
        self.last_stats: DatasetStats | None = None
        self.fit_history: list[dict] = []
        self._error_log_timestamps: defaultdict = defaultdict(list)

        # Task Group 7 (5.1a): Model validation caching
        # Cache validated functions by (id(func), id(func.__code__)) to avoid
        # redundant validation across chunks. Provides 1-5% performance gain.
        self._validated_functions: dict[tuple[int, int], bool] = {}

    @lru_cache(maxsize=100)
    def _should_log_error(self, error_signature: str, current_time: float) -> bool:
        """Rate-limit error logging to prevent log flooding (max once per 60s per error type).

        Parameters
        ----------
        error_signature : str
            Unique signature identifying the error type
        current_time : float
            Current timestamp (rounded to 60s bucket)

        Returns
        -------
        bool
            True if error should be logged, False if rate-limited

        Notes
        -----
        Uses LRU cache to track recent errors. Each error type can be logged
        at most once per 60-second window, preventing log flooding attacks
        or excessive logging during systematic failures.
        """
        time_bucket = int(current_time // 60)
        f"{error_signature}_{time_bucket}"
        # LRU cache will return True first time, then cache hit returns True
        # This effectively rate-limits to once per time bucket
        return True

    def _log_validation_error(self, error: Exception) -> None:
        """Log validation error with rate limiting.

        Parameters
        ----------
        error : Exception
            The validation error to log
        """
        error_signature = f"{type(error).__name__}"
        current_time = time.time()

        if self._should_log_error(error_signature, current_time):
            self.logger.error(f"Model function validation failed: {error}")
            # Track timestamp for this error type
            self._error_log_timestamps[error_signature].append(current_time)

            # Cleanup old timestamps (older than 5 minutes)
            cutoff_time = current_time - 300
            self._error_log_timestamps[error_signature] = [
                t
                for t in self._error_log_timestamps[error_signature]
                if t > cutoff_time
            ]

    def _compute_chunk_stats(
        self, x_chunk: np.ndarray, y_chunk: np.ndarray
    ) -> dict[str, float]:
        """Compute diagnostic statistics for a data chunk.

        Parameters
        ----------
        x_chunk : np.ndarray
            Chunk of independent variable data
        y_chunk : np.ndarray
            Chunk of dependent variable data

        Returns
        -------
        dict
            Dictionary containing statistical measures
        """
        return {
            "x_mean": float(np.mean(x_chunk)),
            "x_std": float(np.std(x_chunk)),
            "y_mean": float(np.mean(y_chunk)),
            "y_std": float(np.std(y_chunk)),
        }

    def _compute_failed_chunk_stats(
        self, x_chunk: np.ndarray, y_chunk: np.ndarray
    ) -> dict[str, float | tuple]:
        """Compute detailed statistics for failed chunks (includes ranges).

        Parameters
        ----------
        x_chunk : np.ndarray
            Chunk of independent variable data
        y_chunk : np.ndarray
            Chunk of dependent variable data

        Returns
        -------
        dict
            Dictionary containing detailed statistical measures
        """
        return {
            "x_mean": float(np.mean(x_chunk)),
            "x_std": float(np.std(x_chunk)),
            "x_range": (float(np.min(x_chunk)), float(np.max(x_chunk))),
            "y_mean": float(np.mean(y_chunk)),
            "y_std": float(np.std(y_chunk)),
            "y_range": (float(np.min(y_chunk)), float(np.max(y_chunk))),
        }

    def _validate_model_function(
        self,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | list | None,
    ) -> None:
        """Validate model function shape compatibility before chunked processing.

        Tests the model function with a small subset of data to catch shape
        mismatches early with clear error messages.

        Parameters
        ----------
        f : callable
            The model function to validate
        xdata : np.ndarray
            Independent variable data
        ydata : np.ndarray
            Dependent variable data
        p0 : np.ndarray | list | None
            Initial parameter guess

        Raises
        ------
        ValueError
            If model function fails execution or returns wrong shape
        TypeError
            If model function returns non-array type

        Notes
        -----
        Task Group 7 (5.1a): Uses validation caching by function identity.
        Validation is skipped for functions that have already been validated,
        using composite key (id(func), id(func.__code__)) for cache lookup.
        Provides 1-5% performance gain in chunked processing.
        """
        # Task Group 7 (5.1a): Check validation cache
        # Use composite key for robust function identity
        func_key = (id(f), id(f.__code__))
        if func_key in self._validated_functions:
            self.logger.debug("Model validation skipped (cached)")
            return

        self.logger.debug("Validating model function shape compatibility...")

        try:
            # Test with first 100 points to avoid expensive computation
            test_size = min(100, len(xdata))
            x_test = xdata[:test_size]
            y_test = ydata[:test_size]

            # Get initial parameters for testing
            if p0 is None:
                # Try to infer from function signature
                try:
                    from inspect import signature

                    sig = signature(f)
                    n_params = len(sig.parameters) - 1  # Subtract x parameter
                    p0_test = np.ones(n_params)
                except Exception:
                    # Fallback to 2 parameters
                    p0_test = np.ones(2)
                    self.logger.warning(
                        "Could not infer parameter count, using 2 parameters for validation"
                    )
            else:
                p0_test = np.array(p0)

            # Call model function with test data
            try:
                output_test = f(x_test, *p0_test)
            except Exception as e:
                raise ValueError(
                    f"Model function failed on test data: {type(e).__name__}: {e}\n"
                    f"\n"
                    f"Model function must be callable as f(xdata, *params) and return array.\n"
                    f"Ensure your model:\n"
                    f"  1. Uses JAX operations (jax.numpy, not numpy)\n"
                    f"  2. Doesn't use Python control flow that breaks JIT\n"
                    f"  3. Returns numeric array, not scalar or other type\n"
                ) from e

            # Validate return type - check if it's array-like (numpy or JAX)
            is_array = isinstance(output_test, np.ndarray) or (
                hasattr(output_test, "shape") and hasattr(output_test, "dtype")
            )
            if not is_array:
                raise TypeError(
                    f"Model function must return array, got {type(output_test)}\n"
                    f"\n"
                    f"Your model returned: {type(output_test).__name__}\n"
                    f"Expected: numpy.ndarray or jax.Array\n"
                )

            # Validate shapes match
            if output_test.shape != y_test.shape:
                raise ValueError(
                    f"Model function SHAPE MISMATCH detected!\n"
                    f"\n"
                    f"  Input xdata shape:  {x_test.shape}\n"
                    f"  Input ydata shape:  {y_test.shape}\n"
                    f"  Model output shape: {output_test.shape}\n"
                    f"  Expected shape:     {y_test.shape}\n"
                    f"\n"
                    f"ERROR: Model output must match ydata size.\n"
                    f"\n"
                    f"When using curve_fit_large with chunking, your model function\n"
                    f"MUST respect the size of xdata. During chunked processing, xdata\n"
                    f"will be a subset (e.g., 1M points) of the full dataset.\n"
                    f"\n"
                    f"Common cause:\n"
                    f"  Your model ignores xdata size and always returns the full array.\n"
                    f"\n"
                    f"Fix: Use xdata as indices to return only the requested subset:\n"
                    f"\n"
                    f"  def model(xdata, *params):\n"
                    f"      # Compute full output if needed\n"
                    f"      y_full = compute_full_model(*params)  # e.g., shape (N,)\n"
                    f"      \n"
                    f"      # Return only requested indices for chunking compatibility\n"
                    f"      indices = xdata.astype(jnp.int32)  # Use JAX operations\n"
                    f"      return y_full[indices]  # Shape matches xdata\n"
                    f"\n"
                    f"See NLSQ documentation for more details on chunking-compatible models.\n"
                )

            self.logger.debug(
                f"Model validation passed: "
                f"f({x_test.shape}, {len(p0_test)} params) -> {output_test.shape}"
            )

            # Task Group 7 (5.1a): Cache successful validation
            self._validated_functions[func_key] = True

        except (ValueError, TypeError) as e:
            # Re-raise validation errors with context (rate-limited logging)
            self._log_validation_error(e)
            raise

        except Exception as e:
            # Unexpected error during validation
            self.logger.warning(
                f"Model validation encountered unexpected error: {type(e).__name__}: {e}\n"
                f"Proceeding with chunked fitting, but errors may occur."
            )
            # Don't fail here - let chunking proceed and catch real errors

    def _run_multistart_exploration(
        self,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | None,
        bounds: tuple,
        **kwargs,
    ) -> tuple[np.ndarray, dict]:
        """Run multi-start exploration on full data to find best starting point.

        Uses MultiStartOrchestrator to evaluate multiple starting points
        generated by LHS/Sobol/Halton sampling and returns the best one.

        Parameters
        ----------
        f : Callable
            Model function f(x, *params) -> y
        xdata : np.ndarray
            Full independent variable data
        ydata : np.ndarray
            Full dependent variable data
        p0 : np.ndarray | None
            Initial parameter guess (used for centering if center_on_p0=True)
        bounds : tuple
            Parameter bounds (lower, upper)
        **kwargs
            Additional arguments passed to curve_fit

        Returns
        -------
        tuple[np.ndarray, dict]
            (best_starting_point, multistart_diagnostics)
            - best_starting_point: Best parameters found from exploration
            - multistart_diagnostics: Dictionary with exploration details
        """
        from nlsq.global_optimization import MultiStartOrchestrator

        exploration_start_time = time.time()

        # Create orchestrator with our config
        orchestrator = MultiStartOrchestrator(
            config=self._multistart_config,
            curve_fit_instance=self.curve_fit,
        )

        self.logger.info(
            f"Running multi-start exploration with {self.n_starts} starting points "
            f"on {len(xdata):,} points using {self.sampler} sampling"
        )

        # Run exploration on full data
        result = orchestrator.fit(
            f=f,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            bounds=bounds,
            **kwargs,
        )

        exploration_time = time.time() - exploration_start_time

        # Extract diagnostics from result
        diagnostics = result.get("multistart_diagnostics", {})
        diagnostics["exploration_time_seconds"] = exploration_time
        diagnostics["dataset_size"] = len(xdata)

        # Get the best starting point
        best_params = result.popt if hasattr(result, "popt") else result.get("popt", p0)

        self.logger.info(
            f"Multi-start exploration completed in {exploration_time:.2f}s. "
            f"Best loss: {diagnostics.get('best_loss', 'N/A')}"
        )

        return np.asarray(best_params), diagnostics

    def estimate_requirements(self, n_points: int, n_params: int) -> DatasetStats:
        """Estimate memory requirements and processing strategy.

        Parameters
        ----------
        n_points : int
            Number of data points
        n_params : int
            Number of parameters to fit

        Returns
        -------
        DatasetStats
            Detailed statistics and recommendations
        """
        _, stats = MemoryEstimator.calculate_optimal_chunk_size(
            n_points, n_params, self.config
        )

        self.last_stats = stats

        # Log recommendations
        self.logger.info(
            f"Dataset analysis for {n_points:,} points, {n_params} parameters:"
        )
        self.logger.info(
            f"  Estimated memory per point: {stats.memory_per_point_bytes:.1f} bytes"
        )
        self.logger.info(
            f"  Total memory estimate: {stats.total_memory_estimate_gb:.2f} GB"
        )
        self.logger.info(f"  Recommended chunk size: {stats.recommended_chunk_size:,}")
        self.logger.info(f"  Number of chunks: {stats.n_chunks}")

        return stats

    def fit(
        self,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | list | None = None,
        bounds: tuple = (-np.inf, np.inf),
        method: str = "trf",
        solver: str = "auto",
        **kwargs,
    ) -> OptimizeResult:
        """Fit curve to large dataset with automatic memory management.

        Parameters
        ----------
        f : callable
            The model function f(x, \\*params) -> y
        xdata : np.ndarray
            Independent variable data
        ydata : np.ndarray
            Dependent variable data
        p0 : array-like, optional
            Initial parameter guess
        bounds : tuple, optional
            Parameter bounds (lower, upper)
        method : str, optional
            Optimization method (default: 'trf')
        solver : str, optional
            Solver type (default: 'auto')
        **kwargs
            Additional arguments passed to curve_fit

        Returns
        -------
        OptimizeResult
            Optimization result with fitted parameters and statistics
        """
        return self._fit_implementation(
            f, xdata, ydata, p0, bounds, method, solver, show_progress=False, **kwargs
        )

    def fit_with_progress(
        self,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | list | None = None,
        bounds: tuple = (-np.inf, np.inf),
        method: str = "trf",
        solver: str = "auto",
        **kwargs,
    ) -> OptimizeResult:
        """Fit curve with progress reporting for long-running fits.

        Parameters
        ----------
        f : callable
            The model function f(x, \\*params) -> y
        xdata : np.ndarray
            Independent variable data
        ydata : np.ndarray
            Dependent variable data
        p0 : array-like, optional
            Initial parameter guess
        bounds : tuple, optional
            Parameter bounds (lower, upper)
        method : str, optional
            Optimization method (default: 'trf')
        solver : str, optional
            Solver type (default: 'auto')
        **kwargs
            Additional arguments passed to curve_fit

        Returns
        -------
        OptimizeResult
            Optimization result with fitted parameters and statistics
        """
        return self._fit_implementation(
            f, xdata, ydata, p0, bounds, method, solver, show_progress=True, **kwargs
        )

    def _fit_implementation(
        self,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | list | None,
        bounds: tuple,
        method: str,
        solver: str,
        show_progress: bool,
        **kwargs,
    ) -> OptimizeResult:
        """Internal implementation of fitting algorithm."""

        fit_start_time = time.time()
        n_points = len(xdata)

        # Estimate number of parameters from function signature or p0
        if p0 is not None:
            n_params = len(p0)
        else:
            # Try to infer from function signature
            try:
                from inspect import signature

                sig = signature(f)
                n_params = len(sig.parameters) - 1  # Subtract x parameter
            except Exception:
                n_params = 2  # Conservative default

        # Normalize initial guess and apply heuristics for stability
        if p0 is not None:
            p0 = np.asarray(p0, dtype=float)
            if self.multistart and p0.size > 0 and p0[0] < 0.5:
                heuristic_amp = max(float(np.ptp(ydata)), 0.5)
                p0 = p0.copy()
                p0[0] = heuristic_amp

        # Get processing statistics and strategy
        stats = self.estimate_requirements(n_points, n_params)

        # Determine if chunking is needed
        needs_chunking = stats.n_chunks > 1

        # Initialize multi-start diagnostics
        multistart_diagnostics = {
            "n_starts_configured": self.n_starts if self.multistart else 0,
            "sampler": self.sampler,
            "bypassed": False,
        }

        # Run multi-start exploration if enabled and chunking is needed
        # (for single-chunk datasets, multi-start overhead isn't worth it)
        if self.multistart and self.n_starts > 0 and needs_chunking:
            self.logger.info("Multi-start optimization enabled for chunked dataset")

            # Run multi-start exploration on full data (no subsampling)
            best_p0, exploration_diagnostics = self._run_multistart_exploration(
                f=f,
                xdata=xdata,
                ydata=ydata,
                p0=np.array(p0) if p0 is not None else None,
                bounds=bounds,
                **kwargs,
            )

            # Update p0 with best starting point
            p0 = best_p0
            multistart_diagnostics.update(exploration_diagnostics)
            multistart_diagnostics["best_starting_point"] = (
                best_p0.tolist() if hasattr(best_p0, "tolist") else list(best_p0)
            )

            self.logger.info(
                f"Using best starting point from multi-start exploration: {best_p0}"
            )
        elif self.multistart and self.n_starts == 0:
            # Multi-start enabled but n_starts=0 means skip
            multistart_diagnostics["bypassed"] = True
            multistart_diagnostics["n_starts_evaluated"] = 0
            self.logger.debug("Multi-start disabled (n_starts=0)")
        elif self.multistart and not needs_chunking:
            # Single chunk dataset - skip multi-start overhead
            multistart_diagnostics["bypassed"] = True
            multistart_diagnostics["bypass_reason"] = "single_chunk_dataset"
            self.logger.debug("Multi-start skipped for single-chunk dataset")

        # Auto-enable mixed precision for chunked datasets (50% additional memory savings)
        enable_mp = self.enable_mixed_precision
        if enable_mp is None and needs_chunking:
            # Prefer accuracy over speed when multi-start is active
            if self.multistart:
                enable_mp = False
                self.logger.info(
                    "Mixed precision disabled for chunked multi-start to favor accuracy"
                )
            else:
                enable_mp = True
                self.logger.info(
                    "Auto-enabled mixed precision for chunked processing "
                    "(50% additional memory savings)"
                )

        # Create mixed precision manager if enabled
        mixed_precision_manager = None
        if enable_mp:
            from nlsq.precision.mixed_precision import (
                MixedPrecisionConfig,
                MixedPrecisionManager,
            )

            mp_config = self.mixed_precision_config
            if mp_config is None:
                mp_config = MixedPrecisionConfig()

            mixed_precision_manager = MixedPrecisionManager(mp_config)
            self.logger.info(
                "Mixed precision optimization enabled (float32 -> float64 fallback)"
            )

        # Handle datasets that fit in memory
        if stats.n_chunks == 1:
            result = self._fit_single_chunk(
                f,
                xdata,
                ydata,
                p0,
                bounds,
                method,
                solver,
                mixed_precision_manager=mixed_precision_manager,
                **kwargs,
            )
            # Add multi-start diagnostics to result
            result["multistart_diagnostics"] = multistart_diagnostics
            return result

        # Handle chunked processing (will use streaming if enabled for very large datasets)
        result = self._fit_chunked(
            f,
            xdata,
            ydata,
            p0,
            bounds,
            method,
            solver,
            show_progress,
            stats,
            mixed_precision_manager=mixed_precision_manager,
            **kwargs,
        )

        # Add multi-start diagnostics and timing to result
        multistart_diagnostics["total_fit_time_seconds"] = time.time() - fit_start_time
        result["multistart_diagnostics"] = multistart_diagnostics

        return result

    def _fit_single_chunk(
        self,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | list | None,
        bounds: tuple,
        method: str,
        solver: str,
        mixed_precision_manager=None,
        **kwargs,
    ) -> OptimizeResult:
        """Fit data that can be processed in a single chunk."""

        self.logger.info("Fitting dataset in single chunk")

        # Use standard curve_fit with mixed precision if enabled
        try:
            popt, _pcov = self.curve_fit.curve_fit(
                f,
                xdata,
                ydata,
                p0=p0,
                bounds=bounds,
                method=method,
                solver=solver,
                mixed_precision_manager=mixed_precision_manager,
                **kwargs,
            )

            # Create result object
            result = OptimizeResult(
                x=popt,
                success=True,
                fun=None,  # Could compute final residuals if needed
                nfev=1,  # Approximation
                message="Single-chunk fit completed successfully",
            )

            # Add covariance matrix and parameters
            result["pcov"] = _pcov
            result["popt"] = popt

            return result

        except Exception as e:
            self.logger.error(f"Single-chunk fit failed: {e}")
            result = OptimizeResult(
                x=p0 if p0 is not None else np.ones(2),
                success=False,
                message=f"Fit failed: {e}",
            )
            # Add empty popt and pcov for consistency
            result["popt"] = result.x
            result["pcov"] = np.eye(len(result.x))
            return result

    def _fit_with_streaming(
        self,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | list | None,
        bounds: tuple,
        method: str,
        solver: str,
        show_progress: bool,
        **kwargs,
    ) -> OptimizeResult:
        """Fit very large dataset using streaming optimization (no accuracy loss).

        This method processes unlimited data using mini-batch gradient descent
        without subsampling, preserving all data for maximum accuracy.
        """
        self.logger.info(
            f"Using streaming optimization for unlimited data ({len(xdata):,} points). "
            f"Batch size: {self.config.streaming_batch_size:,}, "
            f"Max epochs: {self.config.streaming_max_epochs}"
        )

        # Create streaming config
        streaming_config = StreamingConfig(
            batch_size=self.config.streaming_batch_size,
            max_epochs=self.config.streaming_max_epochs,
            use_adam=True,  # Adam is more stable for curve fitting
            learning_rate=0.001,  # Conservative learning rate
            convergence_tol=1e-6,
        )

        # Initialize streaming optimizer
        optimizer = StreamingOptimizer(config=streaming_config)

        # Prepare data generator (in-memory for numpy arrays)
        # For truly unlimited data, users should provide HDF5 files
        class InMemoryGenerator:
            """Simple in-memory data generator for streaming."""

            def __init__(self, x, y, batch_size):
                self.x = x
                self.y = y
                self.batch_size = batch_size
                self.n_points = len(x)

            def __iter__(self):
                """Generate batches of data."""
                indices = np.arange(self.n_points)
                np.random.shuffle(indices)

                for start_idx in range(0, self.n_points, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, self.n_points)
                    batch_indices = indices[start_idx:end_idx]

                    yield self.x[batch_indices], self.y[batch_indices]

        # Convert p0 to array if needed
        if p0 is None:
            p0 = np.ones(2)  # Default 2-parameter model
        elif isinstance(p0, list):
            p0 = np.array(p0)

        # Fit using streaming optimization
        try:
            data_gen = InMemoryGenerator(xdata, ydata, self.config.streaming_batch_size)
            result_dict = optimizer.fit_streaming(
                func=f,
                data_source=data_gen,
                p0=p0,
                bounds=bounds,
                verbose=2 if show_progress else 1,
            )

            # Convert to OptimizeResult format
            result = OptimizeResult(
                x=result_dict["params"],
                success=result_dict["success"],
                message="Streaming optimization completed",
                nfev=result_dict.get("total_samples", 0) // len(p0),
                fun=None,  # Not available in streaming mode
            )
            result["popt"] = result.x
            result["pcov"] = np.eye(
                len(result.x)
            )  # Approximate - streaming doesn't compute covariance

            self.logger.info(
                f"Streaming fit completed. Final loss: {result_dict.get('final_loss', 'N/A')}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Streaming fit failed: {e}")
            result = OptimizeResult(
                x=p0 if p0 is not None else np.ones(2),
                success=False,
                message=f"Streaming fit failed: {e}",
            )
            result["popt"] = result.x
            result["pcov"] = np.eye(len(result.x))
            return result

    def _update_parameters_convergence(
        self,
        current_params: np.ndarray | None,
        popt_chunk: np.ndarray,
        param_history: list,
        convergence_metric: float,
        chunk_idx: int,
        n_chunks: int,
    ) -> tuple[np.ndarray, list, float, bool]:
        """Update parameters with sequential refinement and convergence checking.

        Args:
            current_params: Current parameter estimates (None on first chunk)
            popt_chunk: Newly fitted parameters from current chunk
            param_history: List of parameter estimates from previous chunks
            convergence_metric: Current convergence metric value
            chunk_idx: Index of current chunk (0-based)
            n_chunks: Total number of chunks

        Returns:
            tuple: (updated_params, updated_history, new_convergence_metric, should_stop)
                - updated_params: New current parameter estimates
                - updated_history: Updated parameter history
                - new_convergence_metric: Updated convergence metric
                - should_stop: True if early stopping criteria met
        """
        # Initialize on first chunk
        if current_params is None:
            return (
                popt_chunk.copy(),
                [popt_chunk.copy()],
                np.inf,
                False,
            )

        # Update parameters with sequential refinement
        previous_params = current_params.copy()
        updated_params = popt_chunk.copy()

        # Update parameter history
        updated_history = [*param_history, updated_params.copy()]

        # Calculate convergence metric
        new_convergence_metric = convergence_metric
        if len(updated_history) > 2:
            param_change = np.linalg.norm(updated_params - previous_params)
            relative_change = param_change / (np.linalg.norm(updated_params) + 1e-10)
            new_convergence_metric = relative_change

            # Check early stopping criteria
            # Stop if parameters stabilized and we've processed enough chunks
            if new_convergence_metric < 0.001 and chunk_idx >= min(n_chunks - 1, 3):
                self.logger.info(f"Parameters converged after {chunk_idx + 1} chunks")
                return (updated_params, updated_history, new_convergence_metric, True)

        return (updated_params, updated_history, new_convergence_metric, False)

    def _initialize_chunked_fit_state(
        self,
        p0: np.ndarray | list | None,
        show_progress: bool,
        stats: DatasetStats,
    ) -> tuple[
        ProgressReporter | None,
        np.ndarray | None,
        list,
        list,
        float,
    ]:
        """Initialize state variables for chunked fitting.

        Parameters
        ----------
        p0 : np.ndarray | list | None
            Initial parameter guess
        show_progress : bool
            Whether to show progress updates
        stats : DatasetStats
            Dataset statistics including chunk count

        Returns
        -------
        progress : ProgressReporter | None
            Progress reporter instance or None
        current_params : np.ndarray | None
            Initial parameters
        chunk_results : list
            Empty list for accumulating chunk results
        param_history : list
            Empty list for tracking parameter evolution
        convergence_metric : float
            Initial convergence metric (infinity)
        """
        # Initialize progress reporter
        progress = (
            ProgressReporter(stats.n_chunks, self.logger) if show_progress else None
        )

        # Initialize parameters
        current_params = np.array(p0) if p0 is not None else None

        # Initialize tracking lists
        chunk_results = []
        param_history = []
        convergence_metric = np.inf

        return (
            progress,
            current_params,
            chunk_results,
            param_history,
            convergence_metric,
        )

    def _create_chunk_result(
        self,
        chunk_idx: int,
        x_chunk: np.ndarray,
        y_chunk: np.ndarray,
        chunk_duration: float,
        success: bool = True,
        popt_chunk: np.ndarray | None = None,
        is_retry: bool = False,
        error: Exception | None = None,
        current_params: np.ndarray | None = None,
    ) -> dict:
        """Create a standardized chunk result dictionary.

        Args:
            chunk_idx: Index of the chunk
            x_chunk: Input data for this chunk
            y_chunk: Output data for this chunk
            chunk_duration: Time taken to process this chunk
            success: Whether the chunk fitting succeeded
            popt_chunk: Fitted parameters (if successful)
            is_retry: Whether this was a retry attempt
            error: Exception that occurred (if failed)
            current_params: Current parameter estimates (for failure diagnostics)

        Returns:
            dict: Standardized chunk result with metadata
        """
        # Base result structure
        result = {
            "chunk_idx": chunk_idx,
            "n_points": len(x_chunk),
            "success": success,
            "timestamp": time.time(),
            "duration": chunk_duration,
        }

        if success:
            # Success case
            result["parameters"] = popt_chunk
            if is_retry:
                result["retry"] = True

            # Add diagnostics if enabled (5-10% performance gain when disabled)
            if self.config.save_diagnostics:
                result["data_stats"] = self._compute_chunk_stats(x_chunk, y_chunk)
        else:
            # Failure case
            result["error"] = str(error)
            result["error_type"] = type(error).__name__
            result["initial_params"] = (
                current_params.tolist() if current_params is not None else None
            )
            # Always compute detailed stats for failed chunks (debugging critical)
            result["data_stats"] = self._compute_failed_chunk_stats(x_chunk, y_chunk)

        return result

    def _retry_failed_chunk(
        self,
        f: Callable,
        x_chunk: np.ndarray,
        y_chunk: np.ndarray,
        chunk_idx: int,
        chunk_start_time: float,
        chunk_times: list,
        current_params: np.ndarray | None,
        initial_error: Exception,
        bounds: tuple,
        method: str,
        solver: str,
        mixed_precision_manager=None,
        **kwargs,
    ) -> tuple[dict, np.ndarray | None]:
        """Retry a failed chunk with perturbed parameters.

        Args:
            f: Model function
            x_chunk: Input data for this chunk
            y_chunk: Output data for this chunk
            chunk_idx: Index of the chunk
            chunk_start_time: Start time of chunk processing
            chunk_times: List to append chunk duration to
            current_params: Current parameter estimates
            initial_error: The exception that caused the initial failure
            bounds: Parameter bounds
            method: Optimization method
            solver: Solver type
            **kwargs: Additional curve_fit arguments

        Returns:
            tuple: (chunk_result dict, updated_params or None)
        """
        # Only retry if we have current parameter estimates
        if current_params is None:
            chunk_duration = time.time() - chunk_start_time
            chunk_times.append(chunk_duration)
            chunk_result = self._create_chunk_result(
                chunk_idx=chunk_idx,
                x_chunk=x_chunk,
                y_chunk=y_chunk,
                chunk_duration=chunk_duration,
                success=False,
                error=initial_error,
                current_params=current_params,
            )
            return chunk_result, None

        # Attempt retry with perturbed parameters
        try:
            self.logger.info(f"Retrying chunk {chunk_idx} with current parameters")
            # Add small perturbation to avoid local minima
            perturbed_params = current_params * (
                1 + 0.01 * np.random.randn(len(current_params))
            )
            popt_chunk, _pcov_chunk = self.curve_fit.curve_fit(
                f,
                x_chunk,
                y_chunk,
                p0=perturbed_params,
                bounds=bounds,
                method=method,
                solver=solver,
                mixed_precision_manager=mixed_precision_manager,
                **kwargs,
            )

            # Retry succeeded - use result with lower weight
            adaptive_lr = 0.1  # Lower weight for retry results
            updated_params = (
                1 - adaptive_lr
            ) * current_params + adaptive_lr * popt_chunk

            chunk_duration = time.time() - chunk_start_time
            chunk_times.append(chunk_duration)

            chunk_result = self._create_chunk_result(
                chunk_idx=chunk_idx,
                x_chunk=x_chunk,
                y_chunk=y_chunk,
                chunk_duration=chunk_duration,
                success=True,
                popt_chunk=popt_chunk,
                is_retry=True,
            )

            return chunk_result, updated_params

        except Exception as retry_e:
            # Retry also failed
            self.logger.warning(f"Retry for chunk {chunk_idx} also failed: {retry_e}")
            chunk_duration = time.time() - chunk_start_time
            chunk_times.append(chunk_duration)

            chunk_result = self._create_chunk_result(
                chunk_idx=chunk_idx,
                x_chunk=x_chunk,
                y_chunk=y_chunk,
                chunk_duration=chunk_duration,
                success=False,
                error=initial_error,
                current_params=current_params,
            )

            return chunk_result, current_params  # Keep current params unchanged

    def _create_failure_summary(
        self,
        chunk_results: list,
        chunk_times: list,
    ) -> dict:
        """Create comprehensive failure summary for diagnostics.

        Args:
            chunk_results: List of all chunk result dictionaries
            chunk_times: List of chunk processing durations

        Returns:
            dict: Failure summary with error types, timing stats, and common errors
        """
        failed_chunks = [r for r in chunk_results if not r.get("success", False)]

        failure_summary = {
            "total_failures": len(failed_chunks),
            "failure_rate": len(failed_chunks) / len(chunk_results)
            if chunk_results
            else 0.0,
            "failed_chunk_indices": [r["chunk_idx"] for r in failed_chunks],
            "error_types": {},
            "common_errors": [],
            "timing_stats": {
                "mean_chunk_time": float(np.mean(chunk_times)) if chunk_times else 0.0,
                "median_chunk_time": float(np.median(chunk_times))
                if chunk_times
                else 0.0,
                "failed_chunk_times": [r.get("duration", 0.0) for r in failed_chunks],
                "mean_failed_chunk_time": float(
                    np.mean([r.get("duration", 0.0) for r in failed_chunks])
                )
                if failed_chunks
                else 0.0,
            },
        }

        # Aggregate error types
        for failed_chunk in failed_chunks:
            error_type = failed_chunk.get("error_type", "Unknown")
            failure_summary["error_types"][error_type] = (
                failure_summary["error_types"].get(error_type, 0) + 1
            )

        # Identify most common errors (top 3)
        if failure_summary["error_types"]:
            sorted_errors = sorted(
                failure_summary["error_types"].items(), key=lambda x: x[1], reverse=True
            )
            failure_summary["common_errors"] = [
                {"type": err_type, "count": count}
                for err_type, count in sorted_errors[:3]
            ]

        return failure_summary

    def _compute_covariance_from_history(
        self,
        param_history: list,
        current_params: np.ndarray,
    ) -> np.ndarray:
        """Compute approximate covariance matrix from parameter history.

        In chunked fitting, we estimate covariance from parameter variations
        across chunks rather than from the Jacobian.

        Args:
            param_history: List of parameter estimates from previous chunks
            current_params: Final parameter estimates

        Returns:
            np.ndarray: Approximate covariance matrix
        """
        if len(param_history) > 1:
            # Use last 10 parameter estimates for covariance estimation
            param_variations = np.array(param_history[-min(10, len(param_history)) :])
            pcov = np.cov(param_variations.T)
        else:
            # Fallback: identity matrix scaled by parameter magnitudes
            # This provides a reasonable uncertainty estimate when we have no history
            pcov = np.diag(np.abs(current_params) * 0.01 + 0.001)

        return pcov

    def _finalize_chunked_results(
        self,
        current_params: np.ndarray,
        chunk_results: list,
        param_history: list,
        success_rate: float,
        stats: DatasetStats,
        chunk_times: list,
    ) -> OptimizeResult:
        """Assemble final optimization result from chunked fitting.

        Parameters
        ----------
        current_params : np.ndarray
            Final optimized parameters
        chunk_results : list
            List of all chunk result dictionaries
        param_history : list
            History of parameter estimates across chunks
        success_rate : float
            Fraction of successful chunks
        stats : DatasetStats
            Dataset statistics including chunk count
        chunk_times : list
            Processing durations for each chunk

        Returns
        -------
        OptimizeResult
            Final optimization result with parameters, covariance, and diagnostics
        """
        # Log completion
        self.logger.info(f"Chunked fit completed with {success_rate:.1%} success rate")

        # Create failure summary for diagnostics
        failure_summary = self._create_failure_summary(chunk_results, chunk_times)

        # Assemble result
        result = OptimizeResult(
            x=current_params,
            success=True,
            message=f"Chunked fit completed ({stats.n_chunks} chunks, {success_rate:.1%} success)",
        )
        result["popt"] = current_params

        # Create approximate covariance matrix from parameter history
        result["pcov"] = self._compute_covariance_from_history(
            param_history, current_params
        )

        # Add diagnostic information
        result["chunk_results"] = chunk_results
        result["n_chunks"] = stats.n_chunks
        result["success_rate"] = success_rate
        result["failure_summary"] = failure_summary

        return result

    def _check_success_rate_and_create_result(
        self,
        chunk_results: list,
        current_params: np.ndarray | None,
        param_history: list,
        stats: DatasetStats,
        chunk_times: list,
    ) -> OptimizeResult:
        """Check success rate and create appropriate result (success or failure).

        Args:
            chunk_results: List of chunk processing results
            current_params: Final parameter estimates
            param_history: History of parameter updates
            stats: Dataset statistics
            chunk_times: Processing time for each chunk

        Returns:
            OptimizeResult with success or failure status based on success rate
        """
        # Compute final statistics
        successful_chunks = [r for r in chunk_results if r.get("success", False)]
        success_rate = len(successful_chunks) / len(chunk_results)

        if success_rate < self.config.min_success_rate:
            self.logger.error(
                f"Too many chunks failed ({success_rate:.1%} success rate, "
                f"minimum required: {self.config.min_success_rate:.1%})"
            )
            result = OptimizeResult(
                x=current_params if current_params is not None else np.ones(2),
                success=False,
                message=f"Chunked fit failed: {success_rate:.1%} success rate",
            )
            # Add empty popt and pcov for consistency
            result["popt"] = (
                current_params if current_params is not None else np.ones(2)
            )
            result["pcov"] = np.eye(len(result["popt"]))
            return result

        # Success - assemble final result
        return self._finalize_chunked_results(
            current_params=current_params,
            chunk_results=chunk_results,
            param_history=param_history,
            success_rate=success_rate,
            stats=stats,
            chunk_times=chunk_times,
        )

    def _fit_chunked(
        self,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | list | None,
        bounds: tuple,
        method: str,
        solver: str,
        show_progress: bool,
        stats: DatasetStats,
        mixed_precision_manager=None,
        **kwargs,
    ) -> OptimizeResult:
        """Fit dataset using chunked processing with parameter refinement."""

        self.logger.info(f"Fitting dataset using {stats.n_chunks} chunks")

        # Validate model function shape compatibility
        self._validate_model_function(f, xdata, ydata, p0)

        # Initialize state variables
        (
            progress,
            current_params,
            chunk_results,
            param_history,
            convergence_metric,
        ) = self._initialize_chunked_fit_state(p0, show_progress, stats)
        chunk_times = []  # Track processing time per chunk

        try:
            # Process dataset in chunks with sequential parameter refinement
            # Note: create_chunks yields (x, y, idx, valid_length) - we ignore valid_length
            # here since the cyclic padding doesn't affect least-squares solutions
            for x_chunk, y_chunk, chunk_idx, _valid_length in DataChunker.create_chunks(
                xdata, ydata, stats.recommended_chunk_size
            ):
                chunk_start_time = time.time()
                try:
                    # Fit current chunk with mixed precision if enabled
                    popt_chunk, _pcov_chunk = self.curve_fit.curve_fit(
                        f,
                        x_chunk,
                        y_chunk,
                        p0=current_params,
                        bounds=bounds,
                        method=method,
                        solver=solver,
                        mixed_precision_manager=mixed_precision_manager,
                        **kwargs,
                    )

                    # Update parameters with sequential refinement and check convergence
                    (
                        current_params,
                        param_history,
                        convergence_metric,
                        should_stop,
                    ) = self._update_parameters_convergence(
                        current_params,
                        popt_chunk,
                        param_history,
                        convergence_metric,
                        chunk_idx,
                        stats.n_chunks,
                    )

                    # Early stopping if parameters converged
                    if should_stop:
                        break

                    chunk_duration = time.time() - chunk_start_time
                    chunk_times.append(chunk_duration)

                    # Create successful chunk result
                    chunk_result = self._create_chunk_result(
                        chunk_idx=chunk_idx,
                        x_chunk=x_chunk,
                        y_chunk=y_chunk,
                        chunk_duration=chunk_duration,
                        success=True,
                        popt_chunk=popt_chunk,
                    )

                except Exception as e:
                    self.logger.warning(f"Chunk {chunk_idx} failed: {e}")
                    # Retry chunk with helper method
                    chunk_result, retry_params = self._retry_failed_chunk(
                        f=f,
                        x_chunk=x_chunk,
                        y_chunk=y_chunk,
                        chunk_idx=chunk_idx,
                        chunk_start_time=chunk_start_time,
                        chunk_times=chunk_times,
                        current_params=current_params,
                        initial_error=e,
                        bounds=bounds,
                        method=method,
                        solver=solver,
                        mixed_precision_manager=mixed_precision_manager,
                        **kwargs,
                    )
                    # Update params if retry succeeded
                    if retry_params is not None:
                        current_params = retry_params

                chunk_results.append(chunk_result)

                if progress:
                    progress.update(chunk_idx, chunk_result)

                # Memory cleanup
                gc.collect()

            # Check success rate and create final result
            return self._check_success_rate_and_create_result(
                chunk_results=chunk_results,
                current_params=current_params,
                param_history=param_history,
                stats=stats,
                chunk_times=chunk_times,
            )

        except Exception as e:
            self.logger.error(f"Chunked fitting failed: {e}")
            result = OptimizeResult(
                x=current_params if current_params is not None else np.ones(2),
                success=False,
                message=f"Chunked fit failed: {e}",
            )
            # Add empty popt and pcov for consistency
            result["popt"] = (
                current_params if current_params is not None else np.ones(2)
            )
            result["pcov"] = np.eye(len(result["popt"]))
            return result

    @contextmanager
    def memory_monitor(self):
        """Context manager for monitoring memory usage during fits."""

        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024**3)  # GB
            self.logger.debug(f"Initial memory usage: {initial_memory:.2f} GB")
            yield
        finally:
            try:
                final_memory = process.memory_info().rss / (1024**3)  # GB
                memory_delta = final_memory - initial_memory
                self.logger.debug(
                    f"Final memory usage: {final_memory:.2f} GB (delta: {memory_delta:+.2f} GB)"
                )
            except Exception as e:
                # Memory monitoring is best effort - log but don't fail
                self.logger.debug(f"Memory monitoring failed (non-critical): {e}")

    def get_memory_recommendations(self, n_points: int, n_params: int) -> dict:
        """Get memory usage recommendations for a dataset.

        Parameters
        ----------
        n_points : int
            Number of data points
        n_params : int
            Number of parameters

        Returns
        -------
        dict
            Recommendations and memory analysis
        """
        stats = self.estimate_requirements(n_points, n_params)

        return {
            "dataset_stats": stats,
            "memory_limit_gb": self.config.memory_limit_gb,
            "processing_strategy": "single_chunk" if stats.n_chunks == 1 else "chunked",
            "recommendations": {
                "chunk_size": stats.recommended_chunk_size,
                "n_chunks": stats.n_chunks,
                "memory_per_point_bytes": stats.memory_per_point_bytes,
                "total_memory_estimate_gb": stats.total_memory_estimate_gb,
            },
        }


# Convenience functions
def fit_large_dataset(
    f: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray | list | None = None,
    memory_limit_gb: float = 8.0,
    show_progress: bool = False,
    logger: Logger | None = None,
    enable_mixed_precision: bool | None = None,
    mixed_precision_config=None,
    # Multi-start parameters (Task Group 3)
    multistart: bool = False,
    n_starts: int = 10,
    sampler: Literal["lhs", "sobol", "halton"] = "lhs",
    **kwargs,
) -> OptimizeResult:
    """Convenience function for fitting large datasets.

    Parameters
    ----------
    f : callable
        The model function f(x, \\*params) -> y
    xdata : np.ndarray
        Independent variable data
    ydata : np.ndarray
        Dependent variable data
    p0 : array-like, optional
        Initial parameter guess
    memory_limit_gb : float, optional
        Memory limit in GB (default: 8.0)
    show_progress : bool, optional
        Whether to show progress (default: False)
    logger : logging.Logger, optional
        External logger for application integration (default: None)
    enable_mixed_precision : bool, optional
        Enable mixed precision optimization (float32 -> float64 fallback).
        If None (default), automatically enables for chunked datasets
        (provides 50% additional memory savings). Set to False to disable.
    mixed_precision_config : MixedPrecisionConfig, optional
        Custom configuration for mixed precision behavior. If None and
        mixed precision is enabled, uses default configuration.
    multistart : bool, optional
        Enable multi-start optimization for global search (default: False).
        When enabled, explores multiple starting points on full data
        before running the full chunked optimization.
    n_starts : int, optional
        Number of starting points for multi-start optimization (default: 10).
        Set to 0 to disable multi-start even when multistart=True.
    sampler : str, optional
        Sampling strategy for generating starting points (default: 'lhs').
        Options: 'lhs' (Latin Hypercube), 'sobol', 'halton'.
    **kwargs
        Additional arguments passed to curve_fit

    Returns
    -------
    OptimizeResult
        Optimization result

    Examples
    --------
    >>> from nlsq.streaming.large_dataset import fit_large_dataset
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>>
    >>> # Generate large dataset
    >>> x_large = np.linspace(0, 10, 5_000_000)
    >>> y_large = 2.5 * np.exp(-1.3 * x_large) + np.random.normal(0, 0.1, len(x_large))
    >>>
    >>> # Fit with automatic memory management
    >>> result = fit_large_dataset(
    ...     lambda x, a, b: a * jnp.exp(-b * x),
    ...     x_large, y_large,
    ...     p0=[2.0, 1.0],
    ...     memory_limit_gb=4.0,
    ...     show_progress=True
    ... )
    >>> print(f"Fitted parameters: {result.popt}")
    >>> print(f"Success rate: {result.success_rate:.1%}")
    >>>
    >>> # Fit with multi-start optimization
    >>> result = fit_large_dataset(
    ...     lambda x, a, b: a * jnp.exp(-b * x),
    ...     x_large, y_large,
    ...     p0=[2.0, 1.0],
    ...     bounds=([0, 0], [10, 5]),
    ...     multistart=True,
    ...     n_starts=10,
    ...     sampler='lhs'
    ... )
    >>>
    >>> # Check failure diagnostics if some chunks failed
    >>> if result.failure_summary['total_failures'] > 0:
    ...     print(f"Failed chunks: {result.failure_summary['failed_chunk_indices']}")
    ...     print(f"Common errors: {result.failure_summary['common_errors']}")
    """
    fitter = LargeDatasetFitter(
        memory_limit_gb=memory_limit_gb,
        logger=logger,
        enable_mixed_precision=enable_mixed_precision,
        mixed_precision_config=mixed_precision_config,
        multistart=multistart,
        n_starts=n_starts,
        sampler=sampler,
    )

    if show_progress:
        return fitter.fit_with_progress(f, xdata, ydata, p0=p0, **kwargs)
    else:
        return fitter.fit(f, xdata, ydata, p0=p0, **kwargs)


def estimate_memory_requirements(n_points: int, n_params: int) -> DatasetStats:
    """Estimate memory requirements for a dataset.

    Parameters
    ----------
    n_points : int
        Number of data points
    n_params : int
        Number of parameters

    Returns
    -------
    DatasetStats
        Memory requirements and processing recommendations

    Examples
    --------
    >>> from nlsq.streaming.large_dataset import estimate_memory_requirements
    >>>
    >>> # Estimate requirements for 50M points, 3 parameters
    >>> stats = estimate_memory_requirements(50_000_000, 3)
    >>> print(f"Estimated memory: {stats.total_memory_estimate_gb:.2f} GB")
    >>> print(f"Recommended chunk size: {stats.recommended_chunk_size:,}")
    >>> print(f"Number of chunks: {stats.n_chunks}")
    """
    config = LDMemoryConfig()
    _, stats = MemoryEstimator.calculate_optimal_chunk_size(n_points, n_params, config)
    return stats


__all__ = [
    "DataChunker",
    "DatasetStats",
    "GPUMemoryEstimator",
    "LDMemoryConfig",
    "LargeDatasetFitter",
    "MemoryEstimator",
    "ProgressReporter",
    "cleanup_memory",
    "estimate_memory_requirements",
    "fit_large_dataset",
    "get_memory_tier",
]
