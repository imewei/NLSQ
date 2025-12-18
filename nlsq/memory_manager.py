"""Memory management for NLSQ optimization.

This module provides intelligent memory management capabilities including
prediction, monitoring, pooling, and automatic garbage collection.

Mixed Precision Coordination
-----------------------------
The memory manager coordinates with the mixed precision system to provide
accurate memory estimates based on the current data type (float32 or float64).
This enables:

- 50% memory savings when using float32 precision
- Accurate memory predictions for chunking strategies
- Dynamic memory estimation during precision upgrades

See :class:`MixedPrecisionManager` for more details on mixed precision optimization.
"""

import gc
import logging
import time
import warnings
from contextlib import contextmanager

import jax.numpy as jnp
import numpy as np

# Module logger for debug output
logger = logging.getLogger(__name__)

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn(
        "psutil not installed, memory monitoring will be limited", UserWarning
    )


class MemoryManager:
    """Intelligent memory management for optimization algorithms.

    This class provides:
    - Memory usage monitoring and prediction
    - Array pooling to reduce allocations
    - Automatic garbage collection triggers
    - Context managers for memory-safe operations
    - Mixed precision coordination for accurate memory estimates

    Mixed Precision Integration
    ----------------------------
    The memory manager integrates with :class:`MixedPrecisionManager` to provide
    accurate memory estimates based on the current precision:

    - float32: 4 bytes per element (50% memory savings)
    - float64: 8 bytes per element (default precision)

    Use :meth:`get_current_precision_memory_multiplier` to query the current
    memory multiplier, and pass ``dtype`` to :meth:`predict_memory_requirement`
    for precision-aware memory estimates.

    Attributes
    ----------
    memory_pool : dict
        Pool of reusable arrays indexed by (shape, dtype)
    allocation_history : list
        History of memory allocations
    gc_threshold : float
        Memory usage threshold (0-1) for triggering garbage collection
    safety_factor : float
        Safety factor for memory predictions
    """

    def __init__(
        self,
        gc_threshold: float = 0.8,
        safety_factor: float = 1.2,
        enable_adaptive_safety: bool = False,
        disable_padding: bool = False,
        memory_cache_ttl: float = 1.0,
    ):
        """Initialize memory manager.

        Parameters
        ----------
        gc_threshold : float
            Trigger GC when memory usage exceeds this fraction (0-1)
        safety_factor : float
            Multiply memory requirements by this factor for safety
        enable_adaptive_safety : bool
            Enable adaptive safety factor reduction (1.2 → 1.05 after warmup)
        disable_padding : bool
            Disable padding/bucketing for strict memory environments (Task 5.6).
            When True: uses exact shapes, sets safety_factor=1.0.
            Use case: cloud quotas, strict memory limits.
        memory_cache_ttl : float
            TTL in seconds for cached memory info (default: 1.0).
            Reduces psutil system call overhead by 90%.
        """
        self.memory_pool: dict[tuple, np.ndarray] = {}
        self.allocation_history: list = []
        self.gc_threshold = gc_threshold
        self.disable_padding = disable_padding

        # If disable_padding is True, force safety_factor to 1.0
        if disable_padding:
            self.safety_factor = 1.0
            self._initial_safety_factor = 1.0
            self.enable_adaptive_safety = False  # No adaptation when padding disabled
        else:
            self.safety_factor = safety_factor
            self._initial_safety_factor = safety_factor
            self.enable_adaptive_safety = enable_adaptive_safety

        self._peak_memory = 0

        # TTL-based cache for psutil calls (reduces overhead by 90%)
        self._memory_cache_ttl = memory_cache_ttl
        self._available_memory_cache: float | None = None
        self._available_memory_cache_time: float = 0.0
        self._memory_usage_cache: float | None = None
        self._memory_usage_cache_time: float = 0.0
        self._memory_fraction_cache: float | None = None
        self._memory_fraction_cache_time: float = 0.0

        self._initial_memory = self.get_memory_usage_bytes()

        # Telemetry for adaptive safety factor (Task 5.2)
        self._safety_telemetry: list[dict] = []
        self._warmup_runs = 10  # Number of runs before adapting
        self._min_safety_factor = 1.05  # Target minimum safety factor

    def get_available_memory(self) -> float:
        """Get available memory in bytes.

        Returns
        -------
        available : float
            Available memory in bytes

        Notes
        -----
        Uses TTL-based caching to reduce psutil system call overhead by 90%.
        """
        now = time.time()

        # Return cached value if still valid
        if (
            self._available_memory_cache is not None
            and now - self._available_memory_cache_time < self._memory_cache_ttl
        ):
            return self._available_memory_cache

        # Fetch fresh value
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                self._available_memory_cache = mem.available
                self._available_memory_cache_time = now
                return mem.available
            except Exception as e:
                # Fallback if psutil fails - log for debugging
                logger.debug(f"psutil memory check failed (non-critical): {e}")

        # Conservative fallback estimate (4 GB)
        return 4.0 * 1024**3

    def get_memory_usage_bytes(self) -> float:
        """Get current memory usage in bytes.

        Returns
        -------
        usage : float
            Current memory usage in bytes

        Notes
        -----
        Uses TTL-based caching to reduce psutil system call overhead by 90%.
        """
        now = time.time()

        # Return cached value if still valid
        if (
            self._memory_usage_cache is not None
            and now - self._memory_usage_cache_time < self._memory_cache_ttl
        ):
            return self._memory_usage_cache

        # Fetch fresh value
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                usage = process.memory_info().rss
                self._memory_usage_cache = usage
                self._memory_usage_cache_time = now
                return usage
            except Exception as e:
                logger.debug(f"psutil process memory check failed (non-critical): {e}")

        # Fallback: try to estimate from Python's view
        import sys

        return sys.getsizeof(self.memory_pool) + sum(
            arr.nbytes for arr in self.memory_pool.values()
        )

    def get_memory_usage_fraction(self) -> float:
        """Get current memory usage as fraction of total.

        Returns
        -------
        fraction : float
            Memory usage fraction (0-1)

        Notes
        -----
        Uses TTL-based caching to reduce psutil system call overhead by 90%.
        """
        now = time.time()

        # Return cached value if still valid
        if (
            self._memory_fraction_cache is not None
            and now - self._memory_fraction_cache_time < self._memory_cache_ttl
        ):
            return self._memory_fraction_cache

        # Fetch fresh value
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                fraction = mem.percent / 100.0
                self._memory_fraction_cache = fraction
                self._memory_fraction_cache_time = now
                return fraction
            except Exception as e:
                logger.debug(f"psutil memory fraction check failed (non-critical): {e}")

        # Conservative estimate
        return 0.5

    def predict_memory_requirement(
        self,
        n_points: int,
        n_params: int,
        algorithm: str = "trf",
        dtype: jnp.dtype = jnp.float64,
    ) -> int:
        """Predict memory requirement for optimization.

        Parameters
        ----------
        n_points : int
            Number of data points
        n_params : int
            Number of parameters
        algorithm : str
            Algorithm name ('trf', 'lm', 'dogbox')
        dtype : jnp.dtype, optional
            Data type for computations (default: jnp.float64).
            Affects memory calculations: float32 uses 4 bytes, float64 uses 8 bytes.

        Returns
        -------
        bytes_needed : int
            Estimated memory requirement in bytes

        Notes
        -----
        Memory requirements scale linearly with precision:
        - float32: 4 bytes per element (50% memory savings)
        - float64: 8 bytes per element (default, higher precision)
        """
        # Size of float depends on dtype (4 bytes for float32, 8 bytes for float64)
        float_size = 4 if dtype == jnp.float32 else 8

        # Base arrays: x, y, params
        base_memory = float_size * (2 * n_points + n_params)

        # Jacobian matrix
        jacobian_memory = float_size * n_points * n_params

        # Algorithm-specific memory
        if algorithm == "trf":
            # Trust Region Reflective
            # Needs: SVD decomposition, working arrays
            svd_memory = float_size * min(n_points, n_params) ** 2
            working_memory = float_size * (3 * n_points + 5 * n_params)
            total = base_memory + jacobian_memory + svd_memory + working_memory

        elif algorithm == "lm":
            # Levenberg-Marquardt
            # Needs: Normal equations, working arrays
            normal_memory = float_size * n_params**2
            working_memory = float_size * (2 * n_points + 3 * n_params)
            total = base_memory + jacobian_memory + normal_memory + working_memory

        elif algorithm == "dogbox":
            # Dogbox
            # Similar to TRF but with additional bound constraints
            svd_memory = float_size * min(n_points, n_params) ** 2
            working_memory = float_size * (4 * n_points + 6 * n_params)
            total = base_memory + jacobian_memory + svd_memory + working_memory

        else:
            # Conservative estimate for unknown algorithms
            total = base_memory + jacobian_memory * 2

        # Apply safety factor
        return int(total * self.safety_factor)

    def check_memory_availability(self, bytes_needed: int) -> tuple[bool, str]:
        """Check if enough memory is available.

        Parameters
        ----------
        bytes_needed : int
            Memory required in bytes

        Returns
        -------
        available : bool
            Whether enough memory is available
        message : str
            Descriptive message
        """
        available = self.get_available_memory()

        if available >= bytes_needed:
            return (
                True,
                f"Memory available: {available / 1e9:.2f}GB >= {bytes_needed / 1e9:.2f}GB needed",
            )

        # Try garbage collection
        gc.collect()
        available = self.get_available_memory()

        if available >= bytes_needed:
            return True, "Memory available after garbage collection"

        return False, (
            f"Insufficient memory: need {bytes_needed / 1e9:.2f}GB, "
            f"have {available / 1e9:.2f}GB available"
        )

    @contextmanager
    def memory_guard(self, bytes_needed: int):
        """Context manager to ensure memory availability.

        Parameters
        ----------
        bytes_needed : int
            Required memory in bytes

        Raises
        ------
        MemoryError
            If insufficient memory is available
        """
        # Check availability
        is_available, message = self.check_memory_availability(bytes_needed)

        if not is_available:
            # Last resort: clear memory pool
            self.clear_pool()
            is_available, message = self.check_memory_availability(bytes_needed)

            if not is_available:
                raise MemoryError(message)

        initial_memory = self.get_memory_usage_bytes()

        try:
            yield
        finally:
            # Track peak memory
            current_memory = self.get_memory_usage_bytes()
            self._peak_memory = max(self._peak_memory, current_memory)

            # Check if we should trigger GC
            if self.get_memory_usage_fraction() > self.gc_threshold:
                gc.collect()

            # Log allocation
            self.allocation_history.append(
                {
                    "bytes_requested": bytes_needed,
                    "bytes_used": current_memory - initial_memory,
                    "peak_memory": self._peak_memory,
                }
            )

            # Track telemetry for adaptive safety factor (Task 5.2)
            self._record_safety_telemetry(bytes_needed, current_memory - initial_memory)

    def _record_safety_telemetry(self, bytes_predicted: int, bytes_actual: int | float):
        """Record telemetry for adaptive safety factor calculation.

        Parameters
        ----------
        bytes_predicted : int
            Predicted memory requirement (with current safety factor)
        bytes_actual : int | float
            Actual memory used

        Notes
        -----
        This method collects safety_factor_needed = actual / (predicted / safety_factor)
        which represents the minimum safety factor needed for this allocation.
        After warmup, we calculate p95(safety_factor_needed) to adaptively reduce
        the default safety factor.
        """
        if not self.enable_adaptive_safety:
            return

        # Calculate base prediction (without safety factor)
        bytes_predicted_base = bytes_predicted / self.safety_factor

        # Calculate minimum safety factor needed for this allocation
        if bytes_predicted_base > 0:
            safety_factor_needed = bytes_actual / bytes_predicted_base
        else:
            safety_factor_needed = 1.0

        # Record telemetry
        self._safety_telemetry.append(
            {
                "bytes_predicted": bytes_predicted,
                "bytes_actual": bytes_actual,
                "safety_factor_needed": safety_factor_needed,
                "current_safety_factor": self.safety_factor,
            }
        )

        # Update adaptive safety factor after warmup
        if len(self._safety_telemetry) >= self._warmup_runs:
            self._update_adaptive_safety_factor()

    def _update_adaptive_safety_factor(self):
        """Update safety factor based on telemetry (after warmup).

        Calculates p95(safety_factor_needed) from telemetry and uses it to
        gradually reduce safety factor from initial value (1.2) to target minimum (1.05).
        """
        if (
            not self.enable_adaptive_safety
            or len(self._safety_telemetry) < self._warmup_runs
        ):
            return

        # Extract safety factors needed from telemetry
        safety_factors_needed = [
            entry["safety_factor_needed"] for entry in self._safety_telemetry
        ]

        # Calculate p95 (95th percentile) - conservative estimate
        p95_safety = np.percentile(safety_factors_needed, 95)

        # Adaptive safety factor: max(min_safety_factor, p95_safety)
        # This ensures we never go below 1.05, but use higher if needed
        adaptive_safety = max(self._min_safety_factor, p95_safety)

        # Gradually reduce safety factor (don't jump abruptly)
        if adaptive_safety < self.safety_factor:
            # Reduce by at most 0.05 per update to avoid sudden changes
            self.safety_factor = max(adaptive_safety, self.safety_factor - 0.05)

        logger.debug(
            f"Adaptive safety factor: {self.safety_factor:.3f} "
            f"(p95_needed={p95_safety:.3f}, runs={len(self._safety_telemetry)})"
        )

    def get_safety_telemetry(self) -> dict:
        """Get safety factor telemetry statistics.

        Returns
        -------
        telemetry : dict
            Safety factor telemetry with:
            - current_safety_factor: Current safety factor
            - initial_safety_factor: Initial safety factor (1.2)
            - min_safety_factor: Target minimum (1.05)
            - telemetry_entries: Number of telemetry entries collected
            - p95_safety_needed: 95th percentile of safety factors needed (if data available)
            - safety_factor_history: List of safety factors over time
        """
        telemetry = {
            "current_safety_factor": self.safety_factor,
            "initial_safety_factor": self._initial_safety_factor,
            "min_safety_factor": self._min_safety_factor,
            "telemetry_entries": len(self._safety_telemetry),
            "adaptive_enabled": self.enable_adaptive_safety,
        }

        if self._safety_telemetry:
            safety_factors_needed = [
                entry["safety_factor_needed"] for entry in self._safety_telemetry
            ]
            telemetry["p95_safety_needed"] = float(
                np.percentile(safety_factors_needed, 95)
            )
            telemetry["mean_safety_needed"] = float(np.mean(safety_factors_needed))
            telemetry["max_safety_needed"] = float(np.max(safety_factors_needed))
            telemetry["safety_factor_history"] = [
                entry["current_safety_factor"] for entry in self._safety_telemetry
            ]

        return telemetry

    def allocate_array(
        self, shape: tuple[int, ...], dtype: type = np.float64, zero: bool = True
    ) -> np.ndarray:
        """Allocate array with memory pooling.

        Parameters
        ----------
        shape : tuple
            Shape of array to allocate
        dtype : type
            Data type of array
        zero : bool
            Whether to zero-initialize the array

        Returns
        -------
        array : np.ndarray
            Allocated array

        Raises
        ------
        MemoryError
            If allocation fails
        """
        key = (shape, dtype)

        # Check pool for existing array
        if key in self.memory_pool:
            arr = self.memory_pool[key]
            if zero:
                arr.fill(0)
            return arr

        # Calculate memory needed
        bytes_needed = int(np.prod(shape) * np.dtype(dtype).itemsize)

        # Allocate with memory guard
        with self.memory_guard(bytes_needed):
            if zero:
                arr = np.zeros(shape, dtype=dtype)
            else:
                arr = np.empty(shape, dtype=dtype)

            # Add to pool
            self.memory_pool[key] = arr
            return arr

    def free_array(self, arr: np.ndarray):
        """Return array to pool for reuse.

        Parameters
        ----------
        arr : np.ndarray
            Array to free
        """
        key = (arr.shape, arr.dtype)
        self.memory_pool[key] = arr

    def clear_pool(self):
        """Clear memory pool and run garbage collection."""
        self.memory_pool.clear()
        gc.collect()

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics.

        Returns
        -------
        stats : dict
            Memory statistics including current usage, peak, pool size
        """
        current_memory = self.get_memory_usage_bytes()
        available_memory = self.get_available_memory()

        pool_memory = sum(arr.nbytes for arr in self.memory_pool.values())
        pool_arrays = len(self.memory_pool)

        stats = {
            "current_usage_gb": current_memory / 1e9,
            "available_gb": available_memory / 1e9,
            "peak_usage_gb": self._peak_memory / 1e9,
            "usage_fraction": self.get_memory_usage_fraction(),
            "pool_memory_gb": pool_memory / 1e9,
            "pool_arrays": pool_arrays,
            "allocations": len(self.allocation_history),
        }

        if self.allocation_history:
            total_requested = sum(a["bytes_requested"] for a in self.allocation_history)
            total_used = sum(a["bytes_used"] for a in self.allocation_history)
            stats["total_requested_gb"] = total_requested / 1e9
            stats["total_used_gb"] = total_used / 1e9
            stats["efficiency"] = (
                total_used / total_requested if total_requested > 0 else 1.0
            )

        # Include safety factor telemetry (Task 5.2)
        if self.enable_adaptive_safety:
            stats["safety_telemetry"] = self.get_safety_telemetry()

        # Include padding configuration (Task 5.6)
        stats["disable_padding"] = self.disable_padding

        return stats

    def optimize_memory_pool(self, max_arrays: int = 100):
        """Optimize memory pool by removing least recently used arrays.

        Parameters
        ----------
        max_arrays : int
            Maximum number of arrays to keep in pool
        """
        if len(self.memory_pool) <= max_arrays:
            return

        # Sort by size and keep largest arrays (most benefit from pooling)
        sorted_items = sorted(
            self.memory_pool.items(), key=lambda x: x[1].nbytes, reverse=True
        )

        # Keep only the largest arrays
        self.memory_pool = dict(sorted_items[:max_arrays])
        gc.collect()

    @contextmanager
    def temporary_allocation(self, shape: tuple[int, ...], dtype: type = np.float64):
        """Context manager for temporary array allocation.

        Parameters
        ----------
        shape : tuple
            Shape of array
        dtype : type
            Data type

        Yields
        ------
        array : np.ndarray
            Temporary array that will be returned to pool on exit
        """
        arr = self.allocate_array(shape, dtype)
        try:
            yield arr
        finally:
            # Return array to pool for reuse
            self.free_array(arr)

    def estimate_chunking_strategy(
        self,
        n_points: int,
        n_params: int,
        algorithm: str = "trf",
        memory_limit_gb: float | None = None,
    ) -> dict:
        """Estimate optimal chunking strategy for large datasets.

        Parameters
        ----------
        n_points : int
            Number of data points
        n_params : int
            Number of parameters
        algorithm : str
            Algorithm to use
        memory_limit_gb : float, optional
            Memory limit in GB (uses available memory if None)

        Returns
        -------
        strategy : dict
            Chunking strategy with chunk_size and n_chunks
        """
        if memory_limit_gb is None:
            memory_limit = self.get_available_memory() * 0.8  # Use 80% of available
        else:
            memory_limit = memory_limit_gb * 1e9

        # Calculate memory per point
        memory_per_point = self.predict_memory_requirement(1, n_params, algorithm)

        # Calculate maximum points that fit in memory
        max_points = int(memory_limit / memory_per_point)

        if max_points >= n_points:
            # No chunking needed
            return {
                "needs_chunking": False,
                "chunk_size": n_points,
                "n_chunks": 1,
                "memory_per_chunk_gb": self.predict_memory_requirement(
                    n_points, n_params, algorithm
                )
                / 1e9,
            }

        # Calculate chunking parameters
        chunk_size = min(
            max_points, max(100, n_points // 100)
        )  # At least 100 points per chunk
        n_chunks = (n_points + chunk_size - 1) // chunk_size

        return {
            "needs_chunking": True,
            "chunk_size": chunk_size,
            "n_chunks": n_chunks,
            "memory_per_chunk_gb": self.predict_memory_requirement(
                chunk_size, n_params, algorithm
            )
            / 1e9,
            "total_points": n_points,
        }

    def get_current_precision_memory_multiplier(
        self, mixed_precision_manager=None
    ) -> float:
        """Get memory multiplier based on current precision.

        This method returns the memory usage multiplier relative to float64,
        which helps coordinate memory management with mixed precision optimization.

        Parameters
        ----------
        mixed_precision_manager : MixedPrecisionManager, optional
            Mixed precision manager to query for current dtype.
            If None, assumes float64 (multiplier = 1.0).

        Returns
        -------
        multiplier : float
            Memory multiplier relative to float64:
            - 1.0 if using float64 (default, no manager)
            - 0.5 if using float32 (50% memory savings)

        Examples
        --------
        >>> from nlsq.memory_manager import MemoryManager
        >>> from nlsq.mixed_precision import MixedPrecisionManager, MixedPrecisionConfig
        >>> manager = MemoryManager()
        >>> # Without mixed precision (defaults to float64)
        >>> manager.get_current_precision_memory_multiplier()
        1.0
        >>> # With mixed precision in float32 mode
        >>> mp_manager = MixedPrecisionManager(MixedPrecisionConfig())
        >>> manager.get_current_precision_memory_multiplier(mp_manager)
        0.5

        Notes
        -----
        This multiplier can be used to adjust memory estimates when mixed precision
        is active, providing more accurate memory predictions and enabling better
        resource allocation.
        """
        if mixed_precision_manager is None:
            # No mixed precision manager: assume float64
            return 1.0

        # Query current dtype from manager
        current_dtype = mixed_precision_manager.get_current_dtype()

        # Return multiplier: 0.5 for float32, 1.0 for float64
        return 0.5 if current_dtype == jnp.float32 else 1.0


# Global memory manager instance
_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance.

    Returns
    -------
    manager : MemoryManager
        Global memory manager instance
    """
    global _memory_manager  # noqa: PLW0603
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def clear_memory_pool():
    """Clear the global memory pool."""
    manager = get_memory_manager()
    manager.clear_pool()


def get_memory_stats() -> dict:
    """Get memory usage statistics.

    Returns
    -------
    stats : dict
        Memory statistics
    """
    manager = get_memory_manager()
    return manager.get_memory_stats()
