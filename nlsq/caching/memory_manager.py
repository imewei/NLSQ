"""Memory management for NLSQ optimization.

This module provides intelligent memory management capabilities including
prediction, monitoring, pooling, and automatic garbage collection.

Phase 3 Optimizations (Task Group 9):

- Telemetry Circular Buffer (1.3a): Uses deque(maxlen=1000) for _safety_telemetry
  to prevent memory leak in multi-day optimization runs
"""

import gc
import logging
import threading
import time
import warnings
from collections import OrderedDict, deque
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
        "psutil not installed, memory monitoring will be limited",
        UserWarning,
        stacklevel=2,
    )


class MemoryManager:
    """Intelligent memory management for optimization algorithms.

    This class provides:

    - Memory usage monitoring and prediction
    - Opt-in array pooling (``allocate_array`` / ``free_array``); nothing in
      the library's fit paths uses the pool
    - Automatic garbage collection triggers
    - Context managers for memory-safe operations

    Units: memory sizes reported or accepted in "GB" (``memory_limit_gb``,
    ``*_gb`` stats) are decimal gigabytes (1e9 bytes). The psutil-failure
    fallback of 16 GB available is 16 GiB (``16 * 1024**3`` bytes).

    LRU Memory Pool (Task Group 7 - 1.2a)
    -------------------------------------
    The memory pool uses an OrderedDict to track access order. The pool has
    no automatic capacity limit: ``free_array`` never evicts, and LRU
    eviction happens only when ``optimize_memory_pool(max_arrays)`` is called
    explicitly. ``clear_pool()`` empties it; ``UnifiedCache.clear_cache()``
    does not touch it.

    Telemetry Circular Buffer (Task Group 9 - 1.3a)
    ------------------------------------------------
    The safety telemetry uses a deque with maxlen=1000 to prevent memory leak
    in multi-day optimization runs. This maintains the last 1000 telemetry
    records for adaptive safety factor calculation.

    Attributes
    ----------
    memory_pool : OrderedDict
        Pool of reusable arrays indexed by (shape, dtype) with LRU tracking
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
        adaptive_ttl: bool = True,
    ):
        """Initialize memory manager.

        Parameters
        ----------
        gc_threshold : float
            Trigger GC when memory usage exceeds this fraction (0-1)
        safety_factor : float
            Multiply memory requirements by this factor for safety
        enable_adaptive_safety : bool
            Enable adaptive safety factor reduction (1.2 -> 1.05 after warmup)
        disable_padding : bool
            Disable padding/bucketing for strict memory environments (Task 5.6).
            When True: uses exact shapes, sets safety_factor=1.0.
            Use case: cloud quotas, strict memory limits.
        memory_cache_ttl : float
            TTL in seconds for cached memory info (default: 1.0).
            Reduces psutil system call overhead by 90%.
        adaptive_ttl : bool
            Enable adaptive TTL based on call frequency (default: True).
            High-frequency callers (>100 calls/sec) get 15s effective TTL.
            Medium-frequency callers (>10 calls/sec) get 10s effective TTL.
            Low-frequency callers use the default TTL.
            Reduces psutil overhead in streaming optimization by 15-20%.
        """
        # Task Group 7 (1.2a): Use OrderedDict for LRU memory pool
        # This enables move_to_end() for recently used arrays and
        # popitem(last=False) for LRU eviction when at capacity.
        self.memory_pool: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._pool_lock = threading.Lock()
        self.allocation_history: deque[dict] = deque(maxlen=1000)
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

        self._peak_memory: float = 0.0

        # TTL-based cache for psutil calls (reduces overhead by 90%)
        self._memory_cache_ttl = memory_cache_ttl
        self._available_memory_cache: float | None = None
        self._available_memory_cache_time: float = 0.0
        self._memory_usage_cache: float | None = None
        self._memory_usage_cache_time: float = 0.0
        self._memory_fraction_cache: float | None = None
        self._memory_fraction_cache_time: float = 0.0
        self._fraction_fallback_warned = False

        # Adaptive TTL feature (Task 3 - 1.1a)
        # Tracks timestamps of last 100 calls to compute call frequency
        self._adaptive_ttl = adaptive_ttl
        self._call_frequency_tracker: deque[float] = deque(maxlen=100)

        # Task 9.4 (1.3a): Telemetry Circular Buffer
        # Use deque with maxlen=1000 to prevent memory leak in multi-day runs
        # Maintains last 1000 telemetry records for adaptive safety factor calculation
        self._safety_telemetry: deque[dict] = deque(maxlen=1000)
        self._warmup_runs = 10  # Number of runs before adapting
        self._min_safety_factor = 1.05  # Target minimum safety factor
        # Upper bound for upward adaptation; keeps noisy RSS deltas from
        # inflating the factor without limit.
        self._max_safety_factor = max(3.0, self._initial_safety_factor)
        self._safety_saturated = False
        self._saturation_warned = False

    def _get_effective_ttl(self) -> float:
        """Calculate effective TTL based on call frequency.

        Returns
        -------
        effective_ttl : float
            The effective TTL in seconds based on call frequency:
            - 15.0s for high-frequency callers (>100 calls/sec)
            - 10.0s for medium-frequency callers (>10 calls/sec)
            - default TTL for low-frequency callers

        Notes
        -----
        This method is only used when adaptive_ttl is enabled.
        The call frequency is computed from the time span of the last 100 calls.
        """
        if not self._adaptive_ttl:
            return self._memory_cache_ttl

        # Need at least 2 calls to compute frequency
        if len(self._call_frequency_tracker) < 2:
            return self._memory_cache_ttl

        # Compute time span of tracked calls
        oldest_call = self._call_frequency_tracker[0]
        newest_call = self._call_frequency_tracker[-1]
        time_span = newest_call - oldest_call

        if time_span <= 0:
            # All calls happened at the same time, assume very high frequency
            return 15.0

        # Compute calls per second
        num_calls = len(self._call_frequency_tracker)
        calls_per_sec = num_calls / time_span

        # Determine effective TTL based on frequency thresholds
        # Memory availability changes slowly, so aggressive caching is safe
        if calls_per_sec > 100:
            # High frequency: use 15s TTL (memory is stable)
            return 15.0
        if calls_per_sec > 10:
            # Medium frequency: use 10s TTL
            return 10.0
        # Low frequency: use default TTL
        return self._memory_cache_ttl

    def get_available_memory(self, refresh: bool = False) -> float:
        """Get available memory in bytes.

        Parameters
        ----------
        refresh : bool
            Bypass the TTL cache and query psutil. Used for admission
            decisions, where a value up to ``adaptive_ttl`` seconds old
            (15 s) could admit an allocation that no longer fits.

        Returns
        -------
        available : float
            Available memory in bytes

        Notes
        -----
        Uses TTL-based caching to reduce psutil system call overhead by 90%.
        When adaptive_ttl is enabled, the effective TTL is adjusted based on
        call frequency to further reduce overhead for streaming optimization.
        """
        now = time.monotonic()

        # Track call timestamp for adaptive TTL
        if self._adaptive_ttl:
            self._call_frequency_tracker.append(now)

        # Calculate effective TTL (adaptive or default)
        effective_ttl = self._get_effective_ttl()

        # Return cached value if still valid
        if (
            not refresh
            and self._available_memory_cache is not None
            and now - self._available_memory_cache_time < effective_ttl
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
                # Fallback if psutil fails - warn (not debug-log): a fabricated
                # estimate that's wrong in the optimistic direction can let a
                # memory-constrained caller (e.g. a small container) admit an
                # allocation that OOMs the process, so this must be visible.
                logger.warning(
                    f"psutil memory check failed, using fallback estimate "
                    f"of 16 GB available (non-critical, but downstream memory "
                    f"budget decisions will be based on this guess): {e}",
                )

        # Conservative fallback estimate. Matches the 16 GB fallback used by
        # MemoryEstimator.get_available_memory_gb() (nlsq/streaming/large_dataset.py)
        # for consistency between NLSQ's two memory-estimation code paths.
        return 16.0 * 1024**3

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
        now = time.monotonic()

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
                logger.warning(
                    f"psutil process memory check failed, falling back to a "
                    f"pool-only estimate that undercounts true process memory "
                    f"(non-critical, but usage-fraction decisions will be too "
                    f"optimistic): {e}",
                )

        # Fallback: try to estimate from Python's view. NOTE: this only
        # counts the memory pool's own arrays, not actual process RSS
        # (interpreter, JAX/XLA buffers, etc.), so it systematically
        # undercounts true usage when psutil is unavailable.
        import sys

        with self._pool_lock:
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
        now = time.monotonic()

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
                if not self._fraction_fallback_warned:
                    self._fraction_fallback_warned = True
                    logger.warning(
                        f"psutil memory fraction check failed, using a constant "
                        f"0.5 usage fraction (the memory_guard GC trigger is "
                        f"effectively disabled; non-critical): {e}",
                    )

        # Conservative estimate
        return 0.5

    def predict_memory_requirement(
        self,
        n_points: int,
        n_params: int,
        algorithm: str = "trf",
        dtype: jnp.dtype = jnp.float64,
        bounded: bool = False,
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
            Element size is ``np.dtype(dtype).itemsize`` (float32: 4, float64: 8).
        bounded : bool, optional
            Estimate the bounded trust-region layout (default False). Callers
            with finite bounds must pass True; ``'dogbox'`` is always bounded.

        Returns
        -------
        bytes_needed : int
            Estimated memory requirement in bytes

        Notes
        -----
        Memory requirements scale linearly with precision:
        - float32: 4 bytes per element (50% memory savings)
        - float64: 8 bytes per element (default, higher precision)

        ``'lm'`` is an alias that runs the same trust-region solver as
        ``'trf'`` and shares its estimate. The unbounded TRF step
        (``_svd_no_bounds_jit``) holds ``J``, ``J * d`` and the thin-SVD ``U``,
        three m-by-n arrays. The bounded step (``_svd_bounds_jit``, used by
        ``'dogbox'`` here and by ``'trf'`` with bounds in practice) also holds
        ``J_augmented`` and a ``U`` of shape (m + n) x n. ``'trf'`` and
        ``'lm'`` use the bounded layout only when ``bounded=True``.

        The estimate still sits about 1.3x below measured peak RSS (CPU,
        2e6 x 10). The cause is not verified: SVD workspace inside LAPACK,
        model intermediates and autodiff buffers are all candidates, and none
        is visible in ``trf_jit.py``.
        """
        float_size = np.dtype(dtype).itemsize

        # Base arrays: x, y, params
        base_memory = float_size * (2 * n_points + n_params)

        # Jacobian matrix
        jacobian_memory = float_size * n_points * n_params

        # Algorithm-specific memory
        k = min(n_points, n_params)
        if algorithm in ("trf", "lm", "dogbox"):
            if bounded or algorithm == "dogbox":
                # J_h (m x n), J_augmented ((m+n) x n), U ((m+n) x n), V (n x n).
                aug = n_points + n_params
                svd_memory = float_size * (
                    n_points * n_params + 2 * aug * n_params + n_params**2
                )
            else:
                # J_h = J * d (m x n), thin-SVD U (m x k) and V (n x k); J itself
                # is counted in jacobian_memory.
                svd_memory = float_size * (
                    n_points * n_params + n_points * k + n_params * k
                )
            if algorithm == "dogbox":
                working_memory = float_size * (4 * n_points + 6 * n_params)
            else:
                working_memory = float_size * (3 * n_points + 5 * n_params)
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
        # Admission decision: bypass the TTL cache (up to 15 s stale).
        available = self.get_available_memory(refresh=True)

        if available >= bytes_needed:
            return (
                True,
                f"Memory available: {available / 1e9:.2f}GB >= {bytes_needed / 1e9:.2f}GB needed",
            )

        # Try garbage collection
        gc.collect()
        available = self.get_available_memory(refresh=True)

        if available >= bytes_needed:
            return True, "Memory available after garbage collection"

        return False, (
            f"Insufficient memory: need {bytes_needed / 1e9:.2f}GB, "
            f"have {available / 1e9:.2f}GB available"
        )

    @contextmanager
    def memory_guard(self, bytes_needed: int, record_telemetry: bool = True):
        """Context manager to ensure memory availability.

        Parameters
        ----------
        bytes_needed : int
            Required memory in bytes, including the safety factor (as returned
            by ``predict_memory_requirement``)
        record_telemetry : bool
            Feed the measured RSS delta to the adaptive safety-factor telemetry.
            Pass False when ``bytes_needed`` is a raw size or the RSS delta is
            not meaningful (e.g. lazily-committed ``np.zeros`` pages).

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

        # The prediction that produced bytes_needed used the factor in force now.
        safety_at_entry = self.safety_factor

        # Delta measurement: bypass the TTL cache or both reads return the same RSS.
        self._memory_usage_cache = None
        initial_memory = self.get_memory_usage_bytes()

        try:
            yield
        finally:
            # Track peak memory
            self._memory_usage_cache = None
            current_memory = self.get_memory_usage_bytes()
            self._peak_memory = max(self._peak_memory, current_memory)

            # Check if we should trigger GC
            if self.get_memory_usage_fraction() > self.gc_threshold:
                gc.collect()

            # Log allocation
            self.allocation_history.append(
                {
                    "bytes_requested": bytes_needed,
                    "bytes_used": max(0, current_memory - initial_memory),
                    "peak_memory": self._peak_memory,
                },
            )

            # Track telemetry for adaptive safety factor (Task 5.2)
            # RSS can drop when memory is freed; clamp so a negative delta counts as "used nothing".
            if record_telemetry:
                self._record_safety_telemetry(
                    bytes_needed,
                    max(0, current_memory - initial_memory),
                    safety_at_entry,
                )

    def _record_safety_telemetry(
        self,
        bytes_predicted: int,
        bytes_actual: float,
        safety_factor_used: float | None = None,
    ):
        """Record telemetry for adaptive safety factor calculation.

        Parameters
        ----------
        bytes_predicted : int
            Predicted memory requirement (with current safety factor)
        bytes_actual : int | float
            Actual memory used
        safety_factor_used : float, optional
            Safety factor in force when ``bytes_predicted`` was computed
            (default: the current factor).

        Notes
        -----
        This method collects safety_factor_needed = actual / (predicted / safety_factor)
        which represents the minimum safety factor needed for this allocation.
        After warmup, we calculate p95(safety_factor_needed) to adaptively reduce
        the default safety factor.

        Task 9.4 (1.3a): Uses deque with maxlen=1000 to prevent unbounded growth.
        """
        if not self.enable_adaptive_safety:
            return

        # Calculate base prediction (without safety factor)
        factor_used = (
            self.safety_factor if safety_factor_used is None else safety_factor_used
        )
        bytes_predicted_base = bytes_predicted / factor_used

        # Calculate minimum safety factor needed for this allocation
        if bytes_predicted_base > 0:
            safety_factor_needed = bytes_actual / bytes_predicted_base
        else:
            safety_factor_needed = 1.0

        # Record telemetry (deque automatically evicts oldest if at maxlen)
        self._safety_telemetry.append(
            {
                "bytes_predicted": bytes_predicted,
                "bytes_actual": bytes_actual,
                "safety_factor_needed": safety_factor_needed,
                "current_safety_factor": self.safety_factor,
            },
        )

        # Update adaptive safety factor after warmup
        if len(self._safety_telemetry) >= self._warmup_runs:
            self._update_adaptive_safety_factor()

    def _update_adaptive_safety_factor(self):
        """Update safety factor based on telemetry (after warmup).

        Calculates p95(safety_factor_needed) from telemetry and uses it to
        gradually reduce safety factor from initial value (1.2) to target minimum
        (1.05). When the p95 exceeds the current factor (under-prediction), the
        factor is raised immediately, up to ``_max_safety_factor``; beyond that
        ``safety_saturated`` is set and a warning is logged once.

        RSS deltas that include first-call XLA compile memory can push the p95
        to the cap during warmup; the factor stays high until fewer than 5% of
        the retained records need more.
        """
        telemetry = self._safety_telemetry.copy()
        if not self.enable_adaptive_safety or len(telemetry) < self._warmup_runs:
            return

        # Extract safety factors needed from telemetry
        safety_factors_needed = [entry["safety_factor_needed"] for entry in telemetry]

        # Calculate p95 (95th percentile) - conservative estimate
        p95_safety = np.percentile(safety_factors_needed, 95)

        # Adaptive safety factor: max(min_safety_factor, p95_safety)
        # This ensures we never go below 1.05, but use higher if needed
        adaptive_safety = max(self._min_safety_factor, p95_safety)

        # Needed factor above the cap: predictions stay short of measured usage.
        self._safety_saturated = bool(adaptive_safety > self._max_safety_factor)
        if self._safety_saturated and not self._saturation_warned:
            self._saturation_warned = True
            logger.warning(
                f"Adaptive safety factor saturated at {self._max_safety_factor:.2f} "
                f"(observed p95 need {p95_safety:.2f}): memory predictions "
                f"under-cover the measured usage; treat admission checks as "
                f"optimistic."
            )

        if adaptive_safety > self.safety_factor:
            # Raising is the safe direction: no smoothing.
            self.safety_factor = min(adaptive_safety, self._max_safety_factor)
        elif adaptive_safety < self.safety_factor:
            # Gradually reduce safety factor (don't jump abruptly)
            # Reduce by at most 0.05 per update to avoid sudden changes
            self.safety_factor = max(adaptive_safety, self.safety_factor - 0.05)

        logger.debug(
            f"Adaptive safety factor: {self.safety_factor:.3f} "
            f"(p95_needed={p95_safety:.3f}, runs={len(telemetry)})",
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
        entries = self._safety_telemetry.copy()
        telemetry: dict[str, object] = {
            "current_safety_factor": self.safety_factor,
            "initial_safety_factor": self._initial_safety_factor,
            "min_safety_factor": self._min_safety_factor,
            "telemetry_entries": len(entries),
            "adaptive_enabled": self.enable_adaptive_safety,
            "safety_saturated": self._safety_saturated,
        }

        if entries:
            safety_factors_needed = [entry["safety_factor_needed"] for entry in entries]
            telemetry["p95_safety_needed"] = float(
                np.percentile(safety_factors_needed, 95),
            )
            telemetry["mean_safety_needed"] = float(np.mean(safety_factors_needed))
            telemetry["max_safety_needed"] = float(np.max(safety_factors_needed))
            telemetry["safety_factor_history"] = [
                entry["current_safety_factor"] for entry in entries
            ]

        return telemetry

    def allocate_array(
        self,
        shape: tuple[int, ...],
        dtype: type = np.float64,
        zero: bool = True,
    ) -> np.ndarray:
        """Allocate array with memory pooling and LRU tracking.

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

        Notes
        -----
        Task Group 7 (1.2a): Uses LRU tracking via OrderedDict.
        Arrays are removed from the pool while checked out. Return them with
        ``free_array`` to make them available for reuse and LRU eviction.
        """
        # Normalize so keys match free_array's (arr.shape, arr.dtype)
        shape_t = (shape,) if isinstance(shape, int | np.integer) else tuple(shape)
        key = (shape_t, np.dtype(dtype))

        # Check pool for existing array
        with self._pool_lock:
            arr = self.memory_pool.pop(key, None)
        if arr is not None:
            if zero:
                arr.fill(0)
            return arr

        # Calculate memory needed
        bytes_needed = int(np.prod(shape) * np.dtype(dtype).itemsize)

        # Raw size (no safety factor) and untouched calloc pages: the RSS delta
        # says nothing about the safety factor, so keep it out of telemetry.
        with self.memory_guard(bytes_needed, record_telemetry=False):
            if zero:
                arr = np.zeros(shape, dtype=dtype)
            else:
                arr = np.empty(shape, dtype=dtype)

            return arr

    def free_array(self, arr: np.ndarray):
        """Return array to pool for reuse.

        Parameters
        ----------
        arr : np.ndarray
            Array to free

        Notes
        -----
        Task Group 7 (1.2a): Uses LRU tracking via OrderedDict.
        The array becomes the pooled buffer for its (shape, dtype), replacing
        any earlier one, and is marked as most recently used.
        """
        # allocate_array hands pooled buffers out writable and C-contiguous;
        # a read-only or strided array would break that contract.
        if not isinstance(arr, np.ndarray) or not (
            arr.flags.writeable and arr.flags.c_contiguous
        ):
            return
        key = (arr.shape, arr.dtype)
        # Keep the most recently returned buffer for each shape/dtype.
        with self._pool_lock:
            self.memory_pool[key] = arr
            self.memory_pool.move_to_end(key)

    def clear_pool(self):
        """Clear memory pool and run garbage collection."""
        with self._pool_lock:
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
        # The peak is otherwise only updated inside memory_guard(), so it could
        # read below the current usage.
        self._peak_memory = max(self._peak_memory, current_memory)
        available_memory = self.get_available_memory()

        with self._pool_lock:
            pool_memory = sum(arr.nbytes for arr in self.memory_pool.values())
            pool_arrays = len(self.memory_pool)

        stats: dict[str, object] = {
            "current_usage_gb": current_memory / 1e9,
            "available_gb": available_memory / 1e9,
            "peak_usage_gb": self._peak_memory / 1e9,
            "usage_fraction": self.get_memory_usage_fraction(),
            "pool_memory_gb": pool_memory / 1e9,
            "pool_arrays": pool_arrays,
            "allocations": len(self.allocation_history),  # len() is atomic
        }

        history = list(self.allocation_history)  # snapshot: other threads append
        if history:
            total_requested = sum(a["bytes_requested"] for a in history)
            total_used = sum(a["bytes_used"] for a in history)
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
        """Optimize memory pool using LRU eviction.

        Parameters
        ----------
        max_arrays : int
            Maximum number of arrays to keep in pool

        Notes
        -----
        Task Group 7 (1.2a): Uses LRU eviction via popitem(last=False).
        Arrays are evicted in order of least recent use, keeping the
        most recently used arrays in the pool.
        """
        with self._pool_lock:
            if len(self.memory_pool) <= max_arrays:
                return

            while len(self.memory_pool) > max_arrays:
                self.memory_pool.popitem(last=False)

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
        bounded: bool = False,
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
            Memory limit in GB (uses available memory if None); must be
            finite and positive.
        bounded : bool, optional
            Plan for the bounded trust-region layout (see
            ``predict_memory_requirement``). Default False.

        Returns
        -------
        strategy : dict
            Chunking strategy with chunk_size and n_chunks

        Raises
        ------
        ValueError
            If ``n_points`` or ``n_params`` is negative, or ``memory_limit_gb``
            is not finite and positive.
        MemoryError
            If the whole problem does not fit and the memory budget cannot
            hold even one data point.
        """
        if n_points < 0 or n_params < 0:
            raise ValueError(
                f"n_points and n_params must be >= 0, got {n_points}, {n_params}"
            )
        if memory_limit_gb is None:
            # Fresh reading (not the TTL cache): the plan is checked against a fresh query later
            memory_limit = (
                self.get_available_memory(refresh=True) * 0.8
            )  # 80% of available
        else:
            if not (np.isfinite(memory_limit_gb) and memory_limit_gb > 0):
                raise ValueError(
                    f"memory_limit_gb must be finite and positive, got {memory_limit_gb}"
                )
            memory_limit = memory_limit_gb * 1e9

        def predict(m: int) -> int:
            return self.predict_memory_requirement(
                m, n_params, algorithm, bounded=bounded
            )

        required = predict(n_points)
        if n_points <= 0 or required <= memory_limit:
            # No chunking needed
            return {
                "needs_chunking": False,
                "chunk_size": n_points,
                "n_chunks": 1,
                "memory_per_chunk_gb": required / 1e9,
            }

        one_point = predict(1)
        if one_point > memory_limit:
            raise MemoryError(
                "Memory budget cannot hold even one data point "
                f"(requires approximately {one_point:g} bytes)."
            )

        # predict() is non-decreasing in n_points but not affine below
        # n_points == n_params, so bisect for the largest chunk that fits:
        # predict(lo) <= limit < predict(hi).
        lo, hi = 1, n_points
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if predict(mid) <= memory_limit:
                lo = mid
            else:
                hi = mid
        chunk_size = lo
        n_chunks = (n_points + chunk_size - 1) // chunk_size

        return {
            "needs_chunking": True,
            "chunk_size": chunk_size,
            "n_chunks": n_chunks,
            "memory_per_chunk_gb": predict(chunk_size) / 1e9,
            "total_points": n_points,
        }


# Global memory manager instance
_memory_manager: MemoryManager | None = None
_memory_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance.

    Returns
    -------
    manager : MemoryManager
        Global memory manager instance
    """
    global _memory_manager  # noqa: PLW0603
    if _memory_manager is None:
        with _memory_manager_lock:
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
