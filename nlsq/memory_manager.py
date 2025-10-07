"""Memory management for NLSQ optimization.

This module provides intelligent memory management capabilities including
prediction, monitoring, pooling, and automatic garbage collection.
"""

import gc
import logging
import warnings
from contextlib import contextmanager

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

    def __init__(self, gc_threshold: float = 0.8, safety_factor: float = 1.2):
        """Initialize memory manager.

        Parameters
        ----------
        gc_threshold : float
            Trigger GC when memory usage exceeds this fraction (0-1)
        safety_factor : float
            Multiply memory requirements by this factor for safety
        """
        self.memory_pool: dict[tuple, np.ndarray] = {}
        self.allocation_history: list = []
        self.gc_threshold = gc_threshold
        self.safety_factor = safety_factor
        self._peak_memory = 0
        self._initial_memory = self.get_memory_usage_bytes()

    def get_available_memory(self) -> float:
        """Get available memory in bytes.

        Returns
        -------
        available : float
            Available memory in bytes
        """
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
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
        """
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                return process.memory_info().rss
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
        """
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                return mem.percent / 100.0
            except Exception as e:
                logger.debug(f"psutil memory fraction check failed (non-critical): {e}")

        # Conservative estimate
        return 0.5

    def predict_memory_requirement(
        self, n_points: int, n_params: int, algorithm: str = "trf"
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

        Returns
        -------
        bytes_needed : int
            Estimated memory requirement in bytes
        """
        # Size of double precision float
        float_size = 8

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
