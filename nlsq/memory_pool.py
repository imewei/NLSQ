"""Memory pool for optimization algorithms.

This module provides memory pool allocation to reduce overhead from
repeated array allocations in optimization loops.
"""

import warnings
from typing import Any

import jax.numpy as jnp
import numpy as np


class MemoryPool:
    """Memory pool for reusable array buffers.

    Pre-allocates buffers for common array shapes to avoid repeated
    allocations during optimization iterations.

    Attributes
    ----------
    pools : dict
        Dictionary mapping (shape, dtype) to list of available buffers
    allocated : dict
        Dictionary tracking allocated buffers
    max_pool_size : int
        Maximum number of buffers per shape/dtype combination
    stats : dict
        Statistics on pool usage
    """

    def __init__(self, max_pool_size: int = 10, enable_stats: bool = False):
        """Initialize memory pool.

        Parameters
        ----------
        max_pool_size : int
            Maximum number of buffers to keep per shape/dtype
        enable_stats : bool
            Track allocation statistics
        """
        self.pools: dict[tuple, list[Any]] = {}
        self.allocated: dict[int, tuple] = {}
        self.max_pool_size = max_pool_size
        self.enable_stats = enable_stats

        if enable_stats:
            self.stats = {
                "allocations": 0,
                "reuses": 0,
                "releases": 0,
                "peak_memory": 0,
            }

    def allocate(self, shape: tuple, dtype: type = jnp.float64) -> jnp.ndarray:
        """Allocate array from pool or create new one.

        Parameters
        ----------
        shape : tuple
            Shape of array to allocate
        dtype : type
            Data type of array

        Returns
        -------
        array : jnp.ndarray
            Allocated array (may be reused from pool)
        """
        key = (shape, dtype)

        # Try to reuse from pool
        if self.pools.get(key):
            arr = self.pools[key].pop()
            arr = jnp.zeros(shape, dtype=dtype)  # Reset values
            self.allocated[id(arr)] = key

            if self.enable_stats:
                self.stats["reuses"] += 1

            return arr

        # Allocate new array
        arr = jnp.zeros(shape, dtype=dtype)
        self.allocated[id(arr)] = key

        if self.enable_stats:
            self.stats["allocations"] += 1
            current_mem = sum(
                np.prod(k[0]) * np.dtype(k[1]).itemsize for k in self.allocated.values()
            )
            self.stats["peak_memory"] = max(self.stats["peak_memory"], current_mem)

        return arr

    def release(self, arr: jnp.ndarray):
        """Return array to pool for reuse.

        Parameters
        ----------
        arr : jnp.ndarray
            Array to return to pool
        """
        arr_id = id(arr)

        if arr_id not in self.allocated:
            warnings.warn("Attempting to release array not from pool")
            return

        key = self.allocated.pop(arr_id)

        # Add to pool if not full
        if key not in self.pools:
            self.pools[key] = []

        if len(self.pools[key]) < self.max_pool_size:
            self.pools[key].append(arr)

            if self.enable_stats:
                self.stats["releases"] += 1

    def clear(self):
        """Clear all pools and reset statistics."""
        self.pools.clear()
        self.allocated.clear()

        if self.enable_stats:
            self.stats = {
                "allocations": 0,
                "reuses": 0,
                "releases": 0,
                "peak_memory": 0,
            }

    def get_stats(self) -> dict:
        """Get pool usage statistics.

        Returns
        -------
        stats : dict
            Pool usage statistics
        """
        if not self.enable_stats:
            return {"enabled": False}

        total_ops = self.stats["allocations"] + self.stats["reuses"]
        reuse_rate = self.stats["reuses"] / total_ops if total_ops > 0 else 0.0

        return {
            **self.stats,
            "reuse_rate": reuse_rate,
            "pool_sizes": {k: len(v) for k, v in self.pools.items()},
            "currently_allocated": len(self.allocated),
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clear pool."""
        self.clear()
        return False


class TRFMemoryPool:
    """Specialized memory pool for Trust Region Reflective algorithm.

    Pre-allocates buffers for common TRF operations.

    Parameters
    ----------
    m : int
        Number of residuals
    n : int
        Number of parameters
    dtype : type
        Data type for arrays
    """

    def __init__(self, m: int, n: int, dtype: type = jnp.float64):
        """Initialize TRF memory pool.

        Parameters
        ----------
        m : int
            Number of residuals
        n : int
            Number of parameters
        dtype : type
            Data type
        """
        self.m = m
        self.n = n
        self.dtype = dtype

        # Pre-allocate common buffers
        self.jacobian_buffer = jnp.zeros((m, n), dtype=dtype)
        self.residual_buffer = jnp.zeros(m, dtype=dtype)
        self.gradient_buffer = jnp.zeros(n, dtype=dtype)
        self.step_buffer = jnp.zeros(n, dtype=dtype)
        self.x_buffer = jnp.zeros(n, dtype=dtype)

        # Temporary buffers for trust region subproblem
        self.temp_vec_n = jnp.zeros(n, dtype=dtype)
        self.temp_vec_m = jnp.zeros(m, dtype=dtype)

    def get_jacobian_buffer(self) -> jnp.ndarray:
        """Get Jacobian buffer (m×n)."""
        return self.jacobian_buffer

    def get_residual_buffer(self) -> jnp.ndarray:
        """Get residual buffer (m)."""
        return self.residual_buffer

    def get_gradient_buffer(self) -> jnp.ndarray:
        """Get gradient buffer (n)."""
        return self.gradient_buffer

    def get_step_buffer(self) -> jnp.ndarray:
        """Get step buffer (n)."""
        return self.step_buffer

    def get_x_buffer(self) -> jnp.ndarray:
        """Get parameter buffer (n)."""
        return self.x_buffer

    def reset(self):
        """Reset all buffers to zero."""
        self.jacobian_buffer = jnp.zeros((self.m, self.n), dtype=self.dtype)
        self.residual_buffer = jnp.zeros(self.m, dtype=self.dtype)
        self.gradient_buffer = jnp.zeros(self.n, dtype=self.dtype)
        self.step_buffer = jnp.zeros(self.n, dtype=self.dtype)
        self.x_buffer = jnp.zeros(self.n, dtype=self.dtype)
        self.temp_vec_n = jnp.zeros(self.n, dtype=self.dtype)
        self.temp_vec_m = jnp.zeros(self.m, dtype=self.dtype)


# Global memory pool (optional, for convenience)
_global_pool: MemoryPool | None = None


def get_global_pool(enable_stats: bool = False) -> MemoryPool:
    """Get or create global memory pool.

    Parameters
    ----------
    enable_stats : bool
        Enable statistics tracking

    Returns
    -------
    pool : MemoryPool
        Global memory pool instance
    """
    global _global_pool  # noqa: PLW0603
    if _global_pool is None:
        _global_pool = MemoryPool(enable_stats=enable_stats)
    return _global_pool


def clear_global_pool():
    """Clear the global memory pool."""
    global _global_pool  # noqa: PLW0602
    if _global_pool is not None:
        _global_pool.clear()
