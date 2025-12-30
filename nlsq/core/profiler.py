"""Profiling utilities for Trust Region Reflective optimization.

This module provides profiling classes for timing TRF algorithm operations,
enabling performance analysis and optimization tuning.
"""

from __future__ import annotations

import time

__all__ = [
    "NullProfiler",
    "TRFProfiler",
]


class TRFProfiler:
    """Profiler for timing TRF algorithm operations.

    Records detailed timing information for each operation in the TRF algorithm,
    including GPU synchronization via block_until_ready() for accurate timings.

    This enables performance analysis without duplicating the entire algorithm.

    Attributes
    ----------
    ftimes : list[float]
        Function evaluation times.
    jtimes : list[float]
        Jacobian evaluation times.
    svd_times : list[float]
        SVD computation times.
    ctimes : list[float]
        Cost computation times (JAX).
    gtimes : list[float]
        Gradient computation times (JAX).
    gtimes2 : list[float]
        Gradient norm computation times.
    ptimes : list[float]
        Parameter update times.
    svd_ctimes : list[float]
        SVD conversion times (JAX → NumPy).
    g_ctimes : list[float]
        Gradient conversion times (JAX → NumPy).
    c_ctimes : list[float]
        Cost conversion times (JAX → NumPy).
    p_ctimes : list[float]
        Parameter conversion times (JAX → NumPy).
    """

    __slots__ = (
        "c_ctimes",
        "ctimes",
        "ftimes",
        "g_ctimes",
        "gtimes",
        "gtimes2",
        "jtimes",
        "p_ctimes",
        "ptimes",
        "svd_ctimes",
        "svd_times",
    )

    def __init__(self) -> None:
        """Initialize profiler with empty timing arrays."""
        self.ftimes: list[float] = []
        self.jtimes: list[float] = []
        self.svd_times: list[float] = []
        self.ctimes: list[float] = []
        self.gtimes: list[float] = []
        self.gtimes2: list[float] = []
        self.ptimes: list[float] = []

        # Conversion times (JAX → NumPy)
        self.svd_ctimes: list[float] = []
        self.g_ctimes: list[float] = []
        self.c_ctimes: list[float] = []
        self.p_ctimes: list[float] = []

    def time_operation(self, operation: str, jax_result):
        """Time a JAX operation with GPU synchronization.

        Parameters
        ----------
        operation : str
            Operation name ('fun', 'jac', 'svd', 'cost', 'grad', etc.)
        jax_result :
            JAX array result to synchronize

        Returns
        -------
        result
            The synchronized result (same as input)
        """
        st = time.time()
        result = jax_result.block_until_ready()
        elapsed = time.time() - st

        # Record timing
        if operation == "fun":
            self.ftimes.append(elapsed)
        elif operation == "jac":
            self.jtimes.append(elapsed)
        elif operation == "svd":
            self.svd_times.append(elapsed)
        elif operation == "cost":
            self.ctimes.append(elapsed)
        elif operation == "grad":
            self.gtimes.append(elapsed)
        elif operation == "grad_norm":
            self.gtimes2.append(elapsed)
        elif operation == "param_update":
            self.ptimes.append(elapsed)

        return result

    def time_conversion(self, operation: str, start_time: float) -> None:
        """Record timing for JAX → NumPy conversion.

        Parameters
        ----------
        operation : str
            Conversion operation ('svd_convert', 'grad_convert', 'cost_convert', 'param_convert')
        start_time : float
            Start time from time.time()
        """
        elapsed = time.time() - start_time

        if operation == "svd_convert":
            self.svd_ctimes.append(elapsed)
        elif operation == "grad_convert":
            self.g_ctimes.append(elapsed)
        elif operation == "cost_convert":
            self.c_ctimes.append(elapsed)
        elif operation == "param_convert":
            self.p_ctimes.append(elapsed)

    def get_timing_data(self) -> dict[str, list[float]]:
        """Get all recorded timing data.

        Returns
        -------
        dict[str, list[float]]
            Dictionary containing all timing arrays
        """
        return {
            "ftimes": self.ftimes,
            "jtimes": self.jtimes,
            "svd_times": self.svd_times,
            "ctimes": self.ctimes,
            "gtimes": self.gtimes,
            "gtimes2": self.gtimes2,
            "ptimes": self.ptimes,
            "svd_ctimes": self.svd_ctimes,
            "g_ctimes": self.g_ctimes,
            "c_ctimes": self.c_ctimes,
            "p_ctimes": self.p_ctimes,
        }


class NullProfiler:
    """Null object profiler with zero overhead.

    Provides same interface as TRFProfiler but does nothing,
    enabling profiling to be toggled with no performance impact.
    """

    __slots__ = ()

    def time_operation(self, operation: str, jax_result):
        """No-op timing - returns result unchanged."""
        return jax_result

    def time_conversion(self, operation: str, start_time: float) -> None:
        """No-op conversion timing."""

    def get_timing_data(self) -> dict[str, list[float]]:
        """Returns empty timing data."""
        return {
            "ftimes": [],
            "jtimes": [],
            "svd_times": [],
            "ctimes": [],
            "gtimes": [],
            "gtimes2": [],
            "ptimes": [],
            "svd_ctimes": [],
            "g_ctimes": [],
            "c_ctimes": [],
            "p_ctimes": [],
        }
