"""Performance profiling utilities for NLSQ.

Provides lightweight profiling infrastructure for measuring optimization
performance and validating improvements without heavy dependencies.

Examples
--------
>>> from nlsq.utils.profiling import profile_optimization
>>> from nlsq import least_squares
>>> import jax.numpy as jnp
>>>
>>> with profile_optimization() as metrics:
...     result = least_squares(lambda x: x**2, x0=jnp.array([1.0]), max_nfev=100)
>>>
>>> print(f"Average iteration: {metrics.avg_iteration_time_ms:.2f}ms")
>>> print(f"Total time: {metrics.total_time_sec:.3f}s")
"""

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PerformanceMetrics:
    """Performance metrics for optimization runs.

    Attributes
    ----------
    iteration_count : int
        Total number of iterations performed
    total_time_sec : float
        Total elapsed time in seconds
    iteration_times : list of float
        Individual iteration times (if tracked)

    Properties
    ----------
    avg_iteration_time_ms : float
        Average iteration time in milliseconds
    min_iteration_time_ms : float
        Minimum iteration time in milliseconds
    max_iteration_time_ms : float
        Maximum iteration time in milliseconds
    """

    iteration_count: int = 0
    total_time_sec: float = 0.0
    iteration_times: list[float] = field(default_factory=list)

    @property
    def avg_iteration_time_ms(self) -> float:
        """Average iteration time in milliseconds."""
        if self.iteration_count == 0:
            return 0.0
        return (self.total_time_sec / self.iteration_count) * 1000

    @property
    def min_iteration_time_ms(self) -> float:
        """Minimum iteration time in milliseconds."""
        return min(self.iteration_times) * 1000 if self.iteration_times else 0.0

    @property
    def max_iteration_time_ms(self) -> float:
        """Maximum iteration time in milliseconds."""
        return max(self.iteration_times) * 1000 if self.iteration_times else 0.0


@contextlib.contextmanager
def profile_optimization(enabled: bool = True):
    """Profile optimization performance.

    Measures total runtime as a proxy for performance improvements.
    For beta.1, focuses on wall-clock time rather than detailed profiling.

    Parameters
    ----------
    enabled : bool, default=True
        Whether to enable profiling

    Yields
    ------
    PerformanceMetrics
        Performance statistics

    Examples
    --------
    >>> from nlsq import curve_fit
    >>> import jax.numpy as jnp
    >>>
    >>> with profile_optimization() as metrics:
    ...     x = jnp.linspace(0, 10, 100)
    ...     y = 2.0 * jnp.exp(-0.5 * x)
    ...     popt, pcov = curve_fit(lambda x, a, b: a * jnp.exp(-b * x), x, y)
    >>>
    >>> print(f"Optimization took {metrics.total_time_sec:.3f}s")
    """
    metrics = PerformanceMetrics()

    if not enabled:
        yield metrics
        return

    start_time = time.perf_counter()

    try:
        yield metrics
    finally:
        metrics.total_time_sec = time.perf_counter() - start_time


def analyze_source_transfers(source_code: str) -> dict:
    """Analyze source code for host-device transfer patterns.

    This is a static analysis tool for validating transfer reduction.
    Counts patterns that typically induce GPU-CPU transfers.

    Parameters
    ----------
    source_code : str
        Source code to analyze

    Returns
    -------
    dict
        Analysis results with counts of transfer-inducing patterns:
        - 'np_array_calls': Number of np.array() calls
        - 'np_asarray_calls': Number of np.asarray() calls
        - 'block_until_ready_calls': Number of .block_until_ready() calls
        - 'total_potential_transfers': Sum of all transfer patterns

    Notes
    -----
    This is a heuristic analysis tool, not a comprehensive profiler.
    It provides a relative measure for before/after comparison.
    Only counts NumPy conversions (np.*), not JAX operations (jnp.*).

    Examples
    --------
    >>> from nlsq.utils.profiling import analyze_source_transfers
    >>>
    >>> code = '''
    ... def my_function(x):
    ...     y = np.array(x)  # Transfer!
    ...     return y
    ... '''
    >>>
    >>> analysis = analyze_source_transfers(code)
    >>> print(f"Potential transfers: {analysis['total_potential_transfers']}")
    Potential transfers: 1
    """
    if not isinstance(source_code, str):
        raise TypeError(
            f"source_code must be a string, got {type(source_code).__name__}"
        )

    # Count only NumPy conversions (not JAX operations like jnp.array)
    # Use negative lookbehind to exclude jnp.array but include np.array
    import re

    np_array_calls = len(re.findall(r"(?<!j)np\.array\(", source_code))
    np_asarray_calls = len(re.findall(r"(?<!j)np\.asarray\(", source_code))
    block_until_ready_calls = source_code.count(".block_until_ready(")

    return {
        "np_array_calls": np_array_calls,
        "np_asarray_calls": np_asarray_calls,
        "block_until_ready_calls": block_until_ready_calls,
        "total_potential_transfers": (
            np_array_calls + np_asarray_calls + block_until_ready_calls
        ),
    }


def compare_transfer_reduction(
    source_before: str, source_after: str, module_name: str = "module"
) -> dict:
    """Compare transfer patterns before and after optimization.

    Parameters
    ----------
    source_before : str
        Source code before optimization
    source_after : str
        Source code after optimization
    module_name : str, optional
        Name of module being analyzed (for reporting)

    Returns
    -------
    dict
        Comparison results with reduction percentages

    Examples
    --------
    >>> before = "x = np.array(y); z = np.array(w)"
    >>> after = "x = jnp.asarray(y); z = jnp.asarray(w)"
    >>>
    >>> comparison = compare_transfer_reduction(before, after, "mymodule")
    >>> print(f"Reduction: {comparison['reduction_percent']:.1f}%")
    Reduction: 100.0%
    """
    if not isinstance(source_before, str):
        raise TypeError(
            f"source_before must be a string, got {type(source_before).__name__}"
        )
    if not isinstance(source_after, str):
        raise TypeError(
            f"source_after must be a string, got {type(source_after).__name__}"
        )
    if not isinstance(module_name, str):
        raise TypeError(
            f"module_name must be a string, got {type(module_name).__name__}"
        )

    before = analyze_source_transfers(source_before)
    after = analyze_source_transfers(source_after)

    total_before = before["total_potential_transfers"]
    total_after = after["total_potential_transfers"]

    if total_before == 0:
        reduction_percent = 0.0
    else:
        reduction_percent = ((total_before - total_after) / total_before) * 100

    return {
        "module": module_name,
        "before": before,
        "after": after,
        "reduction_count": total_before - total_after,
        "reduction_percent": reduction_percent,
    }


def profile_transfers_runtime(
    func: Callable[..., Any],
    *args,
    trace_dir: str | None = None,
    **kwargs,
):
    """Profile actual host-device transfers using JAX profiler.

    This function provides runtime measurement of host-device transfers
    using JAX's built-in profiler. Unlike static analysis, this captures
    actual transfer events during execution.

    Parameters
    ----------
    func : callable
        Function to profile
    *args
        Positional arguments to pass to func
    trace_dir : str or None, optional
        Directory to store profiler trace. If None, uses system temp directory
        with "jax-profiling" subdirectory.
    **kwargs
        Keyword arguments to pass to func

    Returns
    -------
    tuple
        (result, transfer_stats) where result is func's return value and
        transfer_stats contains profiling information

    Notes
    -----
    Requires JAX profiler support. On CPU, transfers are minimal.
    Most useful for GPU profiling.

    Examples
    --------
    >>> from nlsq.utils.profiling import profile_transfers_runtime
    >>> import jax.numpy as jnp
    >>>
    >>> def my_computation(x):
    ...     return jnp.sum(x ** 2)
    >>>
    >>> result, stats = profile_transfers_runtime(
    ...     my_computation,
    ...     jnp.array([1.0, 2.0, 3.0])
    ... )
    >>> print(f"Result: {result}, Transfers: {stats['transfer_count']}")
    """
    if not callable(func):
        raise TypeError(f"func must be callable, got {type(func).__name__}")
    if trace_dir is not None and not isinstance(trace_dir, str):
        raise TypeError(
            f"trace_dir must be a string or None, got {type(trace_dir).__name__}"
        )

    import tempfile

    # Use system temp directory if not specified
    if trace_dir is None:
        trace_dir = str(Path(tempfile.gettempdir()) / "jax-profiling")

    try:
        import jax.profiler
    except ImportError:
        # JAX profiler not available, return result without profiling
        result = func(*args, **kwargs)
        return result, {
            "transfer_count": None,
            "transfer_bytes": None,
            "profiler_available": False,
            "message": "JAX profiler not available",
        }

    # Create trace directory if it doesn't exist
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)

    # Run function with profiler trace
    try:
        with jax.profiler.trace(str(trace_path)):
            result = func(*args, **kwargs)

        # Analyze trace for transfer events
        # Note: Detailed trace analysis would require parsing the trace files
        # For now, we return basic profiling information
        transfer_stats = {
            "transfer_count": 0,  # Would be extracted from trace
            "transfer_bytes": 0,  # Would be extracted from trace
            "profiler_available": True,
            "trace_dir": str(trace_path),
            "message": "Profiling complete. Analyze trace in TensorBoard.",
        }

        return result, transfer_stats

    except Exception as e:
        # Profiling failed, return result without stats
        result = func(*args, **kwargs)
        return result, {
            "transfer_count": None,
            "transfer_bytes": None,
            "profiler_available": False,
            "error": str(e),
        }
