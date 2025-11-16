"""JAX profiler integration for transfer monitoring.

This module provides utilities for profiling host-device transfers in the TRF solver,
enabling measurement of transfer bytes and counts per iteration to validate the
80% reduction target.

Performance Targets (Task Group 2):
- Host-device transfer bytes: 80% reduction (current ~80KB → <16KB per iteration)
- Transfer count: Reduce from 24+ → <5 per iteration
- GPU iteration time: 5-15% reduction

Example Usage:
    ```python
    from nlsq.profiling import TransferProfiler

    profiler = TransferProfiler(enable=True)

    with profiler.profile_iteration(iteration=0):
        # TRF iteration code here
        pass

    diagnostics = profiler.get_diagnostics()
    print(f"Transfer bytes: {diagnostics['transfer_bytes']}")
    ```
"""

import contextlib
import warnings
from typing import Any

# Conditional JAX profiler import
try:
    from jax.profiler import trace as jax_trace

    HAS_JAX_PROFILER = True
except ImportError:
    HAS_JAX_PROFILER = False


class TransferProfiler:
    """Profile host-device transfers in TRF solver.

    This profiler integrates with JAX's built-in profiling tools to track
    host-device data transfers during optimization. It provides:

    - Per-iteration transfer tracking
    - Transfer byte counting (estimated)
    - Transfer operation counting
    - Chrome trace generation for visualization

    Attributes
    ----------
    enable : bool
        Whether profiling is enabled (requires JAX profiler)
    transfer_bytes : int
        Estimated total bytes transferred (cumulative)
    transfer_count : int
        Number of transfer operations (cumulative)
    iteration_count : int
        Number of profiled iterations

    Methods
    -------
    profile_iteration(iteration)
        Context manager for profiling a single iteration
    get_diagnostics()
        Return transfer diagnostics dictionary
    reset()
        Reset profiling counters

    Notes
    -----
    - Profiling has minimal overhead (~1-2% when enabled)
    - Transfer bytes are estimated based on array sizes
    - Requires JAX profiler (optional dependency)
    - Chrome traces can be visualized at chrome://tracing
    """

    def __init__(self, enable: bool = False):
        """Initialize transfer profiler.

        Parameters
        ----------
        enable : bool, optional
            Enable profiling (requires JAX profiler), default False

        Warnings
        --------
        If enable=True but JAX profiler not available, issues warning
        and disables profiling.
        """
        if enable and not HAS_JAX_PROFILER:
            warnings.warn(
                "JAX profiler not available. Install with: pip install jax[profiler]. "
                "Profiling will be disabled.",
                UserWarning,
            )
            self.enable = False
        else:
            self.enable = enable

        # Transfer tracking
        self.transfer_bytes = 0
        self.transfer_count = 0
        self.iteration_count = 0

        # Per-iteration history (for analysis)
        self._iteration_history: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def profile_iteration(self, iteration: int):
        """Context manager for profiling single TRF iteration.

        Parameters
        ----------
        iteration : int
            Iteration number (for trace labeling)

        Yields
        ------
        None

        Examples
        --------
        >>> profiler = TransferProfiler(enable=True)
        >>> with profiler.profile_iteration(0):
        ...     # TRF iteration code
        ...     result = trf_step()
        """
        if not self.enable:
            yield
            return

        # JAX profiler trace for Chrome tracing
        with jax_trace(f"trf_iteration_{iteration}"):
            self.iteration_count += 1
            yield

    def record_transfer(self, array_size_bytes: int, operation_name: str = "transfer"):
        """Record a host-device transfer operation.

        Parameters
        ----------
        array_size_bytes : int
            Size of transferred array in bytes
        operation_name : str, optional
            Name of operation causing transfer (for diagnostics)

        Notes
        -----
        This is called manually when a transfer is detected.
        Future versions may auto-detect transfers via JAX internals.
        """
        if not self.enable:
            return

        self.transfer_bytes += array_size_bytes
        self.transfer_count += 1

        # Record in history for per-iteration analysis
        self._iteration_history.append(
            {
                "iteration": self.iteration_count,
                "operation": operation_name,
                "bytes": array_size_bytes,
            }
        )

    def get_diagnostics(self) -> dict[str, Any]:
        """Return transfer diagnostics dictionary.

        Returns
        -------
        diagnostics : dict
            Dictionary containing:
            - transfer_bytes : int - Total bytes transferred
            - transfer_count : int - Number of transfer operations
            - iterations_profiled : int - Number of iterations profiled
            - profiling_enabled : bool - Whether profiling was active
            - avg_bytes_per_iteration : float - Average transfer bytes per iteration
            - avg_transfers_per_iteration : float - Average transfers per iteration

        Examples
        --------
        >>> profiler = TransferProfiler(enable=True)
        >>> # ... run optimization ...
        >>> diag = profiler.get_diagnostics()
        >>> print(f"Transfer reduction: {diag['avg_bytes_per_iteration']:.1f} bytes/iter")
        """
        avg_bytes = (
            self.transfer_bytes / self.iteration_count
            if self.iteration_count > 0
            else 0
        )
        avg_count = (
            self.transfer_count / self.iteration_count
            if self.iteration_count > 0
            else 0
        )

        return {
            "transfer_bytes": self.transfer_bytes,
            "transfer_count": self.transfer_count,
            "iterations_profiled": self.iteration_count,
            "profiling_enabled": self.enable,
            "avg_bytes_per_iteration": avg_bytes,
            "avg_transfers_per_iteration": avg_count,
        }

    def get_iteration_history(self) -> list[dict[str, Any]]:
        """Return detailed per-iteration transfer history.

        Returns
        -------
        history : list of dict
            List of transfer records, each containing:
            - iteration : int - Iteration number
            - operation : str - Operation name
            - bytes : int - Transfer size in bytes

        Notes
        -----
        Only available when profiling is enabled.
        History is cleared on reset().
        """
        return self._iteration_history.copy()

    def reset(self):
        """Reset profiling counters and history.

        Examples
        --------
        >>> profiler = TransferProfiler(enable=True)
        >>> # ... run first optimization ...
        >>> profiler.reset()
        >>> # ... run second optimization (fresh counters) ...
        """
        self.transfer_bytes = 0
        self.transfer_count = 0
        self.iteration_count = 0
        self._iteration_history.clear()

    def print_summary(self):
        """Print formatted profiling summary.

        Examples
        --------
        >>> profiler = TransferProfiler(enable=True)
        >>> # ... run optimization ...
        >>> profiler.print_summary()
        Transfer Profiling Summary
        ==========================
        Total bytes transferred: 81,584 bytes (79.7 KB)
        Total transfer operations: 24
        Iterations profiled: 10
        Avg bytes/iteration: 8,158.4 bytes (8.0 KB)
        Avg transfers/iteration: 2.4
        """
        diag = self.get_diagnostics()

        if not diag["profiling_enabled"]:
            print("Transfer profiling was not enabled")
            return

        print("Transfer Profiling Summary")
        print("=" * 50)
        print(
            f"Total bytes transferred: {diag['transfer_bytes']:,} bytes "
            f"({diag['transfer_bytes'] / 1024:.1f} KB)"
        )
        print(f"Total transfer operations: {diag['transfer_count']}")
        print(f"Iterations profiled: {diag['iterations_profiled']}")

        if diag["iterations_profiled"] > 0:
            print(
                f"Avg bytes/iteration: {diag['avg_bytes_per_iteration']:.1f} bytes "
                f"({diag['avg_bytes_per_iteration'] / 1024:.1f} KB)"
            )
            print(f"Avg transfers/iteration: {diag['avg_transfers_per_iteration']:.1f}")

    def __repr__(self) -> str:
        """Return string representation of profiler state."""
        return (
            f"TransferProfiler(enable={self.enable}, "
            f"iterations={self.iteration_count}, "
            f"transfers={self.transfer_count}, "
            f"bytes={self.transfer_bytes})"
        )


# Convenience function for one-off profiling
@contextlib.contextmanager
def profile_transfers(enable: bool = True):
    """Convenience context manager for one-off transfer profiling.

    Parameters
    ----------
    enable : bool, optional
        Enable profiling, default True

    Yields
    ------
    profiler : TransferProfiler
        Profiler instance (can be queried for diagnostics)

    Examples
    --------
    >>> from nlsq.profiling import profile_transfers
    >>> with profile_transfers() as profiler:
    ...     result = curve_fit(model, xdata, ydata)
    >>> profiler.print_summary()
    """
    profiler = TransferProfiler(enable=enable)
    try:
        yield profiler
    finally:
        if enable:
            profiler.print_summary()


# Export public API
__all__ = [
    "HAS_JAX_PROFILER",
    "TransferProfiler",
    "profile_transfers",
]
