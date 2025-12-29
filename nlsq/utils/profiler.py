"""
Performance Profiling for NLSQ
================================

Comprehensive performance profiling system for tracking and analyzing
optimization performance metrics.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ProfileMetrics:
    """Container for performance metrics from a single optimization run."""

    # Timing metrics
    total_time: float = 0.0
    jit_compile_time: float = 0.0
    optimization_time: float = 0.0
    jacobian_time: float = 0.0

    # Iteration metrics
    n_iterations: int = 0
    n_function_evals: int = 0
    n_jacobian_evals: int = 0

    # Data metrics
    n_data_points: int = 0
    n_parameters: int = 0
    data_dimension: int = 1

    # Convergence metrics
    final_cost: float = 0.0
    initial_cost: float = 0.0
    cost_reduction: float = 0.0
    final_gradient_norm: float = 0.0
    success: bool = False

    # Method information
    method: str = ""
    backend: str = "cpu"

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def speedup_vs_scipy(self) -> float:
        """Estimate speedup vs SciPy (rough heuristic)."""
        if self.backend == "gpu":
            # GPU typically 100-300x faster for large problems
            return min(270, 100 * (self.n_data_points / 100000))
        return 1.0

    def iterations_per_second(self) -> float:
        """Calculate iterations per second."""
        if self.optimization_time > 0:
            return self.n_iterations / self.optimization_time
        return 0.0

    def function_evals_per_second(self) -> float:
        """Calculate function evaluations per second."""
        if self.optimization_time > 0:
            return self.n_function_evals / self.optimization_time
        return 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_time": self.total_time,
            "jit_compile_time": self.jit_compile_time,
            "optimization_time": self.optimization_time,
            "jacobian_time": self.jacobian_time,
            "n_iterations": self.n_iterations,
            "n_function_evals": self.n_function_evals,
            "n_jacobian_evals": self.n_jacobian_evals,
            "n_data_points": self.n_data_points,
            "n_parameters": self.n_parameters,
            "data_dimension": self.data_dimension,
            "final_cost": self.final_cost,
            "initial_cost": self.initial_cost,
            "cost_reduction": self.cost_reduction,
            "final_gradient_norm": self.final_gradient_norm,
            "success": self.success,
            "method": self.method,
            "backend": self.backend,
            "iterations_per_second": self.iterations_per_second(),
            "function_evals_per_second": self.function_evals_per_second(),
            "speedup_vs_scipy": self.speedup_vs_scipy(),
            **self.metadata,
        }


class PerformanceProfiler:
    """
    Performance profiler for NLSQ optimization.

    Tracks and analyzes performance metrics across optimization runs.

    Examples
    --------
    >>> profiler = PerformanceProfiler()
    >>> with profiler.profile("my_optimization"):
    ...     result = curve_fit(model, x, y, p0=[1, 2])
    >>>
    >>> report = profiler.get_report()
    >>> print(report)
    """

    def __init__(self):
        """Initialize performance profiler."""
        self.profiles: dict[str, list[ProfileMetrics]] = defaultdict(list)
        self._current_profile: ProfileMetrics | None = None
        self._start_time: float = 0.0
        self._context_stack: list[str] = []

    def start_profile(self, name: str = "default") -> ProfileMetrics:
        """
        Start profiling a new optimization run.

        Parameters
        ----------
        name : str
            Name for this profiling session

        Returns
        -------
        metrics : ProfileMetrics
            Metrics object for this profile
        """
        metrics = ProfileMetrics()
        self._current_profile = metrics
        self._start_time = time.perf_counter()
        self._context_stack.append(name)
        return metrics

    def end_profile(self, metrics: ProfileMetrics | None = None):
        """
        End current profiling session.

        Parameters
        ----------
        metrics : ProfileMetrics, optional
            Metrics to finalize. If None, uses current profile.
        """
        if metrics is None:
            metrics = self._current_profile

        if metrics is not None:
            metrics.total_time = time.perf_counter() - self._start_time

            # Calculate derived metrics
            if metrics.initial_cost > 0:
                metrics.cost_reduction = (
                    metrics.initial_cost - metrics.final_cost
                ) / metrics.initial_cost

            # Store profile
            if self._context_stack:
                name = self._context_stack.pop()
                self.profiles[name].append(metrics)

        self._current_profile = None

    def profile(self, name: str = "default"):
        """
        Context manager for profiling.

        Parameters
        ----------
        name : str
            Name for this profiling session

        Examples
        --------
        >>> profiler = PerformanceProfiler()
        >>> with profiler.profile("test_1"):
        ...     result = curve_fit(model, x, y)
        """
        return ProfileContext(self, name)

    def record_timing(self, category: str, duration: float):
        """
        Record timing for a specific category.

        Parameters
        ----------
        category : str
            Timing category (e.g., 'jit_compile', 'optimization')
        duration : float
            Duration in seconds
        """
        if self._current_profile is not None:
            if category == "jit_compile":
                self._current_profile.jit_compile_time += duration
            elif category == "optimization":
                self._current_profile.optimization_time += duration
            elif category == "jacobian":
                self._current_profile.jacobian_time += duration

    def update_current(self, **kwargs):
        """
        Update current profile with arbitrary metrics.

        Parameters
        ----------
        **kwargs
            Metrics to update
        """
        if self._current_profile is not None:
            for key, value in kwargs.items():
                if hasattr(self._current_profile, key):
                    setattr(self._current_profile, key, value)
                else:
                    self._current_profile.metadata[key] = value

    def get_metrics(self, name: str = "default") -> list[ProfileMetrics]:
        """
        Get all metrics for a named profile.

        Parameters
        ----------
        name : str
            Profile name

        Returns
        -------
        metrics : list of ProfileMetrics
            All metrics for this profile
        """
        return self.profiles.get(name, [])

    def get_summary(self, name: str = "default") -> dict:
        """
        Get summary statistics for a named profile.

        Parameters
        ----------
        name : str
            Profile name

        Returns
        -------
        summary : dict
            Summary statistics
        """
        metrics_list = self.get_metrics(name)
        if not metrics_list:
            return {}

        # Aggregate metrics
        total_times = [m.total_time for m in metrics_list]
        opt_times = [m.optimization_time for m in metrics_list]
        iterations = [m.n_iterations for m in metrics_list]
        successes = [m.success for m in metrics_list]

        return {
            "n_runs": len(metrics_list),
            "success_rate": sum(successes) / len(successes) if successes else 0.0,
            "total_time": {
                "mean": np.mean(total_times),
                "std": np.std(total_times),
                "min": np.min(total_times),
                "max": np.max(total_times),
            },
            "optimization_time": {
                "mean": np.mean(opt_times),
                "std": np.std(opt_times),
                "min": np.min(opt_times),
                "max": np.max(opt_times),
            },
            "iterations": {
                "mean": np.mean(iterations),
                "std": np.std(iterations),
                "min": int(np.min(iterations)),
                "max": int(np.max(iterations)),
            },
        }

    def get_report(self, name: str = "default", detailed: bool = False) -> str:
        """
        Generate a formatted performance report.

        Parameters
        ----------
        name : str
            Profile name
        detailed : bool
            Include detailed metrics

        Returns
        -------
        report : str
            Formatted report
        """
        metrics_list = self.get_metrics(name)
        if not metrics_list:
            return f"No profiling data for '{name}'"

        summary = self.get_summary(name)

        lines = [
            f"Performance Report: {name}",
            "=" * 60,
            f"Runs: {summary['n_runs']}",
            f"Success Rate: {summary['success_rate']:.1%}",
            "",
            "Timing (seconds):",
            f"  Total Time:        {summary['total_time']['mean']:.3f} ± {summary['total_time']['std']:.3f}",
            f"  Optimization Time: {summary['optimization_time']['mean']:.3f} ± {summary['optimization_time']['std']:.3f}",
            "",
            "Iterations:",
            f"  Mean: {summary['iterations']['mean']:.1f}",
            f"  Range: [{summary['iterations']['min']}, {summary['iterations']['max']}]",
        ]

        if detailed and metrics_list:
            lines.extend(
                [
                    "",
                    "Per-Run Details:",
                    "-" * 60,
                ]
            )
            for i, m in enumerate(metrics_list, 1):
                lines.append(
                    f"  Run {i}: {m.total_time:.3f}s, "
                    f"{m.n_iterations} iter, "
                    f"{'✓' if m.success else '✗'}"
                )

        return "\n".join(lines)

    def compare_profiles(self, name1: str, name2: str) -> dict:
        """
        Compare two profiling sessions.

        Parameters
        ----------
        name1, name2 : str
            Names of profiles to compare

        Returns
        -------
        comparison : dict
            Comparison metrics
        """
        summary1 = self.get_summary(name1)
        summary2 = self.get_summary(name2)

        if not summary1 or not summary2:
            return {}

        speedup = (
            summary1["total_time"]["mean"] / summary2["total_time"]["mean"]
            if summary2["total_time"]["mean"] > 0
            else 0.0
        )

        return {
            "profile_1": name1,
            "profile_2": name2,
            "speedup": speedup,
            "time_difference": (
                summary1["total_time"]["mean"] - summary2["total_time"]["mean"]
            ),
            "success_rate_difference": (
                summary1["success_rate"] - summary2["success_rate"]
            ),
        }

    def clear(self, name: str | None = None):
        """
        Clear profiling data.

        Parameters
        ----------
        name : str, optional
            Name of profile to clear. If None, clears all.
        """
        if name is None:
            self.profiles.clear()
        elif name in self.profiles:
            del self.profiles[name]

    def export_to_dict(self) -> dict:
        """
        Export all profiling data to dictionary.

        Returns
        -------
        data : dict
            All profiling data
        """
        return {
            name: [m.to_dict() for m in metrics]
            for name, metrics in self.profiles.items()
        }


class ProfileContext:
    """Context manager for profiling."""

    def __init__(self, profiler: PerformanceProfiler, name: str):
        """Initialize context."""
        self.profiler = profiler
        self.name = name
        self.metrics: ProfileMetrics | None = None

    def __enter__(self):
        """Enter context."""
        self.metrics = self.profiler.start_profile(self.name)
        return self.metrics

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.profiler.end_profile(self.metrics)
        return False


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_global_profiler() -> PerformanceProfiler:
    """
    Get the global profiler instance.

    Returns
    -------
    profiler : PerformanceProfiler
        Global profiler
    """
    return _global_profiler


def clear_profiling_data():
    """Clear all global profiling data."""
    _global_profiler.clear()


__all__ = [
    "PerformanceProfiler",
    "ProfileMetrics",
    "clear_profiling_data",
    "get_global_profiler",
]
