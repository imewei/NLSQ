"""Mixed Precision Performance Benchmarking.

This module benchmarks the performance and memory characteristics of the
mixed precision fallback system, comparing:
- Float32 only
- Float64 only
- Mixed precision (float32 → float64 automatic fallback)

Metrics:
- Execution time (first run with JIT, cached runs)
- Memory usage
- Convergence quality
- Number of precision upgrades
- Final accuracy
"""

from __future__ import annotations

import contextlib
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit
from nlsq.config import JAXConfig, configure_mixed_precision
from nlsq.mixed_precision import MixedPrecisionConfig


@dataclass
class MixedPrecisionBenchmarkResult:
    """Results from a mixed precision benchmark run."""

    precision_mode: str  # "float32", "float64", or "mixed"
    problem_name: str
    problem_size: int

    # Timing
    first_run_time: float  # Time with JIT compilation
    cached_run_time: float  # Time after JIT compilation

    # Memory
    peak_memory_mb: float
    memory_allocated_mb: float

    # Convergence
    converged: bool
    iterations: int
    final_cost: float

    # Mixed precision specific
    precision_upgraded: bool = False
    upgrade_iteration: int = 0

    # Result quality
    parameter_error: float = 0.0  # |params - true_params|
    relative_error: float = 0.0  # |params - true_params| / |true_params|


@dataclass
class BenchmarkConfig:
    """Configuration for mixed precision benchmarks."""

    problem_sizes: list[int] = field(default_factory=lambda: [100, 1000, 10000])
    n_repeats: int = 5
    warmup_runs: int = 1
    include_float32: bool = True
    include_float64: bool = True
    include_mixed: bool = True


def create_exponential_problem(n_points: int, noise_level: float = 0.01):
    """Create an exponential decay fitting problem.

    Parameters
    ----------
    n_points : int
        Number of data points
    noise_level : float
        Standard deviation of Gaussian noise

    Returns
    -------
    xdata : np.ndarray
        X coordinates
    ydata : np.ndarray
        Y coordinates with noise
    true_params : np.ndarray
        True parameter values [amplitude, decay_rate]
    model : callable
        Model function
    """
    np.random.seed(42)

    # True parameters
    true_amplitude = 5.0
    true_decay = 0.5
    true_params = np.array([true_amplitude, true_decay])

    # Generate data
    xdata = np.linspace(0, 10, n_points)
    ydata_clean = true_amplitude * np.exp(-true_decay * xdata)
    noise = np.random.normal(0, noise_level, n_points)
    ydata = ydata_clean + noise

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    return xdata, ydata, true_params, model


def create_ill_conditioned_problem(n_points: int):
    """Create an ill-conditioned problem that benefits from float64.

    This problem has very small parameter values and steep gradients
    that challenge float32 precision.

    Parameters
    ----------
    n_points : int
        Number of data points

    Returns
    -------
    xdata : np.ndarray
        X coordinates
    ydata : np.ndarray
        Y coordinates
    true_params : np.ndarray
        True parameter values
    model : callable
        Model function
    """
    np.random.seed(42)

    # True parameters (very small values)
    true_params = np.array([1e-6, 1e-7, 2.0])

    # Generate data
    xdata = np.linspace(0, 1, n_points)
    ydata_clean = true_params[0] * xdata**2 + true_params[1] * xdata + true_params[2]
    noise = np.random.normal(0, 1e-8, n_points)
    ydata = ydata_clean + noise

    def model(x, a, b, c):
        return a * x**2 + b * x + c

    return xdata, ydata, true_params, model


def benchmark_single_run(
    xdata: np.ndarray,
    ydata: np.ndarray,
    model: Any,
    p0: np.ndarray,
    precision_mode: str,
    problem_name: str,
    true_params: np.ndarray,
) -> MixedPrecisionBenchmarkResult:
    """Run a single benchmark with specified precision mode.

    Parameters
    ----------
    xdata : np.ndarray
        X data
    ydata : np.ndarray
        Y data
    model : callable
        Model function
    p0 : np.ndarray
        Initial parameters
    precision_mode : str
        One of "float32", "float64", "mixed"
    problem_name : str
        Name of the problem
    true_params : np.ndarray
        True parameter values for error calculation

    Returns
    -------
    result : MixedPrecisionBenchmarkResult
        Benchmark results
    """
    # Configure precision mode
    if precision_mode == "float32":
        configure_mixed_precision(enable=False)
        JAXConfig.enable_x64(False)
    elif precision_mode == "float64":
        configure_mixed_precision(enable=False)
        JAXConfig.enable_x64(True)
    elif precision_mode == "mixed":
        JAXConfig.enable_x64(True)  # Enable x64 for the system
        configure_mixed_precision(
            enable=True,
            max_degradation_iterations=5,
            gradient_explosion_threshold=1e10,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown precision mode: {precision_mode}")

    # First run with JIT compilation and memory tracking
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        result = curve_fit(model, xdata, ydata, p0=p0)
        popt = result.x if hasattr(result, "x") else result[0]
        converged = result.success if hasattr(result, "success") else True
        iterations = result.nfev if hasattr(result, "nfev") else 0
        cost = result.cost if hasattr(result, "cost") else 0.0

        # Check if precision was upgraded (for mixed mode)
        precision_upgraded = False
        upgrade_iteration = 0
        if precision_mode == "mixed" and hasattr(result, "mixed_precision_diagnostics"):
            diag = result.mixed_precision_diagnostics
            precision_upgraded = diag.get("precision_upgraded", False)
            upgrade_iteration = diag.get("upgrade_iteration", 0)

    except Exception as e:
        print(f"Fit failed for {precision_mode} on {problem_name}: {e}")
        popt = p0
        converged = False
        iterations = 0
        cost = float("inf")
        precision_upgraded = False
        upgrade_iteration = 0

    first_run_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Cached run (no JIT compilation)
    start_time = time.perf_counter()
    with contextlib.suppress(Exception):
        _ = curve_fit(model, xdata, ydata, p0=p0)
    cached_run_time = time.perf_counter() - start_time

    # Calculate errors
    param_error = float(np.linalg.norm(popt - true_params))
    relative_error = param_error / (np.linalg.norm(true_params) + 1e-10)

    return MixedPrecisionBenchmarkResult(
        precision_mode=precision_mode,
        problem_name=problem_name,
        problem_size=len(xdata),
        first_run_time=first_run_time,
        cached_run_time=cached_run_time,
        peak_memory_mb=peak / 1024 / 1024,
        memory_allocated_mb=current / 1024 / 1024,
        converged=converged,
        iterations=iterations,
        final_cost=cost,
        precision_upgraded=precision_upgraded,
        upgrade_iteration=upgrade_iteration,
        parameter_error=param_error,
        relative_error=relative_error,
    )


def run_benchmark_suite(
    config: BenchmarkConfig | None = None,
) -> list[MixedPrecisionBenchmarkResult]:
    """Run complete mixed precision benchmark suite.

    Parameters
    ----------
    config : BenchmarkConfig, optional
        Benchmark configuration

    Returns
    -------
    results : list[MixedPrecisionBenchmarkResult]
        All benchmark results
    """
    if config is None:
        config = BenchmarkConfig()

    results = []

    # Problem types
    problems = [
        ("exponential_easy", create_exponential_problem, {"noise_level": 0.01}),
        ("exponential_noisy", create_exponential_problem, {"noise_level": 0.1}),
        ("ill_conditioned", create_ill_conditioned_problem, {}),
    ]

    # Precision modes to test
    modes = []
    if config.include_float32:
        modes.append("float32")
    if config.include_float64:
        modes.append("float64")
    if config.include_mixed:
        modes.append("mixed")

    for problem_name, problem_fn, kwargs in problems:
        for size in config.problem_sizes:
            print(f"\nBenchmarking {problem_name} with {size} points...")

            # Create problem
            xdata, ydata, true_params, model = problem_fn(size, **kwargs)
            p0 = np.ones(len(true_params))

            for mode in modes:
                # Run warmup
                for _ in range(config.warmup_runs):
                    with contextlib.suppress(Exception):
                        benchmark_single_run(
                            xdata, ydata, model, p0, mode, problem_name, true_params
                        )

                # Run actual benchmark (take best of n_repeats)
                run_results = []
                for repeat in range(config.n_repeats):
                    try:
                        result = benchmark_single_run(
                            xdata, ydata, model, p0, mode, problem_name, true_params
                        )
                        run_results.append(result)
                        print(
                            f"  {mode:8s} repeat {repeat + 1}/{config.n_repeats}: "
                            f"{result.cached_run_time * 1000:.2f}ms "
                            f"(first: {result.first_run_time * 1000:.2f}ms, "
                            f"mem: {result.peak_memory_mb:.1f}MB, "
                            f"error: {result.relative_error:.2e})"
                        )
                    except Exception as e:
                        print(
                            f"  {mode:8s} repeat {repeat + 1}/{config.n_repeats}: FAILED - {e}"
                        )

                # Take result with best (lowest) cached run time
                if run_results:
                    best_result = min(run_results, key=lambda r: r.cached_run_time)
                    results.append(best_result)

    return results


def print_summary(results: list[MixedPrecisionBenchmarkResult]):
    """Print summary of benchmark results.

    Parameters
    ----------
    results : list[MixedPrecisionBenchmarkResult]
        Benchmark results to summarize
    """
    print("\n" + "=" * 80)
    print("MIXED PRECISION BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by problem and size
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        key = (r.problem_name, r.problem_size)
        grouped[key].append(r)

    for (problem, size), group in sorted(grouped.items()):
        print(f"\n{problem} (n={size}):")
        print("-" * 80)
        print(
            f"{'Mode':<12} {'Time (ms)':<12} {'Memory (MB)':<12} {'Error':<12} {'Upgraded'}"
        )
        print("-" * 80)

        for r in sorted(group, key=lambda x: x.precision_mode):
            upgraded_str = (
                f"Yes@{r.upgrade_iteration}" if r.precision_upgraded else "No"
            )
            print(
                f"{r.precision_mode:<12} "
                f"{r.cached_run_time * 1000:<12.2f} "
                f"{r.peak_memory_mb:<12.1f} "
                f"{r.relative_error:<12.2e} "
                f"{upgraded_str}"
            )

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    modes = {r.precision_mode for r in results}
    for mode in sorted(modes):
        mode_results = [r for r in results if r.precision_mode == mode]

        avg_time = np.mean([r.cached_run_time for r in mode_results]) * 1000
        avg_memory = np.mean([r.peak_memory_mb for r in mode_results])
        avg_error = np.mean([r.relative_error for r in mode_results])
        convergence_rate = (
            sum(r.converged for r in mode_results) / len(mode_results) * 100
        )

        print(f"\n{mode}:")
        print(f"  Average time:        {avg_time:.2f} ms")
        print(f"  Average memory:      {avg_memory:.1f} MB")
        print(f"  Average rel. error:  {avg_error:.2e}")
        print(f"  Convergence rate:    {convergence_rate:.1f}%")

        if mode == "mixed":
            upgraded = sum(r.precision_upgraded for r in mode_results)
            print(f"  Precision upgrades:  {upgraded}/{len(mode_results)}")


def main():
    """Run mixed precision benchmarks."""
    print("NLSQ Mixed Precision Performance Benchmarking")
    print("=" * 80)

    config = BenchmarkConfig(
        problem_sizes=[100, 1000, 10000],
        n_repeats=3,
        warmup_runs=1,
        include_float32=True,
        include_float64=True,
        include_mixed=True,
    )

    results = run_benchmark_suite(config)
    print_summary(results)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
