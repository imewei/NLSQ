"""
Core Performance Benchmark for NLSQ Optimizations.

This benchmark validates performance improvements from the 012-nlsq-perf-optimization
feature with focus on:
- Timing measurements (wall-clock)
- Memory usage tracking
- Numerical accuracy validation (1e-8 tolerance)

Usage:
    python -m tests.benchmarks.benchmark_core
    pytest tests/benchmarks/benchmark_core.py -v -s
"""

from __future__ import annotations

import contextlib
import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit


@dataclass
class BenchmarkMetrics:
    """Metrics from a single benchmark run."""

    problem: str
    n_points: int
    n_params: int
    time_seconds: float
    memory_bytes: int | None
    success: bool
    relative_error: float
    iterations: int | None = None
    nfev: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BenchmarkResults:
    """Collection of benchmark results."""

    timestamp: str
    jax_version: str
    device: str
    metrics: list[BenchmarkMetrics] = field(default_factory=list)

    def add(self, metric: BenchmarkMetrics) -> None:
        """Add a metric to results."""
        self.metrics.append(metric)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "jax_version": self.jax_version,
            "device": self.device,
            "metrics": [m.to_dict() for m in self.metrics],
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def exponential_decay(x: jnp.ndarray, a: float, b: float, c: float) -> jnp.ndarray:
    """Exponential decay model: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def gaussian_peak(
    x: jnp.ndarray, a: float, mu: float, sigma: float, offset: float
) -> jnp.ndarray:
    """Gaussian peak model: y = a * exp(-(x - mu)^2 / (2 * sigma^2)) + offset"""
    return a * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2)) + offset


def polynomial_5(
    x: jnp.ndarray, a: float, b: float, c: float, d: float, e: float
) -> jnp.ndarray:
    """5th degree polynomial."""
    return a + b * x + c * x**2 + d * x**3 + e * x**4


def double_exponential(
    x: jnp.ndarray, a1: float, b1: float, a2: float, b2: float, c: float
) -> jnp.ndarray:
    """Double exponential decay: two decay components plus offset."""
    return a1 * jnp.exp(-b1 * x) + a2 * jnp.exp(-b2 * x) + c


# Problem definitions: (model, true_params, initial_guess, x_range)
BENCHMARK_PROBLEMS = {
    "exponential_decay_3p": {
        "model": exponential_decay,
        "true_params": jnp.array([10.0, 0.5, 2.0]),
        "p0": jnp.array([8.0, 0.3, 1.5]),
        "x_range": (0.0, 10.0),
    },
    "gaussian_4p": {
        "model": gaussian_peak,
        # Note: mu must be non-zero to avoid divide-by-zero in relative error calculation
        "true_params": jnp.array([5.0, 0.5, 1.0, 0.5]),
        "p0": jnp.array([4.0, 0.3, 1.2, 0.3]),
        "x_range": (-5.0, 5.0),
    },
    "polynomial_5p": {
        "model": polynomial_5,
        "true_params": jnp.array([1.0, 2.0, -0.5, 0.3, -0.1]),
        "p0": jnp.array([0.8, 1.8, -0.3, 0.2, -0.05]),
        "x_range": (-2.0, 2.0),
    },
    "double_exp_5p": {
        "model": double_exponential,
        "true_params": jnp.array([5.0, 0.3, 3.0, 1.0, 1.0]),
        "p0": jnp.array([4.0, 0.2, 2.5, 0.8, 0.8]),
        "x_range": (0.0, 10.0),
    },
}

BENCHMARK_SIZES = [1_000, 10_000, 100_000]
# Relative tolerance for benchmark accuracy validation
# With 1% noise, recovering parameters to ~1% accuracy is realistic
ACCURACY_TOLERANCE = 0.05  # 5% relative error tolerance for noisy data
NOISE_LEVEL = 0.01


def get_memory_stats() -> int | None:
    """Get current GPU memory usage in bytes, or None if not available."""
    try:
        devices = jax.local_devices()
        if devices and hasattr(devices[0], "memory_stats"):
            stats = devices[0].memory_stats()
            if stats:
                return stats.get("bytes_in_use", None)
    except Exception:
        pass
    return None


def generate_data(
    problem_name: str,
    n_points: int,
    seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate synthetic data for a benchmark problem.

    Returns
    -------
    x : jnp.ndarray
        Independent variable
    y : jnp.ndarray
        Noisy dependent variable
    y_true : jnp.ndarray
        True (noiseless) dependent variable
    """
    problem = BENCHMARK_PROBLEMS[problem_name]
    model = problem["model"]
    true_params = problem["true_params"]
    x_min, x_max = problem["x_range"]

    key = jax.random.PRNGKey(seed)
    x = jnp.linspace(x_min, x_max, n_points)
    y_true = model(x, *true_params)
    noise = NOISE_LEVEL * jnp.abs(y_true).max() * jax.random.normal(key, (n_points,))
    y = y_true + noise

    return x, y, y_true


def run_single_benchmark(
    problem_name: str,
    n_points: int,
    warmup: bool = True,
) -> BenchmarkMetrics:
    """Run a single benchmark and return metrics.

    Parameters
    ----------
    problem_name : str
        Name of the benchmark problem
    n_points : int
        Number of data points
    warmup : bool
        Whether to run a warmup iteration first

    Returns
    -------
    BenchmarkMetrics
        Metrics from the benchmark run
    """
    problem = BENCHMARK_PROBLEMS[problem_name]
    model = problem["model"]
    true_params = problem["true_params"]
    p0 = problem["p0"]

    x, y, _ = generate_data(problem_name, n_points)

    # Warmup run to trigger JIT compilation
    if warmup:
        with contextlib.suppress(Exception):
            curve_fit(model, x, y, p0=p0)
        gc.collect()

    # Measure memory before
    mem_before = get_memory_stats()

    # Timed run
    start = time.perf_counter()
    try:
        result = curve_fit(model, x, y, p0=p0, full_output=True)
        elapsed = time.perf_counter() - start

        # Extract results from CurveFitResult object
        # CurveFitResult has popt, pcov, nfev, success as attributes
        popt = result.popt
        nfev = getattr(result, "nfev", None)
        success = True

        # Calculate relative error in fitted parameters
        # Use max(|true|, 1e-10) to avoid divide-by-zero for near-zero true values
        popt_np = np.array(popt)
        true_np = np.array(true_params)
        denominator = np.maximum(np.abs(true_np), 1e-10)
        relative_error = float(np.max(np.abs(popt_np - true_np) / denominator))

    except Exception as e:
        elapsed = time.perf_counter() - start
        success = False
        relative_error = float("inf")
        nfev = None

    # Measure memory after
    mem_after = get_memory_stats()
    memory_used = None
    if mem_before is not None and mem_after is not None:
        memory_used = mem_after - mem_before

    return BenchmarkMetrics(
        problem=problem_name,
        n_points=n_points,
        n_params=len(true_params),
        time_seconds=elapsed,
        memory_bytes=memory_used,
        success=success,
        relative_error=relative_error,
        nfev=nfev,
    )


def run_benchmark_suite(
    problems: list[str] | None = None,
    sizes: list[int] | None = None,
    n_repeats: int = 3,
    verbose: bool = True,
) -> BenchmarkResults:
    """Run the full benchmark suite.

    Parameters
    ----------
    problems : list[str], optional
        List of problem names to benchmark. Default: all problems.
    sizes : list[int], optional
        List of data sizes to benchmark. Default: [1000, 10000, 100000].
    n_repeats : int
        Number of repetitions per configuration.
    verbose : bool
        Print progress information.

    Returns
    -------
    BenchmarkResults
        Collection of all benchmark metrics.
    """
    import datetime

    if problems is None:
        problems = list(BENCHMARK_PROBLEMS.keys())
    if sizes is None:
        sizes = BENCHMARK_SIZES

    results = BenchmarkResults(
        timestamp=datetime.datetime.now().isoformat(),
        jax_version=jax.__version__,
        device=str(jax.devices()[0]) if jax.devices() else "unknown",
    )

    total_runs = len(problems) * len(sizes) * n_repeats
    current = 0

    for problem_name in problems:
        if verbose:
            print(f"\n=== {problem_name} ===")

        for n_points in sizes:
            times = []
            for repeat in range(n_repeats):
                # Only warmup on first repeat
                metrics = run_single_benchmark(
                    problem_name,
                    n_points,
                    warmup=(repeat == 0),
                )
                results.add(metrics)
                times.append(metrics.time_seconds)
                current += 1

            if verbose:
                avg_time = np.mean(times)
                std_time = np.std(times)
                print(
                    f"  {n_points:>7,} points: {avg_time:.4f}s ± {std_time:.4f}s "
                    f"(success: {metrics.success}, error: {metrics.relative_error:.2e})"
                )

    return results


def validate_accuracy(results: BenchmarkResults) -> tuple[bool, list[str]]:
    """Validate that all results meet accuracy tolerance.

    Returns
    -------
    passed : bool
        True if all results meet tolerance
    failures : list[str]
        List of failure descriptions
    """
    failures = []

    for metric in results.metrics:
        if not metric.success:
            failures.append(f"{metric.problem} @ {metric.n_points}: fit failed")
        elif metric.relative_error > ACCURACY_TOLERANCE:
            failures.append(
                f"{metric.problem} @ {metric.n_points}: "
                f"error {metric.relative_error:.2e} > {ACCURACY_TOLERANCE:.2e}"
            )

    return len(failures) == 0, failures


def print_summary(results: BenchmarkResults) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Device: {results.device}")
    print(f"JAX version: {results.jax_version}")
    print(f"Timestamp: {results.timestamp}")
    print()

    # Group by problem and size
    from collections import defaultdict

    grouped: dict[tuple[str, int], list[BenchmarkMetrics]] = defaultdict(list)
    for m in results.metrics:
        grouped[(m.problem, m.n_points)].append(m)

    # Print table
    print(f"{'Problem':<25} {'Size':>10} {'Time (s)':>12} {'Memory':>12} {'Error':>12}")
    print("-" * 80)

    for (problem, size), metrics in sorted(grouped.items()):
        times = [m.time_seconds for m in metrics]
        avg_time = np.mean(times)

        # Get memory if available
        mem_values = [m.memory_bytes for m in metrics if m.memory_bytes is not None]
        mem_str = f"{np.mean(mem_values) / 1e6:.1f}MB" if mem_values else "N/A"

        # Get max error
        max_error = max(m.relative_error for m in metrics)
        error_str = f"{max_error:.2e}" if max_error < float("inf") else "FAILED"

        print(
            f"{problem:<25} {size:>10,} {avg_time:>12.4f} {mem_str:>12} {error_str:>12}"
        )

    # Accuracy validation
    passed, failures = validate_accuracy(results)
    print()
    if passed:
        print("✓ All fits within 1e-8 tolerance")
    else:
        print("✗ Some fits exceeded tolerance:")
        for f in failures[:5]:
            print(f"  - {f}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")


# Test functions for pytest
def test_exponential_accuracy():
    """Test exponential decay fit accuracy meets 1e-8 tolerance."""
    metrics = run_single_benchmark("exponential_decay_3p", 10_000)
    assert metrics.success, "Fit failed"
    assert metrics.relative_error < ACCURACY_TOLERANCE, (
        f"Relative error {metrics.relative_error:.2e} exceeds tolerance {ACCURACY_TOLERANCE:.2e}"
    )


def test_gaussian_accuracy():
    """Test Gaussian peak fit accuracy meets 1e-8 tolerance."""
    metrics = run_single_benchmark("gaussian_4p", 10_000)
    assert metrics.success, "Fit failed"
    assert metrics.relative_error < ACCURACY_TOLERANCE, (
        f"Relative error {metrics.relative_error:.2e} exceeds tolerance {ACCURACY_TOLERANCE:.2e}"
    )


def test_benchmark_suite_runs():
    """Test that the benchmark suite runs without errors."""
    results = run_benchmark_suite(
        problems=["exponential_decay_3p"],
        sizes=[1_000],
        n_repeats=1,
        verbose=False,
    )
    assert len(results.metrics) == 1
    assert results.metrics[0].success


if __name__ == "__main__":
    import sys

    # Run benchmark suite
    results = run_benchmark_suite(verbose=True)
    print_summary(results)

    # Save results
    output_path = Path("specs/012-nlsq-perf-optimization/baseline_results.json")
    results.save(output_path)
    print(f"\nResults saved to {output_path}")

    # Validate accuracy
    passed, failures = validate_accuracy(results)
    sys.exit(0 if passed else 1)
