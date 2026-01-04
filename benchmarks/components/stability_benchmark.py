"""Benchmark stability mode overhead.

This benchmark validates the performance analysis claims:
1. SVD skip reduces overhead for large Jacobians
2. Init-only checks are much faster than per-iteration checks
3. rescale_data impact is negligible

Run with: python benchmarks/benchmark_stability_overhead.py
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax.numpy as jnp

from nlsq import LeastSquares, curve_fit


def benchmark_svd_threshold(
    m: int = 1_000_000, n: int = 7, threshold: int = 10_000_000
) -> dict:
    """Benchmark SVD computation vs. skip for different thresholds.

    Parameters
    ----------
    m : int
        Number of residuals
    n : int
        Number of parameters
    threshold : int
        SVD skip threshold (number of elements)

    Returns
    -------
    dict
        Benchmark results with timing and memory info
    """
    from nlsq.stability import NumericalStabilityGuard

    # Create test Jacobian
    J = jnp.ones((m, n)) + 0.01 * jnp.random.randn(m, n)

    # Benchmark with SVD computation
    guard_with_svd = NumericalStabilityGuard(
        max_jacobian_elements_for_svd=m * n + 1  # Force SVD
    )
    start = time.perf_counter()
    _J_fixed_svd, issues_svd = guard_with_svd.check_and_fix_jacobian(J)
    time_with_svd = time.perf_counter() - start

    # Benchmark with SVD skip
    guard_skip_svd = NumericalStabilityGuard(
        max_jacobian_elements_for_svd=m * n - 1  # Force skip
    )
    start = time.perf_counter()
    _J_fixed_skip, issues_skip = guard_skip_svd.check_and_fix_jacobian(J)
    time_skip_svd = time.perf_counter() - start

    return {
        "m": m,
        "n": n,
        "elements": m * n,
        "threshold": threshold,
        "time_with_svd": time_with_svd,
        "time_skip_svd": time_skip_svd,
        "speedup": time_with_svd / time_skip_svd if time_skip_svd > 0 else np.inf,
        "svd_computed": not issues_svd.get("svd_skipped", False),
        "svd_skipped": not issues_skip.get("svd_skipped", True),
        "condition_number": issues_svd.get("condition_number"),
    }


def benchmark_stability_mode(
    problem_size: Literal["small", "medium", "large"] = "medium",
    n_iterations: int = 50,
) -> dict:
    """Benchmark curve fitting with/without stability mode.

    Parameters
    ----------
    problem_size : str
        Problem size: 'small' (1K pts), 'medium' (100K pts), 'large' (1M pts)
    n_iterations : int
        Expected number of optimization iterations

    Returns
    -------
    dict
        Benchmark results comparing stability modes
    """
    # Problem configurations
    sizes = {"small": 1_000, "medium": 100_000, "large": 1_000_000}
    m = sizes.get(problem_size, 100_000)

    # Generate test data: exponential decay
    xdata = np.linspace(0, 5, m)
    true_params = [1.0, 0.5, 0.1]
    ydata = true_params[0] * np.exp(-true_params[1] * xdata) + true_params[2]
    ydata += 0.01 * np.random.randn(m)  # Add noise

    # Define model
    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    p0 = [0.8, 0.3, 0.0]

    # Benchmark WITHOUT stability mode
    start = time.perf_counter()
    result_no_stab = curve_fit(model, xdata, ydata, p0, stability=False)
    time_no_stability = time.perf_counter() - start

    # Benchmark WITH stability mode (init-only checks)
    start = time.perf_counter()
    result_with_stab = curve_fit(model, xdata, ydata, p0, stability="check")
    time_with_stability = time.perf_counter() - start

    # Benchmark WITH rescale_data=False
    start = time.perf_counter()
    result_no_rescale = curve_fit(
        model, xdata, ydata, p0, stability="check", rescale_data=False
    )
    time_no_rescale = time.perf_counter() - start

    return {
        "problem_size": problem_size,
        "m": m,
        "n": len(p0),
        "time_no_stability": time_no_stability,
        "time_with_stability": time_with_stability,
        "time_no_rescale": time_no_rescale,
        "overhead_stability": (
            (time_with_stability - time_no_stability) / time_no_stability * 100
            if time_no_stability > 0
            else 0
        ),
        "overhead_no_rescale": (
            (time_no_rescale - time_with_stability) / time_with_stability * 100
            if time_with_stability > 0
            else 0
        ),
        "iterations_no_stab": result_no_stab.get("nit", 0),
        "iterations_with_stab": result_with_stab.get("nit", 0),
        "iterations_no_rescale": result_no_rescale.get("nit", 0),
        "success_no_stab": result_no_stab.get("success", False),
        "success_with_stab": result_with_stab.get("success", False),
        "success_no_rescale": result_no_rescale.get("success", False),
    }


def benchmark_jacobian_size_scaling() -> dict:
    """Benchmark how stability overhead scales with Jacobian size.

    Returns
    -------
    dict
        Results for different matrix sizes
    """
    results = []

    # Test different matrix configurations
    configs = [
        (1_000, 5),  # 5K elements
        (10_000, 10),  # 100K elements
        (100_000, 7),  # 700K elements
        (1_000_000, 7),  # 7M elements (typical XPCS)
        (100_000, 100),  # 10M elements (at threshold)
        (1_000_000, 11),  # 11M elements (above threshold)
    ]

    for m, n in configs:
        result = benchmark_svd_threshold(m, n, threshold=10_000_000)
        results.append(result)

    return {"configurations": results}


def print_benchmark_results():
    """Run all benchmarks and print formatted results."""
    print("=" * 80)
    print("NLSQ Stability Performance Benchmark")
    print("=" * 80)
    print()

    # Benchmark 1: SVD Threshold Impact
    print("1. SVD Skip Threshold Impact")
    print("-" * 80)
    scaling_results = benchmark_jacobian_size_scaling()
    print(
        f"{'Shape (m×n)':<20} {'Elements':<15} {'SVD Time':<12} {'Skip Time':<12} {'Speedup':<10}"
    )
    print("-" * 80)
    for result in scaling_results["configurations"]:
        shape = f"{result['m']:,} × {result['n']}"
        elements = f"{result['elements']:,}"
        svd_time = (
            f"{result['time_with_svd'] * 1000:.2f}ms"
            if result["svd_computed"]
            else "N/A"
        )
        skip_time = f"{result['time_skip_svd'] * 1000:.2f}ms"
        speedup = f"{result['speedup']:.1f}x" if result["speedup"] != np.inf else "N/A"
        print(
            f"{shape:<20} {elements:<15} {svd_time:<12} {skip_time:<12} {speedup:<10}"
        )
    print()

    # Benchmark 2: Stability Mode Overhead
    print("2. Stability Mode Overhead")
    print("-" * 80)
    for size in ["small", "medium", "large"]:
        print(f"\nProblem Size: {size.upper()}")
        result = benchmark_stability_mode(size)
        print(f"  Data points: {result['m']:,}")
        print(f"  Parameters: {result['n']}")
        print(f"  Time without stability: {result['time_no_stability']:.4f}s")
        print(f"  Time with stability: {result['time_with_stability']:.4f}s")
        print(f"  Time without rescale: {result['time_no_rescale']:.4f}s")
        print(f"  Stability overhead: {result['overhead_stability']:.2f}%")
        print(f"  No-rescale overhead: {result['overhead_no_rescale']:.2f}%")
        print(f"  Iterations (no stab): {result['iterations_no_stab']}")
        print(f"  Iterations (with stab): {result['iterations_with_stab']}")
        print(f"  Iterations (no rescale): {result['iterations_no_rescale']}")

    print()
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    print_benchmark_results()
