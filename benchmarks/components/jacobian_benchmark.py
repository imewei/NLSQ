#!/usr/bin/env python
"""Benchmark Jacobian mode auto-switching performance.

This benchmark measures the performance difference between jacfwd and jacrev
on high-parameter problems to validate the 10-100x speedup target.

Two benchmark modes:
- direct: Measures JAX jacfwd/jacrev performance directly
- integrated: Measures performance through curve_fit with jacobian_mode parameter

Target Impact: 10-100x Jacobian time reduction on high-parameter problems

Usage:
    python benchmarks/components/jacobian_benchmark.py              # Both modes
    python benchmarks/components/jacobian_benchmark.py --mode=direct
    python benchmarks/components/jacobian_benchmark.py --mode=integrated
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit

from nlsq import curve_fit

# =============================================================================
# Test Problem Creation
# =============================================================================


def create_test_function(n_params: int):
    """Create a JIT-compiled test function for direct Jacobian benchmarking."""

    @jit
    def func(x, params):
        """Sum of exponentials with many parameters."""
        result = jnp.zeros_like(x)
        for i in range(0, len(params), 2):
            if i + 1 < len(params):
                a = params[i]
                b = params[i + 1]
                result = result + a * jnp.exp(-b * x)
        return result

    return func


def create_high_param_problem(n_params: int, n_data: int):
    """Create a high-parameter curve fitting problem for integrated benchmarking.

    Parameters
    ----------
    n_params : int
        Number of parameters (e.g., 1000)
    n_data : int
        Number of data points (e.g., 100)

    Returns
    -------
    model, xdata, ydata, p0
    """

    def model(x, *params):
        """Sum of exponentials model with many parameters."""
        result = jnp.zeros_like(x)
        for i in range(0, len(params), 2):
            if i + 1 < len(params):
                a = params[i]
                b = params[i + 1]
                result = result + a * jnp.exp(-b * x)
            else:
                result = result + params[i]
        return result

    np.random.seed(42)
    xdata = np.linspace(0, 5, n_data)
    p_true = np.random.randn(n_params) * 0.1
    ydata = np.array(model(xdata, *p_true))
    ydata += np.random.randn(n_data) * 0.01
    p0 = np.ones(n_params) * 0.1

    return model, xdata, ydata, p0


# =============================================================================
# Direct JAX Benchmarking
# =============================================================================


def benchmark_jacobian_direct(
    n_params: int,
    n_data: int,
    n_trials: int = 5,
) -> dict:
    """Benchmark jacfwd vs jacrev directly using JAX.

    This measures the raw Jacobian computation time without optimization overhead.
    """
    print(f"\nDirect Jacobian Benchmark: {n_params} params, {n_data} data points")
    print("=" * 70)

    func = create_test_function(n_params)
    x = jnp.linspace(0, 5, n_data)
    params = jnp.ones(n_params) * 0.1

    @jit
    def residual_func(p):
        return func(x, p)

    jac_fwd = jit(jacfwd(residual_func))
    jac_rev = jit(jacrev(residual_func))

    print("\nWarming up (JIT compilation)...")
    _ = jac_fwd(params).block_until_ready()
    _ = jac_rev(params).block_until_ready()
    print("Warmup complete.")

    print(f"\nBenchmarking jacfwd ({n_trials} trials)...")
    times_fwd = []
    for i in range(n_trials):
        start = time.perf_counter()
        _ = jac_fwd(params).block_until_ready()
        elapsed = time.perf_counter() - start
        times_fwd.append(elapsed * 1000)
        print(f"  Trial {i + 1}: {elapsed * 1000:.4f} ms")

    print(f"\nBenchmarking jacrev ({n_trials} trials)...")
    times_rev = []
    for i in range(n_trials):
        start = time.perf_counter()
        _ = jac_rev(params).block_until_ready()
        elapsed = time.perf_counter() - start
        times_rev.append(elapsed * 1000)
        print(f"  Trial {i + 1}: {elapsed * 1000:.4f} ms")

    J_fwd_final = jac_fwd(params)
    J_rev_final = jac_rev(params)
    max_diff = float(jnp.max(jnp.abs(J_fwd_final - J_rev_final)))
    print(f"\nNumerical accuracy: max difference = {max_diff:.2e}")

    avg_fwd = float(np.mean(times_fwd))
    avg_rev = float(np.mean(times_rev))
    std_fwd = float(np.std(times_fwd))
    std_rev = float(np.std(times_rev))
    speedup = avg_fwd / avg_rev if avg_rev > 0 else 0

    print("\n" + "=" * 70)
    print("Results:")
    print(f"  jacfwd:  {avg_fwd:.4f} +/- {std_fwd:.4f} ms")
    print(f"  jacrev:  {avg_rev:.4f} +/- {std_rev:.4f} ms")
    print(f"  Speedup: {speedup:.1f}x (jacfwd / jacrev)")

    if speedup >= 10:
        print(f"  Target achieved: {speedup:.1f}x >= 10x")
    elif speedup >= 5:
        print(f"  Partial success: {speedup:.1f}x >= 5x")
    else:
        print(f"  Target missed: {speedup:.1f}x < 10x")

    return {
        "mode": "direct",
        "n_params": n_params,
        "n_data": n_data,
        "times_fwd_ms": times_fwd,
        "times_rev_ms": times_rev,
        "avg_fwd_ms": avg_fwd,
        "avg_rev_ms": avg_rev,
        "std_fwd_ms": std_fwd,
        "std_rev_ms": std_rev,
        "speedup": speedup,
        "max_diff": max_diff,
    }


# =============================================================================
# Integrated curve_fit Benchmarking
# =============================================================================


def benchmark_jacobian_integrated(
    n_params: int,
    n_data: int,
    n_trials: int = 3,
) -> dict:
    """Benchmark jacobian modes through curve_fit integration.

    This measures real-world performance including optimization overhead.
    """
    print(f"\nIntegrated Benchmark: {n_params} params, {n_data} data points")
    print("=" * 70)

    model, xdata, ydata, p0 = create_high_param_problem(n_params, n_data)

    print("\n1. Testing jacfwd (forward-mode AD)...")
    times_fwd = []
    for trial in range(n_trials):
        start = time.perf_counter()
        try:
            _ = curve_fit(
                model,
                xdata,
                ydata,
                p0=p0,
                jacobian_mode="fwd",
                max_nfev=10,
            )
            elapsed = time.perf_counter() - start
            times_fwd.append(elapsed)
            print(f"   Trial {trial + 1}: {elapsed:.4f}s")
        except Exception as e:
            print(f"   Trial {trial + 1} failed: {e}")
            times_fwd.append(float("inf"))

    print("\n2. Testing jacrev (reverse-mode AD)...")
    times_rev = []
    for trial in range(n_trials):
        start = time.perf_counter()
        try:
            _ = curve_fit(
                model,
                xdata,
                ydata,
                p0=p0,
                jacobian_mode="rev",
                max_nfev=10,
            )
            elapsed = time.perf_counter() - start
            times_rev.append(elapsed)
            print(f"   Trial {trial + 1}: {elapsed:.4f}s")
        except Exception as e:
            print(f"   Trial {trial + 1} failed: {e}")
            times_rev.append(float("inf"))

    print("\n3. Testing auto mode (should select jacrev)...")
    times_auto = []
    for trial in range(n_trials):
        start = time.perf_counter()
        try:
            _ = curve_fit(
                model,
                xdata,
                ydata,
                p0=p0,
                jacobian_mode="auto",
                max_nfev=10,
            )
            elapsed = time.perf_counter() - start
            times_auto.append(elapsed)
            print(f"   Trial {trial + 1}: {elapsed:.4f}s")
        except Exception as e:
            print(f"   Trial {trial + 1} failed: {e}")
            times_auto.append(float("inf"))

    avg_fwd = float(np.mean(times_fwd)) if any(np.isfinite(times_fwd)) else float("inf")
    avg_rev = float(np.mean(times_rev)) if any(np.isfinite(times_rev)) else float("inf")
    avg_auto = (
        float(np.mean(times_auto)) if any(np.isfinite(times_auto)) else float("inf")
    )
    speedup = avg_fwd / avg_rev if avg_rev > 0 else 0

    print("\n" + "=" * 70)
    print("Results:")
    print(f"  jacfwd:  {avg_fwd:.4f}s (average)")
    print(f"  jacrev:  {avg_rev:.4f}s (average)")
    print(f"  auto:    {avg_auto:.4f}s (average)")
    print(f"  Speedup: {speedup:.1f}x (jacfwd / jacrev)")

    if speedup >= 10:
        print(f"  Target achieved: {speedup:.1f}x >= 10x")
    else:
        print(f"  Target missed: {speedup:.1f}x < 10x")

    return {
        "mode": "integrated",
        "n_params": n_params,
        "n_data": n_data,
        "times_fwd": times_fwd,
        "times_rev": times_rev,
        "times_auto": times_auto,
        "avg_fwd": avg_fwd,
        "avg_rev": avg_rev,
        "avg_auto": avg_auto,
        "speedup": speedup,
    }


# =============================================================================
# Benchmark Suite
# =============================================================================


def run_benchmark_suite(mode: str = "both") -> dict:
    """Run complete Jacobian mode benchmark suite."""
    print("=" * 70)
    print("Jacobian Auto-Switch Benchmark Suite")
    print("=" * 70)

    test_cases = [
        (1000, 100, "10-100x"),
        (500, 100, "5-50x"),
        (200, 100, "2-10x"),
        (100, 100, "~1x"),
    ]

    all_results = []

    for n_params, n_data, expected in test_cases:
        print(f"\n\nTest Case: {n_params} params, {n_data} data (expected: {expected})")

        if mode in ("direct", "both"):
            results = benchmark_jacobian_direct(n_params, n_data, n_trials=5)
            all_results.append(results)

        if mode in ("integrated", "both"):
            results = benchmark_jacobian_integrated(n_params, n_data, n_trials=3)
            all_results.append(results)

    baseline_data = {
        "test_cases": all_results,
        "platform": "linux",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "best_speedup": max(
                r["speedup"] for r in all_results if np.isfinite(r["speedup"])
            ),
            "target_achieved": any(
                r["speedup"] >= 10 for r in all_results if np.isfinite(r["speedup"])
            ),
        },
    }

    return baseline_data


def main() -> int:
    """Run Jacobian benchmarks and save results."""
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser(description="Jacobian mode benchmark")
    parser.add_argument(
        "--mode",
        choices=["direct", "integrated", "both"],
        default="both",
        help="Benchmark mode: direct (JAX), integrated (curve_fit), or both",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    baseline = run_benchmark_suite(mode=args.mode)

    baseline_dir = Path(__file__).parent.parent / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    baseline_file = args.output or (baseline_dir / "jacobian_autoswitch.json")

    with open(baseline_file, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\n\n{'=' * 70}")
    print("Benchmark Complete")
    print(f"{'=' * 70}")
    print(f"Baseline saved to: {baseline_file}")
    print(f"Best speedup: {baseline['summary']['best_speedup']:.1f}x")
    print(f"Target (10x) achieved: {baseline['summary']['target_achieved']}")

    print("\n\nSummary:")
    print("-" * 70)
    for result in baseline["test_cases"]:
        mode_str = f"[{result['mode']}]"
        status = (
            "PASS"
            if result["speedup"] >= 10
            else ("PARTIAL" if result["speedup"] >= 5 else "FAIL")
        )
        print(
            f"{status} {mode_str:12s} {result['n_params']:4d} params / "
            f"{result['n_data']:3d} data: {result['speedup']:5.1f}x speedup"
        )

    return 0 if baseline["summary"]["target_achieved"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
