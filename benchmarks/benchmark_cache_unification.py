#!/usr/bin/env python
"""Benchmark for unified cache performance (Task 1.12).

This script measures cache performance including:
- Cold JIT compilation time
- Warm JIT time (cache hits)
- Cache hit rate across batch processing
- Compilation time distribution

Target Performance Metrics:
- Cache hit rate: >80% on batch fitting workflows
- Cold JIT time reduction: 2-5x improvement through better cache reuse
- Warm JIT time: <2ms (cached compilation)

Usage:
    python benchmarks/benchmark_cache_unification.py
"""

import json
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nlsq import curve_fit
from nlsq.unified_cache import clear_cache, get_cache_stats


def exponential_model(x, a, b, c):
    """Exponential model for benchmarking."""
    return a * jnp.exp(-b * x) + c


def benchmark_cold_jit_time(data_sizes=(100, 1000, 10000)):
    """Measure cold JIT compilation time for different data sizes.

    Returns
    -------
    dict
        Cold JIT times in milliseconds (p50, p95)
    """
    times = []

    for size in data_sizes:
        # Clear cache to ensure cold JIT
        clear_cache()

        # Generate data
        x = np.linspace(0, 10, size)
        y_true = 2.5 * np.exp(-0.5 * x) + 1.0
        y = y_true + np.random.normal(0, 0.1, size)

        # Time cold JIT compilation
        start = time.time()
        result = curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])
        _, _ = result
        cold_time_ms = (time.time() - start) * 1000

        times.append(cold_time_ms)
        print(f"  Size {size}: {cold_time_ms:.1f} ms (cold JIT)")

    times = np.array(times)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "mean": float(np.mean(times)),
        "data_sizes": list(data_sizes),
        "all_times": times.tolist(),
    }


def benchmark_warm_jit_time(n_iterations=100):
    """Measure warm JIT time (cache hits).

    Parameters
    ----------
    n_iterations : int
        Number of iterations to run

    Returns
    -------
    dict
        Warm JIT times in milliseconds (p50, p95)
    """
    # Clear cache and run one fit to warm up
    clear_cache()
    x = np.linspace(0, 10, 1000)
    y = 2.5 * np.exp(-0.5 * x) + 1.0 + np.random.normal(0, 0.1, 1000)
    _ = curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])

    # Measure warm times
    times = []
    for _ in range(n_iterations):
        x = np.linspace(0, 10, 1000)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + np.random.normal(0, 0.1, 1000)

        start = time.time()
        result = curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])
        _, _ = result
        warm_time_ms = (time.time() - start) * 1000
        times.append(warm_time_ms)

    times = np.array(times)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "mean": float(np.mean(times)),
        "n_iterations": n_iterations,
    }


def benchmark_cache_hit_rate(n_fits=1000, data_sizes=(100, 200, 500, 1000)):
    """Measure cache hit rate on batch processing workflow.

    Simulates typical batch fitting: same model, varying data sizes.

    Parameters
    ----------
    n_fits : int
        Number of fits to perform
    data_sizes : tuple
        Possible data sizes to randomly sample from

    Returns
    -------
    dict
        Cache statistics including hit rate
    """
    # Clear cache
    clear_cache()

    # Run batch processing
    for i in range(n_fits):
        size = np.random.choice(data_sizes)
        x = np.linspace(0, 10, size)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + np.random.normal(0, 0.1, size)

        result = curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])
        _, _ = result

        if (i + 1) % 100 == 0:
            stats = get_cache_stats()
            print(
                f"  Processed {i + 1}/{n_fits} fits, hit_rate={stats['hit_rate']:.2%}"
            )

    # Get final stats
    final_stats = get_cache_stats()

    return {
        "hit_rate": float(final_stats["hit_rate"]),
        "total_hits": int(final_stats["hits"]),
        "total_misses": int(final_stats["misses"]),
        "total_compilations": int(final_stats["compilations"]),
        "cache_size": int(final_stats["cache_size"]),
        "total_requests": int(final_stats["total_requests"]),
        "n_fits": n_fits,
    }


def calculate_speedup_factor(cold_time, warm_time):
    """Calculate speedup from warm (cached) vs cold execution.

    Parameters
    ----------
    cold_time : dict
        Cold JIT timing results
    warm_time : dict
        Warm JIT timing results

    Returns
    -------
    float
        Speedup factor (cold_time / warm_time)
    """
    cold_p50 = cold_time["p50"]
    warm_p50 = warm_time["p50"]

    if warm_p50 > 0:
        return cold_p50 / warm_p50
    else:
        return 0.0


def main():
    """Run comprehensive cache benchmarks and store results."""
    print("=" * 70)
    print("Unified Cache Performance Benchmark (Task 1.12)")
    print("=" * 70)
    print()

    # System info
    print("System Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  JAX: {jax.__version__}")
    print(f"  Device: {jax.devices()[0].device_kind}")
    print()

    # Benchmark 1: Cold JIT time
    print("Benchmark 1: Cold JIT Compilation Time")
    print("-" * 70)
    cold_time = benchmark_cold_jit_time()
    print(f"  P50: {cold_time['p50']:.1f} ms")
    print(f"  P95: {cold_time['p95']:.1f} ms")
    print(f"  Mean: {cold_time['mean']:.1f} ms")
    print()

    # Benchmark 2: Warm JIT time
    print("Benchmark 2: Warm JIT Time (Cache Hits)")
    print("-" * 70)
    warm_time = benchmark_warm_jit_time(n_iterations=100)
    print(f"  P50: {warm_time['p50']:.2f} ms")
    print(f"  P95: {warm_time['p95']:.2f} ms")
    print(f"  Mean: {warm_time['mean']:.2f} ms")
    print()

    # Calculate speedup
    speedup = calculate_speedup_factor(cold_time, warm_time)
    print(f"  Speedup Factor (Cold/Warm): {speedup:.1f}x")
    print()

    # Benchmark 3: Cache hit rate
    print("Benchmark 3: Cache Hit Rate (1000 fits, varying sizes)")
    print("-" * 70)
    cache_stats = benchmark_cache_hit_rate(n_fits=1000)
    print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
    print(f"  Total Hits: {cache_stats['total_hits']}")
    print(f"  Total Misses: {cache_stats['total_misses']}")
    print(f"  Compilations: {cache_stats['total_compilations']}")
    print(f"  Cache Size: {cache_stats['cache_size']}")
    print()

    # Validation against targets
    print("Target Validation:")
    print("-" * 70)

    target_hit_rate = 0.80
    hit_rate_pass = cache_stats["hit_rate"] >= target_hit_rate
    print(
        f"  Cache Hit Rate: {cache_stats['hit_rate']:.2%} (target: >80%) {'✓' if hit_rate_pass else '✗'}"
    )

    target_speedup = 2.0
    speedup_pass = speedup >= target_speedup
    print(
        f"  Speedup Factor: {speedup:.1f}x (target: 2-5x) {'✓' if speedup_pass else '✗'}"
    )

    target_warm_time = 2.0  # ms
    warm_time_pass = warm_time["p50"] <= target_warm_time
    print(
        f"  Warm JIT Time: {warm_time['p50']:.2f} ms (target: <2ms) {'✓' if warm_time_pass else '✗'}"
    )
    print()

    # Store results
    baseline_dir = Path(__file__).parent / "baselines"
    baseline_dir.mkdir(exist_ok=True)

    baseline_file = baseline_dir / "cache_unification.json"

    results = {
        "platform": platform.system().lower(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "device_kind": jax.devices()[0].device_kind,
        "cold_jit_time_ms": cold_time,
        "warm_jit_time_ms": warm_time,
        "cache_hit_rate": cache_stats["hit_rate"],
        "speedup_factor": speedup,
        "cache_statistics": cache_stats,
        "targets_met": {
            "hit_rate": hit_rate_pass,
            "speedup": speedup_pass,
            "warm_time": warm_time_pass,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(baseline_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {baseline_file}")
    print()

    # Summary
    all_passed = all(results["targets_met"].values())
    print("=" * 70)
    if all_passed:
        print("✓ All performance targets met!")
    else:
        print("✗ Some performance targets not met")
        failed = [k for k, v in results["targets_met"].items() if not v]
        print(f"  Failed: {', '.join(failed)}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
