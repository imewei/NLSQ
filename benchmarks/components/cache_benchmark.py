#!/usr/bin/env python
"""Benchmark JIT compilation cache performance.

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
    python benchmarks/components/cache_benchmark.py          # Full benchmark
    python benchmarks/components/cache_benchmark.py --quick  # Quick validation
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.common.constants import (
    DEFAULT_DATA_SIZES,
    TARGET_CACHE_HIT_RATE,
    TARGET_SPEEDUP_FACTOR,
    TARGET_WARM_JIT_TIME_MS,
)
from nlsq import curve_fit
from nlsq.caching.unified_cache import clear_cache, get_cache_stats


def exponential_model(x: jnp.ndarray, a: float, b: float, c: float) -> jnp.ndarray:
    """Exponential model for benchmarking."""
    return a * jnp.exp(-b * x) + c


def benchmark_cold_jit_time(
    data_sizes: tuple[int, ...] = DEFAULT_DATA_SIZES,
) -> dict:
    """Measure cold JIT compilation time for different data sizes.

    Returns
    -------
    dict
        Cold JIT times in milliseconds (p50, p95)
    """
    times = []

    for size in data_sizes:
        clear_cache()

        x = np.linspace(0, 10, size)
        y_true = 2.5 * np.exp(-0.5 * x) + 1.0
        y = y_true + np.random.normal(0, 0.1, size)

        start = time.time()
        result = curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])
        _, _ = result
        cold_time_ms = (time.time() - start) * 1000

        times.append(cold_time_ms)
        print(f"  Size {size}: {cold_time_ms:.1f} ms (cold JIT)")

    times_arr = np.array(times)
    return {
        "p50": float(np.percentile(times_arr, 50)),
        "p95": float(np.percentile(times_arr, 95)),
        "mean": float(np.mean(times_arr)),
        "data_sizes": list(data_sizes),
        "all_times": times_arr.tolist(),
    }


def benchmark_warm_jit_time(n_iterations: int = 100) -> dict:
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
    clear_cache()
    x = np.linspace(0, 10, 1000)
    y = 2.5 * np.exp(-0.5 * x) + 1.0 + np.random.normal(0, 0.1, 1000)
    _ = curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])

    times = []
    for _ in range(n_iterations):
        x = np.linspace(0, 10, 1000)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + np.random.normal(0, 0.1, 1000)

        start = time.time()
        result = curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])
        _, _ = result
        warm_time_ms = (time.time() - start) * 1000
        times.append(warm_time_ms)

    times_arr = np.array(times)
    return {
        "p50": float(np.percentile(times_arr, 50)),
        "p95": float(np.percentile(times_arr, 95)),
        "mean": float(np.mean(times_arr)),
        "n_iterations": n_iterations,
    }


def benchmark_cache_hit_rate(
    n_fits: int = 1000,
    data_sizes: tuple[int, ...] = (100, 200, 500, 1000),
) -> dict:
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
    clear_cache()

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


def calculate_speedup_factor(cold_time: dict, warm_time: dict) -> float:
    """Calculate speedup from warm (cached) vs cold execution."""
    cold_p50 = cold_time["p50"]
    warm_p50 = warm_time["p50"]

    if warm_p50 > 0:
        return cold_p50 / warm_p50
    return 0.0


def run_quick_benchmark() -> dict:
    """Run quick cache benchmark for validation."""
    print("Quick Cache Benchmark")
    print("=" * 70)

    clear_cache()
    x = np.linspace(0, 10, 1000)
    y = 2.5 * np.exp(-0.5 * x) + 1.0 + np.random.normal(0, 0.1, 1000)

    start = time.time()
    curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])
    cold_time = (time.time() - start) * 1000
    print(f"Cold JIT time: {cold_time:.1f} ms")

    warm_times = []
    for _ in range(10):
        x = np.linspace(0, 10, 1000)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + np.random.normal(0, 0.1, 1000)
        start = time.time()
        curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])
        warm_times.append((time.time() - start) * 1000)

    warm_p50 = float(np.percentile(warm_times, 50))
    print(f"Warm JIT time (P50): {warm_p50:.2f} ms")
    print(f"Speedup: {cold_time / warm_p50:.1f}x")

    stats = get_cache_stats()
    print("\nCache Statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Cache size: {stats['cache_size']}")

    return {
        "cold_jit_time_ms": {"p50": cold_time},
        "warm_jit_time_ms": {"p50": warm_p50},
        "speedup_factor": cold_time / warm_p50,
        "cache_hit_rate": stats["hit_rate"],
    }


def run_full_benchmark() -> dict:
    """Run comprehensive cache benchmarks."""
    print("=" * 70)
    print("Unified Cache Performance Benchmark")
    print("=" * 70)
    print()

    print("System Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  JAX: {jax.__version__}")
    print(f"  Device: {jax.devices()[0].device_kind}")
    print()

    print("Benchmark 1: Cold JIT Compilation Time")
    print("-" * 70)
    cold_time = benchmark_cold_jit_time()
    print(f"  P50: {cold_time['p50']:.1f} ms")
    print(f"  P95: {cold_time['p95']:.1f} ms")
    print(f"  Mean: {cold_time['mean']:.1f} ms")
    print()

    print("Benchmark 2: Warm JIT Time (Cache Hits)")
    print("-" * 70)
    warm_time = benchmark_warm_jit_time(n_iterations=100)
    print(f"  P50: {warm_time['p50']:.2f} ms")
    print(f"  P95: {warm_time['p95']:.2f} ms")
    print(f"  Mean: {warm_time['mean']:.2f} ms")
    print()

    speedup = calculate_speedup_factor(cold_time, warm_time)
    print(f"  Speedup Factor (Cold/Warm): {speedup:.1f}x")
    print()

    print("Benchmark 3: Cache Hit Rate (1000 fits, varying sizes)")
    print("-" * 70)
    cache_stats = benchmark_cache_hit_rate(n_fits=1000)
    print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
    print(f"  Total Hits: {cache_stats['total_hits']}")
    print(f"  Total Misses: {cache_stats['total_misses']}")
    print(f"  Compilations: {cache_stats['total_compilations']}")
    print(f"  Cache Size: {cache_stats['cache_size']}")
    print()

    print("Target Validation:")
    print("-" * 70)

    hit_rate_pass = cache_stats["hit_rate"] >= TARGET_CACHE_HIT_RATE
    print(
        f"  Cache Hit Rate: {cache_stats['hit_rate']:.2%} "
        f"(target: >{TARGET_CACHE_HIT_RATE:.0%}) {'PASS' if hit_rate_pass else 'FAIL'}"
    )

    speedup_pass = speedup >= TARGET_SPEEDUP_FACTOR
    print(
        f"  Speedup Factor: {speedup:.1f}x "
        f"(target: {TARGET_SPEEDUP_FACTOR:.0f}-5x) {'PASS' if speedup_pass else 'FAIL'}"
    )

    warm_time_pass = warm_time["p50"] <= TARGET_WARM_JIT_TIME_MS
    print(
        f"  Warm JIT Time: {warm_time['p50']:.2f} ms "
        f"(target: <{TARGET_WARM_JIT_TIME_MS:.0f}ms) {'PASS' if warm_time_pass else 'FAIL'}"
    )
    print()

    return {
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


def main() -> int:
    """Run cache benchmarks and save results."""
    parser = argparse.ArgumentParser(description="Cache performance benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    if args.quick:
        results = run_quick_benchmark()
    else:
        results = run_full_benchmark()

    baseline_dir = Path(__file__).parent.parent / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    baseline_file = args.output or (baseline_dir / "cache_unification.json")

    with open(baseline_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {baseline_file}")
    print()

    if not args.quick:
        all_passed = all(results["targets_met"].values())
        print("=" * 70)
        if all_passed:
            print("All performance targets met!")
        else:
            print("Some performance targets not met")
            failed = [k for k, v in results["targets_met"].items() if not v]
            print(f"  Failed: {', '.join(failed)}")
        print("=" * 70)
        return 0 if all_passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
