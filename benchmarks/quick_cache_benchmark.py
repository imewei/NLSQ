#!/usr/bin/env python
"""Quick cache benchmark for validation (Task 1.12)."""

import json
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nlsq import curve_fit
from nlsq.unified_cache import clear_cache, get_cache_stats


def exponential_model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


def main():
    """Quick benchmark to validate cache functionality."""
    print("Quick Cache Benchmark (Task 1.12)")
    print("=" * 70)

    # Test 1: Cold JIT time
    clear_cache()
    x = np.linspace(0, 10, 1000)
    y = 2.5 * np.exp(-0.5 * x) + 1.0 + np.random.normal(0, 0.1, 1000)

    start = time.time()
    curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])  # Timing only
    cold_time = (time.time() - start) * 1000
    print(f"Cold JIT time: {cold_time:.1f} ms")

    # Test 2: Warm JIT time (10 iterations)
    warm_times = []
    for _ in range(10):
        x = np.linspace(0, 10, 1000)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + np.random.normal(0, 0.1, 1000)
        start = time.time()
        _result = curve_fit(exponential_model, x, y, p0=[2.0, 0.5, 1.0])
        warm_times.append((time.time() - start) * 1000)

    warm_p50 = np.percentile(warm_times, 50)
    print(f"Warm JIT time (P50): {warm_p50:.2f} ms")
    print(f"Speedup: {cold_time / warm_p50:.1f}x")

    # Test 3: Cache statistics
    stats = get_cache_stats()
    print("\nCache Statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Cache size: {stats['cache_size']}")

    # Save baseline
    baseline_dir = Path(__file__).parent / "baselines"
    baseline_dir.mkdir(exist_ok=True)

    results = {
        "platform": platform.system().lower(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "cold_jit_time_ms": {"p50": cold_time},
        "warm_jit_time_ms": {"p50": warm_p50},
        "speedup_factor": cold_time / warm_p50,
        "cache_hit_rate": stats["hit_rate"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    baseline_file = baseline_dir / "cache_unification.json"
    with open(baseline_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {baseline_file}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
