#!/usr/bin/env python
"""Create performance baseline for v0.3.0-beta.3."""

import json
import platform
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit


def measure_cold_jit_time():
    """Measure cold JIT compilation time."""

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    np.random.seed(42)
    x = jnp.linspace(0, 5, 100)
    y = 2.0 * jnp.exp(-0.5 * x) + 0.05 * np.random.randn(100)

    start = time.perf_counter()
    _popt, _ = curve_fit(model, x, y, p0=[1.0, 0.1])
    cold_jit_time = time.perf_counter() - start

    return cold_jit_time


def measure_hot_path_time():
    """Measure hot path (cached JIT) time."""

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    np.random.seed(42)
    x = jnp.linspace(0, 5, 100)
    y = 2.0 * jnp.exp(-0.5 * x) + 0.05 * np.random.randn(100)

    # Warmup
    _ = curve_fit(model, x, y, p0=[1.0, 0.1])

    # Measure hot path
    times = []
    for _ in range(5):
        start = time.perf_counter()
        _ = curve_fit(model, x, y, p0=[1.0, 0.1])
        times.append(time.perf_counter() - start)

    return np.mean(times)


def create_baseline():
    """Create comprehensive performance baseline."""
    print("Creating performance baseline for v0.3.0-beta.3...")

    cold_jit = measure_cold_jit_time()
    print(f"  Cold JIT time: {cold_jit * 1000:.2f}ms")

    hot_path = measure_hot_path_time()
    print(f"  Hot path time: {hot_path * 1000:.2f}ms")

    baseline = {
        "version": "0.3.0-beta.3",
        "created_at": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
        },
        "metrics": {
            "cold_jit_ms": cold_jit * 1000,
            "hot_path_ms": hot_path * 1000,
        },
        "thresholds": {
            "cold_jit_regression_factor": 1.10,  # 10% slowdown allowed
            "hot_path_regression_factor": 1.10,  # 10% slowdown allowed
        },
    }

    return baseline


if __name__ == "__main__":
    baseline = create_baseline()

    output_file = f"benchmarks/baselines/v0.3.0-beta.3-{platform.system().lower()}.json"
    with open(output_file, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\n✓ Baseline saved to {output_file}")
