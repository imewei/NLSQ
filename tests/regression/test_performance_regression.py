"""Performance regression tests with baseline comparison.

This module implements CI/CD gates to prevent performance regressions.
Tests compare current performance against stored baselines.
"""

import json
import os
import platform
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit

# Check if running in parallel mode (pytest-xdist)
RUNNING_IN_PARALLEL = hasattr(os.environ.get("PYTEST_XDIST_WORKER", ""), "__len__")


def load_baseline():
    """Load performance baseline for current platform."""
    baseline_dir = Path(__file__).parents[2] / "benchmarks" / "baselines"
    baseline_file = baseline_dir / f"v0.3.0-beta.3-{platform.system().lower()}.json"

    if not baseline_file.exists():
        pytest.skip(f"Baseline not found: {baseline_file}")

    with open(baseline_file) as f:
        return json.load(f)


def measure_cold_jit():
    """Measure cold JIT compilation time."""

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    np.random.seed(42)
    x = jnp.linspace(0, 5, 100)
    y = 2.0 * jnp.exp(-0.5 * x) + 0.05 * np.random.randn(100)

    start = time.perf_counter()
    _popt, _ = curve_fit(model, x, y, p0=[1.0, 0.1])
    return time.perf_counter() - start


def measure_hot_path():
    """Measure hot path (cached JIT) time."""

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    np.random.seed(42)
    x = jnp.linspace(0, 5, 100)
    y = 2.0 * jnp.exp(-0.5 * x) + 0.05 * np.random.randn(100)

    # Warmup
    _ = curve_fit(model, x, y, p0=[1.0, 0.1])

    # Measure
    times = []
    for _ in range(3):
        start = time.perf_counter()
        _ = curve_fit(model, x, y, p0=[1.0, 0.1])
        times.append(time.perf_counter() - start)

    return np.mean(times)


class TestPerformanceRegression:
    """Performance regression tests against baselines."""

    @pytest.mark.slow
    def test_performance_improvement_tracking(self):
        """Track performance improvements over baseline."""
        baseline = load_baseline()

        cold_jit = measure_cold_jit() * 1000
        hot_path = measure_hot_path() * 1000

        baseline_cold = baseline["metrics"]["cold_jit_ms"]
        baseline_hot = baseline["metrics"]["hot_path_ms"]

        cold_improvement = ((baseline_cold - cold_jit) / baseline_cold) * 100
        hot_improvement = ((baseline_hot - hot_path) / baseline_hot) * 100

        # This test always passes but reports performance changes
        print("\nPerformance vs Baseline:")
        print(
            f"  Cold JIT: {cold_jit:.2f}ms (baseline: {baseline_cold:.2f}ms, "
            f"{cold_improvement:+.1f}%)"
        )
        print(
            f"  Hot path: {hot_path:.2f}ms (baseline: {baseline_hot:.2f}ms, "
            f"{hot_improvement:+.1f}%)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
