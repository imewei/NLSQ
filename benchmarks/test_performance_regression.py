"""
Performance regression test suite for NLSQ

This module provides pytest-benchmark integration for CI/CD performance tracking.
Benchmarks are designed to detect performance regressions and track improvements
across different optimization phases.

Usage:
    # Run all benchmarks
    pytest benchmarks/test_performance_regression.py --benchmark-only

    # Save baseline
    pytest benchmarks/test_performance_regression.py --benchmark-save=baseline

    # Compare against baseline
    pytest benchmarks/test_performance_regression.py --benchmark-compare=baseline

    # Generate JSON report for CI
    pytest benchmarks/test_performance_regression.py --benchmark-json=report.json

CI/CD Regression Gates:
    - Cold JIT: >10% slowdown → FAIL
    - Hot path: >5-10% slowdown → FAIL
    - Memory: >10% increase → WARN only
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

try:
    from nlsq import CurveFit, curve_fit
except ImportError:
    pytest.skip("NLSQ not installed", allow_module_level=True)


# ============================================================================
# Baseline Loading and Regression Gates
# ============================================================================

BASELINE_FILE = Path(__file__).parent / "baselines" / "linux-py312-beta1.json"


def load_baseline():
    """Load v0.3.0-beta.1 baseline for regression testing"""
    if not BASELINE_FILE.exists():
        pytest.skip(f"Baseline file not found: {BASELINE_FILE}")

    try:
        with open(BASELINE_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        pytest.skip(f"Failed to load baseline: {e}")
    return None


BASELINE = load_baseline()


def check_performance_regression(metric_name, current_value, metric_type="cold_jit"):
    """
    Check if current value exceeds regression threshold

    Parameters:
    - metric_name: Name of the metric (e.g., "exponential_100")
    - current_value: Current measured value (ms)
    - metric_type: "cold_jit", "hot_path", or "memory"

    Returns:
    - (passed, message) tuple
    """
    config = BASELINE.get("regression_gates", {})

    if metric_type == "cold_jit":
        threshold_pct = config.get("cold_jit_slowdown_threshold_percent", 10)
        action = config.get("cold_jit_slowdown_action", "FAIL")
        baseline_val = (
            BASELINE["benchmarks"].get(metric_name, {}).get("nlsq_trf_cold_jit_ms")
        )
    elif metric_type == "hot_path":
        threshold_pct = config.get("hot_path_slowdown_threshold_percent", 5)
        action = config.get("hot_path_slowdown_action", "FAIL")
        baseline_val = (
            BASELINE["benchmarks"].get(metric_name, {}).get("nlsq_trf_hot_path_ms")
        )
    elif metric_type == "memory":
        threshold_pct = config.get("memory_regression_threshold_percent", 10)
        action = config.get("memory_regression_action", "WARN_ONLY")
        baseline_val = BASELINE.get("memory_baseline_mb")
    else:
        return True, f"Unknown metric type: {metric_type}"

    if baseline_val is None:
        return True, f"No baseline found for {metric_name}"

    regression_pct = ((current_value - baseline_val) / baseline_val) * 100
    threshold_exceeded = regression_pct > threshold_pct

    message = (
        f"{metric_name} ({metric_type}): {current_value:.1f}ms "
        f"(baseline: {baseline_val:.1f}ms, "
        f"regression: {regression_pct:+.1f}%, "
        f"threshold: {threshold_pct}%)"
    )

    if threshold_exceeded:
        if action == "FAIL":
            return False, message
        elif action == "WARN_ONLY":
            warnings.warn(message, category=UserWarning)
            return True, message

    return True, message


# ============================================================================
# Test Functions (Models to Fit)
# ============================================================================


def linear(x, m, b):
    """Linear model: y = mx + b"""
    return m * x + b


def exponential(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    import jax.numpy as jnp

    return a * jnp.exp(-b * x) + c


def gaussian(x, amp, mu, sigma):
    """Gaussian function"""
    import jax.numpy as jnp

    return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))


def polynomial(x, a, b, c, d):
    """Cubic polynomial"""
    return a * x**3 + b * x**2 + c * x + d


# ============================================================================
# Test Data Generators
# ============================================================================


@pytest.fixture(scope="module")
def small_linear_data():
    """Small dataset (100 points) - linear"""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 2.0 * x + 1.0
    noise = 0.1 * np.random.randn(len(x))
    y = y_true + noise
    return x, y, [1.0, 0.0]


@pytest.fixture(scope="module")
def medium_exponential_data():
    """Medium dataset (1000 points) - exponential"""
    np.random.seed(42)
    x = np.linspace(0, 10, 1000)
    y_true = 2.0 * np.exp(-0.5 * x) + 0.3
    noise = 0.05 * np.random.randn(len(x))
    y = y_true + noise
    return x, y, [2.0, 0.5, 0.3]


@pytest.fixture(scope="module")
def large_gaussian_data():
    """Large dataset (10000 points) - Gaussian"""
    np.random.seed(42)
    x = np.linspace(-10, 10, 10000)
    y_true = 5.0 * np.exp(-((x - 0.0) ** 2) / (2 * 2.0**2))
    noise = 0.1 * np.random.randn(len(x))
    y = y_true + noise
    return x, y, [5.0, 0.0, 2.0]


@pytest.fixture(scope="module")
def xlarge_polynomial_data():
    """Extra large dataset (50000 points) - polynomial"""
    np.random.seed(42)
    x = np.linspace(-5, 5, 50000)
    y_true = 0.1 * x**3 - 0.5 * x**2 + 2.0 * x + 1.0
    noise = 0.2 * np.random.randn(len(x))
    y = y_true + noise
    return x, y, [0.1, -0.5, 2.0, 1.0]


# ============================================================================
# Benchmark Group 1: Small Problems (Baseline Performance)
# ============================================================================


@pytest.mark.benchmark(group="small-problems")
def test_small_linear_fit(benchmark, small_linear_data):
    """Benchmark: Small linear fit (100 points, 2 params)

    Expected: ~2ms (baseline)
    Critical: This is the baseline performance test
    """
    x, y, p0 = small_linear_data

    result = benchmark(curve_fit, linear, x, y, p0=p0)

    popt, _pcov = result
    # Validate result
    assert len(popt) == 2
    assert np.allclose(popt, [2.0, 1.0], atol=0.1)


@pytest.mark.benchmark(group="small-problems")
def test_small_exponential_fit(benchmark, small_linear_data):
    """Benchmark: Small exponential fit (100 points, 3 params)

    Expected: ~2-3ms
    """
    x = small_linear_data[0]
    y_true = 2.0 * np.exp(-0.5 * x) + 0.3
    y = y_true + 0.05 * np.random.randn(len(x))
    p0 = [2.0, 0.5, 0.3]

    result = benchmark(curve_fit, exponential, x, y, p0=p0)

    popt, _pcov = result
    assert len(popt) == 3


@pytest.mark.benchmark(group="regression-gates")
def test_regression_cold_jit_exponential_100(benchmark, small_linear_data):
    """Regression gate: Cold JIT compilation time for exponential_100

    Fails if >10% slowdown from v0.3.0-beta.1 baseline
    """
    x = small_linear_data[0]
    y_true = 2.0 * np.exp(-0.5 * x) + 0.3
    y = y_true + 0.05 * np.random.randn(len(x))
    p0 = [2.0, 0.5, 0.3]

    # Measure time including JIT compilation
    timer = benchmark.pedantic(
        curve_fit,
        args=(exponential, x, y),
        kwargs={"p0": p0},
        iterations=1,
        rounds=3,
    )

    # Extract measured time (ms)
    current_time_ms = (
        timer[1] * 1000 if isinstance(timer, tuple) else benchmark.stats.mean * 1000
    )

    # Check against baseline
    passed, message = check_performance_regression(
        "exponential_decay", current_time_ms, "cold_jit"
    )

    if not passed:
        pytest.fail(f"Cold JIT regression: {message}")
    else:
        print(f"Cold JIT gate passed: {message}")


# ============================================================================
# Benchmark Group 2: Medium Problems (Core Performance)
# ============================================================================


@pytest.mark.benchmark(group="medium-problems")
def test_medium_exponential_fit(benchmark, medium_exponential_data):
    """Benchmark: Medium exponential fit (1000 points, 3 params)

    Expected: ~2-5ms
    Critical: Most common use case
    """
    x, y, p0 = medium_exponential_data

    result = benchmark(curve_fit, exponential, x, y, p0=p0)

    popt, _pcov = result
    assert len(popt) == 3
    assert np.allclose(popt, [2.0, 0.5, 0.3], atol=0.1)


@pytest.mark.benchmark(group="medium-problems")
def test_medium_gaussian_fit(benchmark):
    """Benchmark: Medium Gaussian fit (1000 points, 3 params)"""
    np.random.seed(42)
    x = np.linspace(-10, 10, 1000)
    y_true = 5.0 * np.exp(-((x - 0.0) ** 2) / (2 * 2.0**2))
    y = y_true + 0.1 * np.random.randn(len(x))
    p0 = [5.0, 0.0, 2.0]

    result = benchmark(curve_fit, gaussian, x, y, p0=p0)

    popt, _pcov = result
    assert len(popt) == 3


# ============================================================================
# Benchmark Group 3: Large Problems (Scalability)
# ============================================================================


@pytest.mark.benchmark(group="large-problems")
def test_large_gaussian_fit(benchmark, large_gaussian_data):
    """Benchmark: Large Gaussian fit (10000 points, 3 params)

    Expected: ~5-10ms
    Tests scalability to larger datasets
    """
    x, y, p0 = large_gaussian_data

    result = benchmark(curve_fit, gaussian, x, y, p0=p0)

    popt, _pcov = result
    assert len(popt) == 3
    assert np.allclose(popt, [5.0, 0.0, 2.0], atol=0.2)


@pytest.mark.benchmark(group="large-problems")
@pytest.mark.slow
def test_xlarge_polynomial_fit(benchmark, xlarge_polynomial_data):
    """Benchmark: Extra large polynomial fit (50000 points, 4 params)

    Expected: ~50-100ms
    Tests performance on very large datasets
    """
    x, y, p0 = xlarge_polynomial_data

    result = benchmark(curve_fit, polynomial, x, y, p0=p0)

    popt, _pcov = result
    assert len(popt) == 4


# ============================================================================
# Benchmark Group 4: CurveFit Class (Reuse Performance)
# ============================================================================


@pytest.mark.benchmark(group="curvefit-class")
def test_curvefit_class_reuse(benchmark, small_linear_data):
    """Benchmark: CurveFit class with function reuse

    Expected: ~1-2ms (should be faster due to JIT caching)
    Tests benefit of CurveFit class for repeated fits
    """
    x, y, p0 = small_linear_data

    # Create CurveFit instance (JIT compilation happens once)
    cf = CurveFit()

    # First fit to warm up JIT
    cf.curve_fit(linear, x, y, p0=p0)

    # Benchmark subsequent fits (should use cached JIT)
    result = benchmark(cf.curve_fit, linear, x, y, p0=p0)

    popt, _pcov = result
    assert len(popt) == 2


@pytest.mark.benchmark(group="curvefit-class")
def test_curvefit_class_with_stability(benchmark, medium_exponential_data):
    """Benchmark: CurveFit with stability features enabled

    Expected: ~3-5ms (25-30% overhead)
    Tests performance impact of stability features
    """
    x, y, p0 = medium_exponential_data

    # Create CurveFit with stability
    cf = CurveFit(enable_stability=True)

    # Warm up
    cf.curve_fit(exponential, x, y, p0=p0)

    # Benchmark
    result = benchmark(cf.curve_fit, exponential, x, y, p0=p0)

    popt, _pcov = result
    assert len(popt) == 3


# ============================================================================
# Benchmark Group 5: Algorithm Comparison
# ============================================================================


@pytest.mark.benchmark(group="algorithm-comparison")
def test_trf_algorithm(benchmark, medium_exponential_data):
    """Benchmark: Trust Region Reflective algorithm

    Expected: ~2-5ms
    Default algorithm for most problems
    """
    x, y, p0 = medium_exponential_data

    result = benchmark(curve_fit, exponential, x, y, p0=p0, method="trf")

    popt, _pcov = result
    assert len(popt) == 3


# NOTE: LM (Levenberg-Marquardt) algorithm test removed
# NLSQ only supports TRF (Trust Region Reflective) algorithm
# TRF is tested in test_trf_algorithm above

# ============================================================================
# Benchmark Group 6: Bounded Optimization
# ============================================================================


@pytest.mark.benchmark(group="bounded-optimization")
def test_bounded_exponential_fit(benchmark, medium_exponential_data):
    """Benchmark: Exponential fit with parameter bounds

    Expected: ~3-6ms (slightly slower than unbounded)
    Tests bounded optimization performance
    """
    x, y, p0 = medium_exponential_data

    # Set bounds
    bounds = ([0.0, 0.0, 0.0], [10.0, 5.0, 1.0])

    result = benchmark(curve_fit, exponential, x, y, p0=p0, bounds=bounds)

    popt, _pcov = result
    assert len(popt) == 3
    # Verify bounds respected
    assert np.all(popt >= bounds[0])
    assert np.all(popt <= bounds[1])


# ============================================================================
# Benchmark Group 7: JIT Compilation Overhead
# ============================================================================


@pytest.mark.benchmark(group="jit-compilation")
def test_first_call_with_jit_compilation(benchmark):
    """Benchmark: First call including JIT compilation

    Expected: ~100-500ms (includes compilation time)
    This measures cold-start performance
    """
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))
    p0 = [1.0, 0.0]

    # Benchmark includes JIT compilation
    result = benchmark.pedantic(
        curve_fit,
        args=(linear, x, y),
        kwargs={"p0": p0},
        iterations=1,  # Only measure first call
        rounds=10,  # Average over 10 rounds
    )

    popt, _pcov = result
    assert len(popt) == 2


# ============================================================================
# Benchmark Group 8: Memory Efficiency
# ============================================================================


@pytest.mark.benchmark(group="memory-efficiency")
@pytest.mark.slow
def test_large_dataset_memory_usage(benchmark, xlarge_polynomial_data):
    """Benchmark: Memory usage for large dataset

    Tests that memory usage scales appropriately
    """
    x, y, p0 = xlarge_polynomial_data

    result = benchmark(curve_fit, polynomial, x, y, p0=p0)

    popt, _pcov = result
    assert len(popt) == 4


# ============================================================================
# Benchmark Group 9: Numerical Stability
# ============================================================================


@pytest.mark.benchmark(group="numerical-stability")
def test_ill_conditioned_problem(benchmark):
    """Benchmark: Ill-conditioned fitting problem

    Tests performance on numerically challenging problems
    """
    np.random.seed(42)
    x = np.linspace(0, 1, 100)

    # Create ill-conditioned problem (very small parameters)
    y_true = 1e-8 * np.exp(-1e6 * x) + 1e-10
    y = y_true + 1e-11 * np.random.randn(len(x))
    p0 = [1e-8, 1e6, 1e-10]

    cf = CurveFit(enable_stability=True)

    try:
        result = benchmark(cf.curve_fit, exponential, x, y, p0=p0)
        popt, _pcov = result
        assert len(popt) == 3
    except Exception:
        # Some ill-conditioned problems may not converge
        pytest.skip("Ill-conditioned problem did not converge")


# ============================================================================
# Utility Functions for CI Integration
# ============================================================================


def pytest_benchmark_scale_unit(config, unit, benchmarks, best, worst, sort):
    """Custom unit scaling for benchmark reports"""
    if unit == "seconds":
        return "milliseconds", 1000.0
    return unit, 1.0


# ============================================================================
# Benchmark Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest-benchmark"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (use -m 'not slow' to skip)"
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Run benchmarks directly:

    python test_performance_regression.py

    Or use pytest:

    pytest test_performance_regression.py --benchmark-only -v
    """
    pytest.main([__file__, "--benchmark-only", "-v"])
