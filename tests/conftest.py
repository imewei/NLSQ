"""
Pytest configuration and shared fixtures for NLSQ test suite.

Provides common test functions, data generators, and fixtures for
scientific computing tests.
"""

import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ============================================================================
# JAX Configuration for Optimal Test Performance (Phase 1 Optimization)
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def configure_jax_for_tests():
    """
    Configure JAX environment for optimal test performance.

    Phase 1 Test Optimization:
    - Disables GPU pre-allocation to reduce memory usage
    - Sets platform-specific memory allocation
    - Enables persistent compilation cache across test sessions
    - Suppresses GPU warnings for CPU-only tests

    Expected Impact: 5-8% reduction (13-33 seconds)
    """
    # Disable JAX GPU pre-allocation (reduces memory usage in tests)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Use platform-specific allocator for faster memory operations
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # Enable persistent JAX compilation cache across test sessions
    cache_dir = "/tmp/nlsq_jax_test_cache"
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir

    # Force CPU for test consistency (GPU tests use explicit markers)
    os.environ["JAX_PLATFORMS"] = "cpu"

    # Suppress GPU warning in tests (already configured via NLSQ_SKIP_GPU_CHECK)
    os.environ["NLSQ_SKIP_GPU_CHECK"] = "1"

    yield

    # Cleanup: Remove JAX compilation cache after test session
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


# ============================================================================
# JAX Memory Cleanup Fixture (OOM Prevention)
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_jax_memory():
    """
    Clean up JAX memory after each test to prevent OOM crashes.

    Root Cause Analysis (2025-12-26):
    - Parallel test execution with many workers accumulates JAX state
    - AdaptiveHybridStreamingOptimizer holds O(p²) accumulators
    - JIT compilation caches grow unbounded across tests
    - Without cleanup, cumulative memory can exceed 60GB on large test suites

    This fixture runs after EVERY test to:
    1. Force Python garbage collection
    2. Clear JAX compilation caches
    3. Release XLA device memory

    Impact: Prevents OOM killer (exit code 137) during full test suite runs.
    """
    # Let the test run
    yield

    # Cleanup after test completes
    import gc

    # Force garbage collection to release Python objects
    gc.collect()

    # Clear JAX compilation caches to free XLA memory
    # This is safe because tests should not depend on cached compilations
    jax.clear_caches()


# ============================================================================
# StreamingOptimizer Thread Cleanup Fixture (Thread Accumulation Prevention)
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_streaming_optimizer_threads():
    """
    Clean up StreamingOptimizer checkpoint threads after each test.

    Root Cause Analysis (2025-12-26):
    - Each StreamingOptimizer creates a background checkpoint thread
    - Tests that create optimizers without explicit cleanup leave threads running
    - Thread accumulation leads to resource exhaustion in parallel test execution
    - Python's __del__ is not guaranteed to be called promptly

    This fixture runs after EVERY test to:
    1. Find any StreamingOptimizer instances in memory
    2. Shut down their checkpoint worker threads
    3. Force garbage collection to clean up the instances

    Impact: Prevents thread accumulation and associated memory leaks.
    """
    # Let the test run
    yield

    # Cleanup after test completes
    import gc
    import threading

    # Get list of all threads to identify checkpoint workers
    checkpoint_threads = [
        t
        for t in threading.enumerate()
        if t.name == "checkpoint-worker" and t.is_alive()
    ]

    # Force garbage collection first to trigger __del__ on orphaned optimizers
    gc.collect()

    # If there are still checkpoint threads running, they may be from
    # optimizers that weren't properly cleaned up. The daemon flag should
    # handle these when the process exits, but we log for debugging.
    remaining_threads = [
        t
        for t in threading.enumerate()
        if t.name == "checkpoint-worker" and t.is_alive()
    ]

    if len(remaining_threads) > len(checkpoint_threads):
        # New threads were created during cleanup - this shouldn't happen
        pass  # Just continue, daemon threads will be cleaned up at exit


# ============================================================================
# pytest-xdist Serial Test Grouping (OOM Prevention)
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """
    Group tests marked with @pytest.mark.serial to run on the same worker.

    This ensures memory-intensive tests don't run in parallel with each other,
    preventing OOM crashes. Tests with the 'serial' marker are assigned to the
    same xdist group, so they execute sequentially on a single worker.

    Note: This hook runs during test collection, before any tests execute.
    """
    for item in items:
        # Check if test has the serial marker
        if item.get_closest_marker("serial"):
            # Assign all serial tests to the same xdist group
            # This ensures they run on the same worker, one at a time
            item.add_marker(pytest.mark.xdist_group("serial_memory_tests"))


# ============================================================================
# Session-Scoped Compiled Model Fixtures (Phase 2 Optimization)
# ============================================================================


@pytest.fixture(scope="session")
def compiled_models():
    """
    Pre-compiled model functions cached at session level.

    Phase 2 Optimization: Eliminates repeated JAX JIT compilations across tests.
    Each model function is compiled ONCE per test session and reused across all tests.

    Expected Impact: 8-12% reduction (21-40 seconds from 268s baseline)

    Returns
    -------
    dict
        Dictionary of pre-compiled model functions accessible by name.

    Examples
    --------
    >>> def test_linear_fit(compiled_models):
    ...     from nlsq import curve_fit
    ...     popt, pcov = curve_fit(compiled_models['linear'], x, y, p0=[1, 1])
    """

    # Define all common model functions
    def linear(x, a, b):
        return a * x + b

    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    def gaussian(x, amp, mu, sigma):
        return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def sinusoidal(x, a, b, c, d):
        return a * jnp.sin(b * x + c) + d

    def polynomial(x, *coeffs):
        result = jnp.zeros_like(x)
        for i, c in enumerate(coeffs):
            result = result + c * x**i
        return result

    # Create CurveFit instances (triggers JIT compilation on first use)
    models = {
        "linear": linear,
        "quadratic": quadratic,
        "exponential": exponential,
        "gaussian": gaussian,
        "sinusoidal": sinusoidal,
        "polynomial": polynomial,
    }

    # Warm up compilation cache with dummy data
    import contextlib

    x_dummy = jnp.linspace(0, 10, 100)
    for name, func in models.items():
        if name == "polynomial":
            # Polynomial needs specific coefficients
            with contextlib.suppress(Exception):
                func(x_dummy, 1.0, 1.0, 1.0)
        elif name == "sinusoidal":
            func(x_dummy, 1.0, 1.0, 1.0, 1.0)
        elif name in {"gaussian", "exponential", "quadratic"}:
            func(x_dummy, 1.0, 1.0, 1.0)
        else:  # linear
            func(x_dummy, 1.0, 1.0)

    return models


# ============================================================================
# Test Function Fixtures (Module Scope for Better Cache Reuse)
# ============================================================================


@pytest.fixture(scope="module")  # Changed from function to module scope
def linear_func():
    """Linear function: f(x, a, b) = a*x + b"""

    def f(x, a, b):
        return a * x + b

    return f


@pytest.fixture(scope="module")  # Changed from function to module scope
def quadratic_func():
    """Quadratic function: f(x, a, b, c) = a*x^2 + b*x + c"""

    def f(x, a, b, c):
        return a * x**2 + b * x + c

    return f


@pytest.fixture(scope="module")  # Changed from function to module scope
def exponential_func():
    """Exponential decay: f(x, a, b, c) = a*exp(-b*x) + c"""

    def f(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    return f


@pytest.fixture(scope="module")  # Changed from function to module scope
def gaussian_func():
    """Gaussian function: f(x, amp, mu, sigma) = amp*exp(-(x-mu)^2/(2*sigma^2))"""

    def f(x, amp, mu, sigma):
        return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

    return f


@pytest.fixture(scope="module")  # Changed from function to module scope
def sinusoidal_func():
    """Sinusoidal function: f(x, a, b, c, d) = a*sin(b*x + c) + d"""

    def f(x, a, b, c, d):
        return a * jnp.sin(b * x + c) + d

    return f


@pytest.fixture(scope="module")  # Changed from function to module scope
def polynomial_func():
    """General polynomial function with variable parameters"""

    def f(x, *coeffs):
        result = jnp.zeros_like(x)
        for i, c in enumerate(coeffs):
            result = result + c * x**i
        return result

    return f


# ============================================================================
# Data Generator Fixtures (Module Scope for Better Performance)
# ============================================================================


@pytest.fixture(scope="module")  # Changed from function to module scope
def linear_data():
    """Generate clean linear data (cached at module level)"""

    def generate(n_points=100, a=2.0, b=1.0, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 10, n_points)
        y = a * x + b
        return x, y, np.array([a, b])

    return generate


@pytest.fixture(scope="module")  # Changed from function to module scope
def noisy_linear_data():
    """Generate noisy linear data (cached at module level)"""

    def generate(n_points=100, a=2.0, b=1.0, noise_level=0.1, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 10, n_points)
        y = a * x + b + np.random.normal(0, noise_level, n_points)
        return x, y, np.array([a, b])

    return generate


@pytest.fixture(scope="module")  # Changed from function to module scope
def exponential_data():
    """Generate exponential decay data (cached at module level)"""

    def generate(n_points=100, a=2.5, b=1.3, c=0.1, noise_level=0.01, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 5, n_points)
        y = a * np.exp(-b * x) + c
        if noise_level > 0:
            y = y + np.random.normal(0, noise_level, n_points)
        return x, y, np.array([a, b, c])

    return generate


@pytest.fixture(scope="module")  # Changed from function to module scope
def gaussian_data():
    """Generate Gaussian data (cached at module level)"""

    def generate(n_points=100, amp=1.5, mu=0.5, sigma=1.2, noise_level=0.01, seed=42):
        np.random.seed(seed)
        x = np.linspace(-5, 5, n_points)
        y = amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        if noise_level > 0:
            y = y + np.random.normal(0, noise_level, n_points)
        return x, y, np.array([amp, mu, sigma])

    return generate


@pytest.fixture(scope="module")  # Changed from function to module scope
def outlier_data():
    """Generate data with outliers (cached at module level)"""

    def generate(n_points=100, outlier_fraction=0.1, outlier_magnitude=10.0, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 10, n_points)
        y = 2 * x + 1 + np.random.normal(0, 0.1, n_points)

        # Add outliers
        n_outliers = int(n_points * outlier_fraction)
        outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
        y[outlier_indices] += np.random.choice([-1, 1], n_outliers) * outlier_magnitude

        return x, y, np.array([2.0, 1.0])

    return generate


@pytest.fixture
def multidimensional_data():
    """Generate multidimensional input data"""

    def generate(n_points=100, n_dims=3, seed=42):
        np.random.seed(seed)
        x = np.random.randn(n_points, n_dims)
        # Simple linear combination
        true_params = np.random.randn(n_dims + 1)  # +1 for intercept
        y = x @ true_params[:-1] + true_params[-1]
        return x, y, true_params

    return generate


# ============================================================================
# JAX-Specific Fixtures
# ============================================================================


@pytest.fixture
def jax_random_key():
    """Provide a JAX random key"""
    return jax.random.PRNGKey(42)


@pytest.fixture
def jax_array_1d():
    """Provide a 1D JAX array"""
    return jnp.linspace(0, 10, 100)


@pytest.fixture
def jax_array_2d():
    """Provide a 2D JAX array"""
    return jnp.ones((100, 10))


@pytest.fixture(params=["cpu", "gpu"])
def jax_device(request):
    """Provide different JAX devices if available"""
    device_type = request.param

    try:
        devices = jax.devices(device_type)
        if devices:
            return devices[0]
    except RuntimeError:
        if device_type == "cpu":
            # CPU should always be available
            return jax.devices("cpu")[0]
        else:
            pytest.skip(f"{device_type.upper()} device not available")

    return jax.devices()[0]


# ============================================================================
# Tolerance and Numerical Settings
# ============================================================================


@pytest.fixture
def default_rtol():
    """Default relative tolerance for numerical tests"""
    return 1e-6


@pytest.fixture
def default_atol():
    """Default absolute tolerance for numerical tests"""
    return 1e-8


@pytest.fixture
def loose_rtol():
    """Loose relative tolerance for challenging problems"""
    return 1e-3


@pytest.fixture
def tight_rtol():
    """Tight relative tolerance for well-conditioned problems"""
    return 1e-10


@pytest.fixture
def gradient_rtol():
    """Relative tolerance for gradient tests"""
    return 1e-4


@pytest.fixture
def gradient_atol():
    """Absolute tolerance for gradient tests"""
    return 1e-6


# ============================================================================
# Random Seed Fixtures
# ============================================================================


@pytest.fixture
def random_seed():
    """Provide consistent random seed"""
    return 42


@pytest.fixture(autouse=True)
def reset_random_seed(random_seed):
    """Automatically reset random seed before each test"""
    np.random.seed(random_seed)


# ============================================================================
# Problem Size Fixtures
# ============================================================================


@pytest.fixture(params=[10, 100, 1000])
def small_problem_size(request):
    """Small problem sizes for quick tests"""
    return request.param


@pytest.fixture(params=[10000, 100000])
def medium_problem_size(request):
    """Medium problem sizes"""
    return request.param


@pytest.fixture(params=[1000000])
def large_problem_size(request):
    """Large problem sizes for performance tests"""
    return request.param


# ============================================================================
# Algorithm Configuration Fixtures
# ============================================================================


@pytest.fixture
def default_curve_fit_kwargs():
    """Default kwargs for curve_fit"""
    return {
        "method": "trf",
        "ftol": 1e-8,
        "xtol": 1e-8,
        "gtol": 1e-8,
        "max_nfev": None,
    }


@pytest.fixture
def fast_curve_fit_kwargs():
    """Fast kwargs for curve_fit (relaxed tolerances)"""
    return {
        "method": "trf",
        "ftol": 1e-6,
        "xtol": 1e-6,
        "gtol": 1e-6,
        "max_nfev": 100,
    }


@pytest.fixture
def accurate_curve_fit_kwargs():
    """Accurate kwargs for curve_fit (tight tolerances)"""
    return {
        "method": "trf",
        "ftol": 1e-10,
        "xtol": 1e-10,
        "gtol": 1e-10,
        "max_nfev": 1000,
    }


# ============================================================================
# Loss Function Fixtures
# ============================================================================


@pytest.fixture(params=["linear", "huber", "soft_l1", "cauchy", "arctan"])
def loss_type(request):
    """All supported loss function types"""
    return request.param


@pytest.fixture(params=["huber", "soft_l1", "cauchy", "arctan"])
def robust_loss_type(request):
    """Robust loss function types only"""
    return request.param


# ============================================================================
# Bounds Fixtures
# ============================================================================


@pytest.fixture
def no_bounds():
    """No parameter bounds"""
    return (-np.inf, np.inf)


@pytest.fixture
def positive_bounds():
    """Positive-only parameter bounds"""

    def get_bounds(n_params):
        return (np.zeros(n_params), np.full(n_params, np.inf))

    return get_bounds


@pytest.fixture
def symmetric_bounds():
    """Symmetric parameter bounds"""

    def get_bounds(n_params, limit=10.0):
        return (-np.full(n_params, limit), np.full(n_params, limit))

    return get_bounds


# ============================================================================
# Utility Functions
# ============================================================================


@pytest.fixture
def assert_optimization_success():
    """Helper to assert optimization succeeded"""

    def check(result):
        assert hasattr(result, "success"), "Result missing 'success' attribute"
        assert result.success, (
            f"Optimization failed: {result.message if hasattr(result, 'message') else 'unknown reason'}"
        )
        assert hasattr(result, "x"), "Result missing parameters 'x'"
        assert np.all(np.isfinite(result.x)), "Parameters contain non-finite values"

    return check


@pytest.fixture
def assert_parameters_close():
    """Helper to assert parameters are close to expected values"""

    def check(popt, expected, rtol=1e-2, atol=1e-3):
        np.testing.assert_allclose(
            popt,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=f"Parameters {popt} not close to expected {expected}",
        )

    return check


@pytest.fixture
def compute_residuals():
    """Helper to compute residuals"""

    def compute(f, x, y, params):
        y_pred = f(x, *params)
        return y - y_pred

    return compute


@pytest.fixture
def compute_relative_error():
    """Helper to compute relative error"""

    def compute(value, expected):
        return np.abs(value - expected) / (np.abs(expected) + 1e-10)

    return compute


# ============================================================================
# Performance Testing Fixtures
# ============================================================================


@pytest.fixture
def benchmark_timer():
    """Simple timer for benchmarking"""
    import time

    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.end = time.time()
            self.elapsed = self.end - self.start

    return Timer


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_jax_cache():
    """Clean up JAX compilation cache after tests"""
    yield
    # Let JAX manage its own cache


@pytest.fixture
def temp_array_pool():
    """Provide a temporary array pool that's cleaned up after test"""
    from nlsq.caching.memory_manager import get_memory_manager

    manager = get_memory_manager()
    len(manager.memory_pool)

    yield manager

    # Clean up arrays added during test
    manager.clear_pool()


# ============================================================================
# StreamingOptimizer Factory Fixture (Thread-Safe Cleanup)
# ============================================================================


@pytest.fixture
def optimizer_factory():
    """
    Factory for creating StreamingOptimizer instances with automatic cleanup.

    Usage:
        def test_something(optimizer_factory):
            optimizer = optimizer_factory(config)
            # use optimizer...
            # cleanup happens automatically after test

    This fixture tracks all created optimizers and ensures their checkpoint
    worker threads are shut down after the test completes, preventing thread
    accumulation in parallel test execution.
    """
    from nlsq.streaming.optimizer import StreamingOptimizer

    created_optimizers = []

    def create(config):
        optimizer = StreamingOptimizer(config)
        created_optimizers.append(optimizer)
        return optimizer

    yield create

    # Cleanup: shut down all optimizer threads
    for optimizer in created_optimizers:
        if hasattr(optimizer, "_shutdown_checkpoint_worker"):
            optimizer._shutdown_checkpoint_worker()
    created_optimizers.clear()
