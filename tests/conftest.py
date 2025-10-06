"""
Pytest configuration and shared fixtures for NLSQ test suite.

Provides common test functions, data generators, and fixtures for
scientific computing tests.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp


# ============================================================================
# Test Function Fixtures
# ============================================================================


@pytest.fixture
def linear_func():
    """Linear function: f(x, a, b) = a*x + b"""
    def f(x, a, b):
        return a * x + b
    return f


@pytest.fixture
def quadratic_func():
    """Quadratic function: f(x, a, b, c) = a*x^2 + b*x + c"""
    def f(x, a, b, c):
        return a * x**2 + b * x + c
    return f


@pytest.fixture
def exponential_func():
    """Exponential decay: f(x, a, b, c) = a*exp(-b*x) + c"""
    def f(x, a, b, c):
        return a * jnp.exp(-b * x) + c
    return f


@pytest.fixture
def gaussian_func():
    """Gaussian function: f(x, amp, mu, sigma) = amp*exp(-(x-mu)^2/(2*sigma^2))"""
    def f(x, amp, mu, sigma):
        return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return f


@pytest.fixture
def sinusoidal_func():
    """Sinusoidal function: f(x, a, b, c, d) = a*sin(b*x + c) + d"""
    def f(x, a, b, c, d):
        return a * jnp.sin(b * x + c) + d
    return f


@pytest.fixture
def polynomial_func():
    """General polynomial function with variable parameters"""
    def f(x, *coeffs):
        result = jnp.zeros_like(x)
        for i, c in enumerate(coeffs):
            result = result + c * x**i
        return result
    return f


# ============================================================================
# Data Generator Fixtures
# ============================================================================


@pytest.fixture
def linear_data():
    """Generate clean linear data"""
    def generate(n_points=100, a=2.0, b=1.0, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 10, n_points)
        y = a * x + b
        return x, y, np.array([a, b])
    return generate


@pytest.fixture
def noisy_linear_data():
    """Generate noisy linear data"""
    def generate(n_points=100, a=2.0, b=1.0, noise_level=0.1, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 10, n_points)
        y = a * x + b + np.random.normal(0, noise_level, n_points)
        return x, y, np.array([a, b])
    return generate


@pytest.fixture
def exponential_data():
    """Generate exponential decay data"""
    def generate(n_points=100, a=2.5, b=1.3, c=0.1, noise_level=0.01, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 5, n_points)
        y = a * np.exp(-b * x) + c
        if noise_level > 0:
            y = y + np.random.normal(0, noise_level, n_points)
        return x, y, np.array([a, b, c])
    return generate


@pytest.fixture
def gaussian_data():
    """Generate Gaussian data"""
    def generate(n_points=100, amp=1.5, mu=0.5, sigma=1.2, noise_level=0.01, seed=42):
        np.random.seed(seed)
        x = np.linspace(-5, 5, n_points)
        y = amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        if noise_level > 0:
            y = y + np.random.normal(0, noise_level, n_points)
        return x, y, np.array([amp, mu, sigma])
    return generate


@pytest.fixture
def outlier_data():
    """Generate data with outliers"""
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
        assert result.success, f"Optimization failed: {result.message if hasattr(result, 'message') else 'unknown reason'}"
        assert hasattr(result, "x"), "Result missing parameters 'x'"
        assert np.all(np.isfinite(result.x)), "Parameters contain non-finite values"
    return check


@pytest.fixture
def assert_parameters_close():
    """Helper to assert parameters are close to expected values"""
    def check(popt, expected, rtol=1e-2, atol=1e-3):
        np.testing.assert_allclose(
            popt, expected, rtol=rtol, atol=atol,
            err_msg=f"Parameters {popt} not close to expected {expected}"
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
    pass


@pytest.fixture
def temp_array_pool():
    """Provide a temporary array pool that's cleaned up after test"""
    from nlsq.memory_manager import get_memory_manager

    manager = get_memory_manager()
    initial_pool_size = len(manager.memory_pool)

    yield manager

    # Clean up arrays added during test
    manager.clear_pool()
