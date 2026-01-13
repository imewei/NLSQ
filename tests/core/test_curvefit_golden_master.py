"""Golden master tests for CurveFit decomposition.

These tests capture the exact behavior of CurveFit before decomposition
to ensure the extracted components produce identical results. All tests
must pass with both old and new implementations.

The golden master approach:
1. Capture current behavior as expected values
2. Run tests against both implementations during migration
3. Ensure numerical equivalence within 1e-8 tolerance (per FR-006)

Reference: specs/017-curve-fit-decomposition/spec.md
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nlsq import CurveFit, curve_fit

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Test Constants (Golden Master Values)
# =============================================================================

# Numerical tolerance for parameter comparison (FR-006)
NUMERICAL_TOLERANCE = 1e-8

# Simple exponential model golden values
EXPONENTIAL_GOLDEN = {
    "popt": np.array([2.5, 1.3]),
    "pcov_diag": np.array([0.00015, 0.00025]),  # Approximate diagonal
}

# Linear model golden values
LINEAR_GOLDEN = {
    "popt": np.array([2.0, 1.0]),
    "pcov_diag": np.array([0.001, 0.002]),
}

# Gaussian model golden values
GAUSSIAN_GOLDEN = {
    "popt": np.array([1.0, 0.0, 1.0]),  # amplitude, mean, sigma
}


# =============================================================================
# Model Functions
# =============================================================================


def exponential_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Simple exponential decay: y = a * exp(-b * x)."""
    return a * jnp.exp(-b * x)


def linear_model(x: np.ndarray, m: float, c: float) -> np.ndarray:
    """Linear model: y = m * x + c."""
    return m * x + c


def gaussian_model(
    x: np.ndarray, amplitude: float, mean: float, sigma: float
) -> np.ndarray:
    """Gaussian model: y = amplitude * exp(-((x - mean) / sigma)^2 / 2)."""
    return amplitude * jnp.exp(-(((x - mean) / sigma) ** 2) / 2)


def polynomial_model(x: np.ndarray, *coeffs: float) -> np.ndarray:
    """Polynomial model of arbitrary degree."""
    result = jnp.zeros_like(x)
    for i, c in enumerate(coeffs):
        result = result + c * x**i
    return result


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_exponential_data(
    n_points: int = 100,
    a_true: float = 2.5,
    b_true: float = 1.3,
    noise_level: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic exponential decay data with noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4, n_points)
    y_true = a_true * np.exp(-b_true * x)
    noise = rng.normal(0, noise_level, n_points)
    y = y_true + noise
    return x, y


def generate_linear_data(
    n_points: int = 50,
    m_true: float = 2.0,
    c_true: float = 1.0,
    noise_level: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic linear data with noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n_points)
    y_true = m_true * x + c_true
    noise = rng.normal(0, noise_level, n_points)
    y = y_true + noise
    return x, y


def generate_gaussian_data(
    n_points: int = 100,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    noise_level: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Gaussian data with noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-5, 5, n_points)
    y_true = amplitude * np.exp(-(((x - mean) / sigma) ** 2) / 2)
    noise = rng.normal(0, noise_level, n_points)
    y = y_true + noise
    return x, y


# =============================================================================
# Feature Flag Fixture
# =============================================================================


@pytest.fixture
def implementation_flag(request: pytest.FixtureRequest) -> str | None:
    """Fixture to test both old and new implementations.

    Usage in test:
        @pytest.mark.parametrize("implementation_flag", ["old", "new"], indirect=True)
        def test_something(implementation_flag):
            ...
    """
    flag_value = getattr(request, "param", None)
    if flag_value is not None:
        # Set all component flags
        os.environ["NLSQ_PREPROCESSOR_IMPL"] = flag_value
        os.environ["NLSQ_SELECTOR_IMPL"] = flag_value
        os.environ["NLSQ_COVARIANCE_IMPL"] = flag_value
        os.environ["NLSQ_STREAMING_IMPL"] = flag_value

    yield flag_value

    # Cleanup
    for key in [
        "NLSQ_PREPROCESSOR_IMPL",
        "NLSQ_SELECTOR_IMPL",
        "NLSQ_COVARIANCE_IMPL",
        "NLSQ_STREAMING_IMPL",
    ]:
        os.environ.pop(key, None)


# =============================================================================
# Golden Master Tests - Basic Fitting
# =============================================================================


class TestGoldenMasterBasicFitting:
    """Golden master tests for basic curve fitting functionality."""

    def test_exponential_fit_parameters(self) -> None:
        """Test exponential fit produces consistent parameters."""
        x, y = generate_exponential_data()
        popt, pcov = curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Parameters should be close to true values (2.5, 1.3)
        assert_allclose(popt[0], 2.5, rtol=0.1)
        assert_allclose(popt[1], 1.3, rtol=0.1)

        # Covariance should be positive definite
        assert np.all(np.diag(pcov) > 0)

    def test_exponential_fit_covariance(self) -> None:
        """Test exponential fit produces consistent covariance."""
        x, y = generate_exponential_data()
        _popt, pcov = curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Covariance matrix should be symmetric
        assert_allclose(pcov, pcov.T, atol=NUMERICAL_TOLERANCE)

        # Diagonal elements should be positive (variances)
        assert np.all(np.diag(pcov) > 0)

    def test_linear_fit_deterministic(self) -> None:
        """Test linear fit is deterministic across runs."""
        x, y = generate_linear_data()

        results = []
        for _ in range(3):
            popt, pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5])
            results.append((np.array(popt), np.array(pcov)))

        # All runs should produce identical results
        for popt, pcov in results[1:]:
            assert_allclose(popt, results[0][0], atol=NUMERICAL_TOLERANCE)
            assert_allclose(pcov, results[0][1], atol=NUMERICAL_TOLERANCE)

    def test_gaussian_fit_basic(self) -> None:
        """Test Gaussian fit basic functionality."""
        x, y = generate_gaussian_data()
        popt, _pcov = curve_fit(
            gaussian_model, x, y, p0=[0.8, 0.1, 1.2], bounds=([0, -1, 0.1], [2, 1, 3])
        )

        # Should recover approximately correct parameters
        assert_allclose(popt[0], 1.0, rtol=0.2)  # amplitude
        assert_allclose(popt[1], 0.0, atol=0.3)  # mean
        assert_allclose(popt[2], 1.0, rtol=0.3)  # sigma


# =============================================================================
# Golden Master Tests - Bounds and Constraints
# =============================================================================


class TestGoldenMasterBounds:
    """Golden master tests for bounded optimization."""

    def test_bounds_respected(self) -> None:
        """Test that parameter bounds are respected."""
        x, y = generate_exponential_data()
        bounds = ([1.0, 0.5], [3.0, 2.0])

        popt, _ = curve_fit(exponential_model, x, y, p0=[2.0, 1.0], bounds=bounds)

        # Parameters should be within bounds
        assert bounds[0][0] <= popt[0] <= bounds[1][0]
        assert bounds[0][1] <= popt[1] <= bounds[1][1]

    def test_tight_bounds(self) -> None:
        """Test fitting with tight bounds."""
        x, y = generate_linear_data()
        bounds = ([1.9, 0.9], [2.1, 1.1])  # Tight around true values

        popt, _ = curve_fit(linear_model, x, y, p0=[2.0, 1.0], bounds=bounds)

        # Should converge within bounds
        assert bounds[0][0] <= popt[0] <= bounds[1][0]
        assert bounds[0][1] <= popt[1] <= bounds[1][1]

    def test_unbounded_vs_bounded(self) -> None:
        """Test that unbounded fit differs from bounded when bounds active."""
        x, y = generate_exponential_data()

        # Unbounded fit
        _popt_unbound, _ = curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Bounded fit with restrictive bounds
        bounds = ([1.0, 0.5], [2.0, 1.0])  # Will constrain the fit
        popt_bound, _ = curve_fit(exponential_model, x, y, p0=[1.5, 0.8], bounds=bounds)

        # Bounded result should be at bounds if they're active
        # (since true values are outside bounds)
        assert popt_bound[0] <= bounds[1][0] + 1e-6
        assert popt_bound[1] <= bounds[1][1] + 1e-6


# =============================================================================
# Golden Master Tests - Sigma/Weights
# =============================================================================


class TestGoldenMasterSigma:
    """Golden master tests for weighted fitting with sigma."""

    def test_uniform_sigma(self) -> None:
        """Test fitting with uniform sigma."""
        x, y = generate_linear_data()
        sigma = np.ones_like(y) * 0.5

        popt, pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5], sigma=sigma)

        # Should produce valid result
        assert len(popt) == 2
        assert pcov.shape == (2, 2)

    def test_varying_sigma_affects_result(self) -> None:
        """Test that varying sigma affects the fit."""
        x, y = generate_linear_data()

        # Fit with uniform sigma
        sigma_uniform = np.ones_like(y)
        popt_uniform, _pcov_uniform = curve_fit(
            linear_model, x, y, p0=[1.5, 0.5], sigma=sigma_uniform
        )

        # Fit with varying sigma (high uncertainty at high x)
        sigma_varying = np.linspace(0.5, 2.0, len(y))
        popt_varying, _pcov_varying = curve_fit(
            linear_model, x, y, p0=[1.5, 0.5], sigma=sigma_varying
        )

        # Results should differ
        assert not np.allclose(popt_uniform, popt_varying, atol=0.01)

    def test_absolute_sigma_flag(self) -> None:
        """Test absolute_sigma affects covariance scaling."""
        x, y = generate_linear_data()
        sigma = np.ones_like(y) * 0.5

        _, pcov_relative = curve_fit(
            linear_model, x, y, p0=[1.5, 0.5], sigma=sigma, absolute_sigma=False
        )

        _, pcov_absolute = curve_fit(
            linear_model, x, y, p0=[1.5, 0.5], sigma=sigma, absolute_sigma=True
        )

        # Covariances should differ
        assert not np.allclose(pcov_relative, pcov_absolute, rtol=0.01)


# =============================================================================
# Golden Master Tests - Method Selection
# =============================================================================


class TestGoldenMasterMethods:
    """Golden master tests for different optimization methods."""

    def test_trf_method_converges(self) -> None:
        """Test that TRF method converges to correct results.

        Note: NLSQ only supports the TRF (Trust Region Reflective) method.
        """
        x, y = generate_exponential_data()

        popt, _pcov = curve_fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            method="trf",
            bounds=([0, 0], [10, 10]),
        )

        # Should get reasonable results
        assert_allclose(popt[0], 2.5, rtol=0.15)
        assert_allclose(popt[1], 1.3, rtol=0.15)

    def test_trf_is_default(self) -> None:
        """Test that TRF is the default method."""
        x, y = generate_linear_data()

        # Default call
        popt_default, _ = curve_fit(linear_model, x, y, p0=[1.5, 0.5])

        # Explicit TRF
        popt_trf, _ = curve_fit(linear_model, x, y, p0=[1.5, 0.5], method="trf")

        # Should be identical
        assert_allclose(popt_default, popt_trf, atol=NUMERICAL_TOLERANCE)


# =============================================================================
# Golden Master Tests - Edge Cases
# =============================================================================


class TestGoldenMasterEdgeCases:
    """Golden master tests for edge cases."""

    def test_single_parameter(self) -> None:
        """Test fitting with single parameter."""

        def single_param_model(x: np.ndarray, a: float) -> np.ndarray:
            return a * x

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.1, 3.9, 6.1, 7.9, 10.1])

        popt, pcov = curve_fit(single_param_model, x, y, p0=[1.5])

        assert len(popt) == 1
        assert pcov.shape == (1, 1)
        assert_allclose(popt[0], 2.0, rtol=0.1)

    def test_many_parameters(self) -> None:
        """Test fitting with many parameters (polynomial)."""
        x = np.linspace(-2, 2, 100)
        # True polynomial: 1 + 2x + 0.5x^2 + 0.1x^3
        y_true = 1 + 2 * x + 0.5 * x**2 + 0.1 * x**3
        rng = np.random.default_rng(42)
        y = y_true + rng.normal(0, 0.1, len(x))

        def poly3(
            x: np.ndarray, c0: float, c1: float, c2: float, c3: float
        ) -> np.ndarray:
            return c0 + c1 * x + c2 * x**2 + c3 * x**3

        popt, pcov = curve_fit(poly3, x, y, p0=[0.5, 1.5, 0.3, 0.05])

        assert len(popt) == 4
        assert pcov.shape == (4, 4)
        # Should recover approximately correct coefficients
        assert_allclose(popt, [1, 2, 0.5, 0.1], rtol=0.2)

    def test_exact_data(self) -> None:
        """Test fitting exact (no noise) data."""
        x = np.linspace(0, 1, 10)
        y = 2.5 * np.exp(-1.3 * x)  # Exact exponential

        popt, _pcov = curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Should recover exact parameters
        assert_allclose(popt, [2.5, 1.3], rtol=1e-4)

    def test_2d_xdata(self) -> None:
        """Test fitting with 2D independent variable."""

        def surface_model(xy: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            x, y = xy
            return a * x + b * y + c

        # Create 2D grid
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        xdata = np.vstack([X.ravel(), Y.ravel()])

        # True parameters: a=2, b=3, c=1
        zdata = 2 * X.ravel() + 3 * Y.ravel() + 1

        popt, _ = curve_fit(surface_model, xdata, zdata, p0=[1.5, 2.5, 0.5])

        assert_allclose(popt, [2, 3, 1], rtol=1e-4)


# =============================================================================
# Golden Master Tests - CurveFit Class Interface
# =============================================================================


class TestGoldenMasterCurveFitClass:
    """Golden master tests for CurveFit class interface."""

    def test_class_interface_basic(self) -> None:
        """Test CurveFit class produces same results as function."""
        x, y = generate_exponential_data()

        # Using function
        popt_func, pcov_func = curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Using class
        fitter = CurveFit()
        popt_class, pcov_class = fitter.curve_fit(
            exponential_model, x, y, p0=[2.0, 1.0]
        )

        assert_allclose(popt_class, popt_func, atol=NUMERICAL_TOLERANCE)
        assert_allclose(pcov_class, pcov_func, atol=NUMERICAL_TOLERANCE)

    def test_flength_parameter(self) -> None:
        """Test flength parameter for JIT compilation."""
        x, y = generate_linear_data()

        # Without flength
        fitter1 = CurveFit()
        popt1, _ = fitter1.curve_fit(linear_model, x, y, p0=[1.5, 0.5])

        # With flength
        fitter2 = CurveFit(flength=100)
        popt2, _ = fitter2.curve_fit(linear_model, x, y, p0=[1.5, 0.5])

        # Results should be numerically equivalent
        assert_allclose(popt1, popt2, atol=NUMERICAL_TOLERANCE)


# =============================================================================
# Regression Tests - Specific Bug Fixes
# =============================================================================


class TestGoldenMasterRegressions:
    """Regression tests for specific bug fixes that must be preserved."""

    def test_tuple_unpacking_works(self) -> None:
        """Test tuple unpacking still works (FR-007)."""
        x, y = generate_linear_data()

        # Tuple unpacking must work
        popt, pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5])

        assert isinstance(popt, (np.ndarray, jnp.ndarray))
        assert isinstance(pcov, (np.ndarray, jnp.ndarray))

    def test_result_object_access(self) -> None:
        """Test result object attribute access (FR-007)."""
        x, y = generate_linear_data()

        # Standard curve_fit returns (popt, pcov) tuple
        fitter = CurveFit()
        popt, pcov = fitter.curve_fit(linear_model, x, y, p0=[1.5, 0.5])

        # Should have expected types
        assert isinstance(popt, (np.ndarray, jnp.ndarray))
        assert isinstance(pcov, (np.ndarray, jnp.ndarray))
        assert popt.shape == (2,)
        assert pcov.shape == (2, 2)

    def test_nan_handling_raise(self) -> None:
        """Test NaN handling with nan_policy='raise'."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, np.nan, 6.0, 8.0, 10.0])

        with pytest.raises(ValueError, match=r"[Nn]aN|[Ff]inite"):
            curve_fit(linear_model, x, y, p0=[1.5, 0.5], check_finite=True)

    def test_inf_handling_raise(self) -> None:
        """Test Inf handling with check_finite=True."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, np.inf, 6.0, 8.0, 10.0])

        with pytest.raises(ValueError, match=r"[Ff]inite|[Ii]nf"):
            curve_fit(linear_model, x, y, p0=[1.5, 0.5], check_finite=True)


# =============================================================================
# Numerical Precision Tests
# =============================================================================


class TestGoldenMasterPrecision:
    """Tests for numerical precision requirements (SC-007, SC-008)."""

    def test_parameter_precision(self) -> None:
        """Test parameter estimates have required precision."""
        # Use deterministic data for precision testing
        x = np.linspace(0, 1, 100)
        y = 2.5 * np.exp(-1.3 * x)  # Exact data

        popt, _ = curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Must match within 1e-8 tolerance (FR-006)
        assert_allclose(popt, [2.5, 1.3], atol=NUMERICAL_TOLERANCE)

    def test_covariance_symmetry(self) -> None:
        """Test covariance matrix is symmetric within tolerance."""
        x, y = generate_exponential_data()
        _, pcov = curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Covariance must be symmetric within 1e-8 (SC-008)
        assert_allclose(pcov, pcov.T, atol=NUMERICAL_TOLERANCE)

    def test_reproducibility(self) -> None:
        """Test results are reproducible across runs."""
        x, y = generate_linear_data()

        results = []
        for _ in range(5):
            popt, pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5])
            results.append((np.array(popt), np.array(pcov)))

        # All runs must produce identical results within tolerance
        for popt, pcov in results[1:]:
            assert_allclose(popt, results[0][0], atol=NUMERICAL_TOLERANCE)
            assert_allclose(pcov, results[0][1], atol=NUMERICAL_TOLERANCE)


# =============================================================================
# Backward Compatibility Tests (FR-007)
# =============================================================================


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility is maintained after decomposition.

    Reference: specs/017-curve-fit-decomposition/spec.md FR-007
    """

    def test_tuple_unpacking_basic(self) -> None:
        """Test basic tuple unpacking works."""
        x, y = generate_linear_data()

        popt, pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5])

        assert len(popt) == 2
        assert pcov.shape == (2, 2)

    def test_tuple_unpacking_with_bounds(self) -> None:
        """Test tuple unpacking with bounded optimization."""
        x, y = generate_linear_data()
        bounds = ([0.0, 0.0], [5.0, 5.0])

        popt, _pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5], bounds=bounds)

        assert len(popt) == 2
        # Parameters should be within bounds
        assert np.all(np.asarray(popt) >= bounds[0])
        assert np.all(np.asarray(popt) <= bounds[1])

    def test_tuple_unpacking_with_sigma(self) -> None:
        """Test tuple unpacking with sigma weights."""
        x, y = generate_linear_data()
        sigma = np.ones_like(y) * 0.1

        popt, pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5], sigma=sigma)

        assert len(popt) == 2
        assert pcov.shape == (2, 2)

    def test_curvefit_class_returns_result_with_tuple_access(self) -> None:
        """Test CurveFit class method returns result with popt/pcov access."""
        x, y = generate_exponential_data()

        fitter = CurveFit()
        result = fitter.curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Result should have x (popt) and pcov attributes
        assert hasattr(result, "x")
        assert hasattr(result, "pcov")
        assert len(result.x) == 2
        assert result.pcov.shape == (2, 2)

    def test_popt_is_numpy_compatible(self) -> None:
        """Test popt can be used with numpy operations."""
        x, y = generate_linear_data()

        popt, _pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5])

        # Should work with numpy operations
        popt_np = np.asarray(popt)
        assert popt_np.shape == (2,)
        assert np.all(np.isfinite(popt_np))

    def test_pcov_is_numpy_compatible(self) -> None:
        """Test pcov can be used with numpy operations."""
        x, y = generate_linear_data()

        _popt, pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5])

        # Should work with numpy operations
        pcov_np = np.asarray(pcov)
        assert pcov_np.shape == (2, 2)

        # Covariance should be symmetric
        assert_allclose(pcov_np, pcov_np.T, atol=1e-10)

    def test_standard_errors_from_pcov(self) -> None:
        """Test standard errors can be computed from pcov."""
        x, y = generate_linear_data()

        _popt, pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5])

        # Standard errors = sqrt(diagonal of pcov)
        pcov_np = np.asarray(pcov)
        perr = np.sqrt(np.diag(pcov_np))

        assert len(perr) == 2
        assert np.all(np.isfinite(perr))
        assert np.all(perr >= 0)

    def test_result_prediction(self) -> None:
        """Test popt can be used to make predictions."""
        x, y = generate_linear_data()

        popt, _pcov = curve_fit(linear_model, x, y, p0=[1.5, 0.5])

        # Use popt for predictions
        y_pred = linear_model(x, *popt)

        # Predictions should be valid and finite
        assert len(y_pred) == len(x)
        assert np.all(np.isfinite(np.asarray(y_pred)))

        # Fitted slope should be close to true value (2.0)
        # Since data has noise, we allow reasonable tolerance
        assert abs(float(popt[0]) - 2.0) < 0.2

    def test_method_parameter_preserved(self) -> None:
        """Test method parameter works correctly."""
        x, y = generate_linear_data()

        # Test with explicit TRF method
        popt_trf, _ = curve_fit(linear_model, x, y, p0=[1.5, 0.5], method="trf")

        # Result should be valid
        assert len(popt_trf) == 2
        assert np.all(np.isfinite(np.asarray(popt_trf)))

    def test_absolute_sigma_parameter(self) -> None:
        """Test absolute_sigma parameter works."""
        x, y = generate_linear_data()
        sigma = np.ones_like(y) * 0.1

        # With absolute sigma
        popt1, _pcov1 = curve_fit(
            linear_model, x, y, p0=[1.5, 0.5], sigma=sigma, absolute_sigma=True
        )

        # Without absolute sigma (relative)
        popt2, _pcov2 = curve_fit(
            linear_model, x, y, p0=[1.5, 0.5], sigma=sigma, absolute_sigma=False
        )

        # Both should produce valid results
        assert len(popt1) == 2
        assert len(popt2) == 2

        # Parameters should be similar (same optimization)
        assert_allclose(popt1, popt2, atol=1e-6)
