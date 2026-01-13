"""Unit tests for OptimizationSelector component.

Tests for parameter detection, method selection, bounds preparation,
and optimization configuration.

Reference: specs/017-curve-fit-decomposition/spec.md FR-002, FR-020
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nlsq.core.orchestration.optimization_selector import OptimizationSelector
from nlsq.interfaces.orchestration_protocol import OptimizationConfig

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def selector() -> OptimizationSelector:
    """Create an OptimizationSelector instance."""
    return OptimizationSelector()


@pytest.fixture
def linear_model():
    """Simple linear model function with 2 parameters."""

    def model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b

    return model


@pytest.fixture
def quadratic_model():
    """Quadratic model function with 3 parameters."""

    def model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * x**2 + b * x + c

    return model


@pytest.fixture
def exponential_model():
    """Exponential decay model function with 2 parameters."""

    def model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        import jax.numpy as jnp

        return a * jnp.exp(-b * x)

    return model


@pytest.fixture
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple test data."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.5, 4.8, 7.1, 9.5, 11.8])
    return x, y


@pytest.fixture
def bounded_data() -> tuple[np.ndarray, np.ndarray]:
    """Data for bounded optimization tests."""
    x = np.linspace(0, 10, 20)
    y = 2.5 * x + 1.0 + np.random.default_rng(42).normal(0, 0.1, 20)
    return x, y


# =============================================================================
# Test OptimizationConfig Result
# =============================================================================


class TestOptimizationConfigResult:
    """Tests for OptimizationConfig result object."""

    def test_returns_optimization_config(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test select returns OptimizationConfig instance."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert isinstance(result, OptimizationConfig)

    def test_optimization_config_has_required_fields(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test OptimizationConfig has all required attributes."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert hasattr(result, "method")
        assert hasattr(result, "tr_solver")
        assert hasattr(result, "n_params")
        assert hasattr(result, "p0")
        assert hasattr(result, "bounds")
        assert hasattr(result, "max_nfev")
        assert hasattr(result, "ftol")
        assert hasattr(result, "xtol")
        assert hasattr(result, "gtol")
        assert hasattr(result, "jac")
        assert hasattr(result, "x_scale")

    def test_n_params_matches_model(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test n_params reflects model parameter count."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert result.n_params == 2  # Linear model has 2 params (a, b)


# =============================================================================
# Test Parameter Count Detection
# =============================================================================


class TestParameterCountDetection:
    """Tests for parameter count detection."""

    def test_detects_linear_model_params(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test detects 2 params for linear model."""
        x, _y = simple_data

        n_params = selector.detect_parameter_count(
            f=linear_model,
            xdata=x,
        )

        assert n_params == 2

    def test_detects_quadratic_model_params(
        self, selector: OptimizationSelector, quadratic_model, simple_data
    ) -> None:
        """Test detects 3 params for quadratic model."""
        x, _y = simple_data

        n_params = selector.detect_parameter_count(
            f=quadratic_model,
            xdata=x,
        )

        assert n_params == 3

    def test_uses_p0_length_if_provided(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test uses p0 length when explicitly provided."""
        x, y = simple_data
        p0 = np.array([1.0, 2.0])

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
            p0=p0,
        )

        assert result.n_params == 2
        assert_allclose(np.asarray(result.p0), p0)

    def test_raises_on_invalid_function(
        self, selector: OptimizationSelector, simple_data
    ) -> None:
        """Test raises ValueError for function with no parameters."""
        x, _y = simple_data

        def invalid_model():
            return 1.0

        with pytest.raises(ValueError, match=r"Parameter|parameter|Fit|fit"):
            selector.detect_parameter_count(
                f=invalid_model,
                xdata=x,
            )


# =============================================================================
# Test Method Selection
# =============================================================================


class TestMethodSelection:
    """Tests for optimization method selection."""

    def test_default_method_is_trf(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test default method is TRF."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert result.method == "trf"

    def test_explicit_method_is_used(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test explicit method overrides default."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
            method="lm",
        )

        assert result.method == "lm"

    def test_dogbox_method(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test dogbox method selection."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
            method="dogbox",
        )

        assert result.method == "dogbox"


# =============================================================================
# Test Bounds Handling
# =============================================================================


class TestBoundsHandling:
    """Tests for bounds preparation and validation."""

    def test_no_bounds_default(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test default bounds are infinite."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        lb, ub = result.bounds
        assert np.all(np.asarray(lb) == -np.inf)
        assert np.all(np.asarray(ub) == np.inf)

    def test_explicit_bounds_used(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test explicit bounds are preserved."""
        x, y = simple_data
        bounds = ([0.0, -10.0], [10.0, 10.0])

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
            bounds=bounds,
        )

        lb, ub = result.bounds
        assert_allclose(np.asarray(lb), bounds[0])
        assert_allclose(np.asarray(ub), bounds[1])

    def test_p0_clipped_to_bounds(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test p0 is clipped to bounds."""
        x, y = simple_data
        p0 = np.array([15.0, 5.0])  # 15.0 outside upper bound
        bounds = ([0.0, -10.0], [10.0, 10.0])

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
            p0=p0,
            bounds=bounds,
        )

        p0_result = np.asarray(result.p0)
        assert p0_result[0] <= 10.0  # Should be clipped to upper bound
        assert p0_result[1] == 5.0  # Should be unchanged


# =============================================================================
# Test Initial Guess Generation
# =============================================================================


class TestInitialGuessGeneration:
    """Tests for automatic initial guess generation."""

    def test_auto_initial_guess_no_bounds(self, selector: OptimizationSelector) -> None:
        """Test auto initial guess without bounds."""
        p0 = selector.auto_initial_guess(
            n_params=3,
            bounds=None,
        )

        assert len(np.asarray(p0)) == 3
        # Default should be ones when no bounds
        assert_allclose(np.asarray(p0), np.ones(3))

    def test_auto_initial_guess_with_bounds(
        self, selector: OptimizationSelector
    ) -> None:
        """Test auto initial guess uses bounds midpoint."""
        import jax.numpy as jnp

        bounds = (jnp.array([0.0, 1.0]), jnp.array([10.0, 5.0]))

        p0 = selector.auto_initial_guess(
            n_params=2,
            bounds=bounds,
        )

        # Should be midpoint of bounds
        expected = np.array([5.0, 3.0])
        assert_allclose(np.asarray(p0), expected)

    def test_auto_initial_guess_infinite_bounds(
        self, selector: OptimizationSelector
    ) -> None:
        """Test auto initial guess with one infinite bound."""
        import jax.numpy as jnp

        bounds = (jnp.array([0.0, -jnp.inf]), jnp.array([10.0, jnp.inf]))

        p0 = selector.auto_initial_guess(
            n_params=2,
            bounds=bounds,
        )

        p0_arr = np.asarray(p0)
        # First param should be midpoint
        assert p0_arr[0] == 5.0
        # Second param should be 1.0 (default for infinite)
        assert p0_arr[1] == 1.0


# =============================================================================
# Test Solver Configuration
# =============================================================================


class TestSolverConfiguration:
    """Tests for trust region solver configuration."""

    def test_default_tr_solver(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test default tr_solver is selected."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        # For small problems, should use exact
        assert result.tr_solver in ("exact", "lsmr", None)

    def test_explicit_tr_solver(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test explicit tr_solver is used."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
            tr_solver="lsmr",
        )

        assert result.tr_solver == "lsmr"


# =============================================================================
# Test Tolerance Settings
# =============================================================================


class TestToleranceSettings:
    """Tests for tolerance configuration."""

    def test_default_tolerances(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test default tolerance values."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert result.ftol == 1e-8
        assert result.xtol == 1e-8
        assert result.gtol == 1e-8

    def test_custom_tolerances(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test custom tolerance values."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
            ftol=1e-10,
            xtol=1e-6,
            gtol=1e-12,
        )

        assert result.ftol == 1e-10
        assert result.xtol == 1e-6
        assert result.gtol == 1e-12


# =============================================================================
# Test Max Function Evaluations
# =============================================================================


class TestMaxFunctionEvaluations:
    """Tests for max_nfev configuration."""

    def test_auto_max_nfev(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test auto max_nfev is computed."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        # Auto max_nfev should be based on data/param size
        assert result.max_nfev > 0

    def test_explicit_max_nfev(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test explicit max_nfev is used."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
            max_nfev=500,
        )

        assert result.max_nfev == 500


# =============================================================================
# Test x_scale Configuration
# =============================================================================


class TestXScaleConfiguration:
    """Tests for x_scale configuration."""

    def test_default_x_scale(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test default x_scale."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        # Default can be 'jac' or 1.0 or array
        assert result.x_scale is not None

    def test_jac_x_scale(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test x_scale='jac' setting."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
            x_scale="jac",
        )

        assert result.x_scale == "jac"

    def test_array_x_scale(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test array x_scale setting."""
        x, y = simple_data
        x_scale = np.array([1.0, 0.1])

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
            x_scale=x_scale,
        )

        assert_allclose(np.asarray(result.x_scale), x_scale)


# =============================================================================
# Test Immutability
# =============================================================================


class TestImmutability:
    """Tests for OptimizationConfig immutability."""

    def test_optimization_config_is_frozen(
        self, selector: OptimizationSelector, linear_model, simple_data
    ) -> None:
        """Test OptimizationConfig cannot be modified."""
        x, y = simple_data

        result = selector.select(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        with pytest.raises((AttributeError, TypeError)):
            result.method = "lm"  # type: ignore[misc]


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_parameter_model(
        self, selector: OptimizationSelector, simple_data
    ) -> None:
        """Test model with single parameter."""
        x, y = simple_data

        def single_param_model(x, a):
            return a * x

        result = selector.select(
            f=single_param_model,
            xdata=x,
            ydata=y,
        )

        assert result.n_params == 1

    def test_many_parameter_model(
        self, selector: OptimizationSelector, simple_data
    ) -> None:
        """Test model with many parameters."""
        x, y = simple_data

        def many_param_model(x, a, b, c, d, e, f):
            return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

        result = selector.select(
            f=many_param_model,
            xdata=x,
            ydata=y,
        )

        assert result.n_params == 6

    def test_2d_xdata(self, selector: OptimizationSelector) -> None:
        """Test with 2D xdata (multiple independent variables)."""
        xy = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        z = np.array([3, 6, 9, 12, 15])

        def surface_model(xy, a, b, c):
            x, y = xy
            return a * x + b * y + c

        result = selector.select(
            f=surface_model,
            xdata=xy,
            ydata=z,
        )

        assert result.n_params == 3
