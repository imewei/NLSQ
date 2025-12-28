"""Tests for fit() function integration.

This module tests the unified fit() entry point from nlsq/minpack.py (Task Group 4).

Tests cover:
- fit() with workflow="auto" returns CurveFitResult
- fit() with named workflow (e.g., "standard")
- fit() with custom config object
- fit() passes **kwargs to underlying curve_fit
- fit() with goal="quality" uses tighter tolerances
- fit() with goal="fast" uses looser tolerances
- fit() coexists with curve_fit() (both callable)
- fit() error handling for invalid workflow names
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.minpack import CurveFit, curve_fit, fit
from nlsq.core.workflow import OptimizationGoal
from nlsq.result import CurveFitResult


# Test model function
def exponential_model(x, a, b):
    """Simple exponential model for testing."""
    return a * jnp.exp(-b * x)


def polynomial_model(x, a, b, c):
    """Simple polynomial model for testing."""
    return a * x**2 + b * x + c


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    x = np.linspace(0, 4, 50)
    # True parameters: a=2.5, b=1.3
    y_true = 2.5 * np.exp(-1.3 * x)
    y = y_true + 0.1 * np.random.normal(size=len(x))
    return x, y


@pytest.fixture
def sample_data_poly():
    """Generate polynomial sample data for testing."""
    np.random.seed(42)
    x = np.linspace(-2, 2, 100)
    # True parameters: a=1.0, b=0.5, c=2.0
    y_true = 1.0 * x**2 + 0.5 * x + 2.0
    y = y_true + 0.2 * np.random.normal(size=len(x))
    return x, y


class TestFitWithWorkflowAuto:
    """Test fit() with workflow='auto' returns CurveFitResult."""

    def test_fit_workflow_auto_returns_curve_fit_result(self, sample_data):
        """Test fit() with workflow='auto' returns CurveFitResult."""
        x, y = sample_data
        result = fit(exponential_model, x, y, p0=[2.0, 1.0], workflow="auto")

        # Should return CurveFitResult
        assert isinstance(result, CurveFitResult)
        # Should have popt and pcov attributes
        assert hasattr(result, "popt")
        assert hasattr(result, "pcov")
        assert result.popt is not None
        assert result.pcov is not None
        # Parameters should be close to true values
        assert len(result.popt) == 2
        assert np.isclose(result.popt[0], 2.5, rtol=0.2)  # a ~ 2.5
        assert np.isclose(result.popt[1], 1.3, rtol=0.2)  # b ~ 1.3

    def test_fit_workflow_auto_tuple_unpacking(self, sample_data):
        """Test fit() with workflow='auto' supports tuple unpacking."""
        x, y = sample_data
        popt, pcov = fit(exponential_model, x, y, p0=[2.0, 1.0], workflow="auto")

        # Should be able to unpack as tuple
        assert len(popt) == 2
        assert pcov.shape == (2, 2)


class TestFitWithNamedWorkflow:
    """Test fit() with named workflow (e.g., 'standard')."""

    def test_fit_with_standard_workflow(self, sample_data):
        """Test fit() with workflow='standard'."""
        x, y = sample_data
        result = fit(exponential_model, x, y, p0=[2.0, 1.0], workflow="standard")

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None
        assert len(result.popt) == 2

    def test_fit_with_quality_workflow(self, sample_data):
        """Test fit() with workflow='quality'."""
        x, y = sample_data
        result = fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            workflow="quality",
            bounds=([0, 0], [10, 5]),
        )

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None


class TestFitWithCustomConfigObject:
    """Test fit() with custom config object."""

    def test_fit_with_ldmemoryconfig(self, sample_data):
        """Test fit() accepts LDMemoryConfig object."""
        from nlsq.streaming.large_dataset import LDMemoryConfig

        x, y = sample_data
        config = LDMemoryConfig(
            memory_limit_gb=4.0,
            min_chunk_size=100,
            max_chunk_size=10000,
        )
        result = fit(exponential_model, x, y, p0=[2.0, 1.0], workflow=config)

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None

    def test_fit_with_hybridstreamingconfig(self, sample_data):
        """Test fit() accepts HybridStreamingConfig object."""
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        x, y = sample_data
        config = HybridStreamingConfig(
            normalize=True,
            warmup_iterations=100,
        )
        result = fit(exponential_model, x, y, p0=[2.0, 1.0], workflow=config)

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None


class TestFitPassesKwargs:
    """Test fit() passes **kwargs to underlying curve_fit."""

    def test_fit_passes_bounds(self, sample_data):
        """Test fit() passes bounds parameter."""
        x, y = sample_data
        bounds = ([0, 0], [10, 5])
        result = fit(exponential_model, x, y, p0=[2.0, 1.0], bounds=bounds)

        # Parameters should be within bounds
        assert result.popt[0] >= 0 and result.popt[0] <= 10
        assert result.popt[1] >= 0 and result.popt[1] <= 5

    def test_fit_passes_sigma(self, sample_data):
        """Test fit() passes sigma parameter."""
        x, y = sample_data
        sigma = np.ones_like(y) * 0.1

        # Should not raise
        result = fit(exponential_model, x, y, p0=[2.0, 1.0], sigma=sigma)
        assert result.popt is not None

    def test_fit_passes_method(self, sample_data):
        """Test fit() passes method parameter."""
        x, y = sample_data

        # Should work with different methods
        result = fit(exponential_model, x, y, p0=[2.0, 1.0], method="trf")
        assert result.popt is not None


class TestFitWithQualityGoal:
    """Test fit() with goal='quality' uses tighter tolerances."""

    def test_fit_goal_quality_works(self, sample_data):
        """Test fit() with goal='quality' completes successfully."""
        x, y = sample_data
        result = fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            goal="quality",
            bounds=([0, 0], [10, 5]),
        )

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None
        # Quality goal should give good fit
        assert np.isclose(result.popt[0], 2.5, rtol=0.2)

    def test_fit_goal_quality_with_optimization_goal_enum(self, sample_data):
        """Test fit() accepts OptimizationGoal enum."""
        x, y = sample_data
        result = fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            goal=OptimizationGoal.QUALITY,
            bounds=([0, 0], [10, 5]),
        )

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None


class TestFitWithFastGoal:
    """Test fit() with goal='fast' uses looser tolerances."""

    def test_fit_goal_fast_works(self, sample_data):
        """Test fit() with goal='fast' completes successfully."""
        x, y = sample_data
        result = fit(exponential_model, x, y, p0=[2.0, 1.0], goal="fast")

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None

    def test_fit_goal_fast_with_optimization_goal_enum(self, sample_data):
        """Test fit() accepts OptimizationGoal.FAST enum."""
        x, y = sample_data
        result = fit(
            exponential_model,
            x,
            y,
            p0=[2.0, 1.0],
            goal=OptimizationGoal.FAST,
        )

        assert isinstance(result, CurveFitResult)
        assert result.popt is not None


class TestFitCoexistsWithCurveFit:
    """Test fit() coexists with curve_fit() (both callable)."""

    def test_both_fit_and_curve_fit_callable(self, sample_data):
        """Test both fit() and curve_fit() can be called."""
        x, y = sample_data

        # fit() should work
        result1 = fit(exponential_model, x, y, p0=[2.0, 1.0])

        # curve_fit() should also work
        result2 = curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Both should return valid results
        assert result1.popt is not None
        assert result2.popt is not None

        # Both should give similar results
        np.testing.assert_allclose(result1.popt, result2.popt, rtol=0.1)

    def test_fit_and_curve_fit_same_result(self, sample_data):
        """Test fit() and curve_fit() give equivalent results."""
        x, y = sample_data
        np.random.seed(42)

        # Both with same parameters should give same result
        popt1, _ = fit(exponential_model, x, y, p0=[2.0, 1.0], workflow="auto")
        popt2, _ = curve_fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Parameters should be very close
        np.testing.assert_allclose(popt1, popt2, rtol=0.1)


class TestFitErrorHandling:
    """Test fit() error handling for invalid workflow names."""

    def test_fit_invalid_workflow_name_raises(self, sample_data):
        """Test fit() raises error for invalid workflow name."""
        x, y = sample_data

        with pytest.raises(ValueError, match="Unknown workflow"):
            fit(exponential_model, x, y, p0=[2.0, 1.0], workflow="nonexistent_workflow")

    def test_fit_invalid_goal_string_raises(self, sample_data):
        """Test fit() raises error for invalid goal string."""
        x, y = sample_data

        with pytest.raises(ValueError, match="Unknown goal"):
            fit(exponential_model, x, y, p0=[2.0, 1.0], goal="invalid_goal")

    def test_fit_empty_data_raises(self):
        """Test fit() raises error for empty data."""
        x = np.array([])
        y = np.array([])

        with pytest.raises(ValueError):
            fit(exponential_model, x, y, p0=[2.0, 1.0])


class TestFitIntegration:
    """Integration tests for fit() function."""

    def test_fit_with_polynomial_model(self, sample_data_poly):
        """Test fit() works with different model functions."""
        x, y = sample_data_poly
        result = fit(polynomial_model, x, y, p0=[1.0, 0.5, 2.0])

        assert isinstance(result, CurveFitResult)
        assert len(result.popt) == 3
        # Check parameters are close to true values
        np.testing.assert_allclose(result.popt, [1.0, 0.5, 2.0], rtol=0.3)

    def test_fit_preserves_xdata_ydata_in_result(self, sample_data):
        """Test fit() preserves xdata and ydata in result."""
        x, y = sample_data
        result = fit(exponential_model, x, y, p0=[2.0, 1.0])

        # Result should have xdata and ydata
        assert hasattr(result, "xdata")
        assert hasattr(result, "ydata")
        np.testing.assert_array_equal(result.xdata, x)
        np.testing.assert_array_equal(result.ydata, y)
