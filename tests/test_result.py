"""Tests for enhanced CurveFitResult functionality."""

import warnings

import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.result import CurveFitResult

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_data():
    """Generate simple exponential decay data for testing."""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 10 * np.exp(-0.5 * x) + 2
    y = y_true + np.random.normal(0, 0.5, size=len(x))
    return x, y


@pytest.fixture
def simple_model():
    """Simple exponential decay model."""
    import jax.numpy as jnp

    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    return model


@pytest.fixture
def simple_result(simple_data, simple_model):
    """Generate a simple CurveFitResult for testing."""
    x, y = simple_data
    result = curve_fit(simple_model, x, y, p0=[10, 0.5, 2])
    return result


# ============================================================================
# Unit Tests - Backward Compatibility
# ============================================================================


def test_tuple_unpacking(simple_data, simple_model):
    """Test backward compatibility with tuple unpacking."""
    x, y = simple_data
    popt, pcov = curve_fit(simple_model, x, y, p0=[10, 0.5, 2])

    assert isinstance(popt, np.ndarray)
    assert isinstance(pcov, np.ndarray)
    assert popt.shape == (3,)
    assert pcov.shape == (3, 3)


def test_enhanced_result_type(simple_result):
    """Test that enhanced result is CurveFitResult instance."""
    assert isinstance(simple_result, CurveFitResult)


def test_both_usage_patterns(simple_data, simple_model):
    """Test both tuple unpacking and enhanced result usage."""
    x, y = simple_data

    # Pattern 1: Tuple unpacking
    popt1, pcov1 = curve_fit(simple_model, x, y, p0=[10, 0.5, 2])

    # Pattern 2: Enhanced result
    result = curve_fit(simple_model, x, y, p0=[10, 0.5, 2])
    popt2, pcov2 = result  # Can still unpack

    # Should be identical
    np.testing.assert_array_almost_equal(popt1, popt2)
    np.testing.assert_array_almost_equal(pcov1, pcov2)


# ============================================================================
# Unit Tests - Statistical Properties
# ============================================================================


def test_r_squared(simple_result):
    """Test R² calculation."""
    r2 = simple_result.r_squared

    # R² should be between 0 and 1 for good fit
    assert 0 <= r2 <= 1
    # Should be high for this clean data
    assert r2 > 0.9


def test_adjusted_r_squared(simple_result):
    """Test adjusted R² calculation."""
    adj_r2 = simple_result.adj_r_squared

    # Adjusted R² should be less than or equal to R²
    assert adj_r2 <= simple_result.r_squared
    # Should be high for this clean data
    assert adj_r2 > 0.9


def test_rmse(simple_result):
    """Test RMSE calculation."""
    rmse = simple_result.rmse

    # RMSE should be positive
    assert rmse > 0
    # Should be close to noise level (0.5)
    assert rmse < 1.0


def test_mae(simple_result):
    """Test MAE calculation."""
    mae = simple_result.mae

    # MAE should be positive
    assert mae > 0
    # MAE should be less than RMSE for normal distribution
    assert mae < simple_result.rmse


def test_aic(simple_result):
    """Test AIC calculation."""
    aic = simple_result.aic

    # AIC should be finite
    assert np.isfinite(aic)


def test_bic(simple_result):
    """Test BIC calculation."""
    bic = simple_result.bic

    # BIC should be finite
    assert np.isfinite(bic)
    # BIC should be greater than AIC (penalizes complexity more)
    assert bic > simple_result.aic


def test_residuals(simple_result):
    """Test residuals calculation."""
    residuals = simple_result.residuals

    # Should be an array
    assert isinstance(residuals, np.ndarray)
    # Should match data length
    assert len(residuals) == len(simple_result.ydata)
    # Mean should be close to zero for good fit
    assert abs(np.mean(residuals)) < 0.1


def test_predictions(simple_result):
    """Test predictions calculation."""
    predictions = simple_result.predictions

    # Should be an array
    assert isinstance(predictions, np.ndarray)
    # Should match data length
    assert len(predictions) == len(simple_result.ydata)
    # Predictions should be close to actual data
    correlation = np.corrcoef(predictions, simple_result.ydata)[0, 1]
    assert correlation > 0.95


def test_predictions_caching(simple_result):
    """Test that predictions are cached."""
    pred1 = simple_result.predictions
    pred2 = simple_result.predictions

    # Should return the same object (cached)
    assert pred1 is pred2


# ============================================================================
# Unit Tests - Confidence Intervals
# ============================================================================


def test_confidence_intervals_default(simple_result):
    """Test confidence intervals with default alpha."""
    ci = simple_result.confidence_intervals()

    # Should return array of shape (n_params, 2)
    assert ci.shape == (3, 2)
    # Lower bound should be less than upper bound
    assert np.all(ci[:, 0] < ci[:, 1])
    # Parameters should be within their confidence intervals
    assert np.all(simple_result.popt >= ci[:, 0])
    assert np.all(simple_result.popt <= ci[:, 1])


def test_confidence_intervals_custom_alpha(simple_result):
    """Test confidence intervals with custom alpha."""
    ci_95 = simple_result.confidence_intervals(alpha=0.95)
    ci_99 = simple_result.confidence_intervals(alpha=0.99)

    # 99% CI should be wider than 95% CI
    width_95 = ci_95[:, 1] - ci_95[:, 0]
    width_99 = ci_99[:, 1] - ci_99[:, 0]
    assert np.all(width_99 > width_95)


def test_confidence_intervals_no_covariance():
    """Test confidence intervals when covariance is not available."""
    # Create result without covariance
    from nlsq.core._optimize import OptimizeResult

    result = CurveFitResult(OptimizeResult())
    result["popt"] = np.array([1.0, 2.0, 3.0])
    result["ydata"] = np.random.randn(100)

    # Should raise AttributeError or similar
    with pytest.raises((AttributeError, KeyError)):
        result.confidence_intervals()


# ============================================================================
# Unit Tests - Prediction Intervals
# ============================================================================


def test_prediction_interval_default(simple_result):
    """Test prediction interval with default alpha."""
    pi = simple_result.prediction_interval()

    # Should return array of shape (n_points, 2)
    n_points = len(simple_result.xdata)
    assert pi.shape == (n_points, 2)
    # Lower bound should be less than upper bound
    assert np.all(pi[:, 0] < pi[:, 1])


def test_prediction_interval_custom_alpha(simple_result):
    """Test prediction interval with custom alpha."""
    pi_95 = simple_result.prediction_interval(alpha=0.95)
    pi_99 = simple_result.prediction_interval(alpha=0.99)

    # 99% PI should be wider than 95% PI
    width_95 = pi_95[:, 1] - pi_95[:, 0]
    width_99 = pi_99[:, 1] - pi_99[:, 0]
    assert np.all(width_99 > width_95)


def test_prediction_interval_custom_x(simple_result):
    """Test prediction interval at custom x values."""
    x_new = np.array([0.5, 1.0, 1.5])
    pi = simple_result.prediction_interval(x=x_new)

    # Should return array of shape (len(x_new), 2)
    assert pi.shape == (len(x_new), 2)


# ============================================================================
# Unit Tests - Plotting
# ============================================================================


def test_plot_basic(simple_result):
    """Test basic plotting functionality."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    # Should create plot without error
    fig, ax = plt.subplots()
    simple_result.plot(ax=ax, show_residuals=False)
    plt.close(fig)


def test_plot_with_residuals(simple_result):
    """Test plotting with residuals."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    # Should create plot with residuals
    simple_result.plot(show_residuals=True)
    plt.close("all")


# ============================================================================
# Unit Tests - Summary
# ============================================================================


def test_summary_output(simple_result):
    """Test summary output format."""
    # Capture output
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()

    try:
        simple_result.summary()
        output = buffer.getvalue()
    finally:
        sys.stdout = old_stdout

    # Check output contains expected sections
    assert "Curve Fit Summary" in output
    assert "R²" in output
    assert "RMSE" in output
    assert "AIC" in output
    assert "BIC" in output


def test_summary_parameters(simple_result):
    """Test summary includes parameter information."""
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()

    try:
        simple_result.summary()
        output = buffer.getvalue()
    finally:
        sys.stdout = old_stdout

    # Should include parameter names
    assert "p0" in output
    assert "p1" in output
    assert "p2" in output


# ============================================================================
# Edge Cases
# ============================================================================


def test_r_squared_constant_data():
    """Test R² with constant data (edge case)."""
    import jax.numpy as jnp

    def model(x, a):
        return a * jnp.ones_like(x)

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # Constant

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = curve_fit(model, x, y, p0=[5.0])
        r2 = result.r_squared

        # Should warn about undefined R²
        assert any(
            "Total sum of squares is zero" in str(warning.message) for warning in w
        )
        # R² should be NaN
        assert np.isnan(r2)


def test_aic_zero_rss():
    """Test AIC with zero residual sum of squares (edge case)."""

    def model(x, a, b):
        return a * x + b

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])  # Perfect linear fit

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = curve_fit(model, x, y, p0=[2.0, 0.0])
        _ = result.aic  # Check property access doesn't raise error

        # May warn about undefined AIC if RSS is exactly zero
        # AIC may be NaN or very negative


def test_missing_model_in_result():
    """Test behavior when model is not stored in result."""
    from nlsq.core._optimize import OptimizeResult

    result = CurveFitResult(OptimizeResult())
    result["popt"] = np.array([1.0, 2.0, 3.0])
    result["xdata"] = np.array([1, 2, 3])
    result["ydata"] = np.array([2, 4, 6])
    # Note: 'model' is not set

    # Should raise error when trying to compute predictions
    with pytest.raises((AttributeError, KeyError)):
        _ = result.predictions


def test_missing_data_in_result():
    """Test behavior when data is not stored in result."""

    from nlsq.core._optimize import OptimizeResult

    def model(x, a):
        return a * x

    result = CurveFitResult(OptimizeResult())
    result["popt"] = np.array([2.0])
    result["model"] = model
    # Note: 'xdata' and 'ydata' are not set

    # Should raise error when trying to compute predictions
    with pytest.raises((AttributeError, KeyError)):
        _ = result.predictions


# ============================================================================
# Integration Tests
# ============================================================================


def test_workflow_statistical_analysis(simple_data, simple_model):
    """Test complete workflow: fit → analyze statistics."""
    x, y = simple_data

    # Fit
    result = curve_fit(simple_model, x, y, p0=[10, 0.5, 2])

    # Analyze
    assert result.r_squared > 0.9
    assert result.rmse < 1.0
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)


def test_workflow_confidence_intervals(simple_data, simple_model):
    """Test complete workflow: fit → confidence intervals."""
    x, y = simple_data

    # Fit
    result = curve_fit(simple_model, x, y, p0=[10, 0.5, 2])

    # Get confidence intervals
    ci = result.confidence_intervals(alpha=0.95)

    # All parameters should be within their CIs
    assert np.all(result.popt >= ci[:, 0])
    assert np.all(result.popt <= ci[:, 1])


def test_workflow_plotting(simple_data, simple_model):
    """Test complete workflow: fit → plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    x, y = simple_data

    # Fit
    result = curve_fit(simple_model, x, y, p0=[10, 0.5, 2])

    # Plot
    result.plot(show_residuals=True)
    plt.close("all")


def test_workflow_model_comparison(simple_data):
    """Test workflow: compare multiple models using AIC/BIC."""
    import jax.numpy as jnp

    x, y = simple_data

    # Model 1: Exponential
    def exp_model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    # Model 2: Linear
    def linear_model(x, a, b):
        return a * x + b

    # Fit both
    result_exp = curve_fit(exp_model, x, y, p0=[10, 0.5, 2])
    result_lin = curve_fit(linear_model, x, y, p0=[0, 10])

    # Compare
    # Exponential should fit better (lower AIC/BIC)
    assert result_exp.aic < result_lin.aic
    assert result_exp.bic < result_lin.bic
    assert result_exp.r_squared > result_lin.r_squared


# ============================================================================
# Performance Tests
# ============================================================================


def test_predictions_computed_once(simple_result):
    """Test that predictions are cached and only computed once."""
    # Access predictions
    pred1 = simple_result.predictions

    # Clear the model to ensure it's not recomputed
    original_model = simple_result.model
    simple_result["model"] = None

    # Access again - should still work (cached)
    pred2 = simple_result.predictions

    # Should be the same object
    assert pred1 is pred2

    # Restore model
    simple_result["model"] = original_model


def test_residuals_computed_once(simple_result):
    """Test that residuals are cached and only computed once."""
    # Access residuals
    res1 = simple_result.residuals

    # Clear ydata to ensure it's not recomputed
    original_ydata = simple_result.ydata
    simple_result["ydata"] = None

    # Access again - should still work (cached)
    res2 = simple_result.residuals

    # Should be the same object
    assert res1 is res2

    # Restore ydata
    simple_result["ydata"] = original_ydata
