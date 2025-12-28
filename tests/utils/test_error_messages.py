"""Tests for enhanced error messages."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.utils.error_messages import (
    OptimizationError,
    analyze_failure,
    format_error_message,
)


class TestEnhancedErrorMessages:
    """Test enhanced error message functionality."""

    def test_error_message_max_iterations(self):
        """Test error message when max iterations reached."""

        def difficult_func(x, a, b):
            """Difficult function to fit."""
            return a * jnp.exp(b * x**2)

        xdata = np.linspace(0, 1, 10)
        ydata = difficult_func(xdata, 1, -5)

        with pytest.raises(OptimizationError) as exc_info:
            curve_fit(difficult_func, xdata, ydata, p0=[0.1, 0.1], max_nfev=5)

        error = exc_info.value
        error_str = str(error)

        # Check error message contains key information
        assert "failed to converge" in error_str.lower()
        assert "max_nfev" in error_str or "maximum" in error_str.lower()
        assert len(error.recommendations) > 0

        # Check recommendations are actionable
        recommendations_str = " ".join(error.recommendations)
        assert (
            "max_nfev" in recommendations_str
            or "increase" in recommendations_str.lower()
        )

    def test_error_message_gradient_tolerance(self):
        """Test that gradient information is included in error diagnostics."""

        def steep_func(x, a, b):
            """Function with steep gradients."""
            return a / (1 + jnp.exp(-b * (x - 5)))

        xdata = np.linspace(0, 10, 20)
        ydata = steep_func(xdata, 10, 5) + np.random.normal(0, 0.1, 20)

        # Use impossible max_nfev to force failure
        with pytest.raises(OptimizationError) as exc_info:
            curve_fit(steep_func, xdata, ydata, p0=[8, 3], max_nfev=1)

        error = exc_info.value
        str(error)

        # Should have diagnostics with gradient information
        assert error.diagnostics is not None
        assert len(error.diagnostics) > 0

        # Diagnostics should include gradient info
        diag_str = str(error.diagnostics)
        assert "gradient" in diag_str.lower() or "Gradient" in str(error.diagnostics)

    def test_error_message_contains_diagnostics(self):
        """Test that error message includes diagnostic information."""

        def simple_exp(x, a, b):
            return a * jnp.exp(-b * x)

        xdata = np.array([1, 2, 3])
        ydata = np.array([1, 0.5, 0.25])

        with pytest.raises(OptimizationError) as exc_info:
            # Bad p0 + low max_nfev to force failure
            curve_fit(simple_exp, xdata, ydata, p0=[0.01, 0.01], max_nfev=3)

        error = exc_info.value

        # Check diagnostics are populated
        assert (
            "Function evaluations" in error.diagnostics or "nfev" in str(error).lower()
        )

    def test_error_message_recommendations(self):
        """Test that recommendations are helpful and actionable."""

        def linear(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        with pytest.raises(OptimizationError) as exc_info:
            # Force failure with max_nfev=1 (impossible to converge)
            curve_fit(linear, xdata, ydata, p0=[1, 1], max_nfev=1)

        error = exc_info.value

        # Should have at least one recommendation
        assert len(error.recommendations) > 0

        # Recommendations should be actionable (contain specific suggestions)
        rec_text = " ".join(error.recommendations).lower()
        assert any(
            keyword in rec_text
            for keyword in ["try", "increase", "check", "consider", "use"]
        )

    def test_analyze_failure_function(self):
        """Test analyze_failure utility function."""

        # Mock result object
        class MockResult:
            def __init__(self):
                self.grad = np.array([0.1, 0.2])
                self.nfev = 150
                self.nit = 50
                self.x = np.array([1.0, 2.0])
                self.cost = 1.234

        result = MockResult()
        gtol = 1e-8
        ftol = 1e-8
        xtol = 1e-8
        max_nfev = 100

        reasons, recommendations = analyze_failure(result, gtol, ftol, xtol, max_nfev)

        # Should identify that max_nfev was reached
        assert any("maximum" in r.lower() for r in reasons)

        # Should have recommendations
        assert len(recommendations) > 0

    def test_format_error_message(self):
        """Test error message formatting."""
        reasons = ["Gradient too large", "Max iterations reached"]
        recommendations = ["Try looser tolerance", "Increase max_nfev"]
        diagnostics = {"Final cost": "1.23e-3", "Iterations": 100}

        msg = format_error_message(reasons, recommendations, diagnostics)

        # Check all sections are present
        assert "Diagnostics:" in msg
        assert "Reasons:" in msg
        assert "Recommendations:" in msg

        # Check content is included
        assert "1.23e-3" in msg
        assert "Gradient too large" in msg
        assert "Try looser tolerance" in msg

    def test_numerical_instability_detection(self):
        """Test that NaN/Inf in parameters is detected."""

        def bad_func(x, a):
            """Function that might produce NaN."""
            return a / x  # Will fail at x=0

        xdata = np.array([0, 1, 2])  # Contains 0!
        ydata = np.array([1, 2, 3])

        try:
            curve_fit(bad_func, xdata, ydata, p0=[1])
        except (OptimizationError, RuntimeError, ValueError) as e:
            # Should catch some kind of error
            error_msg = str(e).lower()
            # May mention NaN, Inf, or numerical issues
            assert any(
                keyword in error_msg
                for keyword in ["nan", "inf", "numerical", "finite", "invalid"]
            )

    def test_error_includes_troubleshooting_link(self):
        """Test that error message includes link to documentation."""

        def exp_func(x, a, b):
            return a * jnp.exp(-b * x)  # Use jnp for JAX compatibility

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 1, 0.5])

        with pytest.raises(OptimizationError) as exc_info:
            curve_fit(exp_func, xdata, ydata, p0=[0.1, 0.1], max_nfev=1)

        error_msg = str(exc_info.value)

        # Should include documentation link
        assert (
            "nlsq.readthedocs.io" in error_msg or "troubleshooting" in error_msg.lower()
        )


class TestErrorMessageContent:
    """Test specific content and quality of error messages."""

    def test_recommendations_are_specific(self):
        """Test that recommendations include specific parameter values."""

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        xdata = np.linspace(0, 10, 20)
        ydata = 2 * xdata**2 + 3 * xdata + 1

        with pytest.raises(OptimizationError) as exc_info:
            curve_fit(quadratic, xdata, ydata, p0=[1, 1, 1], max_nfev=1)

        error = exc_info.value

        # Check that at least one recommendation includes specific values
        rec_text = " ".join(error.recommendations)
        # Should suggest specific max_nfev value (e.g., "max_nfev=2" or similar)
        assert "max_nfev" in rec_text

    def test_error_message_readability(self):
        """Test that error messages are well-formatted and readable."""

        def sigmoid(x, L, x0, k):
            return L / (1 + jnp.exp(-k * (x - x0)))

        xdata = np.linspace(-5, 5, 30)
        ydata = sigmoid(xdata, 1, 0, 1)

        with pytest.raises(OptimizationError) as exc_info:
            curve_fit(sigmoid, xdata, ydata, p0=[0.5, 0, 0.5], max_nfev=3)

        error_msg = str(exc_info.value)

        # Check formatting
        assert "\n" in error_msg  # Multiple lines
        assert "  " in error_msg  # Indentation

        # Should have clear sections
        lines = error_msg.split("\n")
        assert any("Diagnostics:" in line for line in lines)
        assert any("Recommendations:" in line for line in lines)

    def test_multiple_failure_reasons(self):
        """Test handling of multiple failure reasons."""

        class MockResult:
            """Mock result with multiple issues."""

            def __init__(self):
                self.grad = np.array([10.0, 20.0])  # High gradient
                self.nfev = 200  # Max iterations
                self.nit = 200
                self.x = np.array([np.nan, 1.0])  # NaN in solution
                self.cost = 1.0
                self.success = False
                self.status = 0
                self.message = "Multiple issues"

        result = MockResult()
        gtol = 1e-8
        max_nfev = 100

        reasons, recommendations = analyze_failure(result, gtol, 1e-8, 1e-8, max_nfev)

        # Should identify multiple issues
        assert len(reasons) >= 2

        # Should have recommendations for different issues
        assert len(recommendations) >= 2
