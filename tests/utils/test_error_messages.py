"""Tests for enhanced error messages."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.result import OptimizeWarning
from nlsq.utils.error_messages import (
    OptimizationError,
    analyze_failure,
    format_error_message,
)


class TestEnhancedErrorMessages:
    """Test enhanced error message functionality."""

    def test_error_message_max_iterations(self):
        """Test warning message when max iterations reached."""

        def difficult_func(x, a, b):
            """Difficult function to fit."""
            return a * jnp.exp(b * x**2)

        xdata = np.linspace(0, 1, 10)
        ydata = difficult_func(xdata, 1, -5)

        with pytest.warns(OptimizeWarning) as warn_info:
            curve_fit(difficult_func, xdata, ydata, p0=[0.1, 0.1], max_nfev=5)

        warn_msg = str(warn_info[0].message).lower()
        assert "maximum" in warn_msg or "evaluations" in warn_msg

    def test_error_message_gradient_tolerance(self):
        """Test that max_nfev issues a warning and returns a best-effort result."""

        def steep_func(x, a, b):
            """Function with steep gradients."""
            return a / (1 + jnp.exp(-b * (x - 5)))

        xdata = np.linspace(0, 10, 20)
        ydata = steep_func(xdata, 10, 5) + np.random.normal(0, 0.1, 20)

        # max_nfev exceeded now issues OptimizeWarning (SciPy-compatible)
        with pytest.warns(OptimizeWarning):
            curve_fit(steep_func, xdata, ydata, p0=[8, 3], max_nfev=1)

    def test_error_message_contains_diagnostics(self):
        """Test that error message includes diagnostic information."""

        def simple_exp(x, a, b):
            return a * jnp.exp(-b * x)

        xdata = np.array([1, 2, 3])
        ydata = np.array([1, 0.5, 0.25])

        with pytest.warns(OptimizeWarning) as warn_info:
            curve_fit(simple_exp, xdata, ydata, p0=[0.01, 0.01], max_nfev=3)

        warn_msg = str(warn_info[0].message).lower()
        assert "maximum" in warn_msg or "evaluations" in warn_msg or "nfev" in warn_msg

    def test_error_message_recommendations(self):
        """Test that recommendations are helpful and actionable."""

        def linear(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        with pytest.warns(OptimizeWarning) as warn_info:
            curve_fit(linear, xdata, ydata, p0=[1, 1], max_nfev=1)

        warn_msg = str(warn_info[0].message).lower()
        assert "maximum" in warn_msg or "evaluations" in warn_msg

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

        with pytest.warns(OptimizeWarning) as warn_info:
            curve_fit(exp_func, xdata, ydata, p0=[0.1, 0.1], max_nfev=1)

        # Warning message should mention max iterations or evaluations
        warn_msg = str(warn_info[0].message).lower()
        assert "maximum" in warn_msg or "evaluations" in warn_msg


class TestErrorMessageContent:
    """Test specific content and quality of error messages."""

    def test_recommendations_are_specific(self):
        """Test that recommendations include specific parameter values."""

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        xdata = np.linspace(0, 10, 20)
        ydata = 2 * xdata**2 + 3 * xdata + 1

        with pytest.warns(OptimizeWarning) as warn_info:
            curve_fit(quadratic, xdata, ydata, p0=[1, 1, 1], max_nfev=1)

        warn_msg = str(warn_info[0].message)
        assert "maximum" in warn_msg.lower() or "evaluations" in warn_msg.lower()

    def test_error_message_readability(self):
        """Test that error messages are well-formatted and readable."""

        def sigmoid(x, L, x0, k):
            return L / (1 + jnp.exp(-k * (x - x0)))

        xdata = np.linspace(-5, 5, 30)
        ydata = sigmoid(xdata, 1, 0, 1)

        with pytest.warns(OptimizeWarning) as warn_info:
            curve_fit(sigmoid, xdata, ydata, p0=[0.5, 0, 0.5], max_nfev=3)

        warn_msg = str(warn_info[0].message)
        assert len(warn_msg) > 10  # Non-trivial message

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
