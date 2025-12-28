"""Tests for common curve fitting functions."""

import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.core.functions import (
    exponential_decay,
    exponential_growth,
    gaussian,
    linear,
    polynomial,
    power_law,
    sigmoid,
)


class TestLinearFunction:
    """Tests for linear function."""

    def test_linear_auto_p0(self):
        """Test linear function with automatic p0 estimation."""
        np.random.seed(42)

        # True parameters
        a_true, b_true = 2.5, 3.0

        # Generate data
        xdata = np.linspace(0, 10, 50)
        ydata = linear(xdata, a_true, b_true)
        ydata += np.random.normal(0, 0.5, len(xdata))

        # Fit with auto p0
        popt, _pcov = curve_fit(linear, xdata, ydata, p0="auto")

        # Check results
        assert np.abs(popt[0] - a_true) < 0.2, f"Slope mismatch: {popt[0]} vs {a_true}"
        assert np.abs(popt[1] - b_true) < 1.0, (
            f"Intercept mismatch: {popt[1]} vs {b_true}"
        )

    def test_linear_manual_p0(self):
        """Test linear function with manual p0."""
        xdata = np.array([1, 2, 3, 4, 5])
        ydata = np.array([2, 4, 6, 8, 10])

        popt, _pcov = curve_fit(linear, xdata, ydata, p0=[1.5, 0.5])

        assert np.abs(popt[0] - 2.0) < 0.1
        assert np.abs(popt[1] - 0.0) < 0.1

    def test_linear_estimate_p0_method(self):
        """Test linear.estimate_p0() method directly."""
        xdata = np.array([0, 1, 2, 3, 4])
        ydata = np.array([1, 3, 5, 7, 9])

        p0 = linear.estimate_p0(xdata, ydata)

        # Should estimate close to [2, 1]
        assert len(p0) == 2
        assert np.abs(p0[0] - 2.0) < 0.1  # Slope
        assert np.abs(p0[1] - 1.0) < 0.1  # Intercept


class TestExponentialDecay:
    """Tests for exponential_decay function."""

    def test_exponential_decay_auto_p0(self):
        """Test exponential_decay with automatic p0 estimation."""
        np.random.seed(42)

        # True parameters
        a_true, b_true, c_true = 5.0, 0.5, 1.0

        # Generate data
        xdata = np.linspace(0, 10, 100)
        ydata = exponential_decay(xdata, a_true, b_true, c_true)
        ydata += np.random.normal(0, 0.1, len(xdata))

        # Fit with auto p0
        popt, _pcov = curve_fit(exponential_decay, xdata, ydata, p0="auto")

        # Check results (allow 30% tolerance)
        assert np.abs(popt[0] - a_true) / a_true < 0.3, (
            f"Amplitude: {popt[0]} vs {a_true}"
        )
        assert np.abs(popt[1] - b_true) / b_true < 0.3, f"Rate: {popt[1]} vs {b_true}"
        assert np.abs(popt[2] - c_true) < 0.5, f"Offset: {popt[2]} vs {c_true}"

    def test_exponential_decay_estimate_p0(self):
        """Test exponential_decay.estimate_p0() method."""
        xdata = np.linspace(0, 5, 50)
        ydata = exponential_decay(xdata, 3, 0.5, 1)

        p0 = exponential_decay.estimate_p0(xdata, ydata)

        # Should be reasonably close
        assert len(p0) == 3
        assert p0[0] > 0  # Amplitude should be positive
        assert p0[1] > 0  # Rate should be positive

    def test_exponential_decay_bounds(self):
        """Test exponential_decay.bounds() method."""
        lb, ub = exponential_decay.bounds()

        assert len(lb) == 3
        assert len(ub) == 3
        assert lb[0] == 0  # a >= 0
        assert lb[1] == 0  # b >= 0


class TestExponentialGrowth:
    """Tests for exponential_growth function."""

    def test_exponential_growth_auto_p0(self):
        """Test exponential_growth with automatic p0 estimation."""
        np.random.seed(42)

        # True parameters
        a_true, b_true, c_true = 2.0, 0.3, 0.0

        # Generate data
        xdata = np.linspace(0, 5, 50)
        ydata = exponential_growth(xdata, a_true, b_true, c_true)
        ydata += np.random.normal(0, 0.5, len(xdata))

        # Fit with auto p0
        popt, _pcov = curve_fit(exponential_growth, xdata, ydata, p0="auto")

        # Check results (growth can be sensitive, allow 50% tolerance)
        assert np.abs(popt[0] - a_true) / a_true < 0.5, (
            f"Amplitude: {popt[0]} vs {a_true}"
        )
        assert np.abs(popt[1] - b_true) / b_true < 0.5, f"Rate: {popt[1]} vs {b_true}"


class TestGaussian:
    """Tests for gaussian function."""

    def test_gaussian_auto_p0(self):
        """Test gaussian with automatic p0 estimation."""
        np.random.seed(42)

        # True parameters
        amp_true, mu_true, sigma_true = 10.0, 5.0, 1.0

        # Generate data
        xdata = np.linspace(0, 10, 200)
        ydata = gaussian(xdata, amp_true, mu_true, sigma_true)
        ydata += np.random.normal(0, 0.2, len(xdata))

        # Fit with auto p0
        popt, _pcov = curve_fit(gaussian, xdata, ydata, p0="auto")

        # Check results
        assert np.abs(popt[0] - amp_true) < 2.0, f"Amplitude: {popt[0]} vs {amp_true}"
        assert np.abs(popt[1] - mu_true) < 0.5, f"Mean: {popt[1]} vs {mu_true}"
        assert np.abs(popt[2] - sigma_true) < 0.5, f"Sigma: {popt[2]} vs {sigma_true}"

    def test_gaussian_estimate_p0(self):
        """Test gaussian.estimate_p0() method."""
        xdata = np.linspace(-5, 5, 100)
        ydata = gaussian(xdata, 2, 1, 0.5)

        p0 = gaussian.estimate_p0(xdata, ydata)

        # Sanity checks
        assert len(p0) == 3
        assert p0[0] > 0  # Amplitude positive
        assert p0[2] > 0  # Sigma positive
        assert -6 < p0[1] < 6  # Mean in range

    def test_gaussian_peak_detection(self):
        """Test that gaussian correctly identifies peak position."""
        xdata = np.linspace(0, 20, 200)
        ydata = gaussian(xdata, 5, 12, 2)  # Peak at x=12

        popt, _pcov = curve_fit(gaussian, xdata, ydata, p0="auto")

        # Peak position should be close to 12
        assert np.abs(popt[1] - 12) < 1.0


class TestSigmoid:
    """Tests for sigmoid function."""

    def test_sigmoid_auto_p0(self):
        """Test sigmoid with automatic p0 estimation."""
        np.random.seed(42)

        # True parameters
        L_true, x0_true, k_true, b_true = 10.0, 5.0, 1.0, 1.0

        # Generate data
        xdata = np.linspace(0, 10, 100)
        ydata = sigmoid(xdata, L_true, x0_true, k_true, b_true)
        ydata += np.random.normal(0, 0.3, len(xdata))

        # Fit with auto p0
        popt, _pcov = curve_fit(sigmoid, xdata, ydata, p0="auto")

        # Check results
        assert np.abs(popt[0] - L_true) < 3.0, f"L: {popt[0]} vs {L_true}"
        assert np.abs(popt[1] - x0_true) < 2.0, f"x0: {popt[1]} vs {x0_true}"
        assert np.abs(popt[3] - b_true) < 2.0, f"b: {popt[3]} vs {b_true}"

    def test_sigmoid_estimate_p0(self):
        """Test sigmoid.estimate_p0() method."""
        xdata = np.linspace(0, 10, 100)
        ydata = sigmoid(xdata, 5, 5, 2, 1)

        p0 = sigmoid.estimate_p0(xdata, ydata)

        # Sanity checks
        assert len(p0) == 4
        assert p0[0] > 0  # L positive
        assert p0[2] > 0  # k positive


class TestPowerLaw:
    """Tests for power_law function."""

    def test_power_law_auto_p0(self):
        """Test power_law with automatic p0 estimation."""
        np.random.seed(42)

        # True parameters
        a_true, b_true = 3.0, 0.75

        # Generate data (positive x only)
        xdata = np.linspace(1, 100, 50)
        ydata = power_law(xdata, a_true, b_true)
        ydata += np.random.normal(0, 1.0, len(xdata))

        # Fit with auto p0
        popt, _pcov = curve_fit(power_law, xdata, ydata, p0="auto")

        # Check results (power law can be sensitive)
        assert np.abs(popt[0] - a_true) < 2.0, f"Prefactor: {popt[0]} vs {a_true}"
        assert np.abs(popt[1] - b_true) < 0.3, f"Exponent: {popt[1]} vs {b_true}"

    def test_power_law_estimate_p0(self):
        """Test power_law.estimate_p0() method."""
        xdata = np.linspace(1, 10, 20)
        ydata = power_law(xdata, 2, 1.5)

        p0 = power_law.estimate_p0(xdata, ydata)

        # Sanity checks
        assert len(p0) == 2
        assert p0[0] > 0  # Prefactor positive

    def test_power_law_linear_case(self):
        """Test power_law with b=1 (linear case)."""
        xdata = np.linspace(1, 10, 20)
        ydata = power_law(xdata, 5, 1)  # Should be 5*x

        popt, _pcov = curve_fit(power_law, xdata, ydata, p0="auto")

        assert np.abs(popt[0] - 5) < 0.5
        assert np.abs(popt[1] - 1) < 0.2


class TestPolynomial:
    """Tests for polynomial factory function."""

    def test_polynomial_degree_1(self):
        """Test polynomial of degree 1 (linear)."""
        np.random.seed(42)

        # Create linear polynomial
        poly1 = polynomial(1)

        # True coefficients: y = 2*x + 3
        xdata = np.linspace(0, 10, 50)
        ydata = poly1(xdata, 2, 3)
        ydata += np.random.normal(0, 0.5, len(xdata))

        # Fit
        popt, _pcov = curve_fit(poly1, xdata, ydata, p0="auto")

        # Check
        assert np.abs(popt[0] - 2) < 0.3  # Coefficient of x
        assert np.abs(popt[1] - 3) < 1.0  # Constant term

    def test_polynomial_degree_2(self):
        """Test polynomial of degree 2 (quadratic)."""
        np.random.seed(42)

        # Create quadratic polynomial
        poly2 = polynomial(2)

        # True coefficients: y = x² - 2x + 1
        xdata = np.linspace(-5, 5, 100)
        ydata = poly2(xdata, 1, -2, 1)
        ydata += np.random.normal(0, 0.3, len(xdata))

        # Fit
        popt, _pcov = curve_fit(poly2, xdata, ydata, p0="auto")

        # Check (allow larger tolerance)
        assert np.abs(popt[0] - 1) < 0.5  # x² coefficient
        assert np.abs(popt[1] + 2) < 1.0  # x coefficient
        assert np.abs(popt[2] - 1) < 1.5  # Constant

    def test_polynomial_estimate_p0(self):
        """Test polynomial.estimate_p0() method."""
        poly3 = polynomial(3)

        xdata = np.linspace(-2, 2, 50)
        ydata = poly3(xdata, 1, 0, -1, 2)  # x³ - x + 2

        p0 = poly3.estimate_p0(xdata, ydata)

        # Should get 4 coefficients for degree 3
        assert len(p0) == 4

    def test_polynomial_metadata(self):
        """Test polynomial has correct metadata."""
        poly2 = polynomial(2)

        assert poly2.__name__ == "polynomial_degree_2"
        assert "Polynomial of degree 2" in poly2.__doc__
        assert hasattr(poly2, "estimate_p0")
        assert hasattr(poly2, "bounds")


class TestFunctionProperties:
    """Test common properties across all functions."""

    @pytest.mark.parametrize(
        "func",
        [
            linear,
            exponential_decay,
            exponential_growth,
            gaussian,
            sigmoid,
            power_law,
        ],
    )
    def test_has_estimate_p0(self, func):
        """Test that all functions have estimate_p0 method."""
        assert hasattr(func, "estimate_p0")
        assert callable(func.estimate_p0)

    @pytest.mark.parametrize(
        "func",
        [
            linear,
            exponential_decay,
            exponential_growth,
            gaussian,
            sigmoid,
            power_law,
        ],
    )
    def test_has_bounds(self, func):
        """Test that all functions have bounds method."""
        assert hasattr(func, "bounds")
        assert callable(func.bounds)

    @pytest.mark.parametrize(
        "func",
        [
            linear,
            exponential_decay,
            exponential_growth,
            gaussian,
            sigmoid,
            power_law,
        ],
    )
    def test_bounds_structure(self, func):
        """Test that bounds return correct structure."""
        lb, ub = func.bounds()

        assert isinstance(lb, (list, tuple))
        assert isinstance(ub, (list, tuple))
        assert len(lb) == len(ub)

        # Check that lower < upper (or both infinite)
        for lower, upper in zip(lb, ub, strict=False):
            if np.isfinite(lower) and np.isfinite(upper):
                assert lower <= upper


class TestIntegrationWithCurveFit:
    """Integration tests with curve_fit."""

    def test_all_functions_work_with_auto_p0(self):
        """Test that all functions work with p0='auto'."""
        np.random.seed(42)

        # Test each function
        functions_and_data = [
            (linear, np.linspace(0, 10, 30), lambda x: 2 * x + 3, [2, 3]),
            (
                exponential_decay,
                np.linspace(0, 5, 50),
                lambda x: 3 * np.exp(-0.5 * x) + 1,
                [3, 0.5, 1],
            ),
            (
                gaussian,
                np.linspace(-3, 3, 100),
                lambda x: 2 * np.exp(-((x - 0) ** 2) / (2 * 0.5**2)),
                [2, 0, 0.5],
            ),
        ]

        for func, xdata, true_func, expected_params in functions_and_data:
            ydata = true_func(xdata)
            ydata += np.random.normal(0, 0.05 * np.max(np.abs(ydata)), len(xdata))

            # Should not raise error
            popt, pcov = curve_fit(func, xdata, ydata, p0="auto")

            # Should return reasonable results
            assert popt.shape == (len(expected_params),)
            assert pcov.shape == (len(expected_params), len(expected_params))

    def test_function_with_manual_bounds(self):
        """Test using function's default bounds."""
        xdata = np.linspace(0, 10, 50)
        ydata = exponential_decay(xdata, 5, 0.3, 2)
        ydata += np.random.normal(0, 0.1, len(xdata))

        # Use function's default bounds
        bounds = exponential_decay.bounds()

        popt, _pcov = curve_fit(
            exponential_decay, xdata, ydata, p0="auto", bounds=bounds
        )

        # Should work without errors
        assert popt.shape == (3,)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_linear_with_constant_data(self):
        """Test linear function with constant y data."""
        xdata = np.array([1, 2, 3, 4, 5])
        ydata = np.array([5, 5, 5, 5, 5])

        # Should still work (horizontal line)
        popt, _pcov = curve_fit(linear, xdata, ydata, p0="auto")

        assert np.abs(popt[0]) < 0.1  # Slope near 0
        assert np.abs(popt[1] - 5) < 0.1  # Intercept near 5

    def test_gaussian_with_no_peak(self):
        """Test gaussian estimation with monotonic data."""
        xdata = np.linspace(0, 10, 50)
        ydata = np.linspace(1, 2, 50)  # No peak

        # Should still produce some estimate
        p0 = gaussian.estimate_p0(xdata, ydata)

        assert len(p0) == 3
        assert all(np.isfinite(p0))

    def test_power_law_with_negative_x(self):
        """Test power_law estimation handles negative x gracefully."""
        xdata = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        ydata = np.abs(xdata) ** 0.5 + 1

        # Should handle gracefully (will filter out non-positive)
        p0 = power_law.estimate_p0(xdata, ydata)

        assert len(p0) == 2
        assert all(np.isfinite(p0))
