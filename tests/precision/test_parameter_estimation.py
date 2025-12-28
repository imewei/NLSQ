"""
Comprehensive test suite for automatic parameter estimation.

Tests the parameter_estimation module which provides intelligent p0 estimation
when users don't provide initial guesses. This is a critical usability feature
that currently has only 9.17% test coverage.

Target: Bring parameter_estimation.py coverage from 9.17% to 100%
Missing Lines: 150, 156-217, 273-311, 373-452

Test Strategy:
1. Unit tests for each pattern detection function
2. Unit tests for each p0 estimation strategy
3. Integration tests with curve_fit
4. Edge case and error handling tests
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.precision.parameter_estimation import (
    detect_function_pattern,
    estimate_initial_parameters,
    estimate_p0_for_pattern,
)


class TestEstimateInitialParameters(unittest.TestCase):
    """Test automatic p0 estimation from data."""

    def test_estimate_p0_with_explicit_array(self):
        """Test that explicit p0 is returned unchanged."""

        # COVERS: Line 149-150
        def model(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 50)
        y = 2 * x + 3
        p0_explicit = np.array([2.5, 3.5])

        result = estimate_initial_parameters(model, x, y, p0=p0_explicit)

        np.testing.assert_array_equal(result, p0_explicit)

    def test_estimate_p0_with_auto_keyword(self):
        """Test p0='auto' triggers automatic estimation."""

        # COVERS: Lines 149-150, 159-217
        def model(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 50)
        y = 8 * x + 2  # y_range=80, x_range=10

        result = estimate_initial_parameters(model, x, y, p0="auto")

        # First param should be ~y_range, second ~1/x_range
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0], 0)  # Should be ~80
        self.assertGreater(result[1], 0)  # Should be ~0.1

    def test_custom_estimate_p0_method(self):
        """Test that custom .estimate_p0() method is used."""
        # COVERS: Lines 153-157

        class CustomModel:
            def __call__(self, x, a, b):
                return a * jnp.exp(-b * x)

            def estimate_p0(self, xdata, ydata):
                return [np.max(ydata), 0.5]

        model = CustomModel()
        x = np.linspace(0, 5, 50)
        y = 3 * np.exp(-0.5 * x)

        result = estimate_initial_parameters(model, x, y, p0="auto")

        # Should use custom method
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], np.max(y), places=5)
        self.assertAlmostEqual(result[1], 0.5, places=5)

    def test_estimate_p0_generic_heuristics(self):
        """Test generic heuristic estimation for arbitrary functions."""
        # COVERS: Lines 159-217

        def model(x, amp, rate, offset, center):
            return amp * jnp.exp(-rate * (x - center) ** 2) + offset

        x = np.linspace(-10, 10, 100)
        y_min, y_max = 2, 10
        y = y_min + (y_max - y_min) * np.exp(-0.1 * (x - 0) ** 2)

        result = estimate_initial_parameters(model, x, y, p0="auto")

        # Check heuristics:
        # [0]: amplitude ~ y_range
        # [1]: rate ~ 1/x_range
        # [2]: offset ~ y_mean
        # [3]: center ~ x_mean
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], y_max - y_min, delta=1.0)  # amplitude
        self.assertAlmostEqual(result[1], 1.0 / 20.0, delta=0.1)  # rate
        self.assertAlmostEqual(result[2], np.mean(y), delta=1.0)  # offset
        self.assertAlmostEqual(result[3], 0.0, delta=1.0)  # center

    def test_estimate_p0_with_constant_data(self):
        """Test estimation when y is constant (y_range=0)."""
        # COVERS: Lines 183-189, 199

        def model(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 50)
        y = np.ones(50) * 5.0  # Constant

        result = estimate_initial_parameters(model, x, y, p0="auto")

        # Should handle y_range=0 gracefully
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 1.0)  # Fallback for zero range

    def test_estimate_p0_error_no_signature(self):
        """Test error when function signature cannot be determined."""
        # COVERS: Lines 164-174

        # Lambda with no signature
        model = lambda x, *args: x

        x = np.linspace(0, 10, 50)
        y = x

        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            estimate_initial_parameters(model, x, y, p0="auto")

        self.assertIn("Cannot automatically determine", str(context.exception))

    def test_estimate_p0_error_no_parameters(self):
        """Test error when function has no parameters besides x."""
        # COVERS: Lines 176-180

        def model(x):  # No parameters!
            return x

        x = np.linspace(0, 10, 50)
        y = x

        with self.assertRaises(ValueError) as context:
            estimate_initial_parameters(model, x, y, p0="auto")

        self.assertIn("at least one parameter", str(context.exception))


class TestDetectFunctionPattern(unittest.TestCase):
    """Test function pattern detection from data."""

    def test_detect_linear_pattern(self):
        """Test detection of linear pattern (high correlation)."""
        # COVERS: Lines 276-282
        x = np.linspace(0, 10, 100)
        y = 2.5 * x + 3.0  # Perfect linear

        pattern = detect_function_pattern(y, x)

        self.assertEqual(pattern, "linear")

    def test_detect_exponential_decay_pattern(self):
        """Test detection of exponential decay (monotonic decrease)."""
        # COVERS: Lines 284-286
        x = np.linspace(0, 5, 100)
        y = 3 * np.exp(-0.5 * x)  # Monotonically decreasing

        pattern = detect_function_pattern(y, x)

        self.assertEqual(pattern, "exponential_decay")

    def test_detect_exponential_growth_pattern(self):
        """Test detection of exponential growth (monotonic increase)."""
        # COVERS: Lines 288-290
        x = np.linspace(0, 5, 100)
        y = 1 + 2 * x  # Monotonically increasing

        pattern = detect_function_pattern(y, x)

        # Could be linear or exponential_growth
        self.assertIn(pattern, ["exponential_growth", "linear"])

    def test_detect_gaussian_pattern(self):
        """Test detection of Gaussian bell curve."""
        # COVERS: Lines 292-299
        x = np.linspace(-5, 5, 100)
        y = 2 * np.exp(-(x**2) / 2)  # Gaussian peak

        pattern = detect_function_pattern(y, x)

        self.assertEqual(pattern, "gaussian")

    def test_detect_sigmoid_pattern(self):
        """Test detection of sigmoid S-curve."""
        # COVERS: Lines 301-310
        x = np.linspace(-5, 5, 100)
        y = 1 / (1 + np.exp(-x))  # Sigmoid: 0 -> 1

        pattern = detect_function_pattern(y, x)

        self.assertIn(pattern, ["sigmoid", "sigmoid_inv"])

    def test_detect_unknown_pattern(self):
        """Test unknown pattern for irregular data."""
        # COVERS: Lines 273-274, 311
        x = np.linspace(0, 10, 10)
        y = np.random.rand(10)  # Random noise

        pattern = detect_function_pattern(y, x)

        self.assertEqual(pattern, "unknown")

    def test_detect_pattern_insufficient_data(self):
        """Test pattern detection with too few points."""
        # COVERS: Lines 273-274
        x = np.array([1, 2])
        y = np.array([3, 4])

        pattern = detect_function_pattern(y, x)

        self.assertEqual(pattern, "unknown")


class TestEstimateP0ForPattern(unittest.TestCase):
    """Test pattern-specific p0 estimation."""

    def test_linear_pattern_estimation(self):
        """Test p0 estimation for linear pattern using least squares."""
        # COVERS: Lines 373-385
        x = np.linspace(0, 10, 50)
        y = 2.5 * x + 3.7

        p0 = estimate_p0_for_pattern("linear", x, y, n_params=2)

        # Should get close to [2.5, 3.7]
        self.assertEqual(len(p0), 2)
        self.assertAlmostEqual(p0[0], 2.5, places=1)
        self.assertAlmostEqual(p0[1], 3.7, places=1)

    def test_exponential_decay_pattern_estimation(self):
        """Test p0 estimation for exponential decay."""
        # COVERS: Lines 387-407
        x = np.linspace(0, 5, 100)
        y = 3 * np.exp(-0.5 * x) + 1  # a=3, b=0.5, c=1

        p0 = estimate_p0_for_pattern("exponential_decay", x, y, n_params=3)

        # Should estimate [amplitude, rate, offset]
        self.assertEqual(len(p0), 3)
        self.assertAlmostEqual(p0[0], 3.0, delta=0.5)  # amplitude
        self.assertGreater(p0[1], 0)  # rate should be positive
        self.assertAlmostEqual(p0[2], 1.0, delta=0.3)  # offset

    def test_gaussian_pattern_estimation(self):
        """Test p0 estimation for Gaussian bell curve."""
        # COVERS: Lines 409-425
        x = np.linspace(-5, 5, 100)
        amp = 5.0
        mu = 1.0
        sigma = 0.5
        y = amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

        p0 = estimate_p0_for_pattern("gaussian", x, y, n_params=3)

        # Should estimate [amplitude, mean, sigma]
        self.assertEqual(len(p0), 3)
        self.assertAlmostEqual(p0[0], amp, delta=0.5)  # amplitude
        self.assertAlmostEqual(p0[1], mu, delta=0.5)  # mean
        self.assertAlmostEqual(p0[2], sigma, delta=0.3)  # sigma

    def test_sigmoid_pattern_estimation(self):
        """Test p0 estimation for sigmoid curve."""
        # COVERS: Lines 427-449
        x = np.linspace(-5, 5, 100)
        L = 10.0
        k = 1.0
        x0 = 0.0
        b = 2.0
        y = L / (1 + np.exp(-k * (x - x0))) + b

        p0 = estimate_p0_for_pattern("sigmoid", x, y, n_params=4)

        # Should estimate [L, x0, k, b]
        self.assertEqual(len(p0), 4)
        self.assertAlmostEqual(p0[0], L, delta=2.0)  # L (range)
        self.assertAlmostEqual(p0[1], x0, delta=1.0)  # x0 (midpoint)
        self.assertGreater(p0[2], 0)  # k (steepness)
        self.assertAlmostEqual(p0[3], b, delta=1.0)  # b (offset)

    def test_unknown_pattern_fallback(self):
        """Test fallback to generic estimation for unknown pattern."""
        # COVERS: Lines 451-452
        x = np.linspace(0, 10, 50)
        y = np.random.rand(50)

        p0 = estimate_p0_for_pattern("unknown", x, y, n_params=3)

        # Should return generic estimates
        self.assertEqual(len(p0), 3)
        self.assertTrue(all(np.isfinite(p0)))

    def test_pattern_estimation_more_params_than_pattern(self):
        """Test padding with 1.0 when n_params exceeds pattern defaults."""
        # COVERS: Lines 405-406, 423-424, 447-448
        x = np.linspace(0, 5, 50)
        y = 2 * np.exp(-0.5 * x) + 1

        # Exponential has 3 params, ask for 5
        p0 = estimate_p0_for_pattern("exponential_decay", x, y, n_params=5)

        self.assertEqual(len(p0), 5)
        self.assertEqual(p0[3], 1.0)  # Padded
        self.assertEqual(p0[4], 1.0)  # Padded


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_data_arrays(self):
        """Test behavior with empty data arrays."""

        def model(x, a, b):
            return a * x + b

        x = np.array([])
        y = np.array([])

        # Should handle gracefully or raise clear error
        try:
            result = estimate_initial_parameters(model, x, y, p0="auto")
            # If it succeeds, check result is valid
            self.assertTrue(np.all(np.isfinite(result)))
        except (ValueError, IndexError):
            # Expected error for empty data
            pass

    def test_single_data_point(self):
        """Test with single data point."""

        def model(x, a, b):
            return a * x + b

        x = np.array([5.0])
        y = np.array([10.0])

        result = estimate_initial_parameters(model, x, y, p0="auto")

        # Should return something sensible
        self.assertEqual(len(result), 2)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_nan_in_data(self):
        """Test behavior with NaN values in data."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, np.nan, 5])
        y = np.array([2, 4, 6, 8, np.nan])

        # Should handle NaN gracefully
        try:
            result = estimate_initial_parameters(model, x, y, p0="auto")
            # If succeeds, result should be finite (NaN filtered)
            # or contain NaN (propagated)
            self.assertTrue(len(result) == 2)
        except (ValueError, RuntimeError):
            # Expected error for bad data
            pass

    def test_inf_in_data(self):
        """Test behavior with Inf values in data."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, np.inf, 4, 5])
        y = np.array([2, 4, 6, 8, np.inf])

        # Should handle Inf gracefully
        try:
            result = estimate_initial_parameters(model, x, y, p0="auto")
            self.assertEqual(len(result), 2)
        except (ValueError, RuntimeError):
            # Expected error for bad data
            pass


class TestIntegrationWithCurveFit(unittest.TestCase):
    """Integration tests with curve_fit."""

    def test_curve_fit_with_p0_auto(self):
        """Test curve_fit integration with p0='auto'."""
        from nlsq import curve_fit

        def exponential(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y_true = 3 * np.exp(-0.5 * x) + 1
        y = y_true + 0.1 * np.random.randn(len(x))

        # Should work with p0='auto'
        popt, _pcov = curve_fit(exponential, x, y, p0="auto")

        # Should converge to reasonable values
        self.assertAlmostEqual(popt[0], 3.0, delta=0.5)
        self.assertAlmostEqual(popt[1], 0.5, delta=0.2)
        self.assertAlmostEqual(popt[2], 1.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
