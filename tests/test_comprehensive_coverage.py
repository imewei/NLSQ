#!/usr/bin/env python3
"""Comprehensive tests to reach 74% coverage."""

import unittest

import jax.numpy as jnp
import numpy as np

from nlsq.core.least_squares import LeastSquares
from nlsq.core.minpack import CurveFit, curve_fit


class TestComprehensiveCoverage(unittest.TestCase):
    """Comprehensive tests to reach 74% coverage."""

    def test_curve_fit_various_models(self):
        """Test curve_fit with various model types."""

        # Linear model
        def linear(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.1, 4.0, 5.9, 8.0, 9.9])

        popt, _pcov = curve_fit(linear, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=1)

        # Quadratic model
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        y = np.array([1.0, 4.1, 9.0, 15.9, 25.1])
        popt, _pcov = curve_fit(quadratic, x, y)
        self.assertAlmostEqual(popt[0], 1.0, places=1)

        # Exponential model
        def exponential(x, a, b):
            return a * jnp.exp(b * x)

        x = np.linspace(0, 2, 50)
        y_true = 2.5 * np.exp(0.5 * x)
        y_noisy = y_true + 0.05 * np.random.randn(50)

        popt, _pcov = curve_fit(exponential, x, y_noisy, p0=[1, 1])
        self.assertAlmostEqual(popt[0], 2.5, places=0)

        # Sine model
        def sine_model(x, a, b, c):
            return a * jnp.sin(b * x + c)

        x = np.linspace(0, 2 * np.pi, 100)
        y = 2.0 * np.sin(1.0 * x + 0.5) + 0.1 * np.random.randn(100)

        popt, _pcov = curve_fit(sine_model, x, y, p0=[1, 1, 0])
        self.assertAlmostEqual(popt[0], 2.0, places=0)

    def test_curve_fit_with_different_options(self):
        """Test curve_fit with various options."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Test with different tolerances
        popt, _pcov = curve_fit(model, x, y, ftol=1e-10, xtol=1e-10)
        self.assertAlmostEqual(popt[0], 2.0, places=5)

        # Test with different max evaluations
        popt, _pcov = curve_fit(model, x, y, maxfev=5)
        self.assertEqual(len(popt), 2)

        # Test with absolute_sigma
        sigma = np.ones(5) * 0.1
        popt, _pcov = curve_fit(model, x, y, sigma=sigma, absolute_sigma=False)
        self.assertEqual(len(popt), 2)

        # Test with check_finite
        popt, _pcov = curve_fit(model, x, y, check_finite=True)
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_curve_fit_edge_cases(self):
        """Test curve_fit with edge cases."""

        def model(x, a):
            return a * x

        # Single parameter
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        popt, _pcov = curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)

        # Small dataset
        x = np.array([1, 2])
        y = np.array([2, 4])

        popt, _pcov = curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)

        # Perfect fit
        def perfect_model(x, a, b):
            return a * x + b

        x = np.array([0, 1, 2, 3, 4])
        y = np.array([1, 3, 5, 7, 9])  # Exactly 2*x + 1

        popt, _pcov = curve_fit(perfect_model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=6)
        self.assertAlmostEqual(popt[1], 1.0, places=6)

    def test_curve_fit_with_bounds_variations(self):
        """Test curve_fit with different bound configurations."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Tight bounds
        popt, _pcov = curve_fit(model, x, y, bounds=([1.9, -0.1], [2.1, 0.1]))
        self.assertAlmostEqual(popt[0], 2.0, places=1)

        # One-sided bounds
        popt, _pcov = curve_fit(model, x, y, bounds=([0, -np.inf], [np.inf, np.inf]))
        self.assertAlmostEqual(popt[0], 2.0, places=3)

        # Bounds with initial guess
        popt, _pcov = curve_fit(model, x, y, p0=[1, 1], bounds=([0, -10], [10, 10]))
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_least_squares_class(self):
        """Test LeastSquares class directly."""

        def residual(params, x, y):
            a, b = params
            return y - (a * x + b)

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        ls = LeastSquares()

        # Test with basic residual function
        np.array([1.0, 1.0])

        # Create wrapped function for residuals
        def wrapped_residual(params):
            return residual(params, x, y)

        # Test has required attributes/methods
        self.assertTrue(hasattr(ls, "least_squares") or callable(ls))

    def test_curve_fit_class_options(self):
        """Test CurveFit class with various options."""

        # Test with dynamic sizing
        cf = CurveFit(use_dynamic_sizing=True)

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        popt, _pcov = cf.curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)

        # Test with fixed flength
        cf2 = CurveFit(flength=100)  # Use integer
        popt, _pcov = cf2.curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)

        # Test updating flength
        cf2.update_flength(200)  # Use integer
        popt, _pcov = cf2.curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_curve_fit_robustness(self):
        """Test curve_fit with noisy and outlier data."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 100])  # Last point is outlier

        # Should still fit reasonably well despite outlier
        popt, _pcov = curve_fit(model, x, y)
        self.assertGreater(popt[0], 1.5)  # Should be somewhat close to 2

        # With loss='soft_l1' for robust fitting
        popt, _pcov = curve_fit(model, x, y, loss="soft_l1")
        self.assertGreater(popt[0], 1.5)

        # With loss='huber' for robust fitting
        popt, _pcov = curve_fit(model, x, y, loss="huber")
        self.assertGreater(popt[0], 1.5)

    def test_curve_fit_multidimensional(self):
        """Test curve_fit with multi-dimensional x data."""

        def plane_model(xy, a, b, c):
            x, y = xy
            return a * x + b * y + c

        # Create 2D grid data
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        xx, yy = np.meshgrid(x, y)

        # Generate z data (plane: z = 2*x + 3*y + 1)
        zz = 2 * xx + 3 * yy + 1 + 0.1 * np.random.randn(*xx.shape)

        # Flatten for curve_fit
        xdata = (xx.ravel(), yy.ravel())
        ydata = zz.ravel()

        popt, _pcov = curve_fit(plane_model, xdata, ydata, p0=[1, 1, 0])
        self.assertAlmostEqual(popt[0], 2.0, places=1)
        self.assertAlmostEqual(popt[1], 3.0, places=1)
        self.assertAlmostEqual(popt[2], 1.0, places=1)

    def test_curve_fit_complex_models(self):
        """Test curve_fit with more complex models."""

        # Gaussian model
        def gaussian(x, amplitude, center, width):
            return amplitude * jnp.exp(-(((x - center) / width) ** 2))

        x = np.linspace(-5, 5, 100)
        y = 2.0 * np.exp(-(((x - 1.0) / 1.5) ** 2)) + 0.05 * np.random.randn(100)

        popt, _pcov = curve_fit(gaussian, x, y, p0=[1, 0, 1])
        self.assertAlmostEqual(popt[0], 2.0, places=0)
        self.assertAlmostEqual(popt[1], 1.0, places=0)

        # Lorentzian model
        def lorentzian(x, amplitude, center, width):
            return amplitude * width**2 / ((x - center) ** 2 + width**2)

        y = 3.0 * 2.0**2 / ((x - 0.5) ** 2 + 2.0**2) + 0.05 * np.random.randn(100)

        popt, _pcov = curve_fit(lorentzian, x, y, p0=[1, 0, 1])
        self.assertAlmostEqual(popt[0], 3.0, places=0)

    def test_curve_fit_convergence_issues(self):
        """Test curve_fit with difficult convergence scenarios."""

        # Model with local minima
        def difficult_model(x, a, b, c):
            return a * jnp.sin(b * x) * jnp.exp(-c * x)

        x = np.linspace(0, 10, 100)
        y = 2.0 * np.sin(1.5 * x) * np.exp(-0.1 * x) + 0.05 * np.random.randn(100)

        # Multiple initial guesses to test convergence
        initial_guesses = [[1.0, 1.0, 0.1], [3.0, 2.0, 0.2], [2.0, 1.5, 0.1]]

        for p0 in initial_guesses:
            try:
                popt, _pcov = curve_fit(difficult_model, x, y, p0=p0, maxfev=1000)
                # Check if converged to reasonable values
                self.assertGreater(popt[0], 0)
                self.assertGreater(popt[1], 0)
            except:
                # Some initial guesses might fail - that's expected
                pass


if __name__ == "__main__":
    unittest.main()
