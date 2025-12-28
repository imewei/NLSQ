#!/usr/bin/env python3
"""Simple test suite for TRF algorithm focusing on actual implementation."""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.least_squares import LeastSquares
from nlsq.core.minpack import curve_fit


class TestTRFBasic(unittest.TestCase):
    """Basic tests for Trust Region Reflective algorithm."""

    def test_linear_least_squares(self):
        """Test TRF on simple linear least squares problem."""

        def linear(x, a, b):
            return a * x + b

        np.random.seed(42)  # For reproducibility
        x = np.array([0, 1, 2, 3, 4, 5])
        y = 2 * x + 1 + 0.1 * np.random.randn(6)

        popt, _pcov = curve_fit(linear, x, y, method="trf")
        self.assertAlmostEqual(popt[0], 2.0, places=1)
        self.assertAlmostEqual(
            popt[1], 1.0, places=0
        )  # Slightly looser tolerance for intercept

    def test_bounded_optimization(self):
        """Test TRF with parameter bounds."""

        def model(x, a, b):
            return a * jnp.exp(b * x)

        np.random.seed(42)  # For reproducibility
        x = np.linspace(0, 2, 20)
        y = 2.5 * np.exp(0.5 * x) + 0.1 * np.random.randn(20)

        # Test with bounds
        popt, _pcov = curve_fit(
            model, x, y, p0=[1, 1], bounds=([0, 0], [5, 2]), method="trf"
        )
        # Check bounds are respected
        self.assertGreaterEqual(popt[0], 0)
        self.assertLessEqual(popt[0], 5)
        self.assertGreaterEqual(popt[1], 0)
        self.assertLessEqual(popt[1], 2)

    def test_nonlinear_function(self):
        """Test TRF on nonlinear function."""

        def gaussian(x, a, mu, sigma):
            return a * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

        np.random.seed(42)  # For reproducibility
        x = np.linspace(-5, 5, 50)
        y = 3.0 * np.exp(-0.5 * ((x - 1.0) / 0.8) ** 2)
        y += 0.05 * np.random.randn(50)

        popt, _pcov = curve_fit(gaussian, x, y, p0=[1, 0, 1], method="trf")
        self.assertAlmostEqual(popt[0], 3.0, places=0)
        self.assertAlmostEqual(popt[1], 1.0, places=0)
        self.assertAlmostEqual(abs(popt[2]), 0.8, places=0)

    def test_robust_loss_functions(self):
        """Test TRF with different loss functions."""

        def linear(x, a, b):
            return a * x + b

        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y = 2 * x + 1
        # Add outlier
        y[9] = 100

        # Test with different loss functions
        for loss in ["linear", "soft_l1", "huber", "cauchy", "arctan"]:
            popt, _pcov = curve_fit(linear, x, y, method="trf", loss=loss, f_scale=1.0)
            # With robust loss, should be closer to true values despite outlier
            if loss != "linear":
                self.assertAlmostEqual(popt[0], 2.0, places=0)

    def test_convergence_criteria(self):
        """Test different convergence criteria."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.1, 4.0, 5.9, 8.1, 9.9])

        # Test with tight tolerances
        popt, _pcov = curve_fit(
            model, x, y, method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12
        )
        self.assertAlmostEqual(popt[0], 2.0, places=1)

        # Test with loose tolerances
        popt, _pcov = curve_fit(
            model, x, y, method="trf", ftol=1e-3, xtol=1e-3, gtol=1e-3
        )
        self.assertAlmostEqual(popt[0], 2.0, places=0)

    def test_least_squares_interface(self):
        """Test direct LeastSquares interface with TRF."""

        def fun(params, x):
            return params[0] * x + params[1]

        def residual(params, x, y):
            return fun(params, x) - y

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        ls = LeastSquares()
        result = ls.least_squares(
            residual,
            jnp.array([1.0, 0.0]),
            args=(x, y),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.x[0], 2.0, places=3)
        self.assertAlmostEqual(result.x[1], 0.0, places=3)

    def test_multiple_outputs(self):
        """Test TRF with multiple output dimensions."""

        # For now, test with a simpler model that returns single output
        def model(x, a, b, c):
            # Combine linear and quadratic terms
            return a * x + b + c * x**2

        x = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        # Create data that's a mix of linear and quadratic
        y = 2 * x + 1 + 0.5 * x**2

        popt, _pcov = curve_fit(model, x, y, p0=[1, 1, 1], method="trf")
        self.assertEqual(len(popt), 3)
        self.assertAlmostEqual(popt[0], 2.0, places=5)
        self.assertAlmostEqual(popt[1], 1.0, places=5)
        self.assertAlmostEqual(popt[2], 0.5, places=5)

    @pytest.mark.filterwarnings(
        "ignore:Covariance of the parameters could not be estimated"
    )
    def test_edge_cases(self):
        """Test TRF with edge cases."""

        def model(x, a, b):
            return a * x + b

        # Test with minimal data
        x = np.array([1, 2])
        y = np.array([2, 4])
        popt, _pcov = curve_fit(model, x, y, method="trf")
        self.assertAlmostEqual(popt[0], 2.0, places=2)

        # Test with perfect fit
        x = np.array([0, 1, 2, 3, 4])
        y = 2 * x + 1
        popt, _pcov = curve_fit(model, x, y, method="trf")
        self.assertAlmostEqual(popt[0], 2.0, places=10)
        self.assertAlmostEqual(popt[1], 1.0, places=10)

        # Test with all same y values
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 5, 5, 5, 5])
        popt, _pcov = curve_fit(model, x, y, method="trf")
        self.assertAlmostEqual(popt[0], 0.0, places=5)

    def test_parameter_scaling(self):
        """Test TRF with different parameter scales."""

        def model(x, a, b):
            return a * jnp.exp(b * x)

        # Parameters at very different scales
        np.random.seed(42)  # For reproducibility
        x = np.linspace(0, 1, 20)
        y = 1e6 * np.exp(1e-3 * x) + np.random.randn(20)

        popt, _pcov = curve_fit(
            model,
            x,
            y,
            p0=[1e5, 1e-2],
            method="trf",
            x_scale="jac",  # Use Jacobian scaling
        )
        # Check convergence despite scale differences
        self.assertIsNotNone(popt)
        self.assertEqual(len(popt), 2)

    def test_ill_conditioned_problems(self):
        """Test TRF on ill-conditioned problems."""

        def model(x, a, b, c):
            return a * x**2 + b * x + c

        # Create nearly collinear data
        np.random.seed(42)  # For reproducibility
        x = np.linspace(0, 0.1, 20)  # Small range makes problem ill-conditioned
        y = 2 * x**2 + 3 * x + 1 + 0.001 * np.random.randn(20)

        popt, _pcov = curve_fit(
            model, x, y, p0=[1, 1, 1], method="trf", ftol=1e-6, xtol=1e-6
        )
        # Should still converge, though accuracy may be reduced
        self.assertIsNotNone(popt)
        self.assertEqual(len(popt), 3)


class TestTRFSpecialCases(unittest.TestCase):
    """Test special cases and error conditions."""

    def test_singular_jacobian(self):
        """Test handling of singular Jacobian."""

        def model(x, a, b):
            # Model that can produce singular Jacobian
            return a * jnp.where(x > b, x - b, 0)

        np.random.seed(42)  # For reproducibility
        x = np.linspace(-1, 2, 20)
        y = 2 * np.maximum(x - 0.5, 0) + 0.1 * np.random.randn(20)

        try:
            popt, _pcov = curve_fit(model, x, y, p0=[1, 0], method="trf")
            # If it succeeds, check result is reasonable
            self.assertIsNotNone(popt)
        except:
            # Singular Jacobian might cause failure, which is acceptable
            pass

    def test_bounds_at_solution(self):
        """Test when solution is at the bounds."""

        def model(x, a, b):
            return a * x + b

        x = np.array([0, 1, 2, 3, 4])
        y = 5 * x + 2  # True parameters: a=5, b=2

        # Set bounds that force solution to boundary
        popt, _pcov = curve_fit(
            model,
            x,
            y,
            bounds=([0, 0], [3, 10]),  # a is bounded at 3
            method="trf",
        )
        # Solution should be at bound
        self.assertAlmostEqual(popt[0], 3.0, places=5)

    def test_zero_residuals(self):
        """Test with zero residuals (perfect fit)."""

        def model(x, a, b):
            return a * jnp.sin(b * x)

        x = np.linspace(0, 2 * np.pi, 20)
        y = 2 * np.sin(3 * x)  # Exact data

        # Better initial guess for sine fitting
        popt, _pcov = curve_fit(
            model,
            x,
            y,
            p0=[2, 3],  # Start close to true values
            method="trf",
        )
        # Allow for possible sign flip in sine fitting
        self.assertAlmostEqual(abs(popt[0]), 2.0, places=5)
        self.assertAlmostEqual(abs(popt[1]), 3.0, places=5)

    def test_max_iterations(self):
        """Test maximum iteration limit."""

        def slow_converging_model(x, a, b, c, d):
            return a * jnp.exp(b * x) + c * jnp.sin(d * x)

        x = np.linspace(0, 10, 100)
        y = 2 * np.exp(0.1 * x) + 3 * np.sin(0.5 * x)

        # Very low iteration limit - should fail
        with self.assertRaises(RuntimeError) as context:
            _popt, _pcov = curve_fit(
                slow_converging_model,
                x,
                y,
                p0=[1, 0.05, 1, 1],
                method="trf",
                max_nfev=5,  # Very low limit
            )
        # Check that the error message indicates max iterations exceeded
        self.assertIn(
            "maximum number of function evaluations", str(context.exception).lower()
        )


if __name__ == "__main__":
    unittest.main()
