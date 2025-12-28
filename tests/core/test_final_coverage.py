#!/usr/bin/env python3
"""Final tests to reach 74% coverage by testing untested code paths."""

import unittest

import jax.numpy as jnp
import numpy as np

from nlsq import __version__
from nlsq.core.minpack import curve_fit


class TestFinalCoverage(unittest.TestCase):
    """Final tests to reach 74% coverage."""

    def test_import_all(self):
        """Test importing all main functions."""
        from nlsq import CurveFit as CurveFit_from_nlsq
        from nlsq import curve_fit as cf_from_nlsq

        self.assertIsNotNone(cf_from_nlsq)
        self.assertIsNotNone(CurveFit_from_nlsq)

    def test_version(self):
        """Test version is defined."""
        self.assertIsNotNone(__version__)
        self.assertIsInstance(__version__, str)

    def test_linear_regression(self):
        """Test simple linear regression."""

        def linear(x, a, b):
            return a * x + b

        # Perfect linear data
        x = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([1, 3, 5, 7, 9, 11])

        popt, _pcov = curve_fit(linear, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=10)
        self.assertAlmostEqual(popt[1], 1.0, places=10)

        # Check covariance is small for perfect fit
        self.assertLess(np.max(np.abs(_pcov)), 1e-10)

    def test_exponential_decay(self):
        """Test exponential decay fitting."""

        def exp_decay(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        x = np.linspace(0, 4, 50)
        y = 3.0 * np.exp(-0.5 * x) + 1.0
        y += 0.01 * np.random.randn(50)  # Small noise

        popt, _pcov = curve_fit(exp_decay, x, y, p0=[1, 1, 0])
        self.assertAlmostEqual(popt[0], 3.0, places=1)
        self.assertAlmostEqual(popt[1], 0.5, places=1)
        self.assertAlmostEqual(popt[2], 1.0, places=1)

    def test_gaussian_peak(self):
        """Test Gaussian peak fitting."""

        def gaussian(x, a, mu, sigma):
            return a * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

        x = np.linspace(-5, 5, 100)
        y = 2.0 * np.exp(-0.5 * ((x - 1.0) / 0.5) ** 2)
        y += 0.01 * np.random.randn(100)

        popt, _pcov = curve_fit(gaussian, x, y, p0=[1, 0, 1])
        self.assertAlmostEqual(popt[0], 2.0, places=1)
        self.assertAlmostEqual(popt[1], 1.0, places=1)
        self.assertAlmostEqual(abs(popt[2]), 0.5, places=1)

    def test_power_law(self):
        """Test power law fitting."""

        def power_law(x, a, b):
            return a * x**b

        x = np.linspace(0.1, 5, 50)
        y = 2.0 * x**1.5
        y += 0.01 * np.random.randn(50)

        popt, _pcov = curve_fit(power_law, x, y, p0=[1, 1])
        self.assertAlmostEqual(popt[0], 2.0, places=1)
        self.assertAlmostEqual(popt[1], 1.5, places=1)

    def test_logistic_curve(self):
        """Test logistic/sigmoid curve fitting."""

        def logistic(x, L, k, x0):
            return L / (1 + jnp.exp(-k * (x - x0)))

        x = np.linspace(-5, 5, 100)
        y = 10.0 / (1 + np.exp(-2.0 * (x - 1.0)))
        y += 0.1 * np.random.randn(100)

        popt, _pcov = curve_fit(logistic, x, y, p0=[10, 1, 0])
        self.assertAlmostEqual(popt[0], 10.0, places=0)
        self.assertAlmostEqual(popt[1], 2.0, places=0)
        self.assertAlmostEqual(popt[2], 1.0, places=0)

    def test_multiple_gaussians(self):
        """Test fitting sum of two Gaussians."""

        def double_gaussian(x, a1, mu1, sig1, a2, mu2, sig2):
            g1 = a1 * jnp.exp(-0.5 * ((x - mu1) / sig1) ** 2)
            g2 = a2 * jnp.exp(-0.5 * ((x - mu2) / sig2) ** 2)
            return g1 + g2

        x = np.linspace(-5, 5, 200)
        y = 1.0 * np.exp(-0.5 * ((x - (-2)) / 0.5) ** 2) + 2.0 * np.exp(
            -0.5 * ((x - 2) / 0.8) ** 2
        )
        y += 0.02 * np.random.randn(200)

        popt, _pcov = curve_fit(
            double_gaussian, x, y, p0=[1, -2, 1, 1, 2, 1], maxfev=2000
        )
        # Check peaks are roughly in right places
        self.assertTrue(abs(popt[1] - (-2)) < 1 or abs(popt[4] - (-2)) < 1)
        self.assertTrue(abs(popt[1] - 2) < 1 or abs(popt[4] - 2) < 1)

    def test_damped_oscillation(self):
        """Test damped oscillation fitting."""

        def damped_sine(x, a, b, omega, phi, c):
            return a * jnp.exp(-b * x) * jnp.sin(omega * x + phi) + c

        x = np.linspace(0, 10, 200)
        y = 2.0 * np.exp(-0.3 * x) * np.sin(2 * x + 0.5) + 1.0
        y += 0.05 * np.random.randn(200)

        popt, _pcov = curve_fit(damped_sine, x, y, p0=[2, 0.3, 2, 0, 1], maxfev=1000)
        # Check amplitude and decay roughly correct
        self.assertAlmostEqual(abs(popt[0]), 2.0, places=0)
        self.assertAlmostEqual(popt[1], 0.3, places=0)

    def test_rational_function(self):
        """Test rational function fitting."""

        def rational(x, a, b, c):
            return (a * x) / (b + x) + c

        x = np.linspace(0.1, 10, 50)
        y = (5.0 * x) / (2.0 + x) + 1.0
        y += 0.05 * np.random.randn(50)

        popt, _pcov = curve_fit(rational, x, y, p0=[1, 1, 0])
        self.assertAlmostEqual(popt[0], 5.0, places=0)
        self.assertAlmostEqual(popt[1], 2.0, places=0)
        self.assertAlmostEqual(popt[2], 1.0, places=0)

    def test_step_function(self):
        """Test step function (error function) fitting."""
        from jax.scipy.special import erf

        def step_func(x, a, x0, w):
            return a * (1 + erf((x - x0) / w)) / 2

        x = np.linspace(-5, 5, 100)
        y = 3.0 * (1 + erf((x - 1.0) / 0.5)) / 2
        y += 0.05 * np.random.randn(100)

        popt, _pcov = curve_fit(step_func, x, y, p0=[1, 0, 1])
        self.assertAlmostEqual(popt[0], 3.0, places=0)
        self.assertAlmostEqual(popt[1], 1.0, places=0)

    def test_voigt_profile(self):
        """Test Voigt profile (convolution of Gaussian and Lorentzian)."""

        def pseudo_voigt(x, a, x0, gamma, eta):
            # Pseudo-Voigt: linear combination of Gaussian and Lorentzian
            gaussian = jnp.exp(-4 * jnp.log(2) * ((x - x0) / gamma) ** 2)
            lorentzian = 1 / (1 + 4 * ((x - x0) / gamma) ** 2)
            return a * (eta * lorentzian + (1 - eta) * gaussian)

        x = np.linspace(-5, 5, 100)
        y = pseudo_voigt(x, 2.0, 1.0, 1.5, 0.3)
        y += 0.02 * np.random.randn(100)

        popt, _pcov = curve_fit(pseudo_voigt, x, y, p0=[1, 0, 1, 0.5])
        self.assertAlmostEqual(popt[0], 2.0, places=0)
        self.assertAlmostEqual(popt[1], 1.0, places=0)

    def test_periodic_with_trend(self):
        """Test periodic function with linear trend."""

        def periodic_trend(x, a, b, amp, freq, phase):
            return a * x + b + amp * jnp.sin(freq * x + phase)

        x = np.linspace(0, 10, 100)
        y = 0.5 * x + 2.0 + 1.5 * np.sin(3 * x + 0.5)
        y += 0.1 * np.random.randn(100)

        # Increase max iterations for complex optimization
        popt, _pcov = curve_fit(
            periodic_trend,
            x,
            y,
            p0=[0.5, 2, 1.5, 3, 0.5],  # Better initial guess
            max_nfev=1000,
        )
        self.assertAlmostEqual(popt[0], 0.5, places=0)
        self.assertAlmostEqual(popt[1], 2.0, places=0)
        self.assertAlmostEqual(abs(popt[2]), 1.5, places=0)

    def test_asymptotic_function(self):
        """Test asymptotic growth function."""

        def asymptotic(x, a, b, c):
            return a * (1 - jnp.exp(-b * x)) + c

        x = np.linspace(0, 10, 50)
        y = 5.0 * (1 - np.exp(-0.5 * x)) + 2.0
        y += 0.05 * np.random.randn(50)

        popt, _pcov = curve_fit(asymptotic, x, y, p0=[1, 1, 0])
        self.assertAlmostEqual(popt[0], 5.0, places=0)
        self.assertAlmostEqual(popt[1], 0.5, places=0)
        self.assertAlmostEqual(popt[2], 2.0, places=0)

    def test_curve_fit_with_all_options(self):
        """Test curve_fit with many options set."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.1, 3.9, 6.1, 7.9, 10.1])

        popt, _pcov = curve_fit(
            model,
            x,
            y,
            p0=[1, 0],
            bounds=([-10, -10], [10, 10]),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            x_scale="jac",
            loss="linear",
            f_scale=1.0,
            max_nfev=1000,
            verbose=0,
            absolute_sigma=True,
            check_finite=True,
        )
        self.assertAlmostEqual(popt[0], 2.0, places=1)
        self.assertAlmostEqual(popt[1], 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
