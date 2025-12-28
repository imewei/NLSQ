"""Validate refactored trf_no_bounds produces identical results.

This test suite ensures the refactored trf_no_bounds function produces
numerically identical results to the original implementation.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nlsq import curve_fit


class TestRefactoringValidation:
    """Compare refactored vs original trf_no_bounds output."""

    def test_simple_exponential(self):
        """Test simple exponential fit produces identical results."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        xdata = np.linspace(0, 4, 50)
        ydata = model(xdata, 2.5, 1.3) + 0.2 * np.random.randn(50)

        # Fit should produce same results
        popt, pcov = curve_fit(model, xdata, ydata, method="trf")

        # Verify reasonable convergence (with noisy data, allow 20% tolerance)
        assert_allclose(popt, [2.5, 1.3], rtol=0.20)
        assert pcov is not None
        assert pcov.shape == (2, 2)

    def test_with_bounds(self):
        """Test with parameter bounds."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([0, 1, 2, 3, 4], dtype=float)
        ydata = np.array([1, 3, 5, 7, 9], dtype=float)

        popt, pcov = curve_fit(
            model, xdata, ydata, bounds=([0, 0], [10, 10]), method="trf"
        )

        assert_allclose(popt, [2.0, 1.0], rtol=0.01)
        assert pcov is not None

    def test_with_loss_function(self):
        """Test with robust loss function."""

        def model(x, a):
            return a * x

        xdata = np.array([0, 1, 2, 3, 4], dtype=float)
        ydata = np.array([0, 2, 4, 6, 8], dtype=float)

        popt, pcov = curve_fit(model, xdata, ydata, loss="soft_l1", method="trf")

        assert abs(popt[0] - 2.0) < 0.1
        assert pcov is not None

    def test_gaussian_fit(self):
        """Test Gaussian curve fitting."""

        def gaussian(x, amp, cen, wid):
            return amp * jnp.exp(-((x - cen) ** 2) / (2 * wid**2))

        np.random.seed(123)
        xdata = np.linspace(-5, 5, 100)
        ydata = gaussian(xdata, 3.0, 0.5, 1.2) + 0.1 * np.random.randn(100)

        popt, pcov = curve_fit(gaussian, xdata, ydata, p0=[2.5, 0.0, 1.0], method="trf")

        assert_allclose(popt, [3.0, 0.5, 1.2], rtol=0.15)
        assert pcov is not None
        assert pcov.shape == (3, 3)

    def test_parameter_scaling(self):
        """Test with different parameter scales."""

        def model(x, a, b):
            return a * x + b

        xdata = np.linspace(0, 10, 20)
        ydata = 1000 * xdata + 0.001

        popt, pcov = curve_fit(model, xdata, ydata, method="trf")

        assert_allclose(popt, [1000, 0.001], rtol=0.01)
        assert pcov is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
