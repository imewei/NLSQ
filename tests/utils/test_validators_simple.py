"""
Validator tests for Sprint 1 coverage.

Focused on actual validator behavior.
"""

import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit


class TestValidatorIntegration:
    """Test validators through curve_fit integration."""

    def test_valid_basic_fit(self):
        """Test valid inputs work."""

        def model(x, a, b):
            return a * x + b

        xdata = np.array([1, 2, 3])
        ydata = np.array([2, 4, 6])

        popt, _pcov = curve_fit(model, xdata, ydata)
        assert popt is not None

    def test_numpy_arrays(self):
        """Test NumPy arrays accepted."""

        def model(x, a):
            return a * x

        popt, _pcov = curve_fit(model, np.array([1, 2, 3]), np.array([2, 4, 6]))
        assert popt is not None

    def test_with_bounds(self):
        """Test bounded fit."""

        def model(x, a, b):
            return a * x + b

        popt, _pcov = curve_fit(
            model, np.array([1, 2, 3]), np.array([2, 4, 6]), bounds=([0, -10], [10, 10])
        )
        assert popt is not None

    def test_with_scalar_bounds(self):
        """Test scipy-style scalar bounds (0, np.inf) broadcast to all params."""

        def model(x, a, b):
            return a * x + b

        popt, _pcov = curve_fit(
            model,
            np.array([1, 2, 3]),
            np.array([2, 4, 6]),
            bounds=(0, np.inf),
        )
        assert popt is not None
        assert np.all(popt >= 0)

    def test_with_sigma(self):
        """Test weighted fit."""

        def model(x, a, b):
            return a * x + b

        popt, _pcov = curve_fit(
            model,
            np.array([1, 2, 3]),
            np.array([2, 4, 6]),
            sigma=np.array([0.1, 0.2, 0.3]),
        )
        assert popt is not None

    def test_with_p0(self):
        """Test with initial parameters."""

        def model(x, a, b):
            return a * x + b

        popt, _pcov = curve_fit(
            model, np.array([1, 2, 3]), np.array([2, 4, 6]), p0=[1.0, 0.0]
        )
        assert popt is not None

    def test_method_trf(self):
        """Test TRF method."""

        def model(x, a):
            return a * x

        popt, _pcov = curve_fit(
            model, np.array([1, 2, 3]), np.array([2, 4, 6]), method="trf"
        )
        assert popt is not None

    def test_python_lists(self):
        """Test Python lists work."""

        def model(x, a):
            return a * x

        popt, _pcov = curve_fit(model, [1, 2, 3], [2, 4, 6])
        assert popt is not None

    def test_mixed_types(self):
        """Test mixed NumPy/JAX."""

        def model(x, a):
            return a * x

        popt, _pcov = curve_fit(model, np.array([1, 2, 3]), jnp.array([2, 4, 6]))
        assert popt is not None


# 8 focused integration tests


class TestEnableStabilityMultiVariable:
    """xdata (k, M) must validate the same with enable_stability=True."""

    def test_two_variable_xdata_fits(self):
        from nlsq import CurveFit

        rng = np.random.default_rng(0)
        n = 100
        X = rng.uniform(0, 1, (2, n))
        y = 2 * X[0] + 3 * X[1] + rng.normal(0, 0.01, n)

        def f(X, a, b):
            return a * X[0] + b * X[1]

        res = CurveFit(enable_stability=True).curve_fit(f, X, y, p0=[1, 1])
        np.testing.assert_allclose(np.asarray(res.popt), [2.0, 3.0], atol=0.05)

    def test_wide_xdata_with_many_variables(self):
        from nlsq import CurveFit

        rng = np.random.default_rng(1)
        n = 60
        X = rng.uniform(0, 1, (12, n))  # k > 10 rows exercises the test-slice path
        y = X.sum(axis=0)

        def f(X, a):
            return a * X.sum(axis=0)

        res = CurveFit(enable_stability=True).curve_fit(f, X, y, p0=[0.5])
        np.testing.assert_allclose(np.asarray(res.popt), [1.0], atol=1e-3)
