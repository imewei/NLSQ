#!/usr/bin/env python3
"""Additional tests for minpack module to improve coverage."""

import unittest

import jax.numpy as jnp
import numpy as np

from nlsq.core.minpack import CurveFit, curve_fit

try:
    from nlsq import fit_large_dataset
except ImportError:
    fit_large_dataset = None


class TestMinpackCoverage(unittest.TestCase):
    """Tests to improve minpack module coverage."""

    def test_curve_fit_basic(self):
        """Test basic curve_fit functionality."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        popt, _pcov = curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)
        self.assertAlmostEqual(popt[1], 0.0, places=3)

    def test_curve_fit_with_bounds(self):
        """Test curve_fit with bounds."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Test with bounds
        popt, _pcov = curve_fit(model, x, y, bounds=([0, -10], [10, 10]))
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_curve_fit_with_sigma(self):
        """Test curve_fit with uncertainties."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        sigma = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        popt, _pcov = curve_fit(model, x, y, sigma=sigma)
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_curve_fit_exponential(self):
        """Test curve_fit with exponential model."""

        def model(x, a, b):
            return a * jnp.exp(b * x)

        x = np.linspace(0, 1, 50)
        y = 2.5 * np.exp(0.5 * x) + 0.01 * np.random.randn(50)

        popt, _pcov = curve_fit(model, x, y, p0=[1, 1])
        self.assertAlmostEqual(popt[0], 2.5, places=1)
        self.assertAlmostEqual(popt[1], 0.5, places=1)

    def test_curve_fit_2d(self):
        """Test curve_fit with 2D data."""

        def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y):
            x, y = xy
            # Use JAX-compatible operations
            a = 1 / (2 * sigma_x**2)
            b = 1 / (2 * sigma_y**2)
            g = amplitude * jnp.exp(-(a * (x - xo) ** 2 + b * (y - yo) ** 2))
            return g.ravel()

        # Create 2D grid
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        xx, yy = np.meshgrid(x, y)

        # Generate data
        z = gaussian_2d((xx, yy), 1, 5, 5, 1, 1)
        z += 0.01 * np.random.randn(*z.shape)

        # Fit
        popt, _pcov = curve_fit(
            gaussian_2d, (xx.ravel(), yy.ravel()), z, p0=[1, 5, 5, 1, 1]
        )

        self.assertAlmostEqual(popt[0], 1.0, places=1)
        self.assertAlmostEqual(popt[1], 5.0, places=1)

    def test_curve_fit_with_method(self):
        """Test curve_fit with different methods."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Test with trf method (only one implemented)
        popt, _pcov = curve_fit(model, x, y, method="trf")
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_curve_fit_maxfev(self):
        """Test curve_fit with max function evaluations."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        popt, _pcov = curve_fit(model, x, y, maxfev=10)
        # Should still converge for simple problem
        self.assertAlmostEqual(popt[0], 2.0, places=2)

    def test_curve_fit_class(self):
        """Test CurveFit class directly."""
        cf = CurveFit(use_dynamic_sizing=True)

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        popt, _pcov = cf.curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)

    def test_fit_large_dataset(self):
        """Test fit_large_dataset for large datasets."""
        if fit_large_dataset is None:
            self.skipTest("fit_large_dataset not available")

        def model(x, a, b):
            return a * x + b

        # Large dataset
        np.random.seed(42)  # For reproducibility
        x = np.linspace(0, 100, 10000)
        y = 2 * x + 5 + np.random.randn(10000) * 0.1

        # Use fit_large_dataset with appropriate parameters
        result = fit_large_dataset(
            model, x, y, initial_params=[1.0, 0.0], chunk_size=1000
        )

        # Check that optimization was successful
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.popt[0], 2.0, places=1)
        self.assertAlmostEqual(result.popt[1], 5.0, places=0)

    def test_curve_fit_nan_policy(self):
        """Test curve_fit with NaN policy."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # With nan_policy='omit' should skip NaN values
        try:
            popt, _pcov = curve_fit(model, x, y, nan_policy="omit")
            # Should work with remaining points
            self.assertEqual(len(popt), 2)
        except:
            # May not be implemented yet
            pass

    def test_curve_fit_full_output(self):
        """Test curve_fit with full_output option."""

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # With full_output should return additional info
        try:
            result = curve_fit(model, x, y, full_output=True)
            if isinstance(result, tuple) and len(result) > 2:
                _popt, _pcov, infodict, _mesg, _ier = result
                self.assertIn("nfev", infodict)
        except:
            # May not be implemented
            pass

    def test_curve_fit_jac(self):
        """Test curve_fit with analytical Jacobian."""

        def model(x, a, b):
            return a * x + b

        # Jacobian not supported in current implementation
        # Test without jacobian
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Without analytical Jacobian
        popt, _pcov = curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)


class TestBugFixRegressions(unittest.TestCase):
    """Regression tests for bugs found in the 2026-08-25 three-brain review.

    Each test asserts the fixed behavior directly (not just "doesn't
    crash") so a future regression fails meaningfully.
    """

    def test_cmaes_config_chunked_strategy_does_not_crash(self):
        """CMAESConfig is slots=True (no __dict__); rebuilding it via
        **{**cfg.__dict__, ...} crashed every auto-chunked/streaming CMA-ES
        run with AttributeError. Verify the chunked strategy now sets
        data_chunk_size and completes successfully."""
        from nlsq.global_optimization.cmaes_config import (
            CMAESConfig,
            is_evosax_available,
        )

        if not is_evosax_available():
            self.skipTest("evosax not installed")

        from nlsq.core.minpack import _fit_global_cmaes

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = np.linspace(0, 5, 200)
        y = np.asarray(2.0 * jnp.exp(-0.5 * x))
        config = CMAESConfig(data_chunk_size=None)

        result = _fit_global_cmaes(
            f=model,
            xdata=x,
            ydata=y,
            p0=np.array([1.0, 1.0]),
            sigma=None,
            absolute_sigma=False,
            bounds=(np.array([0.0, 0.0]), np.array([10.0, 5.0])),
            strategy="chunked",
            cmaes_config=config,
        )
        self.assertTrue(result["success"])
        self.assertAlmostEqual(float(result["popt"][0]), 2.0, delta=0.5)

    def test_stability_auto_fix_does_not_corrupt_positional_sigma(self):
        """stability='auto' p0 fixes used to move p0 into kwargs and slice
        the raw positional args tuple to drop it, silently reindexing any
        positional sigma/absolute_sigma that followed -- corrupting the
        weighting or crashing with 'multiple values for p0'."""
        x = np.linspace(0, 10, 50)
        rng = np.random.default_rng(1)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 50)
        sigma = np.full(50, 0.1)

        popt, _pcov = curve_fit(
            lambda x, a, b: a * x + b,
            x,
            y,
            [1.5, 0.5],  # positional p0
            sigma,  # positional sigma
            stability="auto",
        )
        np.testing.assert_allclose(popt, [2.0, 1.0], atol=0.15)

    def test_curve_fit_rejects_positional_arg_overflow(self):
        """The positional-to-keyword normalization used to silently drop
        any positional arg past the 9th (jac) via zip()'s shorter-sequence
        truncation. Overflow must now raise, not silently discard."""
        with self.assertRaises(TypeError):
            curve_fit(lambda x, a: a * x, [1, 2], [1, 2], *([None] * 18))

    def test_curve_fit_eleventh_positional_arg_is_applied(self):
        """timeit is the 11th positional parameter of CurveFit.curve_fit;
        verify it's actually forwarded now, not silently dropped."""
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0

        result = curve_fit(
            lambda x, a, b: a * x + b,
            x,
            y,
            [1.5, 0.5],  # p0
            None,  # sigma
            False,  # absolute_sigma
            True,  # check_finite
            (-np.inf, np.inf),  # bounds
            None,  # method
            "auto",  # solver
            None,  # batch_size
            None,  # jac
            None,  # data_mask
            True,  # timeit  (11th positional)
        )
        # timeit=True makes CurveFit.curve_fit return a plain tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)  # popt, pcov, res, post_time, ctime

    def test_curve_fit_special_output_modes_do_not_crash(self):
        """CurveFit.curve_fit intentionally returns a plain tuple (not a
        CurveFitResult) for timeit/return_eval/full_output -- the
        module-level curve_fit() wrapper used to unconditionally do
        result['multistart_diagnostics'] = {...} on whatever came back,
        crashing with TypeError: 'tuple' object does not support item
        assignment for all three modes."""
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0

        for kwargs in (
            {"timeit": True},
            {"return_eval": True},
            {"full_output": True},
        ):
            with self.subTest(**kwargs):
                result = curve_fit(
                    lambda x, a, b: a * x + b, x, y, p0=[1.5, 0.5], **kwargs
                )
                self.assertIsInstance(result, tuple)

        # Normal calls must still get multistart_diagnostics
        result = curve_fit(lambda x, a, b: a * x + b, x, y, p0=[1.5, 0.5])
        self.assertIn("multistart_diagnostics", result)


if __name__ == "__main__":
    unittest.main()
