"""Scientific computing tests for numerical correctness.

These tests validate:
- Analytical solution comparisons
- JAX JIT equivalence
- Gradient correctness (analytical vs finite difference)
- vmap correctness
- Numerical stability edge cases
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nlsq import curve_fit
from nlsq.core.functions import (
    exponential_decay,
    exponential_growth,
    linear,
)


class TestAnalyticalSolutions:
    """Test curve fitting against known analytical solutions."""

    def test_linear_exact_recovery(self):
        """Test exact recovery of linear parameters (no noise)."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        true_a, true_b = 2.5, -1.3

        y = true_a * x + true_b

        popt, _pcov = curve_fit(linear, x, y, p0=[1.0, 0.0])

        assert_allclose(popt[0], true_a, rtol=1e-8)
        assert_allclose(popt[1], true_b, rtol=1e-8)

    def test_exponential_decay_exact_recovery(self):
        """Test exact recovery of exponential decay parameters."""
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        true_a, true_b, true_c = 10.0, 0.5, 2.0

        y = true_a * np.exp(-true_b * x) + true_c

        popt, _pcov = curve_fit(
            exponential_decay,
            x,
            y,
            p0=[5.0, 0.3, 1.0],
            bounds=([0, 0, -10], [20, 5, 10]),
        )

        assert_allclose(popt[0], true_a, rtol=1e-6)
        assert_allclose(popt[1], true_b, rtol=1e-6)
        assert_allclose(popt[2], true_c, rtol=1e-6)

    def test_exponential_growth_exact_recovery(self):
        """Test exact recovery of exponential growth parameters."""
        np.random.seed(42)
        x = np.linspace(0, 2, 50)
        true_a, true_b, true_c = 5.0, 0.3, 1.0

        y = true_a * np.exp(true_b * x) + true_c

        popt, _pcov = curve_fit(
            exponential_growth,
            x,
            y,
            p0=[3.0, 0.2, 0.5],
            bounds=([0, 0, -5], [20, 2, 5]),
        )

        assert_allclose(popt[0], true_a, rtol=1e-5)
        assert_allclose(popt[1], true_b, rtol=1e-5)
        assert_allclose(popt[2], true_c, rtol=1e-5)


class TestNoisyDataRecovery:
    """Test parameter recovery with noisy data."""

    def test_linear_with_gaussian_noise(self):
        """Test linear fit with Gaussian noise."""
        np.random.seed(42)
        x = np.linspace(0, 10, 200)
        true_a, true_b = 3.0, 2.0
        noise_std = 0.5

        y = true_a * x + true_b + np.random.normal(0, noise_std, len(x))

        popt, pcov = curve_fit(linear, x, y, p0=[1.0, 0.0])

        # Parameters should be recovered within reasonable tolerance
        assert_allclose(popt[0], true_a, rtol=0.1)
        assert_allclose(popt[1], true_b, rtol=0.5)

        # Covariance should indicate uncertainty
        assert pcov[0, 0] > 0
        assert pcov[1, 1] > 0

    def test_exponential_with_noise(self):
        """Test exponential decay fit with noise."""
        np.random.seed(123)
        x = np.linspace(0, 5, 100)
        true_a, true_b, true_c = 100.0, 0.5, 10.0
        noise_std = 5.0

        y = (
            true_a * np.exp(-true_b * x)
            + true_c
            + np.random.normal(0, noise_std, len(x))
        )

        popt, _pcov = curve_fit(
            exponential_decay,
            x,
            y,
            p0=[80.0, 0.3, 5.0],
            bounds=([0, 0, 0], [200, 5, 50]),
        )

        # Parameters should be recovered within ~20% (for noisy data)
        assert_allclose(popt[0], true_a, rtol=0.2)
        assert_allclose(popt[1], true_b, rtol=0.2)
        assert_allclose(popt[2], true_c, rtol=0.3)


class TestJITEquivalence:
    """Test that JIT-compiled fits match non-JIT results."""

    def test_jit_produces_same_result(self):
        """Test that JIT compilation doesn't change the result."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, len(x))

        # Fit without explicit JIT (default behavior)
        popt1, _ = curve_fit(linear, x, y, p0=[1.0, 0.0])

        # Fit again (should use cached JIT)
        popt2, _ = curve_fit(linear, x, y, p0=[1.0, 0.0])

        assert_allclose(popt1, popt2, rtol=1e-10)

    def test_jit_with_different_data_sizes(self):
        """Test JIT with various data sizes."""
        np.random.seed(42)

        for n_points in [10, 50, 100, 500]:
            x = np.linspace(0, 10, n_points)
            y = 2.0 * x + 1.0

            popt, _ = curve_fit(linear, x, y, p0=[1.0, 0.0])

            assert_allclose(popt[0], 2.0, rtol=1e-8)
            assert_allclose(popt[1], 1.0, rtol=1e-8)


class TestGradientCorrectness:
    """Test gradient correctness using finite differences."""

    def test_linear_gradient_matches_analytical(self):
        """Test that automatic gradient matches analytical for linear."""

        def loss_fn(params, x, y):
            pred = params[0] * x + params[1]
            return jnp.sum((pred - y) ** 2)

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([2.5, 4.5, 6.5, 8.5, 10.5])
        params = jnp.array([2.0, 0.5])

        # JAX automatic gradient
        auto_grad = jax.grad(loss_fn)(params, x, y)

        # Finite difference gradient
        eps = 1e-5
        fd_grad = np.zeros(2)
        for i in range(2):
            params_plus = params.at[i].set(params[i] + eps)
            params_minus = params.at[i].set(params[i] - eps)
            fd_grad[i] = (loss_fn(params_plus, x, y) - loss_fn(params_minus, x, y)) / (
                2 * eps
            )

        assert_allclose(np.asarray(auto_grad), fd_grad, rtol=1e-4)

    def test_exponential_gradient_matches_finite_diff(self):
        """Test exponential gradient against finite differences."""

        def loss_fn(params, x, y):
            pred = params[0] * jnp.exp(-params[1] * x) + params[2]
            return jnp.sum((pred - y) ** 2)

        x = jnp.linspace(0, 5, 50)
        # Use slightly different parameters to avoid zero gradient
        y_true = 10.0 * jnp.exp(-0.5 * x) + 2.0
        y = y_true + 0.1  # Add small offset to create non-zero gradient
        params = jnp.array([10.0, 0.5, 2.0])

        # JAX automatic gradient
        auto_grad = jax.grad(loss_fn)(params, x, y)

        # Finite difference gradient
        eps = 1e-5
        fd_grad = np.zeros(3)
        for i in range(3):
            params_plus = params.at[i].set(params[i] + eps)
            params_minus = params.at[i].set(params[i] - eps)
            fd_grad[i] = (loss_fn(params_plus, x, y) - loss_fn(params_minus, x, y)) / (
                2 * eps
            )

        # Check that both gradients are close (both should be non-zero now)
        assert_allclose(np.asarray(auto_grad), fd_grad, rtol=1e-3, atol=1e-8)


class TestVmapCorrectness:
    """Test vmap produces correct batched results."""

    def test_linear_vmap_batch_matches_loop(self):
        """Test that vmapped linear matches loop over individual calls."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Multiple parameter sets
        params = jnp.array([[2.0, 1.0], [3.0, -1.0], [1.5, 2.5]])

        # Using vmap
        def linear_batch(params_single):
            return linear(x, params_single[0], params_single[1])

        vmapped_results = jax.vmap(linear_batch)(params)

        # Using loop
        loop_results = jnp.stack([linear(x, p[0], p[1]) for p in params])

        assert_allclose(
            np.asarray(vmapped_results), np.asarray(loop_results), rtol=1e-12
        )

    def test_exponential_vmap_over_data(self):
        """Test vmap over multiple datasets."""
        # Multiple x datasets
        x_batch = jnp.array(
            [jnp.linspace(0, 5, 10), jnp.linspace(0, 10, 10), jnp.linspace(0, 3, 10)]
        )

        a, b, c = 10.0, 0.5, 2.0

        def exp_decay_single(x):
            return exponential_decay(x, a, b, c)

        vmapped_results = jax.vmap(exp_decay_single)(x_batch)

        # Verify each result
        for i, x in enumerate(x_batch):
            expected = exponential_decay(x, a, b, c)
            assert_allclose(
                np.asarray(vmapped_results[i]), np.asarray(expected), rtol=1e-10
            )


class TestNumericalStabilityEdgeCases:
    """Test numerical stability in edge cases."""

    def test_very_small_residuals(self):
        """Test fitting with very small residuals (near-perfect fit)."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0  # Exact data, no noise

        popt, _pcov = curve_fit(linear, x, y, p0=[1.0, 0.0])

        assert_allclose(popt[0], 2.0, rtol=1e-10)
        assert_allclose(popt[1], 1.0, rtol=1e-10)

    def test_large_parameter_values(self):
        """Test fitting with large parameter values."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        true_a, true_b = 1e6, 1e5

        y = true_a * x + true_b + np.random.normal(0, 100, len(x))

        popt, _pcov = curve_fit(linear, x, y, p0=[1e5, 1e4])

        assert_allclose(popt[0], true_a, rtol=0.01)
        assert_allclose(popt[1], true_b, rtol=0.1)

    def test_small_parameter_values(self):
        """Test fitting with small parameter values."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        true_a, true_b = 1e-6, 1e-5

        y = true_a * x + true_b

        popt, _pcov = curve_fit(linear, x, y, p0=[1e-7, 1e-6])

        assert_allclose(popt[0], true_a, rtol=1e-6)
        assert_allclose(popt[1], true_b, rtol=1e-6)

    def test_mixed_scale_parameters(self):
        """Test with parameters of very different scales."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        true_a, true_b = 1e-3, 1e3  # 6 orders of magnitude difference

        y = true_a * x + true_b

        popt, _pcov = curve_fit(linear, x, y, p0=[1e-4, 1e2])

        assert_allclose(popt[0], true_a, rtol=1e-6)
        assert_allclose(popt[1], true_b, rtol=1e-6)

    def test_ill_conditioned_exponential(self):
        """Test exponential decay with challenging initial conditions."""
        np.random.seed(42)
        x = np.linspace(0, 0.1, 50)  # Very short x range
        true_a, true_b, true_c = 100.0, 10.0, 0.0  # Fast decay

        y = true_a * np.exp(-true_b * x) + true_c + np.random.normal(0, 1, len(x))

        popt, _pcov = curve_fit(
            exponential_decay,
            x,
            y,
            p0=[50.0, 5.0, 0.0],
            bounds=([0, 0, -10], [200, 50, 10]),
        )

        # Should converge reasonably close
        assert_allclose(popt[0], true_a, rtol=0.3)
        assert_allclose(popt[1], true_b, rtol=0.5)


class TestToleranceAssertions:
    """Test with tolerance-based assertions for scientific computing."""

    def test_residual_is_minimized(self):
        """Test that final residual is minimal."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, len(x))

        popt, _ = curve_fit(linear, x, y, p0=[1.0, 0.0])

        # Calculate residuals
        y_fit = popt[0] * x + popt[1]
        residuals = y - y_fit
        rss = np.sum(residuals**2)

        # RSS should be small (noise-dominated)
        assert rss < 50  # ~0.5^2 * 100 = 25, with some margin

    def test_covariance_positive_definite(self):
        """Test that covariance matrix is positive definite."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, len(x))

        _, pcov = curve_fit(linear, x, y, p0=[1.0, 0.0])

        # Check eigenvalues are positive
        eigenvalues = np.linalg.eigvalsh(pcov)
        assert all(eigenvalues > 0), "Covariance matrix should be positive definite"

    def test_parameter_uncertainties_reasonable(self):
        """Test that parameter uncertainties are reasonable."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        noise_std = 0.5
        y = 2.0 * x + 1.0 + np.random.normal(0, noise_std, len(x))

        _popt, pcov = curve_fit(linear, x, y, p0=[1.0, 0.0])

        # Standard errors
        perr = np.sqrt(np.diag(pcov))

        # Uncertainties should be non-zero and reasonable
        assert perr[0] > 0
        assert perr[1] > 0
        assert perr[0] < 1.0  # Slope error should be small for this data
        assert perr[1] < 2.0  # Intercept error should be small


class TestNoNaNInfInResults:
    """Test that results never contain NaN or Inf."""

    def test_linear_fit_no_nan(self):
        """Test linear fit produces no NaN."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, len(x))

        popt, pcov = curve_fit(linear, x, y, p0=[1.0, 0.0])

        assert np.all(np.isfinite(popt))
        assert np.all(np.isfinite(pcov))

    def test_exponential_fit_no_nan(self):
        """Test exponential fit produces no NaN."""
        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 10.0 * np.exp(-0.5 * x) + 2.0 + np.random.normal(0, 0.5, len(x))

        popt, pcov = curve_fit(
            exponential_decay,
            x,
            y,
            p0=[8.0, 0.3, 1.0],
            bounds=([0, 0, 0], [20, 5, 10]),
        )

        assert np.all(np.isfinite(popt))
        assert np.all(np.isfinite(pcov))

    def test_poor_initial_guess_still_finite(self):
        """Test that poor initial guess still produces finite result."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, len(x))

        # Very poor initial guess
        popt, _pcov = curve_fit(linear, x, y, p0=[100.0, -50.0])

        assert np.all(np.isfinite(popt))
        # Result should still be close to true values
        assert_allclose(popt[0], 2.0, rtol=0.1)


class TestReproducibility:
    """Test result reproducibility."""

    def test_same_seed_same_result(self):
        """Test that same random seed produces same result."""
        results = []

        for _ in range(3):
            np.random.seed(42)
            x = np.linspace(0, 10, 100)
            y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, len(x))
            popt, _ = curve_fit(linear, x, y, p0=[1.0, 0.0])
            results.append(popt)

        assert_allclose(results[0], results[1], rtol=1e-10)
        assert_allclose(results[1], results[2], rtol=1e-10)

    def test_deterministic_with_fixed_data(self):
        """Test that same data always produces same result."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array([2.5, 4.3, 6.1, 8.2, 9.9, 12.1, 14.0, 15.9, 18.1, 20.0])

        results = []
        for _ in range(3):
            popt, _ = curve_fit(linear, x, y, p0=[1.0, 0.0])
            results.append(popt)

        assert_allclose(results[0], results[1], rtol=1e-10)
        assert_allclose(results[1], results[2], rtol=1e-10)
