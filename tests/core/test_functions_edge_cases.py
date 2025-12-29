"""Edge case tests for nlsq.core.functions module.

Tests numerical edge cases, boundary conditions, and scientific computing scenarios.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nlsq.core.functions import (
    bounds_exponential_decay,
    bounds_linear,
    estimate_p0_exponential_decay,
    estimate_p0_exponential_growth,
    estimate_p0_linear,
    exponential_decay,
    exponential_growth,
    linear,
)


class TestLinearFunction:
    """Test linear function and its utilities."""

    def test_linear_basic(self):
        """Test basic linear function evaluation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = linear(x, 2.0, 3.0)
        expected = 2.0 * x + 3.0
        assert_allclose(result, expected, rtol=1e-12)

    def test_linear_zero_slope(self):
        """Test linear function with zero slope (constant)."""
        x = np.array([1.0, 2.0, 3.0])
        result = linear(x, 0.0, 5.0)
        expected = np.full_like(x, 5.0)
        assert_allclose(result, expected, rtol=1e-12)

    def test_linear_negative_slope(self):
        """Test linear function with negative slope."""
        x = np.array([0.0, 1.0, 2.0])
        result = linear(x, -2.0, 10.0)
        expected = np.array([10.0, 8.0, 6.0])
        assert_allclose(result, expected, rtol=1e-12)

    def test_linear_large_values(self):
        """Test linear function with large values."""
        x = np.array([1e6, 2e6, 3e6])
        result = linear(x, 1e-6, 0.0)
        expected = np.array([1.0, 2.0, 3.0])
        assert_allclose(result, expected, rtol=1e-10)

    def test_linear_jax_array(self):
        """Test linear function with JAX array."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = linear(x, 2.0, 1.0)
        expected = jnp.array([3.0, 5.0, 7.0])
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-12)

    def test_linear_scalar(self):
        """Test linear function with scalar input."""
        result = linear(2.0, 3.0, 1.0)
        assert_allclose(result, 7.0, rtol=1e-12)

    def test_estimate_p0_linear_basic(self):
        """Test linear parameter estimation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 3.0  # a=2, b=3
        p0 = estimate_p0_linear(x, y)
        assert len(p0) == 2
        assert_allclose(p0[0], 2.0, rtol=1e-10)  # slope
        assert_allclose(p0[1], 3.0, rtol=1e-10)  # intercept

    def test_estimate_p0_linear_noisy(self):
        """Test linear parameter estimation with noisy data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 1.5 * x + 2.0 + np.random.normal(0, 0.5, 100)
        p0 = estimate_p0_linear(x, y)
        assert len(p0) == 2
        assert_allclose(p0[0], 1.5, rtol=0.1)  # slope with tolerance
        assert_allclose(p0[1], 2.0, rtol=0.5)  # intercept with tolerance

    def test_estimate_p0_linear_constant(self):
        """Test linear estimation with constant y (slope = 0)."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])
        p0 = estimate_p0_linear(x, y)
        assert len(p0) == 2
        assert_allclose(p0[0], 0.0, atol=1e-10)  # slope
        assert_allclose(p0[1], 5.0, rtol=1e-10)  # intercept

    def test_estimate_p0_linear_single_point(self):
        """Test linear estimation with single point (fallback)."""
        x = np.array([1.0])
        y = np.array([5.0])
        p0 = estimate_p0_linear(x, y)
        # Should return fallback values
        assert len(p0) == 2
        assert np.isfinite(p0[0])
        assert np.isfinite(p0[1])

    def test_bounds_linear(self):
        """Test linear function bounds."""
        lower, upper = bounds_linear()
        assert len(lower) == 2
        assert len(upper) == 2
        assert all(np.isinf(lower))
        assert all(np.isinf(upper))


class TestExponentialDecay:
    """Test exponential decay function and utilities."""

    def test_exponential_decay_basic(self):
        """Test basic exponential decay evaluation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        result = exponential_decay(x, 10.0, 0.5, 2.0)
        expected = 10.0 * np.exp(-0.5 * x) + 2.0
        assert_allclose(result, expected, rtol=1e-10)

    def test_exponential_decay_at_zero(self):
        """Test exponential decay at x=0."""
        result = exponential_decay(0.0, 5.0, 1.0, 3.0)
        expected = 5.0 + 3.0  # a + c
        assert_allclose(result, expected, rtol=1e-12)

    def test_exponential_decay_large_x(self):
        """Test exponential decay as x → ∞ approaches asymptote."""
        x = np.array([100.0])
        result = exponential_decay(x, 10.0, 0.5, 2.0)
        # Should be very close to c=2.0
        assert_allclose(result, 2.0, rtol=1e-10)

    def test_exponential_decay_zero_rate(self):
        """Test exponential decay with zero rate (constant)."""
        x = np.array([0.0, 1.0, 2.0])
        result = exponential_decay(x, 5.0, 0.0, 2.0)
        expected = np.full_like(x, 7.0)  # a + c = 5 + 2
        assert_allclose(result, expected, rtol=1e-12)

    def test_exponential_decay_negative_amplitude(self):
        """Test exponential decay with negative amplitude (growth from below)."""
        x = np.array([0.0, 1.0, 2.0])
        result = exponential_decay(x, -5.0, 0.5, 10.0)
        expected = -5.0 * np.exp(-0.5 * x) + 10.0
        assert_allclose(result, expected, rtol=1e-10)

    def test_exponential_decay_jax_array(self):
        """Test exponential decay with JAX arrays."""
        x = jnp.array([0.0, 1.0, 2.0])
        result = exponential_decay(x, 10.0, 1.0, 0.0)
        expected = 10.0 * jnp.exp(-1.0 * x)
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-10)

    def test_estimate_p0_exponential_decay_basic(self):
        """Test exponential decay parameter estimation."""
        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 10.0 * np.exp(-0.5 * x) + 2.0
        p0 = estimate_p0_exponential_decay(x, y)
        assert len(p0) == 3
        # Amplitude estimate
        assert p0[0] > 0
        # Rate estimate (positive)
        assert p0[1] > 0
        # Offset estimate
        assert np.isfinite(p0[2])

    def test_estimate_p0_exponential_decay_noisy(self):
        """Test exponential decay estimation with noise."""
        np.random.seed(123)
        x = np.linspace(0, 10, 100)
        y = 50.0 * np.exp(-0.3 * x) + 5.0 + np.random.normal(0, 2, 100)
        p0 = estimate_p0_exponential_decay(x, y)
        assert len(p0) == 3
        # Should be reasonable estimates
        assert 30 < p0[0] < 70  # amplitude
        assert 0.01 < p0[1] < 1.0  # rate
        assert 0 < p0[2] < 20  # offset

    def test_estimate_p0_exponential_decay_constant(self):
        """Test estimation with constant data (edge case)."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])
        p0 = estimate_p0_exponential_decay(x, y)
        assert len(p0) == 3
        # All values should be finite
        assert all(np.isfinite(p0))

    def test_estimate_p0_exponential_decay_single_point(self):
        """Test estimation with single data point."""
        x = np.array([1.0])
        y = np.array([10.0])
        p0 = estimate_p0_exponential_decay(x, y)
        assert len(p0) == 3
        assert all(np.isfinite(p0))

    def test_estimate_p0_exponential_decay_negative_x(self):
        """Test estimation with negative x values."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = 10.0 * np.exp(-0.5 * x) + 1.0
        p0 = estimate_p0_exponential_decay(x, y)
        assert len(p0) == 3
        assert all(np.isfinite(p0))
        assert p0[1] > 0  # rate should be positive

    def test_bounds_exponential_decay(self):
        """Test exponential decay bounds."""
        lower, upper = bounds_exponential_decay()
        assert len(lower) == 3
        assert len(upper) == 3
        # a >= 0, b >= 0, c unconstrained
        assert lower[0] == 0
        assert lower[1] == 0
        assert np.isinf(lower[2])


class TestExponentialGrowth:
    """Test exponential growth function and utilities."""

    def test_exponential_growth_basic(self):
        """Test basic exponential growth evaluation."""
        x = np.array([0.0, 1.0, 2.0])
        result = exponential_growth(x, 2.0, 0.5, 1.0)
        expected = 2.0 * np.exp(0.5 * x) + 1.0
        assert_allclose(result, expected, rtol=1e-10)

    def test_exponential_growth_at_zero(self):
        """Test exponential growth at x=0."""
        result = exponential_growth(0.0, 5.0, 1.0, 2.0)
        expected = 5.0 + 2.0  # a + c
        assert_allclose(result, expected, rtol=1e-12)

    def test_exponential_growth_negative_x(self):
        """Test exponential growth with negative x (decay behavior)."""
        x = np.array([-2.0, -1.0, 0.0])
        result = exponential_growth(x, 10.0, 0.5, 0.0)
        expected = 10.0 * np.exp(0.5 * x)
        assert_allclose(result, expected, rtol=1e-10)

    def test_estimate_p0_exponential_growth_basic(self):
        """Test exponential growth parameter estimation."""
        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 2.0 * np.exp(0.3 * x) + 1.0
        p0 = estimate_p0_exponential_growth(x, y)
        assert len(p0) == 3
        assert all(np.isfinite(p0))

    def test_estimate_p0_exponential_growth_noisy(self):
        """Test exponential growth estimation with noise."""
        np.random.seed(456)
        x = np.linspace(0, 3, 50)
        y = 5.0 * np.exp(0.5 * x) + np.random.normal(0, 1, 50)
        p0 = estimate_p0_exponential_growth(x, y)
        assert len(p0) == 3
        assert all(np.isfinite(p0))
        assert p0[1] > 0  # growth rate should be positive


class TestNumericalEdgeCases:
    """Test numerical edge cases and stability."""

    def test_linear_with_inf(self):
        """Test linear function handles infinity gracefully."""
        x = np.array([1.0, 2.0, np.inf])
        result = linear(x, 1.0, 0.0)
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
        assert np.isinf(result[2])

    def test_exponential_decay_overflow_protection(self):
        """Test exponential decay doesn't overflow with large negative x."""
        x = np.array([-1000.0])  # Would cause exp(1000*b) overflow
        result = exponential_decay(x, 1.0, 1.0, 0.0)
        # Result should be inf, not error
        assert np.isinf(result[0])

    def test_exponential_decay_underflow_protection(self):
        """Test exponential decay handles underflow (returns asymptote)."""
        x = np.array([1000.0])  # exp(-1000) ≈ 0
        result = exponential_decay(x, 100.0, 1.0, 5.0)
        assert_allclose(result[0], 5.0, atol=1e-10)

    def test_estimate_p0_with_nan(self):
        """Test parameter estimation handles NaN gracefully."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, np.nan, 3.0])
        # Should not raise, but result may contain NaN
        p0 = estimate_p0_linear(x, y)
        assert len(p0) == 2

    def test_empty_array_handling(self):
        """Test functions handle empty arrays."""
        x = np.array([])
        result = linear(x, 1.0, 0.0)
        assert len(result) == 0

    def test_very_small_values(self):
        """Test functions with very small values."""
        x = np.array([1e-15, 2e-15, 3e-15])
        result = linear(x, 1e15, 0.0)
        expected = np.array([1.0, 2.0, 3.0])
        assert_allclose(result, expected, rtol=1e-10)


class TestFunctionAttributes:
    """Test that functions have correct attributes attached."""

    def test_linear_has_estimate_p0(self):
        """Test linear function has estimate_p0 attribute."""
        assert hasattr(linear, "estimate_p0")
        assert callable(linear.estimate_p0)

    def test_linear_has_bounds(self):
        """Test linear function has bounds attribute."""
        assert hasattr(linear, "bounds")
        assert callable(linear.bounds)

    def test_exponential_decay_has_estimate_p0(self):
        """Test exponential_decay has estimate_p0 attribute."""
        assert hasattr(exponential_decay, "estimate_p0")
        assert callable(exponential_decay.estimate_p0)

    def test_exponential_decay_has_bounds(self):
        """Test exponential_decay has bounds attribute."""
        assert hasattr(exponential_decay, "bounds")
        assert callable(exponential_decay.bounds)

    def test_exponential_growth_has_estimate_p0(self):
        """Test exponential_growth has estimate_p0 attribute."""
        assert hasattr(exponential_growth, "estimate_p0")
        assert callable(exponential_growth.estimate_p0)


class TestJITCompatibility:
    """Test JAX JIT compatibility of functions."""

    def test_linear_jit_compatible(self):
        """Test linear function is JIT-compilable."""
        from jax import jit

        jit_linear = jit(linear)
        x = jnp.array([1.0, 2.0, 3.0])
        result = jit_linear(x, 2.0, 1.0)
        expected = linear(x, 2.0, 1.0)
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-12)

    def test_exponential_decay_jit_compatible(self):
        """Test exponential_decay is JIT-compilable."""
        from jax import jit

        jit_exp = jit(exponential_decay)
        x = jnp.array([0.0, 1.0, 2.0])
        result = jit_exp(x, 10.0, 0.5, 2.0)
        expected = exponential_decay(x, 10.0, 0.5, 2.0)
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-10)

    def test_exponential_growth_jit_compatible(self):
        """Test exponential_growth is JIT-compilable."""
        from jax import jit

        jit_exp = jit(exponential_growth)
        x = jnp.array([0.0, 1.0, 2.0])
        result = jit_exp(x, 5.0, 0.3, 1.0)
        expected = exponential_growth(x, 5.0, 0.3, 1.0)
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-10)


class TestGradientCorrectness:
    """Test that JAX gradients are correct."""

    def test_linear_gradient(self):
        """Test gradient of linear function."""
        from jax import grad

        # Gradient w.r.t. 'a' (slope)
        def linear_a(a):
            return jnp.sum(linear(jnp.array([1.0, 2.0, 3.0]), a, 0.0))

        grad_a = grad(linear_a)(2.0)
        expected = 1.0 + 2.0 + 3.0  # sum of x
        assert_allclose(grad_a, expected, rtol=1e-10)

    def test_exponential_decay_gradient(self):
        """Test gradient of exponential decay is correct."""
        from jax import grad

        def exp_decay_sum(a, b, c):
            x = jnp.array([0.0, 1.0, 2.0])
            return jnp.sum(exponential_decay(x, a, b, c))

        # Gradient w.r.t. amplitude at x=0 should be 1 per point at x=0
        grad_a = grad(exp_decay_sum, argnums=0)(10.0, 0.5, 2.0)
        assert np.isfinite(grad_a)
        assert grad_a > 0  # Positive gradient (increasing a increases output)

    def test_exponential_growth_gradient(self):
        """Test gradient of exponential growth is correct."""
        from jax import grad

        def exp_growth_sum(a, b, c):
            x = jnp.array([0.0, 1.0, 2.0])
            return jnp.sum(exponential_growth(x, a, b, c))

        grad_a = grad(exp_growth_sum, argnums=0)(5.0, 0.3, 1.0)
        assert np.isfinite(grad_a)
        assert grad_a > 0


class TestVmapCompatibility:
    """Test JAX vmap compatibility."""

    def test_linear_vmap(self):
        """Test linear function works with vmap."""
        from jax import vmap

        # Batch over different slopes
        slopes = jnp.array([1.0, 2.0, 3.0])
        x = jnp.array([1.0, 2.0, 3.0])

        # vmap over first parameter (slope)
        batched_linear = vmap(lambda a: linear(x, a, 0.0))
        results = batched_linear(slopes)

        expected = jnp.array(
            [
                x * 1.0,
                x * 2.0,
                x * 3.0,
            ]
        )
        assert_allclose(np.asarray(results), np.asarray(expected), rtol=1e-12)

    def test_exponential_decay_vmap(self):
        """Test exponential_decay works with vmap over parameters."""
        from jax import vmap

        rates = jnp.array([0.1, 0.5, 1.0])
        x = jnp.array([0.0, 1.0, 2.0])

        # vmap over decay rate
        batched_exp = vmap(lambda b: exponential_decay(x, 10.0, b, 0.0))
        results = batched_exp(rates)

        assert results.shape == (3, 3)
        # At x=0, all should equal amplitude=10
        assert_allclose(np.asarray(results[:, 0]), 10.0, rtol=1e-10)
