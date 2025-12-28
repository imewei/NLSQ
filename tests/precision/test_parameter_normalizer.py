"""Tests for parameter normalization and model wrapper.

This module tests the ParameterNormalizer and NormalizedModelWrapper classes
which handle automatic parameter scaling to improve gradient signals and
convergence speed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.precision.parameter_normalizer import (
    NormalizedModelWrapper,
    ParameterNormalizer,
)


class TestParameterNormalizer:
    """Tests for ParameterNormalizer class."""

    def test_bounds_based_normalization_to_unit_interval(self):
        """Test bounds-based normalization maps parameters to [0, 1]."""
        # Setup: parameters in range [10, 100] and [0, 1]
        p0 = jnp.array([50.0, 0.5])
        bounds = (jnp.array([10.0, 0.0]), jnp.array([100.0, 1.0]))

        # Create normalizer with bounds strategy
        normalizer = ParameterNormalizer(p0=p0, bounds=bounds, strategy="bounds")

        # Normalize parameters
        normalized = normalizer.normalize(p0)

        # Expected: (50-10)/(100-10) = 40/90 ≈ 0.444, (0.5-0)/(1-0) = 0.5
        expected = jnp.array([40.0 / 90.0, 0.5])
        np.testing.assert_allclose(normalized, expected, rtol=1e-7)

        # Check bounds are normalized to [0, 1]
        normalized_lb = normalizer.normalize(bounds[0])
        normalized_ub = normalizer.normalize(bounds[1])
        np.testing.assert_allclose(normalized_lb, jnp.array([0.0, 0.0]), atol=1e-10)
        np.testing.assert_allclose(normalized_ub, jnp.array([1.0, 1.0]), atol=1e-10)

    def test_p0_based_normalization_scale_by_magnitude(self):
        """Test p0-based normalization scales by initial parameter magnitudes."""
        # Setup: parameters with different magnitudes
        p0 = jnp.array([1000.0, 1.0, 0.001])

        # Create normalizer with p0 strategy (no bounds)
        normalizer = ParameterNormalizer(p0=p0, bounds=None, strategy="p0")

        # Normalize parameters
        normalized = normalizer.normalize(p0)

        # Expected: p0 / |p0| = [1.0, 1.0, 1.0] (normalized to magnitude 1)
        expected = jnp.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(normalized, expected, rtol=1e-7)

        # Check different parameters are scaled proportionally
        params_test = jnp.array([2000.0, 2.0, 0.002])
        normalized_test = normalizer.normalize(params_test)
        expected_test = jnp.array([2.0, 2.0, 2.0])
        np.testing.assert_allclose(normalized_test, expected_test, rtol=1e-7)

    def test_normalize_denormalize_are_inverses(self):
        """Test that normalize() and denormalize() are exact inverses."""
        # Test with bounds-based normalization
        p0 = jnp.array([5.0, 15.0, 25.0])
        bounds = (jnp.array([0.0, 10.0, 20.0]), jnp.array([10.0, 20.0, 30.0]))
        normalizer = ParameterNormalizer(p0=p0, bounds=bounds, strategy="bounds")

        # Round-trip: normalize then denormalize
        normalized = normalizer.normalize(p0)
        denormalized = normalizer.denormalize(normalized)
        np.testing.assert_allclose(denormalized, p0, rtol=1e-10)

        # Test with p0-based normalization
        p0_nobounds = jnp.array([100.0, 0.1, 10.0])
        normalizer_p0 = ParameterNormalizer(p0=p0_nobounds, bounds=None, strategy="p0")

        normalized_p0 = normalizer_p0.normalize(p0_nobounds)
        denormalized_p0 = normalizer_p0.denormalize(normalized_p0)
        np.testing.assert_allclose(denormalized_p0, p0_nobounds, rtol=1e-10)

    def test_normalization_jacobian_analytical(self):
        """Test normalization Jacobian is computed analytically and correct."""
        # Bounds-based: Jacobian should be diag(ub - lb)
        p0 = jnp.array([5.0, 15.0])
        bounds = (jnp.array([0.0, 10.0]), jnp.array([10.0, 20.0]))
        normalizer = ParameterNormalizer(p0=p0, bounds=bounds, strategy="bounds")

        jacobian = normalizer.normalization_jacobian

        # Expected: diagonal matrix with (ub - lb) on diagonal
        expected_diag = jnp.array([10.0, 10.0])
        expected_jacobian = jnp.diag(expected_diag)
        np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-10)

        # p0-based: Jacobian should be diag(scale_factors)
        p0_nobounds = jnp.array([100.0, 0.01])
        normalizer_p0 = ParameterNormalizer(p0=p0_nobounds, bounds=None, strategy="p0")

        jacobian_p0 = normalizer_p0.normalization_jacobian

        # Expected: diagonal matrix with p0 magnitudes
        expected_diag_p0 = jnp.abs(p0_nobounds)
        expected_jacobian_p0 = jnp.diag(expected_diag_p0)
        np.testing.assert_allclose(jacobian_p0, expected_jacobian_p0, rtol=1e-10)

    def test_bounds_transformation_to_normalized_space(self):
        """Test bounds are correctly transformed to normalized space."""
        # Original bounds
        p0 = jnp.array([50.0, 0.5])
        bounds = (jnp.array([10.0, 0.0]), jnp.array([100.0, 1.0]))

        # Bounds-based normalization
        normalizer = ParameterNormalizer(p0=p0, bounds=bounds, strategy="bounds")

        lb_norm, ub_norm = normalizer.transform_bounds()

        # Bounds should map to [0, 1] for bounds-based
        np.testing.assert_allclose(lb_norm, jnp.array([0.0, 0.0]), atol=1e-10)
        np.testing.assert_allclose(ub_norm, jnp.array([1.0, 1.0]), atol=1e-10)

        # p0-based normalization
        p0_nobounds = jnp.array([10.0, 2.0])
        bounds_nobounds = (jnp.array([5.0, 1.0]), jnp.array([20.0, 4.0]))
        normalizer_p0 = ParameterNormalizer(
            p0=p0_nobounds, bounds=bounds_nobounds, strategy="p0"
        )

        lb_norm_p0, ub_norm_p0 = normalizer_p0.transform_bounds()

        # Bounds should be scaled by p0 magnitudes
        expected_lb = jnp.array([5.0 / 10.0, 1.0 / 2.0])  # [0.5, 0.5]
        expected_ub = jnp.array([20.0 / 10.0, 4.0 / 2.0])  # [2.0, 2.0]
        np.testing.assert_allclose(lb_norm_p0, expected_lb, rtol=1e-7)
        np.testing.assert_allclose(ub_norm_p0, expected_ub, rtol=1e-7)

    def test_normalization_disabled_identity_transform(self):
        """Test normalization='none' acts as identity transform."""
        # Create normalizer with 'none' strategy
        p0 = jnp.array([5.0, 15.0, 25.0])
        bounds = (jnp.array([0.0, 10.0, 20.0]), jnp.array([10.0, 20.0, 30.0]))
        normalizer = ParameterNormalizer(p0=p0, bounds=bounds, strategy="none")

        # Normalize and denormalize should return unchanged values
        normalized = normalizer.normalize(p0)
        np.testing.assert_allclose(normalized, p0, rtol=1e-10)

        denormalized = normalizer.denormalize(p0)
        np.testing.assert_allclose(denormalized, p0, rtol=1e-10)

        # Jacobian should be identity
        jacobian = normalizer.normalization_jacobian
        expected_jacobian = jnp.eye(len(p0))
        np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-10)

    def test_auto_strategy_selection(self):
        """Test 'auto' strategy selects bounds or p0 appropriately."""
        # With bounds: should use bounds-based
        p0 = jnp.array([50.0, 0.5])
        bounds = (jnp.array([10.0, 0.0]), jnp.array([100.0, 1.0]))
        normalizer_with_bounds = ParameterNormalizer(
            p0=p0, bounds=bounds, strategy="auto"
        )

        # Should normalize to [0, 1] like bounds-based
        normalized = normalizer_with_bounds.normalize(p0)
        expected = jnp.array([40.0 / 90.0, 0.5])
        np.testing.assert_allclose(normalized, expected, rtol=1e-7)

        # Without bounds: should use p0-based
        p0_nobounds = jnp.array([100.0, 1.0])
        normalizer_no_bounds = ParameterNormalizer(
            p0=p0_nobounds, bounds=None, strategy="auto"
        )

        # Should normalize to magnitude 1 like p0-based
        normalized_nobounds = normalizer_no_bounds.normalize(p0_nobounds)
        expected_nobounds = jnp.array([1.0, 1.0])
        np.testing.assert_allclose(normalized_nobounds, expected_nobounds, rtol=1e-7)

    def test_zero_parameter_handling(self):
        """Test normalizer handles zero parameters gracefully."""
        # p0 with zero value
        p0 = jnp.array([10.0, 0.0, 5.0])

        # p0-based normalization should add epsilon to prevent division by zero
        normalizer = ParameterNormalizer(p0=p0, bounds=None, strategy="p0")

        # Should not raise error
        normalized = normalizer.normalize(p0)
        assert jnp.all(jnp.isfinite(normalized))

        # Zero parameter should be handled with small epsilon
        # The second element should be normalized to 0.0 (0 / epsilon ≈ 0)
        assert normalized[1] == 0.0


class TestNormalizedModelWrapper:
    """Tests for NormalizedModelWrapper class."""

    def test_wrapper_preserves_function_output(self):
        """Test wrapped model produces identical output to original."""

        # Define simple model
        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        # Create normalizer
        p0 = jnp.array([2.0, 0.5, 1.0])
        bounds = (jnp.array([0.0, 0.0, 0.0]), jnp.array([10.0, 2.0, 5.0]))
        normalizer = ParameterNormalizer(p0=p0, bounds=bounds, strategy="bounds")

        # Wrap model
        wrapped_model = NormalizedModelWrapper(model, normalizer)

        # Test data
        x = jnp.linspace(0, 5, 10)

        # Compute output with original model and original parameters
        original_output = model(x, 2.0, 0.5, 1.0)

        # Compute output with wrapped model and normalized parameters
        normalized_params = normalizer.normalize(p0)
        wrapped_output = wrapped_model(x, *normalized_params)

        # Outputs should be identical
        np.testing.assert_allclose(wrapped_output, original_output, rtol=1e-7)

    def test_jit_compatibility_of_wrapped_model(self):
        """Test wrapped model is JIT-compatible."""

        # Define model
        def model(x, a, b):
            return a * x + b

        # Create normalizer
        p0 = jnp.array([5.0, 10.0])
        normalizer = ParameterNormalizer(p0=p0, bounds=None, strategy="p0")

        # Wrap model
        wrapped_model = NormalizedModelWrapper(model, normalizer)

        # JIT compile wrapped model
        @jax.jit
        def jitted_wrapped_model(x, a_norm, b_norm):
            return wrapped_model(x, a_norm, b_norm)

        # Test data
        x = jnp.array([1.0, 2.0, 3.0])
        normalized_params = normalizer.normalize(p0)

        # Should compile and run without error
        output = jitted_wrapped_model(x, *normalized_params)

        # Verify output is correct
        expected = model(x, 5.0, 10.0)
        np.testing.assert_allclose(output, expected, rtol=1e-7)

        # Test that recompilation works with different shapes (static shapes)
        x2 = jnp.array([1.0, 2.0, 3.0, 4.0])
        output2 = jitted_wrapped_model(x2, *normalized_params)
        expected2 = model(x2, 5.0, 10.0)
        np.testing.assert_allclose(output2, expected2, rtol=1e-7)

    def test_wrapper_with_vectorized_operations(self):
        """Test wrapper works correctly with vmap and batch operations."""

        # Define model
        def model(x, a, b):
            return a * jnp.sin(b * x)

        # Create normalizer
        p0 = jnp.array([3.0, 2.0])
        normalizer = ParameterNormalizer(p0=p0, bounds=None, strategy="p0")

        # Wrap model
        wrapped_model = NormalizedModelWrapper(model, normalizer)

        # Test with vmap over x
        x_batch = jnp.array([[1.0], [2.0], [3.0]])
        normalized_params = normalizer.normalize(p0)

        # Vmap over first dimension of x
        vmapped_model = jax.vmap(
            lambda x_single: wrapped_model(x_single, *normalized_params)
        )

        output = vmapped_model(x_batch.squeeze())

        # Verify output
        expected = model(x_batch.squeeze(), 3.0, 2.0)
        np.testing.assert_allclose(output, expected, rtol=1e-7)

    def test_wrapper_gradient_computation(self):
        """Test gradients can be computed through wrapped model."""

        # Define model
        def model(x, a, b):
            return a * x**2 + b * x

        # Create normalizer
        p0 = jnp.array([2.0, 3.0])
        normalizer = ParameterNormalizer(p0=p0, bounds=None, strategy="p0")

        # Wrap model
        wrapped_model = NormalizedModelWrapper(model, normalizer)

        # Define loss function in normalized space
        def loss_fn(params_norm):
            x = jnp.array([1.0, 2.0, 3.0])
            y_true = jnp.array([5.0, 14.0, 27.0])  # a=2, b=3
            y_pred = wrapped_model(x, *params_norm)
            return jnp.sum((y_pred - y_true) ** 2)

        # Compute gradient
        normalized_params = normalizer.normalize(p0)
        grad = jax.grad(loss_fn)(normalized_params)

        # Gradient should be finite and non-zero (model is trainable)
        assert jnp.all(jnp.isfinite(grad))

        # At optimal parameters, gradient should be near zero
        # (though not exactly due to normalization)
        assert jnp.linalg.norm(grad) < 1e-10
