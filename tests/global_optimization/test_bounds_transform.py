"""Tests for sigmoid bound transformation functions.

These tests verify the correctness of the sigmoid/logit transformations
used for bounded optimization with CMA-ES.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization.bounds_transform import (
    compute_default_popsize,
    transform_from_bounds,
    transform_to_bounds,
)


class TestTransformToBounds:
    """Tests for transform_to_bounds function."""

    def test_zero_maps_to_midpoint(self) -> None:
        """Test that x=0 maps to the midpoint of bounds."""
        x = jnp.array([0.0, 0.0, 0.0])
        lb = jnp.array([0.0, -10.0, 100.0])
        ub = jnp.array([10.0, 10.0, 200.0])

        result = transform_to_bounds(x, lb, ub)

        # sigmoid(0) = 0.5, so x_bounded = lb + 0.5 * (ub - lb)
        expected = jnp.array([5.0, 0.0, 150.0])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_large_positive_maps_near_upper(self) -> None:
        """Test that large positive values map near upper bound."""
        x = jnp.array([10.0])
        lb = jnp.array([0.0])
        ub = jnp.array([1.0])

        result = transform_to_bounds(x, lb, ub)

        # sigmoid(10) ≈ 0.99995
        assert float(result[0]) > 0.999

    def test_large_negative_maps_near_lower(self) -> None:
        """Test that large negative values map near lower bound."""
        x = jnp.array([-10.0])
        lb = jnp.array([0.0])
        ub = jnp.array([1.0])

        result = transform_to_bounds(x, lb, ub)

        # sigmoid(-10) ≈ 0.00005
        assert float(result[0]) < 0.001

    def test_respects_bounds(self) -> None:
        """Test that output is always within bounds."""
        # Test with extreme values
        x = jnp.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        lb = jnp.full(5, 0.0)
        ub = jnp.full(5, 1.0)

        result = transform_to_bounds(x, lb, ub)

        assert jnp.all(result >= lb).item()
        assert jnp.all(result <= ub).item()

    def test_batched_input(self) -> None:
        """Test with batched (population) input."""
        # Shape: (3 individuals, 2 parameters)
        x = jnp.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]])
        lb = jnp.array([0.0, 10.0])
        ub = jnp.array([10.0, 20.0])

        result = transform_to_bounds(x, lb, ub)

        assert result.shape == (3, 2)
        # All values should be within bounds
        assert jnp.all(result >= lb).item()
        assert jnp.all(result <= ub).item()

    def test_asymmetric_bounds(self) -> None:
        """Test with asymmetric bounds (e.g., multi-scale parameters)."""
        x = jnp.array([0.0, 0.0])
        lb = jnp.array([1e-6, 1e3])  # Very different scales
        ub = jnp.array([1e-3, 1e6])

        result = transform_to_bounds(x, lb, ub)

        # Midpoints of each range
        expected_0 = 1e-6 + 0.5 * (1e-3 - 1e-6)
        expected_1 = 1e3 + 0.5 * (1e6 - 1e3)
        np.testing.assert_allclose(result[0], expected_0, rtol=1e-5)
        np.testing.assert_allclose(result[1], expected_1, rtol=1e-5)


class TestTransformFromBounds:
    """Tests for transform_from_bounds function."""

    def test_midpoint_maps_to_zero(self) -> None:
        """Test that midpoint maps to x=0."""
        lb = jnp.array([0.0, -10.0, 100.0])
        ub = jnp.array([10.0, 10.0, 200.0])
        x = (lb + ub) / 2  # Midpoints

        result = transform_from_bounds(x, lb, ub)

        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_near_upper_maps_to_large_positive(self) -> None:
        """Test that values near upper bound map to large positive."""
        lb = jnp.array([0.0])
        ub = jnp.array([1.0])
        x = jnp.array([0.99])

        result = transform_from_bounds(x, lb, ub)

        assert float(result[0]) > 4.0  # logit(0.99) ≈ 4.6

    def test_near_lower_maps_to_large_negative(self) -> None:
        """Test that values near lower bound map to large negative."""
        lb = jnp.array([0.0])
        ub = jnp.array([1.0])
        x = jnp.array([0.01])

        result = transform_from_bounds(x, lb, ub)

        assert float(result[0]) < -4.0  # logit(0.01) ≈ -4.6

    def test_roundtrip_accuracy(self) -> None:
        """Test that transform_from_bounds(transform_to_bounds(x)) ≈ x."""
        # Test roundtrip in both directions
        lb = jnp.array([0.0, 1.0, -100.0])
        ub = jnp.array([10.0, 100.0, 100.0])

        # Forward then inverse
        x_unbounded = jnp.array([-2.0, 0.0, 2.0])
        x_bounded = transform_to_bounds(x_unbounded, lb, ub)
        x_recovered = transform_from_bounds(x_bounded, lb, ub)
        np.testing.assert_allclose(x_recovered, x_unbounded, rtol=1e-4)

        # Inverse then forward
        x_bounded2 = jnp.array([5.0, 50.0, 0.0])
        x_unbounded2 = transform_from_bounds(x_bounded2, lb, ub)
        x_recovered2 = transform_to_bounds(x_unbounded2, lb, ub)
        np.testing.assert_allclose(x_recovered2, x_bounded2, rtol=1e-4)

    def test_handles_edge_values(self) -> None:
        """Test that edge values are handled without NaN/Inf."""
        lb = jnp.array([0.0])
        ub = jnp.array([1.0])

        # Test at bounds (should be clamped by epsilon)
        x_at_lb = jnp.array([0.0])
        x_at_ub = jnp.array([1.0])

        result_lb = transform_from_bounds(x_at_lb, lb, ub)
        result_ub = transform_from_bounds(x_at_ub, lb, ub)

        # Should not be NaN or Inf
        assert jnp.isfinite(result_lb).all().item()
        assert jnp.isfinite(result_ub).all().item()

    def test_batched_roundtrip(self) -> None:
        """Test roundtrip with batched input."""
        lb = jnp.array([0.0, 10.0])
        ub = jnp.array([10.0, 20.0])

        # Population of unbounded values
        x_unbounded = jnp.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])

        x_bounded = transform_to_bounds(x_unbounded, lb, ub)
        x_recovered = transform_from_bounds(x_bounded, lb, ub)

        np.testing.assert_allclose(x_recovered, x_unbounded, rtol=1e-4)


class TestComputeDefaultPopsize:
    """Tests for compute_default_popsize function."""

    def test_minimum_popsize(self) -> None:
        """Test that minimum population size is 4."""
        assert compute_default_popsize(1) >= 4
        assert compute_default_popsize(2) >= 4

    def test_increases_with_dimensions(self) -> None:
        """Test that popsize increases with number of parameters."""
        popsize_5 = compute_default_popsize(5)
        popsize_10 = compute_default_popsize(10)
        popsize_20 = compute_default_popsize(20)

        assert popsize_5 < popsize_10 < popsize_20

    def test_known_values(self) -> None:
        """Test against known formula: int(4 + 3 * log(n))."""
        import math

        for n in [5, 10, 20, 50, 100]:
            expected = int(4 + 3 * math.log(n))
            assert compute_default_popsize(n) == expected

    def test_zero_params_handled(self) -> None:
        """Test that n_params=0 doesn't cause errors."""
        # Should handle gracefully (minimum 4)
        result = compute_default_popsize(0)
        assert result >= 4


class TestNumericalStability:
    """Tests for numerical stability of transformations."""

    def test_extreme_bounds(self) -> None:
        """Test with extreme bound values (multi-scale parameters)."""
        lb = jnp.array([1e-10, 1e6])
        ub = jnp.array([1e-6, 1e10])

        x = jnp.array([0.0, 0.0])  # Should map to midpoint
        result = transform_to_bounds(x, lb, ub)

        assert jnp.isfinite(result).all().item()
        assert jnp.all(result > lb).item()
        assert jnp.all(result < ub).item()

    def test_no_nan_with_large_input(self) -> None:
        """Test that large inputs don't produce NaN."""
        lb = jnp.array([0.0])
        ub = jnp.array([1.0])

        # Very large positive and negative values
        x = jnp.array([1000.0, -1000.0])

        result = transform_to_bounds(x, lb, ub)

        assert jnp.isfinite(result).all().item()
        assert float(result[0]) > 0.999999  # Near upper bound
        assert float(result[1]) < 0.000001  # Near lower bound

    @pytest.mark.parametrize("n_params", [1, 5, 10, 50, 100])
    def test_roundtrip_various_dimensions(self, n_params: int) -> None:
        """Test roundtrip accuracy for various parameter counts."""
        key = jnp.array([0, n_params], dtype=jnp.uint32)  # Simple seed

        lb = jnp.ones(n_params) * 0.0
        ub = jnp.ones(n_params) * 10.0

        # Random unbounded values
        x_unbounded = jnp.linspace(-3.0, 3.0, n_params)

        x_bounded = transform_to_bounds(x_unbounded, lb, ub)
        x_recovered = transform_from_bounds(x_bounded, lb, ub)

        np.testing.assert_allclose(x_recovered, x_unbounded, rtol=1e-4)
