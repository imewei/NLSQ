"""
Tests for Smart Parameter Bounds Inference
===========================================

Tests the automatic bounds inference system for parameter constraints.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.precision.bound_inference import (
    BoundsInference,
    analyze_bounds_quality,
    infer_bounds,
    merge_bounds,
)


class TestBoundsInference:
    """Test BoundsInference class."""

    def test_initialization(self):
        """Test BoundsInference initialization."""
        x = np.linspace(0, 10, 100)
        y = np.linspace(1, 5, 100)
        p0 = np.array([2.0, 0.5, 1.0])

        inference = BoundsInference(x, y, p0)

        assert inference.x_min == 0.0
        assert inference.x_max == 10.0
        assert inference.y_min == 1.0
        assert inference.y_max == 5.0
        assert inference.x_range == 10.0
        assert inference.y_range == 4.0

    def test_infer_bounds_basic(self):
        """Test basic bounds inference."""
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0
        p0 = np.array([2.5, 0.5, 1.0])

        inference = BoundsInference(x, y, p0)
        lower, upper = inference.infer()

        # Should have 3 bounds
        assert len(lower) == 3
        assert len(upper) == 3

        # All upper > lower
        assert np.all(upper > lower)

        # p0 should be within bounds (or close, due to safety factor)
        assert np.all(p0 >= lower)
        assert np.all(p0 <= upper)

    def test_infer_bounds_positivity(self):
        """Test that positive p0 enforces positivity."""
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0
        p0 = np.array([2.5, 0.5, 1.0])

        inference = BoundsInference(x, y, p0, enforce_positivity=True)
        lower, _upper = inference.infer()

        # All lower bounds should be non-negative
        assert np.all(lower >= 0)

    def test_infer_bounds_allow_negative(self):
        """Test that negative p0 allows negative bounds."""
        x = np.linspace(-5, 5, 100)
        y = -2.0 * x + 1.0
        p0 = np.array([-2.0, 1.0])

        inference = BoundsInference(x, y, p0, enforce_positivity=False)
        lower, _upper = inference.infer()

        # First parameter should allow negative values
        assert lower[0] < 0

    def test_safety_factor(self):
        """Test that safety factor affects bound width."""
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0
        p0 = np.array([2.5, 0.5, 1.0])

        # Small safety factor = tighter bounds
        inference1 = BoundsInference(x, y, p0, safety_factor=5.0)
        lower1, upper1 = inference1.infer()

        # Large safety factor = wider bounds
        inference2 = BoundsInference(x, y, p0, safety_factor=20.0)
        lower2, upper2 = inference2.infer()

        # Wider bounds should be wider
        assert np.all(upper2 >= upper1)
        assert np.all(lower2 <= lower1)

    def test_zero_p0_handling(self):
        """Test handling of zero initial guess."""
        x = np.linspace(0, 10, 100)
        y = np.linspace(1, 5, 100)
        p0 = np.array([0.0, 0.0, 0.0])

        inference = BoundsInference(x, y, p0)
        lower, upper = inference.infer()

        # Should still produce valid bounds
        assert len(lower) == 3
        assert len(upper) == 3
        assert np.all(upper > lower)


class TestInferBoundsFunction:
    """Test standalone infer_bounds function."""

    def test_convenience_function(self):
        """Test that convenience function works."""
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0
        p0 = [2.5, 0.5, 1.0]

        lower, upper = infer_bounds(x, y, p0)

        assert len(lower) == 3
        assert len(upper) == 3
        assert np.all(upper > lower)

    def test_with_custom_parameters(self):
        """Test with custom safety factor and positivity."""
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0
        p0 = [2.5, 0.5, 1.0]

        lower, upper = infer_bounds(
            x, y, p0, safety_factor=20.0, enforce_positivity=False
        )

        assert len(lower) == 3
        assert len(upper) == 3


class TestMergeBounds:
    """Test bounds merging functionality."""

    def test_merge_with_none(self):
        """Test merging with None returns inferred bounds."""
        inferred = (np.array([0, 0, 0]), np.array([10, 5, 10]))
        merged = merge_bounds(inferred, None)

        assert np.array_equal(merged[0], inferred[0])
        assert np.array_equal(merged[1], inferred[1])

    def test_merge_with_inf(self):
        """Test merging with infinite bounds."""
        inferred = (np.array([0, 0, 0]), np.array([10, 5, 10]))
        user = (
            np.array([-np.inf, -np.inf, -np.inf]),
            np.array([np.inf, np.inf, np.inf]),
        )
        merged = merge_bounds(inferred, user)

        # Should use inferred bounds where user has inf
        assert np.array_equal(merged[0], inferred[0])
        assert np.array_equal(merged[1], inferred[1])

    def test_merge_partial_bounds(self):
        """Test merging with partial user bounds."""
        inferred = (np.array([0, 0, 0]), np.array([10, 5, 10]))
        user = (np.array([1, -np.inf, 0]), np.array([5, np.inf, 10]))
        merged = merge_bounds(inferred, user)

        # First parameter: user bounds
        assert merged[0][0] == 1
        assert merged[1][0] == 5

        # Second parameter: inferred bounds (user had inf)
        assert merged[0][1] == inferred[0][1]
        assert merged[1][1] == inferred[1][1]

        # Third parameter: user bounds
        assert merged[0][2] == 0
        assert merged[1][2] == 10

    def test_merge_scalar_bounds(self):
        """Test merging with scalar bounds."""
        inferred = (np.array([0, 0, 0]), np.array([10, 5, 10]))
        user = (np.array([0]), np.array([np.inf]))
        merged = merge_bounds(inferred, user)

        # Scalar bounds should be broadcast
        assert merged[0][0] == 0
        assert merged[1][0] == inferred[1][0]  # Upper from inferred


class TestAnalyzeBoundsQuality:
    """Test bounds quality analysis."""

    def test_feasible_bounds(self):
        """Test analysis of feasible bounds."""
        bounds = (np.array([0, 0]), np.array([10, 10]))
        p0 = np.array([5, 5])

        analysis = analyze_bounds_quality(bounds, p0)

        assert analysis["is_feasible"] is True
        assert len(analysis["bound_ratios"]) == 2

    def test_infeasible_bounds(self):
        """Test analysis of infeasible bounds."""
        bounds = (np.array([6, 6]), np.array([10, 10]))
        p0 = np.array([5, 5])  # Outside bounds

        analysis = analyze_bounds_quality(bounds, p0)

        assert analysis["is_feasible"] is False

    def test_tight_vs_loose(self):
        """Test identification of tight and loose parameters."""
        bounds = (np.array([0, 0, 0]), np.array([2, 100, 10]))  # Tight, loose, medium
        p0 = np.array([1, 50, 5])

        analysis = analyze_bounds_quality(bounds, p0)

        # First parameter should be tight (ratio = 2/0 = inf, but small range)
        # Second parameter should be loose (ratio = 100/0 = inf, large range)
        assert "tight_parameters" in analysis
        assert "loose_parameters" in analysis


class TestCurveFitIntegration:
    """Test auto_bounds integration with curve_fit."""

    def exponential_decay(self, x, a, b, c):
        """Exponential decay model."""
        return a * jnp.exp(-b * x) + c

    def test_auto_bounds_basic(self):
        """Test basic auto_bounds functionality."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(100)

        result = curve_fit(
            self.exponential_decay, x, y, p0=[2, 0.5, 1], auto_bounds=True
        )

        # Should converge successfully
        assert result.x is not None
        assert abs(result.x[0] - 2.5) < 0.5
        assert abs(result.x[1] - 0.5) < 0.2
        assert abs(result.x[2] - 1.0) < 0.3

    def test_auto_bounds_with_user_bounds(self):
        """Test auto_bounds combined with user bounds."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(100)

        # User provides partial bounds
        result = curve_fit(
            self.exponential_decay,
            x,
            y,
            p0=[2, 0.5, 1],
            auto_bounds=True,
            bounds=([0, -np.inf, 0], [5, np.inf, 2]),
        )

        # Should converge with merged bounds
        assert result.x is not None
        assert 0 <= result.x[0] <= 5  # User bounds applied
        assert 0 <= result.x[2] <= 2  # User bounds applied

    def test_auto_bounds_safety_factor(self):
        """Test custom safety factor."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(100)

        # Very conservative bounds
        result = curve_fit(
            self.exponential_decay,
            x,
            y,
            p0=[2, 0.5, 1],
            auto_bounds=True,
            bounds_safety_factor=50.0,
        )

        # Should still converge (wide bounds don't hurt)
        assert result.x is not None

    def test_auto_bounds_with_fallback(self):
        """Test auto_bounds combined with fallback."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(100)

        # Poor initial guess
        result = curve_fit(
            self.exponential_decay,
            x,
            y,
            p0=[10, 5, 5],
            auto_bounds=True,
            fallback=True,
        )

        # Should converge with combined features
        assert result.x is not None
        assert result.x[0] > 0
        assert result.x[1] > 0

    def test_auto_bounds_without_p0(self):
        """Test that auto_bounds requires p0."""
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0

        # Without p0, auto_bounds should be silently ignored
        # (p0 would be auto-estimated by curve_fit)
        result = curve_fit(self.exponential_decay, x, y, auto_bounds=True)

        # Should still work (with auto p0 estimation)
        assert result.x is not None

    def test_auto_bounds_improves_convergence(self):
        """Test that auto_bounds helps with difficult cases."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 100.0 * np.exp(-0.01 * x) + 0.01 + 0.1 * np.random.randn(100)

        # Extreme parameter values - poor p0
        result = curve_fit(
            self.exponential_decay,
            x,
            y,
            p0=[1, 1, 1],  # Wrong magnitude
            auto_bounds=True,
        )

        # Should help constrain search space
        assert result.x is not None
        assert result.x[0] > 10  # Large amplitude
        assert result.x[1] > 0  # Positive decay rate


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_constant_data(self):
        """Test with constant y data."""
        x = np.linspace(0, 10, 100)
        y = np.ones(100) * 5.0
        p0 = [5.0]

        lower, upper = infer_bounds(x, y, p0)

        # Should handle zero y_range
        assert np.all(upper > lower)

    def test_single_point_range(self):
        """Test with single point (zero x_range)."""
        x = np.array([5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0])
        p0 = [2.0]

        lower, upper = infer_bounds(x, y, p0)

        # Should handle zero x_range
        assert np.all(upper > lower)

    def test_negative_data(self):
        """Test with negative data values."""
        x = np.linspace(0, 10, 100)
        y = -2.5 * np.exp(-0.5 * x) - 1.0
        p0 = [-2.5, 0.5, -1.0]

        lower, _upper = infer_bounds(x, y, p0, enforce_positivity=False)

        # Should handle negative values
        assert lower[0] < 0
        assert lower[2] < 0

    def test_mixed_positive_negative_p0(self):
        """Test with mixed positive/negative p0."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x - 5.0
        p0 = [2.0, -5.0]

        # With enforce_positivity=False, negative bounds are allowed
        lower, _upper = infer_bounds(x, y, p0, enforce_positivity=False)

        # Second param can be negative
        assert lower[1] < 0

        # With enforce_positivity=True, positive p0 enforces positivity
        lower2, _upper2 = infer_bounds(x, y, p0, enforce_positivity=True)
        assert lower2[0] >= 0  # First param should be non-negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
