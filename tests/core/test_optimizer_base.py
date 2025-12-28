"""Comprehensive tests for nlsq.optimizer_base module.

This test suite covers:
- OptimizerBase abstract base class
- TrustRegionOptimizerBase class
- Counter management (nfev, njev)
- Trust region radius updates
- Convergence checking
- Result creation
"""

import unittest

import numpy as np
import pytest

from nlsq.core._optimize import OptimizeResult
from nlsq.core.optimizer_base import OptimizerBase, TrustRegionOptimizerBase


# Concrete implementation for testing OptimizerBase
class ConcreteOptimizer(OptimizerBase):
    """Concrete implementation of OptimizerBase for testing."""

    def optimize(self, fun, x0, jac=None, bounds=(-np.inf, np.inf), **kwargs):
        """Simple implementation that just returns initial guess."""
        self.increment_nfev()
        if jac is not None:
            self.increment_njev()

        result = self.create_result(
            x=x0,
            fun=fun(x0),
            success=True,
            status=0,
            message="Test optimization completed",
        )
        return result


# Concrete implementation for testing TrustRegionOptimizerBase
class ConcreteTrustRegionOptimizer(TrustRegionOptimizerBase):
    """Concrete implementation of TrustRegionOptimizerBase for testing."""

    def optimize(self, fun, x0, jac=None, bounds=(-np.inf, np.inf), **kwargs):
        """Simple implementation that uses trust region."""
        self.increment_nfev()

        result = self.create_result(
            x=x0,
            fun=fun(x0),
            success=True,
            status=0,
            message="Trust region optimization completed",
            trust_radius=self.trust_radius,
        )
        return result


class TestOptimizerBaseInitialization(unittest.TestCase):
    """Tests for OptimizerBase initialization."""

    def test_default_initialization(self):
        """Test initialization with default name."""
        optimizer = ConcreteOptimizer()

        self.assertEqual(optimizer.name, "optimizer")
        self.assertEqual(optimizer.nfev, 0)
        self.assertEqual(optimizer.njev, 0)
        self.assertIsNotNone(optimizer.logger)

    def test_custom_name_initialization(self):
        """Test initialization with custom name."""
        optimizer = ConcreteOptimizer(name="custom_optimizer")

        self.assertEqual(optimizer.name, "custom_optimizer")
        self.assertEqual(optimizer.nfev, 0)
        self.assertEqual(optimizer.njev, 0)

    def test_is_abstract_base_class(self):
        """Test that OptimizerBase is an abstract base class."""
        # Cannot instantiate OptimizerBase directly
        with self.assertRaises(TypeError):
            OptimizerBase()

    def test_logger_created(self):
        """Test that logger is properly created."""
        optimizer = ConcreteOptimizer(name="test_opt")

        self.assertIsNotNone(optimizer.logger)
        # Logger name should include the optimizer name
        self.assertIn("test_opt", optimizer.logger.name)


class TestOptimizerBaseCounters(unittest.TestCase):
    """Tests for counter management in OptimizerBase."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = ConcreteOptimizer(name="test")

    def test_initial_counters_zero(self):
        """Test that counters start at zero."""
        self.assertEqual(self.optimizer.nfev, 0)
        self.assertEqual(self.optimizer.njev, 0)

    def test_increment_nfev(self):
        """Test incrementing function evaluation counter."""
        self.optimizer.increment_nfev()
        self.assertEqual(self.optimizer.nfev, 1)

        self.optimizer.increment_nfev()
        self.assertEqual(self.optimizer.nfev, 2)

    def test_increment_njev(self):
        """Test incrementing Jacobian evaluation counter."""
        self.optimizer.increment_njev()
        self.assertEqual(self.optimizer.njev, 1)

        self.optimizer.increment_njev()
        self.assertEqual(self.optimizer.njev, 2)

    def test_increment_nfev_multiple(self):
        """Test incrementing nfev by custom amount."""
        self.optimizer.increment_nfev(5)
        self.assertEqual(self.optimizer.nfev, 5)

        self.optimizer.increment_nfev(3)
        self.assertEqual(self.optimizer.nfev, 8)

    def test_increment_njev_multiple(self):
        """Test incrementing njev by custom amount."""
        self.optimizer.increment_njev(4)
        self.assertEqual(self.optimizer.njev, 4)

        self.optimizer.increment_njev(2)
        self.assertEqual(self.optimizer.njev, 6)

    def test_reset_counters(self):
        """Test resetting counters to zero."""
        self.optimizer.increment_nfev(10)
        self.optimizer.increment_njev(5)

        self.assertEqual(self.optimizer.nfev, 10)
        self.assertEqual(self.optimizer.njev, 5)

        self.optimizer.reset_counters()

        self.assertEqual(self.optimizer.nfev, 0)
        self.assertEqual(self.optimizer.njev, 0)

    def test_counters_independent(self):
        """Test that nfev and njev are independent."""
        self.optimizer.increment_nfev(3)
        self.assertEqual(self.optimizer.nfev, 3)
        self.assertEqual(self.optimizer.njev, 0)

        self.optimizer.increment_njev(5)
        self.assertEqual(self.optimizer.nfev, 3)
        self.assertEqual(self.optimizer.njev, 5)


class TestOptimizerBaseCreateResult(unittest.TestCase):
    """Tests for create_result method."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = ConcreteOptimizer(name="test")

    def test_create_result_basic(self):
        """Test creating a basic OptimizeResult."""
        x = np.array([1.0, 2.0, 3.0])
        fun = np.array([0.1, -0.2, 0.15])

        # Increment counters to test that they're included in result
        self.optimizer.increment_nfev(10)

        result = self.optimizer.create_result(
            x=x, success=True, status=1, message="Success", fun=fun
        )

        self.assertIsInstance(result, OptimizeResult)
        np.testing.assert_array_equal(result.x, x)
        self.assertTrue(result.success)
        self.assertEqual(result.status, 1)
        self.assertEqual(result.message, "Success")
        np.testing.assert_array_equal(result.fun, fun)
        self.assertEqual(result.nfev, 10)

    def test_create_result_with_optional_fields(self):
        """Test creating result with optional fields."""
        x = np.array([1.0, 2.0])

        # Increment counters to test that they're included in result
        self.optimizer.increment_nfev(25)
        self.optimizer.increment_njev(25)

        result = self.optimizer.create_result(
            x=x,
            success=True,
            status=2,
            message="Converged",
            fun=np.array([0.01]),
            nit=10,
            cost=0.0001,
            optimality=1e-8,
            jac=np.random.rand(1, 2),
        )

        self.assertEqual(result.nfev, 25)
        self.assertEqual(result.njev, 25)
        self.assertEqual(result.nit, 10)
        self.assertAlmostEqual(result.cost, 0.0001)
        self.assertAlmostEqual(result.optimality, 1e-8)
        self.assertIsNotNone(result.jac)

    def test_create_result_minimal(self):
        """Test creating result with minimal required fields."""
        x = np.array([1.0])
        fun = np.array([0.5])

        result = self.optimizer.create_result(x=x, fun=fun)

        np.testing.assert_array_equal(result.x, x)
        np.testing.assert_array_equal(result.fun, fun)
        self.assertIn("x", result)
        self.assertIn("fun", result)

    def test_create_result_with_kwargs(self):
        """Test creating result with additional kwargs."""
        x = np.array([1.0, 2.0])
        fun = np.array([1.0])

        result = self.optimizer.create_result(
            x=x,
            fun=fun,
            success=True,
            custom_field="custom_value",
            another_field=42,
        )

        self.assertEqual(result.custom_field, "custom_value")
        self.assertEqual(result.another_field, 42)


class TestOptimizerBaseCheckConvergence(unittest.TestCase):
    """Tests for check_convergence method."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = ConcreteOptimizer(name="test")

    def test_ftol_convergence(self):
        """Test convergence due to ftol criterion."""
        # Small actual_reduction relative to cost
        # actual_reduction < ftol * cost => 1e-10 < 1e-8 * 1.0 => True
        status = self.optimizer.check_convergence(
            actual_reduction=1e-10,
            cost=1.0,
            step_norm=1.0,
            x_norm=1.0,
            ratio=1.0,
            ftol=1e-8,
            xtol=1e-8,
        )

        self.assertEqual(status, 2)  # ftol convergence

    def test_xtol_convergence(self):
        """Test convergence due to xtol criterion."""
        # Small step_norm relative to (xtol + x_norm)
        # step_norm < xtol * (xtol + x_norm) => 1e-10 < 1e-8 * (1e-8 + 1.0) => True
        status = self.optimizer.check_convergence(
            actual_reduction=1.0,
            cost=1.0,
            step_norm=1e-10,
            x_norm=1.0,
            ratio=1.0,
            ftol=1e-8,
            xtol=1e-8,
        )

        self.assertEqual(status, 3)  # xtol convergence

    def test_both_tolerances_convergence(self):
        """Test convergence when both tolerances are satisfied."""
        # Both ftol and xtol criteria satisfied
        # actual_reduction < ftol * cost AND step_norm < xtol * (xtol + x_norm)
        status = self.optimizer.check_convergence(
            actual_reduction=1e-10,
            cost=1.0,
            step_norm=1e-10,
            x_norm=1.0,
            ratio=1.0,
            ftol=1e-8,
            xtol=1e-8,
        )

        # Should return 2 (ftol) since it's checked first, unless both are needed for status 4
        # Looking at the code: status 4 requires BOTH conditions in an AND check
        # But status 2 and 3 are checked first, so this should return 2
        self.assertEqual(status, 2)

    def test_no_convergence(self):
        """Test when no convergence criteria are met."""
        status = self.optimizer.check_convergence(
            actual_reduction=1.0,
            cost=1.0,
            step_norm=1.0,
            x_norm=1.0,
            ratio=1.0,
            ftol=1e-8,
            xtol=1e-8,
        )

        self.assertIsNone(status)

    def test_edge_case_zero_actual_reduction(self):
        """Test edge case with zero actual_reduction."""
        # actual_reduction=0 < ftol * cost => 0 < 1e-8 * 1.0 => True
        status = self.optimizer.check_convergence(
            actual_reduction=0.0,
            cost=1.0,
            step_norm=1.0,
            x_norm=1.0,
            ratio=0.0,
            ftol=1e-8,
            xtol=1e-8,
        )

        # Should trigger ftol convergence
        self.assertEqual(status, 2)

    def test_edge_case_zero_step_norm(self):
        """Test edge case with zero step_norm."""
        # step_norm=0 < xtol * (xtol + x_norm) => 0 < 1e-8 * (1e-8 + 1.0) => True
        status = self.optimizer.check_convergence(
            actual_reduction=1.0,
            cost=1.0,
            step_norm=0.0,
            x_norm=1.0,
            ratio=1.0,
            ftol=1e-8,
            xtol=1e-8,
        )

        # Should trigger xtol convergence (status 3) only if ftol not satisfied
        # But ftol check first: actual_reduction < ftol * cost => 1.0 < 1e-8 * 1.0 => False
        # So should return status 3
        self.assertEqual(status, 3)


class TestOptimizerBaseOptimize(unittest.TestCase):
    """Tests for optimize method integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = ConcreteOptimizer(name="test")

    def test_optimize_increments_nfev(self):
        """Test that optimize increments nfev."""

        def fun(x):
            return np.sum(x**2)

        x0 = np.array([1.0, 2.0])

        self.optimizer.optimize(fun, x0)

        self.assertEqual(self.optimizer.nfev, 1)

    def test_optimize_with_jac_increments_njev(self):
        """Test that optimize with jac increments njev."""

        def fun(x):
            return np.sum(x**2)

        def jac(x):
            return 2 * x

        x0 = np.array([1.0, 2.0])

        result = self.optimizer.optimize(fun, x0, jac=jac)

        self.assertEqual(self.optimizer.nfev, 1)
        self.assertEqual(self.optimizer.njev, 1)

    def test_optimize_returns_result(self):
        """Test that optimize returns OptimizeResult."""

        def fun(x):
            return np.sum(x**2)

        x0 = np.array([1.0, 2.0, 3.0])

        result = self.optimizer.optimize(fun, x0)

        self.assertIsInstance(result, OptimizeResult)
        self.assertTrue(result.success)
        np.testing.assert_array_equal(result.x, x0)

    def test_optimize_multiple_calls(self):
        """Test multiple optimize calls."""

        def fun(x):
            return np.sum(x**2)

        x0 = np.array([1.0])

        self.optimizer.optimize(fun, x0)
        self.assertEqual(self.optimizer.nfev, 1)

        self.optimizer.optimize(fun, x0)
        self.assertEqual(self.optimizer.nfev, 2)


class TestTrustRegionOptimizerBaseInitialization(unittest.TestCase):
    """Tests for TrustRegionOptimizerBase initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        optimizer = ConcreteTrustRegionOptimizer()

        self.assertEqual(optimizer.name, "trust_region")
        self.assertEqual(optimizer.trust_radius, 1.0)
        self.assertEqual(optimizer.nfev, 0)

    def test_custom_name_initialization(self):
        """Test initialization with custom name."""
        optimizer = ConcreteTrustRegionOptimizer(name="trf_optimizer")

        self.assertEqual(optimizer.name, "trf_optimizer")
        self.assertEqual(optimizer.trust_radius, 1.0)

    def test_inherits_from_optimizer_base(self):
        """Test that TrustRegionOptimizerBase inherits from OptimizerBase."""
        optimizer = ConcreteTrustRegionOptimizer()

        self.assertIsInstance(optimizer, OptimizerBase)
        self.assertIsInstance(optimizer, TrustRegionOptimizerBase)

    def test_has_counter_methods(self):
        """Test that inherited counter methods work."""
        optimizer = ConcreteTrustRegionOptimizer()

        optimizer.increment_nfev()
        self.assertEqual(optimizer.nfev, 1)

        optimizer.reset_counters()
        self.assertEqual(optimizer.nfev, 0)


class TestTrustRegionRadius(unittest.TestCase):
    """Tests for trust region radius management."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = ConcreteTrustRegionOptimizer()

    def test_initial_trust_radius(self):
        """Test initial trust radius value."""
        self.assertEqual(self.optimizer.trust_radius, 1.0)

    def test_set_trust_radius(self):
        """Test setting trust radius via property."""
        self.optimizer.trust_radius = 2.5

        self.assertEqual(self.optimizer.trust_radius, 2.5)

    def test_trust_radius_positive(self):
        """Test that trust radius can be any positive value."""
        self.optimizer.trust_radius = 0.1
        self.assertEqual(self.optimizer.trust_radius, 0.1)

        self.optimizer.trust_radius = 100.0
        self.assertEqual(self.optimizer.trust_radius, 100.0)

    def test_trust_radius_very_small(self):
        """Test trust radius with very small values."""
        self.optimizer.trust_radius = 1e-10

        self.assertAlmostEqual(self.optimizer.trust_radius, 1e-10)


class TestUpdateTrustRadius(unittest.TestCase):
    """Tests for update_trust_radius method."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = ConcreteTrustRegionOptimizer()

    def test_increase_radius_good_step(self):
        """Test that radius increases for very good steps."""
        Delta = 1.0
        actual_reduction = 0.9
        predicted_reduction = 0.8
        step_norm = 0.95  # Close to boundary

        new_radius, ratio = self.optimizer.update_trust_radius(
            Delta,
            actual_reduction,
            predicted_reduction,
            step_norm,
            step_at_boundary=True,
        )

        # Good step at boundary should increase radius
        self.assertGreater(new_radius, Delta)
        self.assertAlmostEqual(ratio, actual_reduction / predicted_reduction)

    def test_decrease_radius_bad_step(self):
        """Test that radius decreases for bad steps."""
        Delta = 1.0
        actual_reduction = 0.1
        predicted_reduction = 0.8
        step_norm = 0.5

        new_radius, _ratio = self.optimizer.update_trust_radius(
            Delta,
            actual_reduction,
            predicted_reduction,
            step_norm,
            step_at_boundary=False,
        )

        # Bad step should decrease radius
        self.assertLess(new_radius, Delta)

    def test_maintain_radius_acceptable_step(self):
        """Test that radius is maintained for acceptable steps."""
        Delta = 1.0
        actual_reduction = 0.6
        predicted_reduction = 0.8
        step_norm = 0.5

        new_radius, _ratio = self.optimizer.update_trust_radius(
            Delta,
            actual_reduction,
            predicted_reduction,
            step_norm,
            step_at_boundary=False,
        )

        # Acceptable step away from boundary should maintain radius
        self.assertAlmostEqual(new_radius, Delta, places=5)

    def test_ratio_calculation(self):
        """Test that ratio is calculated correctly."""
        actual_reduction = 0.75
        predicted_reduction = 1.0

        _, ratio = self.optimizer.update_trust_radius(
            Delta=1.0,
            actual_reduction=actual_reduction,
            predicted_reduction=predicted_reduction,
            step_norm=0.5,
            step_at_boundary=False,
        )

        self.assertAlmostEqual(ratio, 0.75)

    def test_step_at_boundary_flag(self):
        """Test behavior with step_at_boundary flag."""
        # Good step at boundary
        new_radius_at_boundary, _ = self.optimizer.update_trust_radius(
            Delta=1.0,
            actual_reduction=0.8,
            predicted_reduction=0.8,
            step_norm=0.99,
            step_at_boundary=True,
        )

        # Same good step not at boundary
        new_radius_not_at_boundary, _ = self.optimizer.update_trust_radius(
            Delta=1.0,
            actual_reduction=0.8,
            predicted_reduction=0.8,
            step_norm=0.5,
            step_at_boundary=False,
        )

        # Radius at boundary should be larger
        self.assertGreaterEqual(new_radius_at_boundary, new_radius_not_at_boundary)

    def test_zero_predicted_reduction(self):
        """Test handling of zero predicted reduction."""
        # This should handle division by zero gracefully
        new_radius, ratio = self.optimizer.update_trust_radius(
            Delta=1.0,
            actual_reduction=0.1,
            predicted_reduction=0.0,
            step_norm=0.5,
            step_at_boundary=False,
        )

        # Should return valid values even with zero predicted reduction
        self.assertIsInstance(new_radius, (float, np.floating))
        self.assertIsInstance(ratio, (float, np.floating))


class TestStepAccepted(unittest.TestCase):
    """Tests for step_accepted method."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = ConcreteTrustRegionOptimizer()

    def test_accept_good_step(self):
        """Test that good steps are accepted."""
        ratio = 0.75

        accepted = self.optimizer.step_accepted(ratio)

        self.assertTrue(accepted)

    def test_reject_bad_step(self):
        """Test that bad steps are rejected."""
        ratio = -0.5

        accepted = self.optimizer.step_accepted(ratio)

        self.assertFalse(accepted)

    def test_threshold_boundary(self):
        """Test behavior at acceptance threshold."""
        # Just above threshold
        accepted_above = self.optimizer.step_accepted(1e-4 + 1e-5)
        self.assertTrue(accepted_above)

        # Just below threshold
        accepted_below = self.optimizer.step_accepted(1e-4 - 1e-5)
        self.assertFalse(accepted_below)

    def test_custom_threshold(self):
        """Test with custom acceptance threshold."""
        ratio = 0.05

        # Should be rejected with default threshold
        accepted_default = self.optimizer.step_accepted(ratio, threshold=1e-4)
        self.assertTrue(accepted_default)

        # Should be rejected with higher threshold
        accepted_high = self.optimizer.step_accepted(ratio, threshold=0.1)
        self.assertFalse(accepted_high)

    def test_zero_ratio(self):
        """Test with zero ratio."""
        accepted = self.optimizer.step_accepted(0.0)

        self.assertFalse(accepted)

    def test_negative_ratio(self):
        """Test with negative ratio (increasing cost)."""
        accepted = self.optimizer.step_accepted(-1.0)

        self.assertFalse(accepted)

    def test_very_good_ratio(self):
        """Test with ratio > 1 (better than predicted)."""
        accepted = self.optimizer.step_accepted(1.5)

        self.assertTrue(accepted)


class TestTrustRegionOptimizerIntegration(unittest.TestCase):
    """Integration tests for TrustRegionOptimizerBase."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = ConcreteTrustRegionOptimizer(name="trf_test")

    def test_optimize_uses_trust_radius(self):
        """Test that optimize method can access trust_radius."""

        def fun(x):
            return np.sum(x**2)

        x0 = np.array([1.0, 2.0])

        result = self.optimizer.optimize(fun, x0)

        # Result should contain trust_radius
        self.assertIn("trust_radius", result)
        self.assertEqual(result.trust_radius, 1.0)

    def test_trust_radius_modification(self):
        """Test modifying trust radius during optimization."""

        def fun(x):
            return np.sum(x**2)

        x0 = np.array([1.0, 2.0])

        # Change trust radius before optimization
        self.optimizer.trust_radius = 2.5

        result = self.optimizer.optimize(fun, x0)

        self.assertEqual(result.trust_radius, 2.5)

    def test_multiple_optimizations_with_radius_changes(self):
        """Test multiple optimizations with trust radius changes."""

        def fun(x):
            return np.sum(x**2)

        x0 = np.array([1.0])

        # First optimization
        self.optimizer.trust_radius = 1.0
        result1 = self.optimizer.optimize(fun, x0)
        self.assertEqual(result1.trust_radius, 1.0)

        # Update radius
        self.optimizer.trust_radius = 0.5

        # Second optimization
        result2 = self.optimizer.optimize(fun, x0)
        self.assertEqual(result2.trust_radius, 0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
