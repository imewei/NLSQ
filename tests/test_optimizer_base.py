"""Tests for the optimizer_base module."""
import unittest
import numpy as np
import jax.numpy as jnp
from nlsq.optimizer_base import OptimizerBase, TrustRegionOptimizerBase
from nlsq._optimize import OptimizeResult


class TestOptimizerBase(unittest.TestCase):
    """Test the OptimizerBase abstract class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class ConcreteOptimizer(OptimizerBase):
            def optimize(self, fun, x0, jac=None, bounds=(-np.inf, np.inf), **kwargs):
                # Simple implementation that just returns initial point
                self._nfev += 1
                return OptimizeResult(
                    x=x0,
                    success=True,
                    status=1,
                    message="Test completed",
                    fun=fun(x0) if callable(fun) else np.zeros(1),
                    jac=np.eye(len(x0)),
                    nfev=self._nfev,
                    njev=self._njev,
                    nit=1
                )

        self.optimizer = ConcreteOptimizer(name="test_optimizer")

    def test_initialization(self):
        """Test OptimizerBase initialization."""
        self.assertEqual(self.optimizer.name, "test_optimizer")
        self.assertEqual(self.optimizer._nfev, 0)
        self.assertEqual(self.optimizer._njev, 0)
        self.assertIsNotNone(self.optimizer.logger)

    def test_optimize(self):
        """Test the optimize method."""
        def test_fun(x):
            return np.sum(x**2)

        x0 = np.array([1.0, 2.0])
        result = self.optimizer.optimize(test_fun, x0)

        self.assertIsInstance(result, OptimizeResult)
        self.assertTrue(result.success)
        self.assertEqual(result.status, 1)
        np.testing.assert_array_equal(result.x, x0)
        self.assertEqual(result.nfev, 1)

    def test_optimize_with_bounds(self):
        """Test optimization with bounds."""
        def test_fun(x):
            return np.sum(x**2)

        x0 = np.array([0.5, 0.5])
        bounds = (0, 1)
        result = self.optimizer.optimize(test_fun, x0, bounds=bounds)

        self.assertTrue(result.success)
        np.testing.assert_array_equal(result.x, x0)

    def test_optimize_with_jacobian(self):
        """Test optimization with Jacobian."""
        def test_fun(x):
            return np.sum(x**2)

        def test_jac(x):
            return 2 * x

        x0 = np.array([1.0, 1.0])
        result = self.optimizer.optimize(test_fun, x0, jac=test_jac)

        self.assertTrue(result.success)
        np.testing.assert_array_equal(result.x, x0)

    def test_reset_counters(self):
        """Test counter reset functionality."""
        # Create a new optimizer with method to reset counters
        class ResettableOptimizer(OptimizerBase):
            def optimize(self, fun, x0, jac=None, bounds=(-np.inf, np.inf), **kwargs):
                self._nfev += 1
                if jac is not None:
                    self._njev += 1
                return OptimizeResult(
                    x=x0,
                    success=True,
                    status=1,
                    message="Test",
                    fun=np.zeros(1),
                    jac=np.eye(len(x0)),
                    nfev=self._nfev,
                    njev=self._njev
                )

            def reset_counters(self):
                self._nfev = 0
                self._njev = 0

        optimizer = ResettableOptimizer(name="resettable")

        def dummy_fun(x):
            return x

        def dummy_jac(x):
            return np.ones_like(x)

        # Run optimization to increment counters
        optimizer.optimize(dummy_fun, np.array([1.0]), jac=dummy_jac)
        self.assertEqual(optimizer._nfev, 1)
        self.assertEqual(optimizer._njev, 1)

        # Reset counters
        optimizer.reset_counters()
        self.assertEqual(optimizer._nfev, 0)
        self.assertEqual(optimizer._njev, 0)


class TestTrustRegionOptimizerBase(unittest.TestCase):
    """Test the TrustRegionOptimizerBase abstract class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class ConcreteTrustRegion(TrustRegionOptimizerBase):
            def optimize(self, fun, x0, jac=None, bounds=(-np.inf, np.inf), **kwargs):
                # Simple implementation
                self._nfev += 1
                return OptimizeResult(
                    x=x0,
                    success=True,
                    status=1,
                    message="Trust region test",
                    fun=fun(x0) if callable(fun) else np.zeros(1),
                    jac=np.eye(len(x0)),
                    nfev=self._nfev
                )

        self.optimizer = ConcreteTrustRegion(name="test_trust_region")

    def test_initialization(self):
        """Test TrustRegionOptimizerBase initialization."""
        self.assertEqual(self.optimizer.name, "test_trust_region")
        self.assertIsNotNone(self.optimizer.logger)

    def test_trust_radius_property(self):
        """Test trust radius property."""
        # Get initial radius
        self.assertEqual(self.optimizer.trust_radius, 1.0)

        # Set new radius
        self.optimizer.trust_radius = 2.5
        self.assertEqual(self.optimizer.trust_radius, 2.5)

        # Set negative radius (should be clamped to 0)
        self.optimizer.trust_radius = -1.0
        self.assertEqual(self.optimizer.trust_radius, 0.0)

    def test_update_trust_radius(self):
        """Test trust radius update logic."""
        # Test ratio < 0.25 (poor step)
        Delta_new, ratio = self.optimizer.update_trust_radius(
            Delta=1.0,
            actual_reduction=0.1,
            predicted_reduction=1.0,
            step_norm=0.5,
            step_at_boundary=False
        )
        self.assertEqual(ratio, 0.1)
        self.assertEqual(Delta_new, 0.125)  # 0.25 * step_norm

        # Test ratio > 0.75 with step at boundary (good step)
        Delta_new, ratio = self.optimizer.update_trust_radius(
            Delta=1.0,
            actual_reduction=0.9,
            predicted_reduction=1.0,
            step_norm=1.0,
            step_at_boundary=True
        )
        self.assertEqual(ratio, 0.9)
        self.assertEqual(Delta_new, 2.0)  # 2 * Delta

        # Test ratio in middle range
        Delta_new, ratio = self.optimizer.update_trust_radius(
            Delta=1.0,
            actual_reduction=0.5,
            predicted_reduction=1.0,
            step_norm=0.8,
            step_at_boundary=False
        )
        self.assertEqual(ratio, 0.5)
        self.assertEqual(Delta_new, 1.0)  # Unchanged

    def test_step_accepted(self):
        """Test step acceptance logic."""
        # Good ratio - should accept
        accepted = self.optimizer.step_accepted(ratio=0.1, threshold=0.01)
        self.assertTrue(accepted)

        # Bad ratio - should reject
        accepted = self.optimizer.step_accepted(ratio=0.001, threshold=0.01)
        self.assertFalse(accepted)

        # Edge case - exactly at threshold (should reject since it's not greater than)
        accepted = self.optimizer.step_accepted(ratio=0.01, threshold=0.01)
        self.assertFalse(accepted)


if __name__ == '__main__':
    unittest.main()