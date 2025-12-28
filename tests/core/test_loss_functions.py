"""
Comprehensive test suite for loss_functions module.

Tests robust loss functions (Huber, Soft L1, Cauchy, Arctan) for:
- Numerical correctness
- JAX compatibility
- Gradient correctness
- Robustness properties
"""

import unittest

import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from numpy.testing import assert_allclose

from nlsq.core.loss_functions import LossFunctionsJIT

# Property-based testing
try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


class TestLossFunctionsJITBasic(unittest.TestCase):
    """Test suite for LossFunctionsJIT basic functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.loss_func_jit = LossFunctionsJIT()

    def test_initialization(self):
        """Test LossFunctionsJIT initializes correctly."""
        assert self.loss_func_jit is not None
        assert hasattr(self.loss_func_jit, "loss_funcs")
        assert hasattr(self.loss_func_jit, "IMPLEMENTED_LOSSES")

    def test_implemented_losses(self):
        """Test all implemented loss functions are available."""
        expected_losses = ["linear", "huber", "soft_l1", "cauchy", "arctan"]
        assert set(self.loss_func_jit.IMPLEMENTED_LOSSES.keys()) == set(expected_losses)

    def test_linear_loss_returns_none(self):
        """Test linear loss returns None (no transformation needed)."""
        loss_func = self.loss_func_jit.get_loss_function("linear")
        assert loss_func is None

    def test_huber_loss_callable(self):
        """Test Huber loss is callable."""
        loss_func = self.loss_func_jit.get_loss_function("huber")
        assert loss_func is not None
        assert callable(loss_func)

    def test_soft_l1_loss_callable(self):
        """Test Soft L1 loss is callable."""
        loss_func = self.loss_func_jit.get_loss_function("soft_l1")
        assert loss_func is not None
        assert callable(loss_func)

    def test_cauchy_loss_callable(self):
        """Test Cauchy loss is callable."""
        loss_func = self.loss_func_jit.get_loss_function("cauchy")
        assert loss_func is not None
        assert callable(loss_func)

    def test_arctan_loss_callable(self):
        """Test Arctan loss is callable."""
        loss_func = self.loss_func_jit.get_loss_function("arctan")
        assert loss_func is not None
        assert callable(loss_func)


class TestLossFunctionsComputation(unittest.TestCase):
    """Test loss function computation."""

    def setUp(self):
        """Set up test fixtures."""
        self.loss_func_jit = LossFunctionsJIT()
        self.residuals = jnp.array([0.5, 1.0, 2.0, 3.0])
        self.f_scale = 1.0
        self.data_mask = jnp.ones_like(self.residuals, dtype=bool)

    def test_huber_cost_computation(self):
        """Test Huber loss cost computation."""
        loss_func = self.loss_func_jit.get_loss_function("huber")
        cost = loss_func(self.residuals, self.f_scale, self.data_mask, cost_only=True)

        assert cost >= 0
        assert jnp.isfinite(cost)

    def test_huber_rho_computation(self):
        """Test Huber loss rho computation with derivatives."""
        loss_func = self.loss_func_jit.get_loss_function("huber")
        rho = loss_func(self.residuals, self.f_scale, self.data_mask, cost_only=False)

        # Should return array with shape (3, len(residuals))
        assert rho.shape == (3, len(self.residuals))
        # All values should be finite
        assert jnp.all(jnp.isfinite(rho))

    def test_soft_l1_cost_computation(self):
        """Test Soft L1 loss cost computation."""
        loss_func = self.loss_func_jit.get_loss_function("soft_l1")
        cost = loss_func(self.residuals, self.f_scale, self.data_mask, cost_only=True)

        assert cost >= 0
        assert jnp.isfinite(cost)

    def test_cauchy_cost_computation(self):
        """Test Cauchy loss cost computation."""
        loss_func = self.loss_func_jit.get_loss_function("cauchy")
        cost = loss_func(self.residuals, self.f_scale, self.data_mask, cost_only=True)

        assert cost >= 0
        assert jnp.isfinite(cost)

    def test_arctan_cost_computation(self):
        """Test Arctan loss cost computation."""
        loss_func = self.loss_func_jit.get_loss_function("arctan")
        cost = loss_func(self.residuals, self.f_scale, self.data_mask, cost_only=True)

        assert cost >= 0
        assert jnp.isfinite(cost)


class TestLossFunctionsEdgeCases(unittest.TestCase):
    """Edge case tests for loss functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.loss_func_jit = LossFunctionsJIT()
        self.f_scale = 1.0

    def test_single_residual(self):
        """Test with single residual."""
        single = jnp.array([1.5])
        data_mask = jnp.ones_like(single, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)
            cost = loss_func(single, self.f_scale, data_mask, cost_only=True)

            assert jnp.isfinite(cost)
            assert cost >= 0

    def test_zero_residuals(self):
        """Test with all zero residuals."""
        zeros = jnp.zeros(10)
        data_mask = jnp.ones_like(zeros, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)
            cost = loss_func(zeros, self.f_scale, data_mask, cost_only=True)

            assert cost == 0.0 or abs(cost) < 1e-12

    def test_large_residuals(self):
        """Test numerical stability with large residuals."""
        large = jnp.array([100.0, 1000.0, 10000.0])
        data_mask = jnp.ones_like(large, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)
            cost = loss_func(large, self.f_scale, data_mask, cost_only=True)

            assert jnp.isfinite(cost), f"{loss_type} produced non-finite result"

    def test_small_residuals(self):
        """Test numerical stability with small residuals."""
        small = jnp.array([1e-10, 1e-8, 1e-6])
        data_mask = jnp.ones_like(small, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)
            cost = loss_func(small, self.f_scale, data_mask, cost_only=True)

            assert jnp.isfinite(cost)
            assert cost >= 0

    def test_mixed_signs(self):
        """Test with mixed positive and negative residuals."""
        mixed = jnp.array([1.0, -1.0, 2.0, -2.0, 0.0])
        data_mask = jnp.ones_like(mixed, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)
            cost = loss_func(mixed, self.f_scale, data_mask, cost_only=True)

            assert jnp.isfinite(cost)
            assert cost >= 0

    def test_invalid_loss_type(self):
        """Test invalid loss type raises appropriate error."""
        with self.assertRaises((ValueError, KeyError)):
            self.loss_func_jit.get_loss_function("invalid_loss")


class TestLossFunctionsCorrectness(unittest.TestCase):
    """Numerical correctness tests for loss functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.loss_func_jit = LossFunctionsJIT()
        self.f_scale = 1.0

    def test_huber_quadratic_region(self):
        """Test Huber loss is quadratic for small residuals (|f/f_scale| <= 1)."""
        residuals = jnp.array([0.1, 0.5, 0.9])
        data_mask = jnp.ones_like(residuals, dtype=bool)
        loss_func = self.loss_func_jit.get_loss_function("huber")

        cost = loss_func(residuals, self.f_scale, data_mask, cost_only=True)
        # For small residuals, Huber should be approximately quadratic
        expected = 0.5 * self.f_scale**2 * jnp.sum((residuals / self.f_scale) ** 2)

        assert_allclose(float(cost), float(expected), rtol=1e-6)

    def test_loss_symmetry(self):
        """Test loss functions are symmetric in residuals."""
        residuals = jnp.array([1.0, 2.0, 3.0])
        data_mask = jnp.ones_like(residuals, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)

            cost_pos = loss_func(residuals, self.f_scale, data_mask, cost_only=True)
            cost_neg = loss_func(-residuals, self.f_scale, data_mask, cost_only=True)

            assert_allclose(
                float(cost_pos),
                float(cost_neg),
                rtol=1e-12,
                err_msg=f"{loss_type} not symmetric",
            )

    def test_loss_at_zero(self):
        """Test all losses are zero at zero residual."""
        zero_residual = jnp.array([0.0])
        data_mask = jnp.ones_like(zero_residual, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)
            cost = loss_func(zero_residual, self.f_scale, data_mask, cost_only=True)

            assert abs(cost) < 1e-12, f"{loss_type} not zero at zero residual"

    def test_rho_components_shape(self):
        """Test that rho returns correct shape with derivatives."""
        residuals = jnp.array([0.5, 1.0, 2.0])
        data_mask = jnp.ones_like(residuals, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)
            rho = loss_func(residuals, self.f_scale, data_mask, cost_only=False)

            # Should return (3, n) array: [rho, rho', rho'']
            self.assertEqual(rho.shape[0], 3)
            self.assertEqual(rho.shape[1], len(residuals))

    def test_data_mask_effect(self):
        """Test that data mask correctly excludes points."""
        residuals = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask_all = jnp.ones_like(residuals, dtype=bool)
        mask_half = jnp.array([True, False, True, False])

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)

            cost_all = loss_func(residuals, self.f_scale, mask_all, cost_only=True)
            cost_half = loss_func(residuals, self.f_scale, mask_half, cost_only=True)

            # Masked cost should be less than full cost
            self.assertLess(float(cost_half), float(cost_all))


class TestLossFunctionsJAX(unittest.TestCase):
    """JAX-specific tests for loss functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.loss_func_jit = LossFunctionsJIT()
        self.f_scale = 1.0

    def test_jit_compilation(self):
        """Test that loss functions work with JIT compilation."""
        residuals = jnp.array([0.5, 1.5, 3.0])
        data_mask = jnp.ones_like(residuals, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)

            # JIT compile the function
            jitted_func = jit(
                lambda r, loss=loss_func: loss(
                    r, self.f_scale, data_mask, cost_only=True
                )
            )

            # Should work without errors
            cost = jitted_func(residuals)
            self.assertTrue(jnp.isfinite(cost))

    def test_gradient_shape(self):
        """Test gradient computation via JAX autodiff."""
        residuals = jnp.array([0.5, 1.5, 3.0])
        data_mask = jnp.ones_like(residuals, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)

            # Define function for gradient
            def cost_fn(r, loss=loss_func):
                return loss(r, self.f_scale, data_mask, cost_only=True)

            # Compute gradient
            grad_fn = grad(cost_fn)
            gradient = grad_fn(residuals)

            # Gradient should have same shape as residuals
            self.assertEqual(gradient.shape, residuals.shape)
            self.assertTrue(jnp.all(jnp.isfinite(gradient)))

    def test_gradient_at_zero(self):
        """Test gradient is zero at zero residual."""
        residuals = jnp.array(
            [1e-10, 1e-10, 1e-10]
        )  # Use small but non-zero to avoid NaN
        data_mask = jnp.ones_like(residuals, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)

            def cost_fn(r, loss=loss_func):
                return loss(r, self.f_scale, data_mask, cost_only=True)

            grad_fn = grad(cost_fn)
            gradient = grad_fn(residuals)

            # Gradient should be close to zero for very small residuals
            assert_allclose(gradient, jnp.zeros_like(residuals), atol=1e-6)

    def test_different_f_scales(self):
        """Test loss functions with different f_scale values."""
        residuals = jnp.array([1.0, 2.0, 3.0])
        data_mask = jnp.ones_like(residuals, dtype=bool)

        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)

            cost_1 = loss_func(residuals, 1.0, data_mask, cost_only=True)
            cost_2 = loss_func(residuals, 2.0, data_mask, cost_only=True)

            # Different scales should give different costs
            self.assertNotAlmostEqual(float(cost_1), float(cost_2))


class TestLossFunctionsIntegration(unittest.TestCase):
    """Integration tests for loss functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.loss_func_jit = LossFunctionsJIT()
        np.random.seed(42)

    def test_with_outliers(self):
        """Test that robust losses handle outliers better."""
        # Create data with outliers
        residuals = jnp.array(np.random.randn(100) * 0.1)
        # Add outliers
        residuals = residuals.at[10].set(10.0)
        residuals = residuals.at[20].set(-8.0)
        residuals = residuals.at[30].set(5.0)

        f_scale = 1.0
        data_mask = jnp.ones_like(residuals, dtype=bool)

        # Compute costs
        costs = {}
        for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
            loss_func = self.loss_func_jit.get_loss_function(loss_type)
            costs[loss_type] = loss_func(residuals, f_scale, data_mask, cost_only=True)

        # All costs should be finite and positive
        for loss_type, cost in costs.items():
            self.assertTrue(jnp.isfinite(cost), f"{loss_type} cost not finite")
            self.assertGreater(float(cost), 0, f"{loss_type} cost not positive")


if HYPOTHESIS_AVAILABLE:

    class TestLossFunctionsProperties(unittest.TestCase):
        """Property-based tests using Hypothesis."""

        def setUp(self):
            """Set up test fixtures."""
            self.loss_func_jit = LossFunctionsJIT()

        @given(
            residuals=st.lists(
                st.floats(min_value=-100, max_value=100), min_size=1, max_size=50
            )
        )
        @settings(
            max_examples=50,
            deadline=None,
            suppress_health_check=[HealthCheck.function_scoped_fixture],
        )
        def test_loss_non_negative(self, residuals):
            """Property: All loss functions are non-negative."""
            residuals_arr = jnp.array(residuals)
            data_mask = jnp.ones_like(residuals_arr, dtype=bool)
            f_scale = 1.0

            for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
                loss_func = self.loss_func_jit.get_loss_function(loss_type)
                cost = loss_func(residuals_arr, f_scale, data_mask, cost_only=True)

                self.assertGreaterEqual(float(cost), 0)

        @given(
            residuals=st.lists(
                st.floats(min_value=-100, max_value=100), min_size=1, max_size=50
            )
        )
        @settings(
            max_examples=50,
            deadline=None,
            suppress_health_check=[HealthCheck.function_scoped_fixture],
        )
        def test_loss_finite(self, residuals):
            """Property: All loss functions produce finite values."""
            residuals_arr = jnp.array(residuals)
            data_mask = jnp.ones_like(residuals_arr, dtype=bool)
            f_scale = 1.0

            for loss_type in ["huber", "soft_l1", "cauchy", "arctan"]:
                loss_func = self.loss_func_jit.get_loss_function(loss_type)
                cost = loss_func(residuals_arr, f_scale, data_mask, cost_only=True)

                self.assertTrue(jnp.isfinite(cost))


if __name__ == "__main__":
    unittest.main()
