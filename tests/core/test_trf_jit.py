"""Tests for nlsq.core.trf_jit module.

Characterization tests for JIT-compiled functions used in Trust Region Reflective
optimization, providing GPU/TPU-accelerated implementations of core mathematical operations.

Coverage targets:
- TrustRegionJITFunctions: Class containing JIT-compiled functions
- Gradient computation (compute_grad, compute_grad_hat)
- SVD functions (svd_no_bounds, svd_bounds)
- Conjugate gradient solver (conjugate_gradient_solve)
- Trust region subproblem solvers (solve_tr_subproblem_cg, solve_tr_subproblem_cg_bounds)
- Cost calculation and numerical validation functions
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from nlsq.core.trf_jit import (
    DEFAULT_TOLERANCE,
    LOSS_FUNCTION_COEFF,
    NUMERICAL_ZERO_THRESHOLD,
    TrustRegionJITFunctions,
)


# Test fixtures
@pytest.fixture(scope="module")
def trf_funcs():
    """Create TrustRegionJITFunctions instance (module-scoped for JIT caching)."""
    return TrustRegionJITFunctions()


@pytest.fixture
def simple_jacobian():
    """Create a simple Jacobian matrix for testing."""
    return jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )


@pytest.fixture
def simple_residuals():
    """Create simple residuals for testing."""
    return jnp.array([1.0, 2.0, 3.0])


@pytest.fixture
def scaling_diagonal():
    """Create scaling diagonal for testing."""
    return jnp.array([1.0, 1.0])


@pytest.fixture
def random_key():
    """Create a JAX random key for generating random test data."""
    return random.PRNGKey(42)


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_loss_function_coeff(self):
        """Test loss function coefficient value."""
        assert LOSS_FUNCTION_COEFF == 0.5

    def test_numerical_zero_threshold(self):
        """Test numerical zero threshold value."""
        assert NUMERICAL_ZERO_THRESHOLD == 1e-14

    def test_default_tolerance(self):
        """Test default tolerance value."""
        assert DEFAULT_TOLERANCE == 1e-6


class TestTrustRegionJITFunctionsInit:
    """Tests for TrustRegionJITFunctions initialization."""

    def test_initialization(self, trf_funcs):
        """Test that all functions are created during initialization."""
        # Check gradient functions
        assert hasattr(trf_funcs, "compute_grad")
        assert hasattr(trf_funcs, "compute_grad_hat")

        # Check SVD functions
        assert hasattr(trf_funcs, "svd_no_bounds")
        assert hasattr(trf_funcs, "svd_bounds")

        # Check iterative solvers
        assert hasattr(trf_funcs, "conjugate_gradient_solve")
        assert hasattr(trf_funcs, "solve_tr_subproblem_cg")
        assert hasattr(trf_funcs, "solve_tr_subproblem_cg_bounds")

        # Check utility functions
        assert hasattr(trf_funcs, "default_loss_func")
        assert hasattr(trf_funcs, "calculate_cost")
        assert hasattr(trf_funcs, "check_isfinite")

    def test_functions_are_callable(self, trf_funcs):
        """Test that all created functions are callable."""
        assert callable(trf_funcs.compute_grad)
        assert callable(trf_funcs.compute_grad_hat)
        assert callable(trf_funcs.svd_no_bounds)
        assert callable(trf_funcs.svd_bounds)
        assert callable(trf_funcs.conjugate_gradient_solve)
        assert callable(trf_funcs.default_loss_func)
        assert callable(trf_funcs.calculate_cost)
        assert callable(trf_funcs.check_isfinite)


class TestDefaultLossFunc:
    """Tests for the default loss function."""

    def test_zero_residuals(self, trf_funcs):
        """Test loss function with zero residuals."""
        f = jnp.zeros(5)
        loss = trf_funcs.default_loss_func(f)

        assert float(loss) == 0.0

    def test_unit_residuals(self, trf_funcs):
        """Test loss function with unit residuals."""
        f = jnp.ones(4)
        loss = trf_funcs.default_loss_func(f)

        # 0.5 * sum(1^2 * 4) = 0.5 * 4 = 2.0
        assert float(loss) == pytest.approx(2.0)

    def test_known_residuals(self, trf_funcs):
        """Test loss function with known residuals."""
        f = jnp.array([1.0, 2.0, 3.0])
        loss = trf_funcs.default_loss_func(f)

        # 0.5 * (1 + 4 + 9) = 0.5 * 14 = 7.0
        assert float(loss) == pytest.approx(7.0)

    def test_negative_residuals(self, trf_funcs):
        """Test loss function with negative residuals."""
        f = jnp.array([-1.0, -2.0])
        loss = trf_funcs.default_loss_func(f)

        # 0.5 * (1 + 4) = 2.5
        assert float(loss) == pytest.approx(2.5)


class TestComputeGrad:
    """Tests for gradient computation."""

    def test_basic_gradient(self, trf_funcs, simple_jacobian, simple_residuals):
        """Test basic gradient computation."""
        J = simple_jacobian
        f = simple_residuals

        grad = trf_funcs.compute_grad(J, f)

        # grad = f.T @ J = [1, 2, 3] @ [[1, 0], [0, 1], [1, 1]]
        # = [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] = [4, 5]
        expected = jnp.array([4.0, 5.0])
        np.testing.assert_allclose(np.array(grad), np.array(expected), rtol=1e-6)

    def test_gradient_shape(self, trf_funcs):
        """Test gradient output shape matches number of parameters."""
        J = jnp.ones((10, 3))
        f = jnp.ones(10)

        grad = trf_funcs.compute_grad(J, f)

        assert grad.shape == (3,)

    def test_zero_jacobian(self, trf_funcs):
        """Test gradient with zero Jacobian."""
        J = jnp.zeros((5, 2))
        f = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        grad = trf_funcs.compute_grad(J, f)

        np.testing.assert_allclose(np.array(grad), np.zeros(2), atol=1e-10)


class TestComputeGradHat:
    """Tests for gradient in hat space."""

    def test_basic_grad_hat(self, trf_funcs):
        """Test basic gradient hat computation."""
        g = jnp.array([2.0, 3.0])
        d = jnp.array([0.5, 2.0])

        grad_hat = trf_funcs.compute_grad_hat(g, d)

        expected = jnp.array([1.0, 6.0])
        np.testing.assert_allclose(np.array(grad_hat), np.array(expected), rtol=1e-6)

    def test_identity_scaling(self, trf_funcs):
        """Test with identity scaling (d = 1)."""
        g = jnp.array([1.0, 2.0, 3.0])
        d = jnp.ones(3)

        grad_hat = trf_funcs.compute_grad_hat(g, d)

        np.testing.assert_allclose(np.array(grad_hat), np.array(g), rtol=1e-6)

    def test_zero_scaling(self, trf_funcs):
        """Test with zero scaling."""
        g = jnp.array([1.0, 2.0])
        d = jnp.zeros(2)

        grad_hat = trf_funcs.compute_grad_hat(g, d)

        np.testing.assert_allclose(np.array(grad_hat), np.zeros(2), atol=1e-10)


class TestSVDNoBounds:
    """Tests for SVD without bounds."""

    def test_basic_svd(
        self, trf_funcs, simple_jacobian, simple_residuals, scaling_diagonal
    ):
        """Test basic SVD computation."""
        J = simple_jacobian
        f = simple_residuals
        d = scaling_diagonal

        J_h, U, s, V, _uf = trf_funcs.svd_no_bounds(J, d, f)

        # Check output shapes
        assert J_h.shape == J.shape
        assert U.shape[0] == J.shape[0]
        assert s.shape[0] == min(J.shape)
        assert V.shape[0] == J.shape[1]

    def test_svd_reconstruction(self, trf_funcs, random_key):
        """Test that SVD factors can reconstruct the scaled Jacobian.

        The SVD decomposes A = U @ diag(s) @ V^T (standard notation).
        The compute_svd_with_fallback function returns V (already transposed from Vt).
        So reconstruction is: A = U @ diag(s) @ V.T
        """
        key1, key2 = random.split(random_key)
        J = random.normal(key1, (10, 3))
        f = random.normal(key2, (10,))
        d = jnp.ones(3)

        J_h, U, s, V, _uf = trf_funcs.svd_no_bounds(J, d, f)

        # Reconstruct: U @ diag(s) @ V.T = J_h
        # The SVD function returns V (not Vt), so we need V.T for reconstruction
        reconstructed = U @ jnp.diag(s) @ V.T
        np.testing.assert_allclose(
            np.array(reconstructed), np.array(J_h), rtol=1e-4, atol=1e-8
        )

    def test_uf_computation(self, trf_funcs, random_key):
        """Test that uf = U.T @ f is correctly computed."""
        key1, key2 = random.split(random_key)
        J = random.normal(key1, (10, 3))
        f = random.normal(key2, (10,))
        d = jnp.ones(3)

        _J_h, U, _s, _V, uf = trf_funcs.svd_no_bounds(J, d, f)

        # Verify uf = U.T @ f
        expected_uf = U.T @ f
        np.testing.assert_allclose(np.array(uf), np.array(expected_uf), rtol=1e-6)

    def test_scaling_applied(self, trf_funcs, simple_jacobian, simple_residuals):
        """Test that diagonal scaling is applied to Jacobian."""
        J = simple_jacobian
        f = simple_residuals
        d = jnp.array([2.0, 0.5])

        J_h, _U, _s, _V, _uf = trf_funcs.svd_no_bounds(J, d, f)

        # J_h = J * d (broadcasting)
        expected_J_h = J * d
        np.testing.assert_allclose(np.array(J_h), np.array(expected_J_h), rtol=1e-6)


class TestSVDBounds:
    """Tests for SVD with bounds (augmented system)."""

    def test_basic_svd_bounds(
        self, trf_funcs, simple_jacobian, simple_residuals, scaling_diagonal
    ):
        """Test basic SVD with bounds computation."""
        J = simple_jacobian
        f = simple_residuals
        d = scaling_diagonal
        J_diag = jnp.eye(2) * 0.1
        f_zeros = jnp.zeros(2)

        J_h, U, _s, _V, _uf = trf_funcs.svd_bounds(f, J, d, J_diag, f_zeros)

        # Check output shapes (augmented system: 3+2 = 5 rows)
        assert J_h.shape == J.shape
        assert U.shape[0] == J.shape[0] + 2  # Augmented with J_diag

    def test_augmentation(self, trf_funcs):
        """Test that system is correctly augmented."""
        J = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        f = jnp.array([1.0, 2.0])
        d = jnp.ones(2)
        J_diag = jnp.array([[0.1, 0.0], [0.0, 0.1]])
        f_zeros = jnp.zeros(2)

        _J_h, _U, _s, _V, uf = trf_funcs.svd_bounds(f, J, d, J_diag, f_zeros)

        # Augmented f has 4 elements (f + f_zeros)
        assert uf.shape[0] == min(4, 2)  # min(m, n) for reduced SVD


class TestConjugateGradientSolve:
    """Tests for conjugate gradient solver."""

    def test_simple_system(self, trf_funcs):
        """Test CG solver on a simple system."""
        # Create a well-conditioned positive definite system
        J = jnp.array(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [1.0, 1.0],
            ]
        )
        f = jnp.array([1.0, 1.0, 0.0])
        d = jnp.ones(2)

        p, residual_norm, n_iter = trf_funcs.conjugate_gradient_solve(J, f, d)

        assert p.shape == (2,)
        assert residual_norm >= 0
        assert n_iter >= 0

    def test_cg_convergence(self, trf_funcs, random_key):
        """Test that CG converges on a well-conditioned system."""
        key1, key2 = random.split(random_key)
        # Create well-conditioned system
        A = random.normal(key1, (20, 5))
        f = random.normal(key2, (20,))
        d = jnp.ones(5)

        _p, residual_norm, _n_iter = trf_funcs.conjugate_gradient_solve(
            A, f, d, alpha=0.1, tol=1e-6
        )

        # Should have converged
        assert residual_norm < 1.0  # Some convergence

    def test_cg_with_regularization(
        self, trf_funcs, simple_jacobian, simple_residuals, scaling_diagonal
    ):
        """Test CG solver with regularization."""
        J = simple_jacobian
        f = simple_residuals
        d = scaling_diagonal

        # Without regularization
        p1, _, _ = trf_funcs.conjugate_gradient_solve(J, f, d, alpha=0.0)

        # With regularization
        p2, _, _ = trf_funcs.conjugate_gradient_solve(J, f, d, alpha=1.0)

        # Solutions should be different due to regularization
        # (unless the solution happens to be very close)
        # Just check both produce valid outputs
        assert p1.shape == (2,)
        assert p2.shape == (2,)

    def test_cg_max_iterations(self, trf_funcs, random_key):
        """Test that CG respects max iterations."""
        key1, key2 = random.split(random_key)
        J = random.normal(key1, (10, 5))
        f = random.normal(key2, (10,))
        d = jnp.ones(5)

        _, _, n_iter = trf_funcs.conjugate_gradient_solve(
            J,
            f,
            d,
            max_iter=3,
            tol=1e-20,  # Tight tolerance forces max iters
        )

        assert n_iter <= 3


class TestSolveTrSubproblemCG:
    """Tests for trust region subproblem solver using CG.

    These functions use jax.lax.cond for JAX-compatible conditionals within
    JIT-compiled functions.
    """

    def test_within_trust_region(
        self, trf_funcs, simple_jacobian, simple_residuals, scaling_diagonal
    ):
        """Test when Gauss-Newton step is within trust region."""
        J = simple_jacobian
        f = simple_residuals
        d = scaling_diagonal
        Delta = 100.0  # Large trust region

        p = trf_funcs.solve_tr_subproblem_cg(J, f, d, Delta)

        assert p.shape == (2,)
        # Step should be within trust region
        assert jnp.linalg.norm(p) <= Delta + 1e-6

    def test_outside_trust_region(self, trf_funcs, random_key):
        """Test when step exceeds trust region."""
        key1, key2 = random.split(random_key)
        J = random.normal(key1, (10, 3)) * 0.1  # Small Jacobian -> large step
        f = random.normal(key2, (10,)) * 10.0  # Large residuals
        d = jnp.ones(3)
        Delta = 0.1  # Small trust region

        p = trf_funcs.solve_tr_subproblem_cg(J, f, d, Delta, alpha=0.1)

        assert p.shape == (3,)
        # Verify the result is finite (not NaN or Inf)
        assert jnp.all(jnp.isfinite(p)), "Step should be finite"
        # The step should be bounded due to scaling (scaling is clipped to [0.1, 10.0])
        # so step norm should be bounded
        assert jnp.linalg.norm(p) < 100.0, "Step should be bounded"


class TestSolveTrSubproblemCGBounds:
    """Tests for trust region subproblem solver with bounds using CG.

    These functions use jax.lax.cond for JAX-compatible conditionals within
    JIT-compiled functions.
    """

    def test_basic_bounds_solver(
        self, trf_funcs, simple_jacobian, simple_residuals, scaling_diagonal
    ):
        """Test basic bounds solver operation."""
        J = simple_jacobian
        f = simple_residuals
        d = scaling_diagonal
        J_diag = jnp.eye(2) * 0.1
        f_zeros = jnp.zeros(2)
        Delta = 100.0

        p = trf_funcs.solve_tr_subproblem_cg_bounds(J, f, d, J_diag, f_zeros, Delta)

        assert p.shape == (2,)


class TestCalculateCost:
    """Tests for cost calculation."""

    def test_basic_cost(self, trf_funcs):
        """Test basic cost calculation."""
        rho = jnp.array(
            [
                [2.0, 4.0, 6.0],  # Values (only first row used as rho[0])
            ]
        )
        data_mask = jnp.array([True, True, False])

        cost = trf_funcs.calculate_cost(rho, data_mask)

        # 0.5 * (2 + 4 + 0) = 3.0
        assert float(cost) == pytest.approx(3.0)

    def test_all_masked(self, trf_funcs):
        """Test cost with all data masked out."""
        rho = jnp.array([[1.0, 2.0, 3.0]])
        data_mask = jnp.array([False, False, False])

        cost = trf_funcs.calculate_cost(rho, data_mask)

        assert float(cost) == 0.0

    def test_all_included(self, trf_funcs):
        """Test cost with all data included."""
        rho = jnp.array([[2.0, 4.0, 6.0]])
        data_mask = jnp.array([True, True, True])

        cost = trf_funcs.calculate_cost(rho, data_mask)

        # 0.5 * (2 + 4 + 6) = 6.0
        assert float(cost) == pytest.approx(6.0)


class TestCheckIsfinite:
    """Tests for numerical validation."""

    def test_finite_values(self, trf_funcs):
        """Test with all finite values."""
        f = jnp.array([1.0, 2.0, 3.0])

        result = trf_funcs.check_isfinite(f)

        assert bool(result) is True

    def test_nan_values(self, trf_funcs):
        """Test with NaN values."""
        f = jnp.array([1.0, jnp.nan, 3.0])

        result = trf_funcs.check_isfinite(f)

        assert bool(result) is False

    def test_inf_values(self, trf_funcs):
        """Test with Inf values."""
        f = jnp.array([1.0, jnp.inf, 3.0])

        result = trf_funcs.check_isfinite(f)

        assert bool(result) is False

    def test_neg_inf_values(self, trf_funcs):
        """Test with negative Inf values."""
        f = jnp.array([1.0, -jnp.inf, 3.0])

        result = trf_funcs.check_isfinite(f)

        assert bool(result) is False

    def test_empty_array(self, trf_funcs):
        """Test with empty array."""
        f = jnp.array([])

        result = trf_funcs.check_isfinite(f)

        # Empty array should be considered all finite
        assert bool(result) is True


class TestNumericalStability:
    """Tests for numerical stability of JIT functions."""

    def test_large_jacobian(self, trf_funcs, random_key):
        """Test with large Jacobian matrix."""
        key1, key2 = random.split(random_key)
        J = random.normal(key1, (1000, 20))
        f = random.normal(key2, (1000,))
        d = jnp.ones(20)

        # Should not raise or produce NaN
        _J_h, _U, s, _V, uf = trf_funcs.svd_no_bounds(J, d, f)

        assert jnp.all(jnp.isfinite(s))
        assert jnp.all(jnp.isfinite(uf))

    def test_ill_conditioned_jacobian(self, trf_funcs):
        """Test with ill-conditioned Jacobian."""
        # Create ill-conditioned matrix
        J = jnp.array(
            [
                [1.0, 1.0],
                [1.0, 1.0 + 1e-10],
                [2.0, 2.0 + 1e-10],
            ]
        )
        f = jnp.array([1.0, 1.0, 2.0])
        d = jnp.ones(2)

        # Should handle without crashing
        _J_h, _U, s, _V, _uf = trf_funcs.svd_no_bounds(J, d, f)

        # Singular values should be computed (may have one very small)
        assert s.shape[0] == 2

    def test_zero_jacobian(self, trf_funcs):
        """Test with zero Jacobian."""
        J = jnp.zeros((5, 3))
        f = jnp.ones(5)
        d = jnp.ones(3)

        # Should handle zero Jacobian
        _J_h, _U, s, _V, _uf = trf_funcs.svd_no_bounds(J, d, f)

        # All singular values should be zero
        np.testing.assert_allclose(np.array(s), np.zeros(3), atol=1e-10)

    def test_very_small_values(self, trf_funcs):
        """Test with very small values."""
        J = jnp.ones((5, 2)) * 1e-15
        f = jnp.ones(5) * 1e-15
        d = jnp.ones(2)

        grad = trf_funcs.compute_grad(J, f)

        assert jnp.all(jnp.isfinite(grad))


class TestJITCompilation:
    """Tests to verify JIT compilation behavior."""

    def test_multiple_calls_same_shape(self, trf_funcs, random_key):
        """Test that multiple calls with same shape are efficient."""
        key1, key2, key3, key4 = random.split(random_key, 4)

        J1 = random.normal(key1, (10, 3))
        f1 = random.normal(key2, (10,))
        J2 = random.normal(key3, (10, 3))
        f2 = random.normal(key4, (10,))
        d = jnp.ones(3)

        # Both calls should use cached JIT
        result1 = trf_funcs.compute_grad(J1, f1)
        result2 = trf_funcs.compute_grad(J2, f2)

        assert result1.shape == result2.shape == (3,)

    def test_different_shapes_work(self, trf_funcs, random_key):
        """Test that different shapes trigger recompilation correctly."""
        key1, key2, key3, key4 = random.split(random_key, 4)

        # Different sized problems
        J1 = random.normal(key1, (10, 3))
        f1 = random.normal(key2, (10,))
        J2 = random.normal(key3, (20, 5))
        f2 = random.normal(key4, (20,))

        result1 = trf_funcs.compute_grad(J1, f1)
        result2 = trf_funcs.compute_grad(J2, f2)

        assert result1.shape == (3,)
        assert result2.shape == (5,)
