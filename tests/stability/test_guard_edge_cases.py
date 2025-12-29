"""Edge case tests for nlsq.stability.guard module.

Tests numerical stability guards, edge cases, and scientific computing scenarios.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nlsq.stability.guard import (
    NumericalStabilityGuard,
    solve_with_cholesky_fallback,
)


class TestSolveWithCholeskyFallback:
    """Tests for solve_with_cholesky_fallback function."""

    def test_positive_definite_uses_cholesky(self):
        """Test that positive definite matrices use Cholesky."""
        A = jnp.array([[4.0, 2.0], [2.0, 3.0]])  # Positive definite
        b = jnp.array([1.0, 2.0])
        x, used_cholesky = solve_with_cholesky_fallback(A, b)

        # Verify solution
        assert np.allclose(A @ x, b, rtol=1e-10)
        # Should use Cholesky for positive definite
        assert used_cholesky is True or bool(used_cholesky)

    def test_non_positive_definite_uses_fallback(self):
        """Test that non-positive definite matrices use eigenvalue fallback."""
        # Indefinite matrix (has negative eigenvalue)
        A = jnp.array([[1.0, 2.0], [2.0, 1.0]])
        b = jnp.array([1.0, 1.0])
        x, _used_cholesky = solve_with_cholesky_fallback(A, b)

        # Solution should still be reasonable
        assert all(np.isfinite(x))

    def test_nearly_singular_matrix(self):
        """Test with nearly singular matrix."""
        A = jnp.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]])
        b = jnp.array([2.0, 2.0])
        x, _used_cholesky = solve_with_cholesky_fallback(A, b)

        # Should not crash, solution may be regularized
        assert all(np.isfinite(x))

    def test_identity_matrix(self):
        """Test with identity matrix (trivial case)."""
        A = jnp.eye(3)
        b = jnp.array([1.0, 2.0, 3.0])
        x, used_cholesky = solve_with_cholesky_fallback(A, b)

        assert_allclose(np.asarray(x), np.asarray(b), rtol=1e-10)
        assert used_cholesky is True or bool(used_cholesky)

    def test_diagonal_matrix(self):
        """Test with diagonal positive definite matrix."""
        A = jnp.diag(jnp.array([4.0, 9.0, 16.0]))
        b = jnp.array([8.0, 27.0, 64.0])
        x, _used_cholesky = solve_with_cholesky_fallback(A, b)

        expected = jnp.array([2.0, 3.0, 4.0])
        assert_allclose(np.asarray(x), np.asarray(expected), rtol=1e-10)

    def test_asymmetric_input_symmetrized(self):
        """Test that asymmetric input is symmetrized."""
        # Asymmetric input - function should symmetrize it
        A = jnp.array([[4.0, 3.0], [1.0, 3.0]])
        b = jnp.array([1.0, 1.0])
        x, _ = solve_with_cholesky_fallback(A, b)

        # Should not crash
        assert all(np.isfinite(x))

    def test_large_condition_number(self):
        """Test matrix with large condition number."""
        # Create ill-conditioned matrix
        A = jnp.array([[1e10, 0.0], [0.0, 1e-10]])
        b = jnp.array([1e10, 1e-10])
        x, _ = solve_with_cholesky_fallback(A, b)

        # Solution should be [1.0, 1.0]
        assert_allclose(np.asarray(x), [1.0, 1.0], rtol=1e-5)

    def test_jit_compatibility(self):
        """Test that solve_with_cholesky_fallback is JIT-compatible."""
        from jax import jit

        jit_solve = jit(solve_with_cholesky_fallback)
        A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
        b = jnp.array([1.0, 2.0])

        x, _used_cholesky = jit_solve(A, b)
        assert all(np.isfinite(x))


class TestNumericalStabilityGuardInit:
    """Tests for NumericalStabilityGuard initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        guard = NumericalStabilityGuard()

        assert guard.eps > 0
        assert guard.max_float > 1e300
        assert guard.min_float > 0
        assert guard.condition_threshold == 1e12
        assert guard.regularization_factor == 1e-10
        assert guard.max_exp_arg == 700
        assert guard.min_exp_arg == -700
        assert guard.max_jacobian_elements_for_svd == 10_000_000

    def test_custom_max_jacobian_elements(self):
        """Test custom max_jacobian_elements_for_svd."""
        guard = NumericalStabilityGuard(max_jacobian_elements_for_svd=1_000_000)
        assert guard.max_jacobian_elements_for_svd == 1_000_000

    def test_jit_functions_created(self):
        """Test that JIT functions are created."""
        guard = NumericalStabilityGuard()

        assert hasattr(guard, "_safe_exp_jit")
        assert hasattr(guard, "_safe_log_jit")
        assert hasattr(guard, "_safe_divide_jit")
        assert hasattr(guard, "_safe_sqrt_jit")
        assert hasattr(guard, "_check_jacobian_fast_jit")
        assert hasattr(guard, "_check_gradient_jit")
        assert hasattr(guard, "_safe_norm_jit")


class TestNumericalStabilityGuardSafeMath:
    """Tests for safe mathematical operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = NumericalStabilityGuard()

    def test_safe_exp_normal_values(self):
        """Test safe exp with normal values."""
        x = jnp.array([0.0, 1.0, -1.0])
        result = self.guard._safe_exp_jit(x)
        expected = jnp.exp(x)
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-10)

    def test_safe_exp_overflow_protection(self):
        """Test safe exp prevents overflow."""
        x = jnp.array([1000.0])  # Would cause overflow
        result = self.guard._safe_exp_jit(x)

        # Should be clipped to max_exp_arg and not inf
        assert np.isfinite(result[0])
        assert result[0] > 0

    def test_safe_exp_underflow_protection(self):
        """Test safe exp handles extreme negative values."""
        x = jnp.array([-1000.0])  # Would underflow to 0
        result = self.guard._safe_exp_jit(x)

        # Should be very small but finite
        assert np.isfinite(result[0])
        assert result[0] >= 0

    def test_safe_log_normal_values(self):
        """Test safe log with normal values."""
        x = jnp.array([1.0, 2.718, 10.0])
        result = self.guard._safe_log_jit(x)
        expected = jnp.log(x)
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-10)

    def test_safe_log_zero_protection(self):
        """Test safe log with zero input."""
        x = jnp.array([0.0])
        result = self.guard._safe_log_jit(x)

        # Should return log of min_float, not -inf
        assert np.isfinite(result[0])

    def test_safe_log_negative_protection(self):
        """Test safe log with negative input."""
        x = jnp.array([-1.0])
        result = self.guard._safe_log_jit(x)

        # Should clamp to min_float
        assert np.isfinite(result[0])

    def test_safe_divide_normal_values(self):
        """Test safe divide with normal values."""
        num = jnp.array([1.0, 2.0, 3.0])
        denom = jnp.array([2.0, 4.0, 6.0])
        result = self.guard._safe_divide_jit(num, denom)
        expected = jnp.array([0.5, 0.5, 0.5])
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-10)

    def test_safe_divide_by_zero(self):
        """Test safe divide handles division by zero."""
        num = jnp.array([1.0])
        denom = jnp.array([0.0])
        result = self.guard._safe_divide_jit(num, denom)

        # Should not be inf
        assert np.isfinite(result[0])

    def test_safe_divide_by_small_value(self):
        """Test safe divide with very small denominator."""
        num = jnp.array([1.0])
        denom = jnp.array([1e-320])  # Subnormal
        result = self.guard._safe_divide_jit(num, denom)

        assert np.isfinite(result[0])

    def test_safe_sqrt_normal_values(self):
        """Test safe sqrt with normal values."""
        x = jnp.array([0.0, 1.0, 4.0, 9.0])
        result = self.guard._safe_sqrt_jit(x)
        expected = jnp.array([0.0, 1.0, 2.0, 3.0])
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-10)

    def test_safe_sqrt_negative_input(self):
        """Test safe sqrt with negative input."""
        x = jnp.array([-1.0, -4.0])
        result = self.guard._safe_sqrt_jit(x)

        # Should clamp to 0 and return 0
        assert_allclose(np.asarray(result), [0.0, 0.0], atol=1e-10)


class TestNumericalStabilityGuardJacobianCheck:
    """Tests for Jacobian checking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = NumericalStabilityGuard()

    def test_check_jacobian_clean(self):
        """Test Jacobian check with clean matrix."""
        J = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        J_fixed, issues = self.guard.check_and_fix_jacobian(J)

        assert_allclose(np.asarray(J_fixed), np.asarray(J), rtol=1e-12)
        assert issues["has_nan"] is False
        assert issues["has_inf"] is False

    def test_check_jacobian_with_nan(self):
        """Test Jacobian check detects and fixes NaN."""
        J = jnp.array([[1.0, np.nan], [3.0, 4.0]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            J_fixed, issues = self.guard.check_and_fix_jacobian(J)

            # Should warn about NaN
            assert any("NaN" in str(warning.message) for warning in w)

        assert issues["has_nan"] is True
        # NaN should be replaced with 0
        assert jnp.isfinite(J_fixed).all()

    def test_check_jacobian_with_inf(self):
        """Test Jacobian check detects and fixes Inf."""
        J = jnp.array([[1.0, np.inf], [3.0, 4.0]])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            J_fixed, issues = self.guard.check_and_fix_jacobian(J)

        assert issues["has_inf"] is True
        assert jnp.isfinite(J_fixed).all()

    def test_check_jacobian_all_zeros(self):
        """Test Jacobian check handles all-zero matrix."""
        J = jnp.zeros((3, 2))

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            J_fixed, _issues = self.guard.check_and_fix_jacobian(J)

        # Should add small perturbation (eps ~ 2.22e-16)
        # Use exact comparison since the values should be exactly eps
        assert jnp.any(J_fixed != 0.0)

    def test_check_jacobian_large_skips_svd(self):
        """Test that large Jacobians skip SVD computation."""
        # Create Jacobian larger than threshold
        guard = NumericalStabilityGuard(max_jacobian_elements_for_svd=100)
        J = jnp.ones((20, 10))  # 200 elements > 100

        _J_fixed, issues = guard.check_and_fix_jacobian(J)

        assert issues["svd_skipped"] is True
        assert "reason" in issues
        assert "too large" in issues["reason"]

    def test_check_jacobian_small_uses_svd(self):
        """Test that small Jacobians use SVD for condition number."""
        guard = NumericalStabilityGuard(max_jacobian_elements_for_svd=10_000_000)
        J = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # 4 elements

        _J_fixed, issues = guard.check_and_fix_jacobian(J)

        # SVD should be used
        assert issues.get("svd_skipped", False) is False


class TestNumericalStabilityGuardGradientCheck:
    """Tests for gradient checking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = NumericalStabilityGuard()

    def test_check_gradient_clean(self):
        """Test gradient check with clean gradient."""
        gradient = jnp.array([1.0, 2.0, 3.0])
        max_grad_norm = 100.0

        result = self.guard._check_gradient_jit(gradient, max_grad_norm)
        grad_fixed, has_invalid, needs_clipping, _grad_norm = result

        assert_allclose(np.asarray(grad_fixed), np.asarray(gradient), rtol=1e-12)
        assert has_invalid is False or bool(has_invalid) is False
        assert needs_clipping is False or bool(needs_clipping) is False

    def test_check_gradient_with_nan(self):
        """Test gradient check replaces NaN with zero."""
        gradient = jnp.array([1.0, np.nan, 3.0])
        max_grad_norm = 100.0

        result = self.guard._check_gradient_jit(gradient, max_grad_norm)
        grad_fixed, has_invalid, _, _ = result

        assert has_invalid is True or bool(has_invalid) is True
        assert jnp.isfinite(grad_fixed).all()
        assert float(grad_fixed[1]) == 0.0  # NaN replaced with 0

    def test_check_gradient_clips_large(self):
        """Test gradient check clips large gradients."""
        gradient = jnp.array([100.0, 200.0, 300.0])
        max_grad_norm = 10.0

        result = self.guard._check_gradient_jit(gradient, max_grad_norm)
        grad_fixed, _, needs_clipping, _ = result

        assert needs_clipping is True or bool(needs_clipping) is True
        assert (
            float(jnp.linalg.norm(grad_fixed)) <= max_grad_norm * 1.01
        )  # Small tolerance


class TestNumericalStabilityGuardSafeNorm:
    """Tests for safe norm computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = NumericalStabilityGuard()

    def test_safe_norm_normal_values(self):
        """Test safe norm with normal values."""
        x = jnp.array([3.0, 4.0])
        scale_factor = 1.0

        result = self.guard._safe_norm_jit(x, scale_factor)
        expected = 5.0
        assert_allclose(float(result), expected, rtol=1e-10)

    def test_safe_norm_with_scaling(self):
        """Test safe norm with scaling factor."""
        x = jnp.array([3e10, 4e10])
        scale_factor = 1e10

        result = self.guard._safe_norm_jit(x, scale_factor)
        expected = 5e10
        assert_allclose(float(result), expected, rtol=1e-6)

    def test_safe_norm_prevents_overflow(self):
        """Test safe norm prevents overflow with large values."""
        x = jnp.array([1e154, 1e154])  # Would overflow without scaling
        scale_factor = 1e154

        result = self.guard._safe_norm_jit(x, scale_factor)
        expected = np.sqrt(2) * 1e154
        assert np.isfinite(result)
        assert_allclose(float(result), expected, rtol=1e-6)


class TestNumericalStabilityGuardEdgeCases:
    """Edge case tests for NumericalStabilityGuard."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = NumericalStabilityGuard()

    def test_empty_jacobian(self):
        """Test with empty Jacobian (edge case)."""
        J = jnp.zeros((0, 0))
        # Should handle gracefully or raise informative error
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _J_fixed, issues = self.guard.check_and_fix_jacobian(J)
            # If it works, verify issues dict exists
            assert isinstance(issues, dict)
        except (ValueError, IndexError):
            # Acceptable to raise for empty input
            pass

    def test_single_element_jacobian(self):
        """Test with 1x1 Jacobian."""
        J = jnp.array([[5.0]])
        J_fixed, _issues = self.guard.check_and_fix_jacobian(J)

        assert_allclose(np.asarray(J_fixed), np.asarray(J), rtol=1e-12)

    def test_tall_jacobian(self):
        """Test with tall Jacobian (m >> n)."""
        J = jnp.ones((1000, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            J_fixed, _issues = self.guard.check_and_fix_jacobian(J)

        assert J_fixed.shape == (1000, 3)
        assert jnp.isfinite(J_fixed).all()

    def test_wide_jacobian(self):
        """Test with wide Jacobian (m << n)."""
        J = jnp.ones((3, 100))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            J_fixed, _issues = self.guard.check_and_fix_jacobian(J)

        assert J_fixed.shape == (3, 100)
        assert jnp.isfinite(J_fixed).all()

    def test_mixed_nan_inf_values(self):
        """Test Jacobian with both NaN and Inf."""
        J = jnp.array([[np.nan, np.inf], [-np.inf, 1.0]])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            J_fixed, issues = self.guard.check_and_fix_jacobian(J)

        assert jnp.isfinite(J_fixed).all()
        assert issues["has_nan"] is True
        assert issues["has_inf"] is True

    def test_subnormal_values(self):
        """Test handling of subnormal (denormalized) numbers."""
        J = jnp.array([[1e-310, 1e-315], [1e-320, 1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            J_fixed, _issues = self.guard.check_and_fix_jacobian(J)

        # Subnormals should be preserved (they're valid finite numbers)
        assert jnp.isfinite(J_fixed).all()


class TestJITCompatibility:
    """Tests for JIT compatibility of guard operations."""

    def test_safe_exp_jit_multiple_calls(self):
        """Test safe_exp JIT compilation is stable across calls."""
        guard = NumericalStabilityGuard()

        x1 = jnp.array([1.0, 2.0])
        x2 = jnp.array([3.0, 4.0])

        r1 = guard._safe_exp_jit(x1)
        r2 = guard._safe_exp_jit(x2)

        assert_allclose(np.asarray(r1), np.exp([1.0, 2.0]), rtol=1e-10)
        assert_allclose(np.asarray(r2), np.exp([3.0, 4.0]), rtol=1e-10)

    def test_check_jacobian_jit_stable(self):
        """Test check_and_fix_jacobian is stable across different inputs."""
        guard = NumericalStabilityGuard()

        J1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        J2 = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        _, issues1 = guard.check_and_fix_jacobian(J1)
        _, issues2 = guard.check_and_fix_jacobian(J2)

        # Both should succeed without errors
        assert "has_nan" in issues1
        assert "has_nan" in issues2


class TestVmapCompatibility:
    """Tests for vmap compatibility of guard operations."""

    def test_safe_exp_vmap(self):
        """Test safe_exp works with vmap."""
        from jax import vmap

        guard = NumericalStabilityGuard()

        # Batch of vectors
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        vmapped_exp = vmap(guard._safe_exp_jit)
        results = vmapped_exp(batch)

        expected = jnp.exp(batch)
        assert_allclose(np.asarray(results), np.asarray(expected), rtol=1e-10)

    def test_safe_sqrt_vmap(self):
        """Test safe_sqrt works with vmap."""
        from jax import vmap

        guard = NumericalStabilityGuard()

        batch = jnp.array([[1.0, 4.0], [9.0, 16.0]])

        vmapped_sqrt = vmap(guard._safe_sqrt_jit)
        results = vmapped_sqrt(batch)

        expected = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert_allclose(np.asarray(results), np.asarray(expected), rtol=1e-10)
