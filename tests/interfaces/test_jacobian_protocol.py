"""Protocol contract tests for JacobianProtocol.

This module tests that Jacobian implementations conform to the
JacobianProtocol defined in nlsq.interfaces.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from nlsq.interfaces.jacobian_protocol import (
    AutodiffJacobian,
    JacobianProtocol,
    SparseJacobianProtocol,
)


class TestJacobianProtocolDefinition:
    """Test that JacobianProtocol is correctly defined."""

    def test_protocol_is_runtime_checkable(self):
        """JacobianProtocol should be runtime_checkable."""

        class MockJacobian:
            def compute(
                self,
                fun: Callable[..., np.ndarray],
                x: np.ndarray,
                args: tuple[Any, ...] = (),
            ) -> np.ndarray:
                return np.eye(len(x))

        assert isinstance(MockJacobian(), JacobianProtocol)

    def test_protocol_requires_compute_method(self):
        """Classes missing compute() should not satisfy protocol."""

        class IncompleteJacobian:
            def jacobian(self, fun, x):
                pass

        assert not isinstance(IncompleteJacobian(), JacobianProtocol)


class TestAutodiffJacobianConformance:
    """Test that AutodiffJacobian conforms to JacobianProtocol."""

    def test_autodiff_jacobian_satisfies_protocol(self):
        """AutodiffJacobian should satisfy JacobianProtocol."""
        jac = AutodiffJacobian()
        assert isinstance(jac, JacobianProtocol)

    def test_autodiff_jacobian_compute(self):
        """AutodiffJacobian should correctly compute Jacobians."""
        import jax.numpy as jnp

        jac = AutodiffJacobian()

        # Simple linear function using JAX arrays
        def linear_fun(x):
            return jnp.array([2 * x[0], 3 * x[1]])

        x = np.array([1.0, 2.0])
        J = jac.compute(linear_fun, x)

        # Expected Jacobian: [[2, 0], [0, 3]]
        expected = np.array([[2.0, 0.0], [0.0, 3.0]])
        np.testing.assert_allclose(J, expected, rtol=1e-5)

    def test_autodiff_jacobian_with_args(self):
        """AutodiffJacobian should pass args to the function."""
        import jax.numpy as jnp

        jac = AutodiffJacobian()

        # Function with args using JAX arrays
        def scaled_fun(x, a, b):
            return jnp.array([a * x[0], b * x[1]])

        x = np.array([1.0, 2.0])
        J = jac.compute(scaled_fun, x, args=(2.0, 3.0))

        expected = np.array([[2.0, 0.0], [0.0, 3.0]])
        np.testing.assert_allclose(J, expected, rtol=1e-5)

    def test_autodiff_jacobian_nonlinear(self):
        """AutodiffJacobian should handle nonlinear functions."""
        import jax.numpy as jnp

        jac = AutodiffJacobian()

        # Nonlinear function using JAX arrays
        def nonlinear_fun(x):
            return jnp.array([x[0] ** 2, x[0] * x[1]])

        x = np.array([2.0, 3.0])
        J = jac.compute(nonlinear_fun, x)

        # Jacobian: [[2*x[0], 0], [x[1], x[0]]] = [[4, 0], [3, 2]]
        expected = np.array([[4.0, 0.0], [3.0, 2.0]])
        np.testing.assert_allclose(J, expected, rtol=1e-5)


class TestSparseJacobianProtocol:
    """Test SparseJacobianProtocol requirements."""

    def test_protocol_is_runtime_checkable(self):
        """SparseJacobianProtocol should be runtime_checkable."""

        class MockSparseJacobian:
            def compute(self, fun, x, args=()):
                return np.eye(len(x))

            def compute_sparse(self, fun, x, args=(), sparsity_threshold=0.5):
                return np.eye(len(x))

            @property
            def sparsity_pattern(self):
                return None

        assert isinstance(MockSparseJacobian(), SparseJacobianProtocol)

    def test_protocol_requires_all_methods(self):
        """SparseJacobianProtocol requires compute, compute_sparse, sparsity_pattern."""

        class MissingComputeSparse:
            def compute(self, fun, x, args=()):
                return np.eye(len(x))

            @property
            def sparsity_pattern(self):
                return None

        assert not isinstance(MissingComputeSparse(), SparseJacobianProtocol)


class TestJacobianUsagePatterns:
    """Test common Jacobian usage patterns."""

    def test_jacobian_shape(self):
        """Jacobian should have shape (n_residuals, n_params)."""
        import jax.numpy as jnp

        jac = AutodiffJacobian()

        # Function: R^3 -> R^5 using JAX arrays
        def fun(x):
            return jnp.array([x[0], x[1], x[2], x[0] + x[1], x[1] + x[2]])

        x = np.array([1.0, 2.0, 3.0])
        J = jac.compute(fun, x)

        assert J.shape == (5, 3)

    def test_jacobian_with_scalar_output(self):
        """Jacobian should work with scalar-like output."""
        import jax.numpy as jnp

        jac = AutodiffJacobian()

        # Function: R^2 -> R^1 using JAX arrays
        def fun(x):
            return jnp.array([x[0] ** 2 + x[1] ** 2])

        x = np.array([1.0, 2.0])
        J = jac.compute(fun, x)

        assert J.shape == (1, 2)
        # Gradient: [2*x[0], 2*x[1]] = [2, 4]
        np.testing.assert_allclose(J[0], [2.0, 4.0], rtol=1e-5)


class TestCustomJacobianImplementation:
    """Test custom Jacobian implementations."""

    def test_finite_difference_jacobian(self):
        """Custom finite difference Jacobian should work."""

        class FiniteDifferenceJacobian:
            def __init__(self, eps: float = 1e-7):
                self.eps = eps

            def compute(
                self,
                fun: Callable[..., np.ndarray],
                x: np.ndarray,
                args: tuple[Any, ...] = (),
            ) -> np.ndarray:
                f0 = fun(x, *args)
                n_residuals = len(f0)
                n_params = len(x)
                J = np.zeros((n_residuals, n_params))

                for j in range(n_params):
                    x_plus = x.copy()
                    x_plus[j] += self.eps
                    f_plus = fun(x_plus, *args)
                    J[:, j] = (f_plus - f0) / self.eps

                return J

        fd_jac = FiniteDifferenceJacobian()
        assert isinstance(fd_jac, JacobianProtocol)

        def fun(x):
            return np.array([x[0] ** 2, x[1] ** 2])

        x = np.array([2.0, 3.0])
        J = fd_jac.compute(fun, x)

        expected = np.array([[4.0, 0.0], [0.0, 6.0]])
        np.testing.assert_allclose(J, expected, rtol=1e-4)
