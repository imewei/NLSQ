"""Unit tests for JIT-compiled bounds helper functions in common_jax.py.

Tests CL_scaling_vector_jax, in_bounds_jax, and make_strictly_feasible_jax
against their NumPy equivalents in common_scipy.py for numerical equivalence.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.common_jax import (
    CL_scaling_vector_jax,
    in_bounds_jax,
    make_strictly_feasible_jax,
)
from nlsq.common_scipy import CL_scaling_vector, in_bounds, make_strictly_feasible


class TestCLScalingVectorJax:
    """Test CL_scaling_vector_jax matches NumPy version."""

    def _compare(self, x, g, lb, ub):
        v_jax, dv_jax = CL_scaling_vector_jax(
            jnp.array(x), jnp.array(g), jnp.array(lb), jnp.array(ub)
        )
        v_np, dv_np = CL_scaling_vector(
            np.array(x, dtype=float),
            np.array(g, dtype=float),
            np.array(lb, dtype=float),
            np.array(ub, dtype=float),
        )
        np.testing.assert_allclose(v_jax, v_np, atol=1e-15)
        np.testing.assert_allclose(dv_jax, dv_np, atol=1e-15)

    def test_basic(self):
        self._compare(
            x=[1.0, 2.0, 3.0],
            g=[-0.5, 0.3, 0.0],
            lb=[0.0, 0.0, 0.0],
            ub=[5.0, 5.0, 5.0],
        )

    def test_all_interior(self):
        """g=0 everywhere → v=1, dv=0."""
        self._compare(x=[1.0, 2.0], g=[0.0, 0.0], lb=[0.0, 0.0], ub=[5.0, 5.0])

    def test_negative_gradient_upper_bound(self):
        """g<0 with finite ub → v = ub - x, dv = -1."""
        self._compare(x=[1.0], g=[-1.0], lb=[0.0], ub=[5.0])

    def test_positive_gradient_lower_bound(self):
        """g>0 with finite lb → v = x - lb, dv = 1."""
        self._compare(x=[3.0], g=[1.0], lb=[1.0], ub=[5.0])

    def test_infinite_upper_bound(self):
        """g<0 but ub=inf → v=1 (no finite bound to measure distance to)."""
        self._compare(x=[1.0], g=[-1.0], lb=[0.0], ub=[np.inf])

    def test_infinite_lower_bound(self):
        """g>0 but lb=-inf → v=1."""
        self._compare(x=[1.0], g=[1.0], lb=[-np.inf], ub=[5.0])

    def test_both_infinite(self):
        """Both bounds infinite → v=1 regardless of gradient."""
        self._compare(x=[1.0], g=[-1.0], lb=[-np.inf], ub=[np.inf])

    def test_on_lower_bound(self):
        """x == lb, g > 0 → v = 0, dv = 1."""
        self._compare(x=[0.0], g=[1.0], lb=[0.0], ub=[5.0])

    def test_on_upper_bound(self):
        """x == ub, g < 0 → v = 0, dv = -1."""
        self._compare(x=[5.0], g=[-1.0], lb=[0.0], ub=[5.0])

    def test_mixed_cases(self):
        """Multiple parameters with different gradient/bound combinations."""
        self._compare(
            x=[1.0, 2.0, 3.0, 4.0],
            g=[-1.0, 1.0, 0.0, -0.5],
            lb=[-np.inf, 0.0, 0.0, 0.0],
            ub=[10.0, np.inf, 5.0, 8.0],
        )


class TestInBoundsJax:
    """Test in_bounds_jax matches NumPy version."""

    def test_inside(self):
        x = jnp.array([1.0, 2.0, 3.0])
        lb = jnp.array([0.0, 0.0, 0.0])
        ub = jnp.array([5.0, 5.0, 5.0])
        assert bool(in_bounds_jax(x, lb, ub)) is True
        assert in_bounds(np.array(x), np.array(lb), np.array(ub))

    def test_outside_upper(self):
        x = jnp.array([6.0, 2.0, 3.0])
        lb = jnp.array([0.0, 0.0, 0.0])
        ub = jnp.array([5.0, 5.0, 5.0])
        assert bool(in_bounds_jax(x, lb, ub)) is False

    def test_outside_lower(self):
        x = jnp.array([-1.0, 2.0, 3.0])
        lb = jnp.array([0.0, 0.0, 0.0])
        ub = jnp.array([5.0, 5.0, 5.0])
        assert bool(in_bounds_jax(x, lb, ub)) is False

    def test_on_boundary(self):
        x = jnp.array([0.0, 5.0])
        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([5.0, 5.0])
        assert bool(in_bounds_jax(x, lb, ub)) is True

    def test_scalar(self):
        assert (
            bool(in_bounds_jax(jnp.array([3.0]), jnp.array([0.0]), jnp.array([5.0])))
            is True
        )
        assert (
            bool(in_bounds_jax(jnp.array([6.0]), jnp.array([0.0]), jnp.array([5.0])))
            is False
        )

    def test_infinite_bounds(self):
        x = jnp.array([1e10, -1e10])
        lb = jnp.array([-jnp.inf, -jnp.inf])
        ub = jnp.array([jnp.inf, jnp.inf])
        assert bool(in_bounds_jax(x, lb, ub)) is True


class TestMakeStrictlyFeasibleJax:
    """Test make_strictly_feasible_jax matches NumPy version (rstep=0)."""

    def _compare(self, x, lb, ub):
        x_jnp, lb_jnp, ub_jnp = jnp.array(x), jnp.array(lb), jnp.array(ub)
        x_np = np.array(x, dtype=float)
        lb_np, ub_np = np.array(lb, dtype=float), np.array(ub, dtype=float)

        result_jax = make_strictly_feasible_jax(x_jnp, lb_jnp, ub_jnp)
        result_np = make_strictly_feasible(x_np, lb_np, ub_np, rstep=0)

        # Both must be strictly inside bounds (the essential contract)
        assert np.all(np.asarray(result_jax) >= lb_np)
        assert np.all(np.asarray(result_jax) <= ub_np)

        # Interior points must match exactly
        interior = (x_np > lb_np) & (x_np < ub_np)
        np.testing.assert_allclose(
            np.asarray(result_jax)[interior], result_np[interior], atol=1e-15
        )

        # Boundary points: both versions nudge inward, but JAX uses a DAZ-safe
        # guard (tiny instead of nextafter-denormal) on CPU. Verify both are
        # strictly inside and close to the boundary.
        boundary = ~interior
        if np.any(boundary):
            jax_boundary = np.asarray(result_jax)[boundary]
            assert np.all(jax_boundary > lb_np[boundary]) or np.all(
                lb_np[boundary] == ub_np[boundary]
            )
            assert np.all(jax_boundary < ub_np[boundary]) or np.all(
                lb_np[boundary] == ub_np[boundary]
            )

    def test_interior_point_unchanged(self):
        """Interior point should not be modified."""
        x = jnp.array([1.0, 2.0, 3.0])
        lb = jnp.array([0.0, 0.0, 0.0])
        ub = jnp.array([5.0, 5.0, 5.0])
        result = make_strictly_feasible_jax(x, lb, ub)
        np.testing.assert_array_equal(result, x)

    def test_on_lower_bound(self):
        """Point on lower bound should be nudged inside."""
        self._compare(x=[0.0, 2.0], lb=[0.0, 0.0], ub=[5.0, 5.0])

    def test_on_upper_bound(self):
        """Point on upper bound should be nudged inside."""
        self._compare(x=[5.0, 2.0], lb=[0.0, 0.0], ub=[5.0, 5.0])

    def test_on_both_bounds(self):
        """Points on both bounds simultaneously."""
        self._compare(x=[0.0, 5.0], lb=[0.0, 0.0], ub=[5.0, 5.0])

    def test_tight_bounds(self):
        """lb == ub → midpoint (0.5 * (lb + ub))."""
        self._compare(x=[1.0, 2.0], lb=[1.0, 2.0], ub=[1.0, 2.0])

    def test_near_tight_bounds(self):
        """Very close bounds where nextafter may overshoot."""
        eps = np.finfo(float).tiny
        self._compare(x=[1.0], lb=[1.0], ub=[1.0 + eps])

    def test_mixed_boundary_interior(self):
        """Mix of boundary and interior points."""
        self._compare(
            x=[0.0, 2.5, 5.0, 3.0],
            lb=[0.0, 0.0, 0.0, 0.0],
            ub=[5.0, 5.0, 5.0, 5.0],
        )

    def test_result_strictly_inside(self):
        """All results must be strictly lb < x < ub (when lb != ub)."""
        x = jnp.array([0.0, 5.0, 2.5])
        lb = jnp.array([0.0, 0.0, 0.0])
        ub = jnp.array([5.0, 5.0, 5.0])
        result = make_strictly_feasible_jax(x, lb, ub)
        assert jnp.all(result > lb)
        assert jnp.all(result < ub)

    @pytest.mark.parametrize("n", [1, 10, 100])
    def test_random_boundary_points(self, n):
        """Random points on boundaries should be nudged inside."""
        rng = np.random.default_rng(42)
        lb = rng.uniform(-10, 0, size=n)
        ub = rng.uniform(0.1, 10, size=n)
        # Place half on lower bound, half on upper
        x = np.where(np.arange(n) % 2 == 0, lb, ub)
        self._compare(x=x, lb=lb, ub=ub)
