"""Regression tests for the 2026-11-04 review (CMA-ES bounds)."""

import jax.numpy as jnp
import pytest

from nlsq.global_optimization.cmaes_optimizer import _require_finite_bounds


@pytest.mark.parametrize(
    "lb, ub",
    [([10.0, 2.0], [0.1, 0.01]), ([1.0, 0.0], [1.0, 1.0])],
)
def test_inverted_or_degenerate_bounds_raise(lb, ub):
    with pytest.raises(ValueError, match="lower < upper"):
        _require_finite_bounds(jnp.asarray(lb), jnp.asarray(ub))


def test_valid_bounds_pass():
    _require_finite_bounds(jnp.asarray([0.0, -1.0]), jnp.asarray([1.0, 2.0]))
