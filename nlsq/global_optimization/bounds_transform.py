"""Sigmoid bound transformation for CMA-ES optimization.

CMA-ES operates in unbounded space. This module provides smooth, differentiable
transformations between bounded parameter space and unbounded CMA-ES space.

The sigmoid transformation is used:
- Forward: x_bounded = lb + (ub - lb) * sigmoid(x_unbounded)
- Inverse: x_unbounded = logit((x_bounded - lb) / (ub - lb))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array

__all__ = [
    "compute_default_popsize",
    "transform_from_bounds",
    "transform_to_bounds",
]


def transform_to_bounds(
    x_unbounded: Array,
    lower_bounds: Array,
    upper_bounds: Array,
) -> Array:
    """Transform unbounded CMA-ES samples to bounded parameter space.

    Uses sigmoid transformation for smooth, differentiable bounds handling:
    x_bounded = lb + (ub - lb) * sigmoid(x_unbounded)

    Parameters
    ----------
    x_unbounded : Array
        Unbounded parameter values from CMA-ES, shape (n_params,) or
        (population_size, n_params).
    lower_bounds : Array
        Lower bounds for each parameter, shape (n_params,).
    upper_bounds : Array
        Upper bounds for each parameter, shape (n_params,).

    Returns
    -------
    Array
        Bounded parameter values, same shape as x_unbounded.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.array([0.0, 1.0, -1.0])
    >>> lb = jnp.array([0.0, 0.0, 0.0])
    >>> ub = jnp.array([1.0, 10.0, 100.0])
    >>> transform_to_bounds(x, lb, ub)
    Array([0.5, 7.310586, 26.894142], dtype=float32)
    """
    # Sigmoid: 1 / (1 + exp(-x))
    sigmoid_x = jnp.where(
        x_unbounded >= 0,
        1.0 / (1.0 + jnp.exp(-x_unbounded)),
        jnp.exp(x_unbounded) / (1.0 + jnp.exp(x_unbounded)),
    )

    # Map to bounds: lb + (ub - lb) * sigmoid(x)
    return lower_bounds + (upper_bounds - lower_bounds) * sigmoid_x


def transform_from_bounds(
    x_bounded: Array,
    lower_bounds: Array,
    upper_bounds: Array,
    epsilon: float = 1e-8,
) -> Array:
    """Transform bounded parameters to unbounded CMA-ES space.

    Uses logit (inverse sigmoid) transformation:
    x_unbounded = log(ratio / (1 - ratio)) where ratio = (x - lb) / (ub - lb)

    Parameters
    ----------
    x_bounded : Array
        Bounded parameter values, shape (n_params,) or (population_size, n_params).
    lower_bounds : Array
        Lower bounds for each parameter, shape (n_params,).
    upper_bounds : Array
        Upper bounds for each parameter, shape (n_params,).
    epsilon : float, optional
        Small value to prevent log(0) or log(1). Default: 1e-8.

    Returns
    -------
    Array
        Unbounded parameter values for CMA-ES, same shape as x_bounded.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.array([0.5, 5.0, 50.0])
    >>> lb = jnp.array([0.0, 0.0, 0.0])
    >>> ub = jnp.array([1.0, 10.0, 100.0])
    >>> transform_from_bounds(x, lb, ub)
    Array([0., 0., 0.], dtype=float32)
    """
    # Normalize to [0, 1]
    ratio = (x_bounded - lower_bounds) / (upper_bounds - lower_bounds)

    # Clamp to avoid log(0) or log(inf)
    ratio_clamped = jnp.clip(ratio, epsilon, 1.0 - epsilon)

    # Logit: log(ratio / (1 - ratio))
    return jnp.log(ratio_clamped / (1.0 - ratio_clamped))


def compute_default_popsize(n_params: int) -> int:
    """Compute default CMA-ES population size.

    Uses the standard CMA-ES formula: int(4 + 3 * log(n))

    Parameters
    ----------
    n_params : int
        Number of parameters being optimized.

    Returns
    -------
    int
        Default population size, minimum 4.

    Examples
    --------
    >>> compute_default_popsize(5)
    8
    >>> compute_default_popsize(20)
    12
    """
    import math

    return max(4, int(4 + 3 * math.log(max(1, n_params))))
