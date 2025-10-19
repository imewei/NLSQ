"""Type aliases and protocols for NLSQ.

This module provides type hints for the NLSQ public API to improve IDE support,
documentation, and static type checking with mypy.

Note: These types are primarily for documentation and tooling. Python's duck typing
means functions will work with any compatible objects at runtime.
"""

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias

import numpy as np
import jax.numpy as jnp

# Array-like types
ArrayLike: TypeAlias = np.ndarray | jnp.ndarray | list | tuple
"""Array-like type that can be converted to numpy/JAX arrays."""

FloatArray: TypeAlias = np.ndarray
"""NumPy array of floating point numbers."""

JAXArray: TypeAlias = jnp.ndarray
"""JAX array for GPU/TPU-accelerated computations."""

# Function types
ModelFunction: TypeAlias = Callable[..., ArrayLike]
"""Model function f(x, *params) -> y_pred.

The model function takes independent variable(s) x and fit parameters,
returning predicted dependent variable values.

Examples:
    - Linear: f(x, a, b) = a*x + b
    - Exponential: f(x, a, b) = a * exp(-b * x)
    - Multi-parameter: f(x, p1, p2, ..., pN) = ...
"""

JacobianFunction: TypeAlias = Callable[..., ArrayLike]
"""Jacobian function jac(x, *params) -> J.

The Jacobian function computes the matrix of partial derivatives:
J[i, j] = ∂f[i]/∂params[j]

Parameters:
    x: Independent variable(s)
    *params: Fit parameters

Returns:
    J: Jacobian matrix of shape (m, n) where m = len(f(x)) and n = len(params)
"""

CallbackFunction: TypeAlias = Callable[[FloatArray, FloatArray], bool | None]
"""Callback function for monitoring optimization progress.

Parameters:
    params: Current parameter estimates
    residuals: Current residual values

Returns:
    True to stop optimization, False/None to continue
"""

LossFunction: TypeAlias = Callable[[FloatArray], FloatArray]
"""Loss function rho(z) for robust fitting.

Robust loss functions reduce the influence of outliers by applying
a non-linear transformation to residuals.

Parameters:
    z: Squared residuals (z = residuals**2)

Returns:
    rho: Transformed residuals for robust fitting
"""

# Bounds types
BoundsTuple: TypeAlias = tuple[ArrayLike, ArrayLike]
"""Parameter bounds as (lower, upper) tuple.

Examples:
    - Unbounded: (-np.inf, np.inf)
    - Lower only: (0, np.inf) for positive parameters
    - Both: ([-1, 0], [1, 10]) for constrained parameters
"""

# Result types (using dict for backward compatibility)
OptimizeResultDict: TypeAlias = dict[str, Any]
"""Optimization result dictionary with parameters and diagnostics.

Common fields:
    - x: Optimized parameters
    - success: Whether optimization converged
    - message: Optimization status message
    - fun: Final residual values (optional)
    - jac: Final Jacobian (optional)
    - cost: Final cost function value
    - optimality: Final gradient norm
    - nfev: Number of function evaluations
    - njev: Number of Jacobian evaluations (optional)
"""

# Configuration types
MethodLiteral: TypeAlias = str  # "trf" | "dogbox" | "lm"
"""Optimization method name.

Options:
    - "trf": Trust Region Reflective (default, supports bounds)
    - "dogbox": Dogleg algorithm for box-constrained problems
    - "lm": Levenberg-Marquardt (unconstrained only, faster)
"""

SolverLiteral: TypeAlias = str  # "exact" | "lsmr"
"""Linear solver for trust region subproblems.

Options:
    - "exact": Direct solver using SVD (default, more accurate)
    - "lsmr": Iterative solver (faster for large problems)
"""


# Protocols for structural typing
class HasShape(Protocol):
    """Protocol for objects with a shape attribute."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""
        ...


class SupportsFloat(Protocol):
    """Protocol for objects that can be converted to float."""

    def __float__(self) -> float:
        """Convert to float."""
        ...


# Re-export commonly used types from dependencies
__all__ = [
    # Array types
    "ArrayLike",
    "FloatArray",
    "JAXArray",
    # Function types
    "ModelFunction",
    "JacobianFunction",
    "CallbackFunction",
    "LossFunction",
    # Bounds and results
    "BoundsTuple",
    "OptimizeResultDict",
    # Method/solver literals
    "MethodLiteral",
    "SolverLiteral",
    # Protocols
    "HasShape",
    "SupportsFloat",
]
