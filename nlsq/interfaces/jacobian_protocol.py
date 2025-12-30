"""Protocol definition for Jacobian computation strategies.

This module defines the JacobianProtocol that Jacobian computation
strategies should implement, enabling different approaches (autodiff,
finite differences, analytical, sparse).
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class JacobianProtocol(Protocol):
    """Protocol for Jacobian computation strategies.

    This protocol defines the interface for computing Jacobians of
    residual functions, allowing different strategies to be swapped.

    Methods
    -------
    compute(fun, x, args)
        Compute the Jacobian matrix at point x.
    """

    def compute(
        self,
        fun: Callable[..., np.ndarray],
        x: np.ndarray,
        args: tuple[Any, ...] = (),
    ) -> np.ndarray:
        """Compute Jacobian matrix.

        Parameters
        ----------
        fun : Callable
            Residual function ``f(x, *args) -> residuals``.
        x : np.ndarray
            Point at which to evaluate Jacobian.
        args : tuple
            Additional arguments to pass to fun.

        Returns
        -------
        np.ndarray
            Jacobian matrix of shape (n_residuals, n_params).
        """
        ...


@runtime_checkable
class SparseJacobianProtocol(Protocol):
    """Protocol for sparse Jacobian computation.

    Extended protocol for Jacobians that may have sparse structure,
    returning sparse matrices when beneficial.
    """

    def compute(
        self,
        fun: Callable[..., np.ndarray],
        x: np.ndarray,
        args: tuple[Any, ...] = (),
    ) -> np.ndarray:
        """Compute Jacobian matrix (possibly sparse)."""
        ...

    def compute_sparse(
        self,
        fun: Callable[..., np.ndarray],
        x: np.ndarray,
        args: tuple[Any, ...] = (),
        sparsity_threshold: float = 0.5,
    ) -> Any:
        """Compute sparse Jacobian if beneficial.

        Parameters
        ----------
        fun : Callable
            Residual function.
        x : np.ndarray
            Point at which to evaluate.
        args : tuple
            Additional arguments.
        sparsity_threshold : float
            Fraction of zeros above which to use sparse format.

        Returns
        -------
        np.ndarray or scipy.sparse matrix
            Jacobian in appropriate format.
        """
        ...

    @property
    def sparsity_pattern(self) -> np.ndarray | None:
        """Known sparsity pattern, if any."""
        ...


class AutodiffJacobian:
    """Jacobian computation using JAX automatic differentiation.

    This is the default Jacobian computation strategy for NLSQ.

    Parameters
    ----------
    use_forward_mode : bool, default=False
        Use forward-mode AD (vmap of jvp) instead of reverse-mode.
        Forward mode is faster when n_params << n_residuals.
    """

    __slots__ = ("_use_forward_mode",)

    def __init__(self, use_forward_mode: bool = False) -> None:
        self._use_forward_mode = use_forward_mode

    def compute(
        self,
        fun: Callable[..., np.ndarray],
        x: np.ndarray,
        args: tuple[Any, ...] = (),
    ) -> np.ndarray:
        """Compute Jacobian using JAX autodiff."""
        import jax
        import jax.numpy as jnp

        x_jax = jnp.asarray(x)

        def fun_x(params: jnp.ndarray):
            return fun(params, *args)

        jacobian = jax.jacfwd(fun_x)(x_jax)
        return np.asarray(jacobian)
