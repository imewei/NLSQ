"""Protocol definition for optimizers.

This module defines the OptimizerProtocol that all optimization algorithms
should implement, enabling loose coupling and dependency injection.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Protocol for optimization algorithms.

    This protocol defines the interface that all optimization algorithms
    must implement. It allows modules to depend on the abstraction rather
    than concrete implementations, breaking circular dependencies.

    Methods
    -------
    optimize(fun, x0, args, **kwargs)
        Perform optimization on the given objective function.

    Examples
    --------
    >>> class MyOptimizer:
    ...     def optimize(self, fun, x0, args=(), **kwargs):
    ...         # Implementation
    ...         pass
    >>> isinstance(MyOptimizer(), OptimizerProtocol)
    True
    """

    def optimize(
        self,
        fun: Callable[..., np.ndarray],
        x0: np.ndarray,
        args: tuple[Any, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Perform optimization on the given objective function.

        Parameters
        ----------
        fun : Callable
            Objective function to minimize. Should return residuals array.
        x0 : np.ndarray
            Initial parameter guess.
        args : tuple
            Additional arguments to pass to the objective function.
        **kwargs : Any
            Additional keyword arguments for the optimizer.

        Returns
        -------
        Any
            Optimization result (typically an OptimizeResult-like object).
        """
        ...


@runtime_checkable
class LeastSquaresOptimizerProtocol(Protocol):
    """Protocol for least squares optimizers.

    Extended protocol for optimizers that specifically handle least squares
    problems with Jacobian computation and bounds support.
    """

    def least_squares(
        self,
        fun: Callable[..., np.ndarray],
        x0: np.ndarray,
        jac: Callable[..., np.ndarray] | str | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        args: tuple[Any, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Perform least squares optimization.

        Parameters
        ----------
        fun : Callable
            Residual function to minimize.
        x0 : np.ndarray
            Initial parameter guess.
        jac : Callable, str, or None
            Jacobian of the residual function, or 'auto' for autodiff.
        bounds : tuple or None
            (lower, upper) bounds for parameters.
        args : tuple
            Additional arguments to pass to fun and jac.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            Optimization result.
        """
        ...


@runtime_checkable
class CurveFitProtocol(Protocol):
    """Protocol for curve fitting interfaces.

    This protocol defines the interface for curve_fit-like functions,
    enabling different implementations (standard, streaming, global).
    """

    def curve_fit(
        self,
        f: Callable[..., np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a function to data.

        Parameters
        ----------
        f : Callable
            Model function ``f(x, *params) -> y``.
        xdata : np.ndarray
            Independent variable data.
        ydata : np.ndarray
            Dependent variable data.
        p0 : np.ndarray or None
            Initial parameter guess.
        sigma : np.ndarray or None
            Uncertainty in ydata.
        bounds : tuple or None
            (lower, upper) bounds for parameters.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (popt, pcov) - optimal parameters and covariance matrix.
        """
        ...
