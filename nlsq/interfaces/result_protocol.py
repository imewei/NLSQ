"""Protocol definition for optimization results.

This module defines the ResultProtocol that optimization results should
implement, enabling consistent access to optimization outcomes.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ResultProtocol(Protocol):
    """Protocol for optimization results.

    This protocol defines the minimum interface that optimization results
    must implement, enabling consistent handling across different optimizers.

    Attributes
    ----------
    x : np.ndarray
        Optimal parameter values.
    success : bool
        Whether optimization converged successfully.
    message : str
        Description of termination reason.
    """

    @property
    def x(self) -> np.ndarray:
        """Optimal parameter values."""
        ...

    @property
    def success(self) -> bool:
        """Whether optimization converged successfully."""
        ...

    @property
    def message(self) -> str:
        """Description of termination reason."""
        ...


@runtime_checkable
class LeastSquaresResultProtocol(Protocol):
    """Protocol for least squares optimization results.

    Extended protocol with additional attributes specific to least squares.

    Attributes
    ----------
    x : np.ndarray
        Optimal parameter values.
    cost : float
        Final cost value (0.5 * sum(residuals**2)).
    fun : np.ndarray
        Residuals at the solution.
    jac : np.ndarray
        Jacobian at the solution.
    success : bool
        Whether optimization converged successfully.
    message : str
        Description of termination reason.
    nfev : int
        Number of function evaluations.
    njev : int
        Number of Jacobian evaluations.
    """

    @property
    def x(self) -> np.ndarray:
        """Optimal parameter values."""
        ...

    @property
    def cost(self) -> float:
        """Final cost value."""
        ...

    @property
    def fun(self) -> np.ndarray:
        """Residuals at the solution."""
        ...

    @property
    def jac(self) -> np.ndarray:
        """Jacobian at the solution."""
        ...

    @property
    def success(self) -> bool:
        """Whether optimization converged successfully."""
        ...

    @property
    def message(self) -> str:
        """Description of termination reason."""
        ...

    @property
    def nfev(self) -> int:
        """Number of function evaluations."""
        ...

    @property
    def njev(self) -> int:
        """Number of Jacobian evaluations."""
        ...


@runtime_checkable
class CurveFitResultProtocol(Protocol):
    """Protocol for curve fit results.

    Extended protocol with covariance information for curve fitting.

    Attributes
    ----------
    popt : np.ndarray
        Optimal parameter values.
    pcov : np.ndarray
        Covariance matrix of parameters.
    success : bool
        Whether fitting converged successfully.
    """

    @property
    def popt(self) -> np.ndarray:
        """Optimal parameter values."""
        ...

    @property
    def pcov(self) -> np.ndarray:
        """Covariance matrix of parameters."""
        ...

    @property
    def success(self) -> bool:
        """Whether fitting converged successfully."""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute by name with default."""
        ...
