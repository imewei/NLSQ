"""Base class for optimization algorithms in NLSQ."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

_jax_config = JAXConfig()

from nlsq.core._optimize import OptimizeResult
from nlsq.utils.logging import get_logger


class OptimizerBase(ABC):
    """Abstract base class for optimization algorithms.

    This class provides a common interface for optimization algorithms
    used in NLSQ. It includes common functionality like logging,
    result creation, and defines the interface that subclasses must implement.
    """

    def __init__(self, name: str = "optimizer"):
        """Initialize the optimizer base class.

        Parameters
        ----------
        name : str
            Name of the optimizer for logging purposes
        """
        self.name = name
        self.logger = get_logger(f"optimizer.{name}")
        self._nfev = 0
        self._njev = 0

    @abstractmethod
    def optimize(
        self,
        fun: Callable,
        x0: np.ndarray,
        jac: Callable | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | tuple[float, float] = (-np.inf, np.inf),
        **kwargs,
    ) -> OptimizeResult:
        """Perform optimization.

        This is the main optimization method that must be implemented
        by subclasses.

        Parameters
        ----------
        fun : callable
            The objective function to minimize
        x0 : np.ndarray
            Initial guess for parameters
        jac : callable, optional
            Jacobian function
        bounds : tuple of arrays
            Lower and upper bounds for parameters
        **kwargs
            Additional optimization parameters

        Returns
        -------
        OptimizeResult
            The optimization result
        """

    def reset_counters(self):
        """Reset function evaluation counters."""
        self._nfev = 0
        self._njev = 0
        self.logger.debug("Counters reset", nfev=0, njev=0)

    def increment_nfev(self, count: int = 1):
        """Increment function evaluation counter."""
        self._nfev += count

    def increment_njev(self, count: int = 1):
        """Increment Jacobian evaluation counter."""
        self._njev += count

    @property
    def nfev(self) -> int:
        """Number of function evaluations."""
        return self._nfev

    @property
    def njev(self) -> int:
        """Number of Jacobian evaluations."""
        return self._njev

    def create_result(
        self,
        x: np.ndarray,
        fun: np.ndarray,
        jac: np.ndarray | None = None,
        cost: float | None = None,
        status: int = 0,
        message: str = "",
        optimality: float | None = None,
        active_mask: np.ndarray | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Create a standardized optimization result.

        Parameters
        ----------
        x : np.ndarray
            Optimized parameters
        fun : np.ndarray
            Function values at optimized parameters
        jac : np.ndarray, optional
            Jacobian at optimized parameters
        cost : float, optional
            Final cost value
        status : int
            Termination status code
        message : str
            Termination message
        optimality : float, optional
            First-order optimality measure
        active_mask : np.ndarray, optional
            Active constraints mask
        **kwargs
            Additional result attributes

        Returns
        -------
        OptimizeResult
            Standardized optimization result
        """
        result = OptimizeResult(
            x=x,
            fun=fun,
            jac=jac,
            cost=cost,
            nfev=self.nfev,
            njev=self.njev,
            status=status,
            message=message,
            optimality=optimality,
            active_mask=active_mask,
            **kwargs,
        )

        self.logger.debug(
            "Result created",
            final_cost=cost,
            nfev=self.nfev,
            njev=self.njev,
            status=status,
        )

        return result

    def check_convergence(
        self,
        actual_reduction: float,
        cost: float,
        step_norm: float,
        x_norm: float,
        ratio: float,
        ftol: float,
        xtol: float,
    ) -> int | None:
        """Check convergence criteria.

        Parameters
        ----------
        actual_reduction : float
            Actual reduction in cost function
        cost : float
            Current cost value
        step_norm : float
            Norm of the optimization step
        x_norm : float
            Norm of the parameter vector
        ratio : float
            Ratio of actual to predicted reduction
        ftol : float
            Function tolerance
        xtol : float
            Parameter tolerance

        Returns
        -------
        int or None
            Termination status code if converged, None otherwise
        """
        # Check function tolerance
        if actual_reduction < ftol * cost:
            self.logger.debug(
                "Convergence: function tolerance satisfied",
                actual_reduction=actual_reduction,
                ftol=ftol,
                cost=cost,
            )
            return 2

        # Check parameter tolerance
        if step_norm < xtol * (xtol + x_norm):
            self.logger.debug(
                "Convergence: parameter tolerance satisfied",
                step_norm=step_norm,
                xtol=xtol,
                x_norm=x_norm,
            )
            return 3

        # Check combined criteria
        if actual_reduction < ftol * cost and step_norm < xtol * (xtol + x_norm):
            self.logger.debug("Convergence: both tolerances satisfied")
            return 4

        return None

    def log_iteration(
        self,
        iteration: int,
        cost: float,
        gradient_norm: float | None = None,
        step_size: float | None = None,
        **kwargs,
    ):
        """Log optimization iteration details.

        Parameters
        ----------
        iteration : int
            Iteration number
        cost : float
            Current cost value
        gradient_norm : float, optional
            Gradient norm
        step_size : float, optional
            Step size
        **kwargs
            Additional logging parameters
        """
        self.logger.optimization_step(
            iteration=iteration,
            cost=cost,
            gradient_norm=gradient_norm,
            step_size=step_size,
            nfev=self.nfev,
            **kwargs,
        )

    def log_convergence(
        self,
        reason: str,
        iterations: int,
        final_cost: float,
        time_elapsed: float | None = None,
        **kwargs,
    ):
        """Log convergence information.

        Parameters
        ----------
        reason : str
            Convergence reason
        iterations : int
            Number of iterations
        final_cost : float
            Final cost value
        time_elapsed : float, optional
            Total optimization time
        **kwargs
            Additional logging parameters
        """
        self.logger.convergence(
            reason=reason,
            iterations=iterations,
            final_cost=final_cost,
            time_elapsed=time_elapsed,
            **kwargs,
        )


class TrustRegionOptimizerBase(OptimizerBase):
    """Base class for trust region optimization algorithms.

    This class extends OptimizerBase with trust region specific functionality
    like trust region radius management and step acceptance criteria.
    """

    def __init__(self, name: str = "trust_region"):
        """Initialize trust region optimizer."""
        super().__init__(name)
        self._trust_radius = 1.0

    @property
    def trust_radius(self) -> float:
        """Current trust region radius."""
        return self._trust_radius

    @trust_radius.setter
    def trust_radius(self, value: float):
        """Set trust region radius."""
        self._trust_radius = max(0.0, value)

    def update_trust_radius(
        self,
        Delta: float,
        actual_reduction: float,
        predicted_reduction: float,
        step_norm: float,
        step_at_boundary: bool = False,
    ) -> tuple[float, float]:
        """Update trust region radius based on step quality.

        Parameters
        ----------
        Delta : float
            Current trust region radius
        actual_reduction : float
            Actual reduction in objective
        predicted_reduction : float
            Predicted reduction from model
        step_norm : float
            Norm of the step taken
        step_at_boundary : bool
            Whether step reached trust region boundary

        Returns
        -------
        tuple
            New trust region radius and reduction ratio
        """
        if predicted_reduction <= 0:
            ratio = 0.0
        else:
            ratio = actual_reduction / predicted_reduction

        if ratio < 0.25:
            Delta_new = 0.25 * step_norm
        elif ratio > 0.75 and step_at_boundary:
            Delta_new = min(2.0 * Delta, 1000.0)
        else:
            Delta_new = Delta

        self.trust_radius = Delta_new

        self.logger.debug(
            "Trust radius updated",
            old_radius=Delta,
            new_radius=Delta_new,
            ratio=ratio,
            step_norm=step_norm,
            at_boundary=step_at_boundary,
        )

        return Delta_new, ratio

    def step_accepted(self, ratio: float, threshold: float = 1e-4) -> bool:
        """Check if optimization step should be accepted.

        Parameters
        ----------
        ratio : float
            Ratio of actual to predicted reduction
        threshold : float
            Minimum acceptable ratio

        Returns
        -------
        bool
            True if step should be accepted
        """
        accepted = ratio > threshold
        self.logger.debug(
            "Step acceptance check", ratio=ratio, threshold=threshold, accepted=accepted
        )
        return accepted
