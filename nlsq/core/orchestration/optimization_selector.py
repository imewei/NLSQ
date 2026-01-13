"""OptimizationSelector component for CurveFit decomposition.

Handles parameter detection, method selection, bounds preparation,
and solver configuration for curve fitting operations.

Reference: specs/017-curve-fit-decomposition/spec.md FR-002
"""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Literal, cast

import jax.numpy as jnp
import numpy as np

from nlsq.interfaces.orchestration_protocol import OptimizationConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

    from nlsq.types import ArrayLike


def prepare_bounds(bounds: tuple | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Prepare bounds for optimization.

    Args:
        bounds: Tuple of (lower, upper) bounds or None for unbounded
        n: Number of parameters

    Returns:
        Tuple of (lower_bounds, upper_bounds) arrays
    """
    if bounds is None or bounds == (-np.inf, np.inf):
        lb = np.full(n, -np.inf)
        ub = np.full(n, np.inf)
    else:
        lb = np.atleast_1d(np.asarray(bounds[0], dtype=float))
        ub = np.atleast_1d(np.asarray(bounds[1], dtype=float))

        # Broadcast scalar bounds to array
        if lb.size == 1 and n > 1:
            lb = np.full(n, lb[0])
        if ub.size == 1 and n > 1:
            ub = np.full(n, ub[0])

    return lb, ub


def _initialize_feasible(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Initialize parameters to feasible starting point.

    Uses midpoint for bounded parameters, 1.0 for unbounded.

    Args:
        lb: Lower bounds
        ub: Upper bounds

    Returns:
        Initial parameter array
    """
    n = len(lb)
    p0 = np.empty(n)

    for i in range(n):
        if np.isfinite(lb[i]) and np.isfinite(ub[i]):
            # Both bounds finite - use midpoint
            p0[i] = 0.5 * (lb[i] + ub[i])
        elif np.isfinite(lb[i]):
            # Only lower bound - use lb + 1
            p0[i] = lb[i] + 1.0
        elif np.isfinite(ub[i]):
            # Only upper bound - use ub - 1
            p0[i] = ub[i] - 1.0
        else:
            # Both infinite - use 1.0
            p0[i] = 1.0

    return p0


class OptimizationSelector:
    """Selector for optimization method and configuration.

    Handles:
    1. Parameter count detection from function signature
    2. Method selection based on bounds and problem type
    3. Bounds validation and preparation
    4. Initial guess generation if not provided
    5. Solver configuration validation

    Example:
        >>> selector = OptimizationSelector()
        >>> config = selector.select(
        ...     f=my_model,
        ...     xdata=x_values,
        ...     ydata=y_values,
        ...     bounds=([0, 0], [10, 10]),
        ... )
        >>> print(f"Method: {config.method}, Params: {config.n_params}")
    """

    def select(
        self,
        f: Callable[..., ArrayLike],
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        p0: ArrayLike | None = None,
        bounds: tuple[ArrayLike, ArrayLike] | None = None,
        method: str | None = None,
        jac: str | Callable | None = None,
        tr_solver: str | None = None,
        x_scale: ArrayLike | str | float = 1.0,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        max_nfev: int | None = None,
    ) -> OptimizationConfig:
        """Select optimization method and prepare configuration.

        Args:
            f: Model function to fit
            xdata: Independent variable data
            ydata: Dependent variable data
            p0: Initial parameter guess (auto-detected if None)
            bounds: Parameter bounds as (lower, upper)
            method: Optimization method ('trf', 'lm', 'dogbox', or None for auto)
            jac: Jacobian computation method
            tr_solver: Trust region solver ('exact', 'lsmr', or None for auto)
            x_scale: Parameter scaling
            ftol: Function tolerance
            xtol: Parameter tolerance
            gtol: Gradient tolerance
            max_nfev: Maximum function evaluations (auto if None)

        Returns:
            OptimizationConfig with all settings resolved

        Raises:
            ValueError: If configuration is invalid
        """
        # Convert to numpy for processing
        xdata_np = np.asarray(xdata)
        ydata_np = np.asarray(ydata)

        # Step 1: Determine parameter count
        n_params, p0_validated = self._determine_parameter_count(f, p0, xdata_np)

        # Step 2: Prepare bounds and initial guess
        lb, ub, p0_final = self._prepare_bounds_and_initial_guess(
            bounds, n_params, p0_validated
        )

        # Step 3: Select method
        selected_method = self._select_method(method)

        # Step 4: Select trust region solver
        m = len(ydata_np)
        selected_tr_solver = self._select_tr_solver(tr_solver, m, n_params)

        # Step 5: Calculate max_nfev if not provided
        if max_nfev is None:
            max_nfev = 100 * (n_params + 1)

        # Step 6: Convert to JAX arrays for output
        jnp_p0 = jnp.asarray(p0_final)
        jnp_lb = jnp.asarray(lb)
        jnp_ub = jnp.asarray(ub)

        # Handle x_scale
        if isinstance(x_scale, str):
            jnp_x_scale: jax.Array | str = x_scale
        else:
            jnp_x_scale = jnp.asarray(x_scale)

        return OptimizationConfig(
            method=selected_method,
            tr_solver=selected_tr_solver,
            n_params=n_params,
            p0=jnp_p0,
            bounds=(jnp_lb, jnp_ub),
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            jac=jac,
            x_scale=jnp_x_scale,
        )

    def detect_parameter_count(
        self,
        f: Callable[..., ArrayLike],
        xdata: ArrayLike,
    ) -> int:
        """Detect number of parameters from function signature.

        Uses inspection of function signature to determine parameter count.

        Args:
            f: Model function to analyze
            xdata: Sample data (not used currently, for future probing)

        Returns:
            Number of parameters (excluding x)

        Raises:
            ValueError: If parameter count cannot be determined
        """
        sig = signature(f)
        args = sig.parameters
        if len(args) < 2:
            msg = "Unable to determine number of fit parameters."
            raise ValueError(msg)
        return len(args) - 1

    def auto_initial_guess(
        self,
        n_params: int,
        bounds: tuple[jax.Array, jax.Array] | None,
    ) -> jax.Array:
        """Generate automatic initial parameter guess.

        Uses bounds midpoint if available, otherwise ones.

        Args:
            n_params: Number of parameters
            bounds: Parameter bounds or None

        Returns:
            Initial guess array of shape (n_params,)
        """
        if bounds is None:
            return jnp.ones(n_params)

        lb = np.asarray(bounds[0])
        ub = np.asarray(bounds[1])
        p0 = _initialize_feasible(lb, ub)
        return jnp.asarray(p0)

    def _determine_parameter_count(
        self,
        f: Callable,
        p0: ArrayLike | None,
        xdata: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray | None]:
        """Determine number of fit parameters from p0 or function signature.

        Args:
            f: The fit function
            p0: Initial parameter guess (None to detect from signature)
            xdata: Independent variable data (for auto p0 estimation)

        Returns:
            Tuple of (n_params, validated_p0)
        """
        # If p0 is explicitly provided, use it
        if p0 is not None:
            p0_arr = np.atleast_1d(np.asarray(p0))
            n = p0_arr.size
            return n, p0_arr

        # Fall back: determine n from function signature
        sig = signature(f)
        args = sig.parameters
        if len(args) < 2:
            msg = "Unable to determine number of fit parameters."
            raise ValueError(msg)
        n = len(args) - 1

        return n, None

    def _prepare_bounds_and_initial_guess(
        self, bounds: tuple | None, n: int, p0: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare bounds and initialize p0 if needed.

        Args:
            bounds: Bounds tuple (lower, upper)
            n: Number of parameters
            p0: Initial parameter guess

        Returns:
            Tuple of (lb, ub, p0)
        """
        lb, ub = prepare_bounds(bounds, n)
        if p0 is None:
            p0 = _initialize_feasible(lb, ub)
        else:
            # Clip p0 to bounds to ensure feasibility
            p0 = np.clip(p0, lb, ub)

        return lb, ub, p0

    def _select_method(self, method: str | None) -> Literal["trf", "lm", "dogbox"]:
        """Select optimization method.

        Args:
            method: Requested method or None for auto

        Returns:
            Selected method ('trf', 'lm', or 'dogbox')
        """
        if method is None:
            return "trf"

        valid_methods = {"trf", "lm", "dogbox"}
        if method not in valid_methods:
            msg = f"Invalid method '{method}'. Must be one of {valid_methods}."
            raise ValueError(msg)

        return cast(Literal["trf", "lm", "dogbox"], method)

    def _select_tr_solver(
        self, tr_solver: str | None, m: int, n: int
    ) -> Literal["exact", "lsmr"] | None:
        """Select trust region solver based on problem size.

        Args:
            tr_solver: Requested solver or None for auto
            m: Number of data points
            n: Number of parameters

        Returns:
            Selected solver ('exact', 'lsmr', or None)
        """
        if tr_solver is not None:
            valid_solvers = {"exact", "lsmr"}
            if tr_solver not in valid_solvers:
                msg = (
                    f"Invalid tr_solver '{tr_solver}'. Must be one of {valid_solvers}."
                )
                raise ValueError(msg)
            return cast(Literal["exact", "lsmr"], tr_solver)

        # Auto-select based on problem size
        if m * n < 10000:
            return "exact"  # SVD-based for small problems
        else:
            return "lsmr"  # Iterative for large problems
