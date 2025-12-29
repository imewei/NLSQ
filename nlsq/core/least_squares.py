"""Generic interface for least-squares minimization."""

# mypy: ignore-errors

from collections.abc import Callable, Sequence
from typing import Any, Literal
from warnings import warn

import numpy as np

# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

__jax_config = JAXConfig()
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit
from jax.scipy.linalg import solve_triangular as jax_solve_triangular

from nlsq.caching.memory_manager import get_memory_manager
from nlsq.caching.unified_cache import get_global_cache
from nlsq.common_scipy import EPS, in_bounds, make_strictly_feasible
from nlsq.constants import DEFAULT_FTOL, DEFAULT_GTOL, DEFAULT_XTOL
from nlsq.core.loss_functions import LossFunctionsJIT
from nlsq.core.trf import TrustRegionReflective
from nlsq.stability.guard import NumericalStabilityGuard
from nlsq.types import ArrayLike, BoundsTuple, CallbackFunction, MethodLiteral
from nlsq.utils.diagnostics import OptimizationDiagnostics
from nlsq.utils.logging import get_logger


def jacobian_mode_selector(
    n_params: int, n_residuals: int, mode: str = "auto"
) -> tuple[str, str]:
    """Select Jacobian automatic differentiation mode based on problem dimensions.

    Automatically chooses between forward-mode (jacfwd) and reverse-mode (jacrev)
    automatic differentiation based on the Jacobian shape to minimize computational cost.

    Parameters
    ----------
    n_params : int
        Number of parameters (columns in Jacobian)
    n_residuals : int
        Number of residuals (rows in Jacobian)
    mode : {'auto', 'fwd', 'rev'}, optional
        Jacobian mode selection. Default is 'auto'.
        - 'auto': Automatically select based on problem dimensions
        - 'fwd': Force forward-mode AD (jacfwd)
        - 'rev': Force reverse-mode AD (jacrev)

    Returns
    -------
    selected_mode : str
        Selected mode ('fwd' or 'rev')
    rationale : str
        Human-readable explanation of the selection

    Raises
    ------
    ValueError
        If mode is not one of 'auto', 'fwd', 'rev'

    Notes
    -----
    Selection heuristic for 'auto' mode:

    - Use jacrev when n_params > n_residuals (tall Jacobian, more params than residuals)

      - Reverse-mode is O(n_residuals) operations
      - Forward-mode would be O(n_params) operations

    - Use jacfwd when n_params <= n_residuals (wide Jacobian, more residuals than params)

      - Forward-mode is O(n_params) operations
      - Reverse-mode would be O(n_residuals) operations

    For high-parameter problems (e.g., 1000 params, 100 residuals), jacrev can be
    10-100x faster than jacfwd.

    Examples
    --------
    >>> from nlsq.core.least_squares import jacobian_mode_selector
    >>> # Tall Jacobian (many parameters, few residuals)
    >>> mode, rationale = jacobian_mode_selector(1000, 100, mode='auto')
    >>> print(mode, rationale)
    rev jacrev (1000 params > 100 residuals)

    >>> # Wide Jacobian (few parameters, many residuals)
    >>> mode, rationale = jacobian_mode_selector(100, 1000, mode='auto')
    >>> print(mode, rationale)
    fwd jacfwd (100 params <= 1000 residuals)

    >>> # Manual override
    >>> mode, rationale = jacobian_mode_selector(1000, 100, mode='fwd')
    >>> print(mode, rationale)
    fwd explicit override: fwd
    """
    if mode == "auto":
        # Heuristic: use jacrev for tall Jacobians (n_params > n_residuals)
        # because reverse-mode is O(n_residuals) vs forward-mode O(n_params)
        if n_params > n_residuals:
            return "rev", f"jacrev ({n_params} params > {n_residuals} residuals)"
        else:
            return "fwd", f"jacfwd ({n_params} params <= {n_residuals} residuals)"
    elif mode in ("fwd", "rev"):
        return mode, f"explicit override: {mode}"
    else:
        raise ValueError(
            f"Invalid jacobian_mode: {mode}. Must be 'auto', 'fwd', or 'rev'"
        )


TERMINATION_MESSAGES = {
    -3: "Inner optimization loop exceeded maximum iterations.",
    -2: "Maximum iterations reached.",
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied.",
}


def prepare_bounds(bounds, n) -> tuple[np.ndarray, np.ndarray]:
    """Prepare bounds for optimization.

    This function prepares the bounds for the optimization by ensuring that
    they are both 1-D arrays of length `n`. If either bound is a scalar, it is
    resized to an array of length `n`.

    Parameters
    ----------
    bounds : Tuple[np.ndarray, np.ndarray]
        The lower and upper bounds for the optimization.
    n : int
        The length of the bounds arrays.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The prepared lower and upper bounds arrays.
    """
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = np.resize(lb, n)

    if ub.ndim == 0:
        ub = np.resize(ub, n)

    return lb, ub


def check_tolerance(
    ftol: float, xtol: float, gtol: float, method: str
) -> tuple[float, float, float]:
    """Check and prepare tolerance values for optimization.

    This function checks the tolerance values for the optimization and
    prepares them for use. If any of the tolerances is `None`, it is set to
    0. If any of the tolerances is lower than the machine epsilon, a warning
    is issued and the tolerance is set to the machine epsilon. If all
    tolerances are lower than the machine epsilon, a `ValueError` is raised.

    Parameters
    ----------
    ftol : float
        The tolerance for the optimization function value.
    xtol : float
        The tolerance for the optimization variable values.
    gtol : float
        The tolerance for the optimization gradient values.
    method : str
        The name of the optimization method.

    Returns
    -------
    Tuple[float, float, float]
        The prepared tolerance values.
    """

    def check(tol: float, name: str) -> float:
        if tol is None:
            tol = 0
        elif tol < EPS:
            warn(
                f"Setting `{name}` below the machine epsilon ({EPS:.2e}) effectively "
                "disables the corresponding termination condition.",
                stacklevel=2,
            )
        return tol

    ftol = check(ftol, "ftol")
    xtol = check(xtol, "xtol")
    gtol = check(gtol, "gtol")

    if ftol < EPS and xtol < EPS and gtol < EPS:
        raise ValueError(
            "At least one of the tolerances must be higher than "
            f"machine epsilon ({EPS:.2e})."
        )

    return ftol, xtol, gtol


def check_x_scale(
    x_scale: str | Sequence[float], x0: Sequence[float]
) -> str | Sequence[float]:
    """Check and prepare the `x_scale` parameter for optimization.

    This function checks and prepares the `x_scale` parameter for the
    optimization. `x_scale` can either be 'jac' or an array_like with positive
    numbers. If it's 'jac' the jacobian is used as the scaling.

    Parameters
    ----------
    x_scale : str | Sequence[float]
        The scaling for the optimization variables.
    x0 : Sequence[float]
        The initial guess for the optimization variables.

    Returns
    -------
    str | Sequence[float]
        The prepared `x_scale` parameter.
    """

    if isinstance(x_scale, str) and x_scale == "jac":
        return x_scale

    try:
        x_scale = np.asarray(x_scale, dtype=float)
        valid = np.all(np.isfinite(x_scale)) and np.all(x_scale > 0)
    except (ValueError, TypeError):
        valid = False

    if not valid:
        raise ValueError("`x_scale` must be 'jac' or array_like with positive numbers.")

    if x_scale.ndim == 0:
        x_scale = np.resize(x_scale, x0.shape)

    if x_scale.shape != x0.shape:
        raise ValueError("Inconsistent shapes between `x_scale` and `x0`.")

    return x_scale


class AutoDiffJacobian:
    """Wraps the residual fit function such that automatic differentiation is performed.

    Supports both forward-mode (jacfwd) and reverse-mode (jacrev) automatic differentiation.
    This needs to be a class since we need to maintain in memory three different versions
    of the Jacobian for different sigma/covariance cases.
    """

    def create_ad_jacobian(
        self, func: Callable, num_args: int, masked: bool = True, mode: str = "fwd"
    ) -> Callable:
        """Creates a function that returns the autodiff jacobian of the
        residual fit function. The Jacobian of the residual fit function is
        equivalent to the Jacobian of the fit function.

        Parameters
        ----------
        func : Callable
            The function to take the jacobian of.
        num_args : int
            The number of arguments the function takes.
        masked : bool, optional
            Whether to use a masked jacobian, by default True
        mode : str, optional
            Jacobian mode ('fwd' or 'rev'), by default 'fwd'

        Returns
        -------
        Callable
            The function that returns the autodiff jacobian of the given
            function.
        """

        # create a list of argument indices for the wrapped function which
        # will correspond to the arguments of the residual fit function and
        # will be passed to JAX's jacfwd/jacrev function.
        arg_list = [4 + i for i in range(num_args)]

        # Select the appropriate JAX differentiation function
        jac_func_ad = jacfwd if mode == "fwd" else jacrev

        # Note: Uses @jit (not cached_jit) because these closures capture 'func'
        # which changes each call, so caching based on source wouldn't work
        @jit
        def wrap_func(*all_args) -> jnp.ndarray:
            """Wraps the residual fit function such that it can be passed to the
            jacfwd/jacrev function. Both require the function to have a single list
            of arguments.
            """
            xdata, ydata, data_mask, atransform = all_args[:4]
            args = jnp.array(all_args[4:])
            return func(args, xdata, ydata, data_mask, atransform)

        @jit
        def jac_func(
            args: jnp.ndarray,
            xdata: jnp.ndarray,
            ydata: jnp.ndarray,
            data_mask: jnp.ndarray,
            atransform: jnp.ndarray,
        ) -> jnp.ndarray:
            """Returns the jacobian. Places all the residual fit function
            arguments into a single list for the wrapped residual fit function.
            Then calls the jacfwd or jacrev function on the wrapped function with
            the arglist of the arguments to differentiate with respect to which
            is only the arguments of the original fit function.
            """

            fixed_args = [xdata, ydata, data_mask, atransform]
            all_args = [*fixed_args, *args]
            jac_result = jac_func_ad(wrap_func, argnums=arg_list)(*all_args)
            return jnp.array(jac_result)

        @jit
        def masked_jac(
            args: jnp.ndarray,
            xdata: jnp.ndarray,
            ydata: jnp.ndarray,
            data_mask: jnp.ndarray,
            atransform: jnp.ndarray,
        ) -> jnp.ndarray:
            """Returns the masked jacobian."""
            Jt = jac_func(args, xdata, ydata, data_mask, atransform)
            J = jnp.where(data_mask, Jt, 0).T
            return jnp.atleast_2d(J)

        @jit
        def no_mask_jac(
            args: jnp.ndarray,
            xdata: jnp.ndarray,
            ydata: jnp.ndarray,
            data_mask: jnp.ndarray,
            atransform: jnp.ndarray,
        ) -> jnp.ndarray:
            """Returns the unmasked jacobian."""
            J = jac_func(args, xdata, ydata, data_mask, atransform).T
            return jnp.atleast_2d(J)

        if masked:
            self.jac = masked_jac
        else:
            self.jac = no_mask_jac
        return self.jac


class LeastSquares:
    """Core least squares optimization engine with JAX acceleration.

    This class implements the main optimization algorithms for nonlinear least squares
    problems, including Trust Region Reflective (TRF) and Levenberg-Marquardt (LM).
    It handles automatic differentiation, bound constraints, loss functions, and
    uncertainty propagation.

    The class maintains separate automatic differentiation instances for different
    sigma configurations (no sigma, 1D sigma, 2D covariance matrix) to optimize
    compilation and execution performance.

    Attributes
    ----------
    trf : TrustRegionReflective
        Trust Region Reflective algorithm implementation
    ls : LossFunctionsJIT
        JIT-compiled loss function implementations
    logger : Logger
        Internal logger for debugging and performance tracking
    f : callable
        Current objective function being optimized
    jac : callable or None
        Current Jacobian function (None for automatic differentiation)
    adjn : AutoDiffJacobian
        Automatic differentiation instance for unweighted problems
    adj1d : AutoDiffJacobian
        Automatic differentiation instance for 1D sigma weighting
    adj2d : AutoDiffJacobian
        Automatic differentiation instance for 2D covariance matrix weighting

    Methods
    -------
    least_squares : Main optimization method
    """

    def __init__(
        self,
        enable_stability: bool = False,
        enable_diagnostics: bool = False,
        max_jacobian_elements_for_svd: int = 10_000_000,
    ) -> None:
        """Initialize LeastSquares with optimization algorithms and autodiff instances.

        Sets up the Trust Region Reflective solver, loss functions, and separate
        automatic differentiation instances for different weighting schemes to
        maximize JAX compilation efficiency.

        Parameters
        ----------
        enable_stability : bool, default False
            Enable numerical stability checks and fixes
        enable_diagnostics : bool, default False
            Enable optimization diagnostics collection
        max_jacobian_elements_for_svd : int, default 10_000_000
            Maximum Jacobian size (m × n elements) for SVD computation during
            stability checks. SVD is skipped for larger Jacobians.
        """
        super().__init__()  # not sure if this is needed
        self.trf = TrustRegionReflective()
        self.ls = LossFunctionsJIT()
        self.logger = get_logger("least_squares")
        # initialize jacobian to None and f to a dummy function
        self.f = lambda x: None
        self.jac: Callable[..., jnp.ndarray] | None = None

        # need a separate instance of the autodiff class for each of the
        # the different sigma/covariance cases
        self.adjn = AutoDiffJacobian()
        self.adj1d = AutoDiffJacobian()
        self.adj2d = AutoDiffJacobian()

        # Initialize unified cache for JIT compilation tracking
        self.cache = get_global_cache()

        # Stability and diagnostics systems
        self.enable_stability = enable_stability
        self.enable_diagnostics = enable_diagnostics
        self.max_jacobian_elements_for_svd = max_jacobian_elements_for_svd

        if enable_stability:
            self.stability_guard = NumericalStabilityGuard(
                max_jacobian_elements_for_svd=max_jacobian_elements_for_svd
            )
            self.memory_manager = get_memory_manager()

        if enable_diagnostics:
            self.diagnostics = OptimizationDiagnostics()

    def _validate_least_squares_inputs(
        self,
        x0: np.ndarray,
        bounds: tuple,
        method: str,
        jac,
        loss: str,
        verbose: int,
        max_nfev: float | None,
        ftol: float,
        xtol: float,
        gtol: float,
        x_scale,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
        """Validate and prepare least squares inputs.

        Returns
        -------
        x0 : np.ndarray
            Validated initial guess
        lb : np.ndarray
            Lower bounds
        ub : np.ndarray
            Upper bounds
        ftol : float
            Function tolerance
        xtol : float
            Parameter tolerance
        gtol : float
            Gradient tolerance
        x_scale : np.ndarray
            Parameter scaling
        """
        # Validate loss function
        if loss not in self.ls.IMPLEMENTED_LOSSES and not callable(loss):
            raise ValueError(
                f"`loss` must be one of {self.ls.IMPLEMENTED_LOSSES.keys()} or a callable."
            )

        # Validate method
        if method not in ["trf"]:
            raise ValueError("`method` must be 'trf'")

        # Validate jac parameter
        if jac not in [None] and not callable(jac):
            raise ValueError("`jac` must be None or callable.")

        # Validate verbose level
        if verbose not in [0, 1, 2]:
            raise ValueError("`verbose` must be in [0, 1, 2].")

        # Validate bounds
        if len(bounds) != 2:
            raise ValueError("`bounds` must contain 2 elements.")

        # Validate max_nfev
        if max_nfev is not None and max_nfev <= 0:
            raise ValueError("`max_nfev` must be None or positive integer.")

        # Validate x0
        if np.iscomplexobj(x0):
            raise ValueError("`x0` must be real.")

        x0 = np.atleast_1d(x0).astype(float)

        if x0.ndim > 1:
            raise ValueError("`x0` must have at most 1 dimension.")

        # Prepare bounds
        lb, ub = prepare_bounds(bounds, x0.shape[0])

        if lb.shape != x0.shape or ub.shape != x0.shape:
            raise ValueError("Inconsistent shapes between bounds and `x0`.")

        if np.any(lb >= ub):
            raise ValueError(
                "Each lower bound must be strictly less than each upper bound."
            )

        if not in_bounds(x0, lb, ub):
            raise ValueError("`x0` is infeasible.")

        # Check and prepare scaling/tolerances
        x_scale = check_x_scale(x_scale, x0)
        ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol, method)
        x0 = make_strictly_feasible(x0, lb, ub)

        return x0, lb, ub, ftol, xtol, gtol, x_scale

    def _setup_functions(
        self,
        fun: Callable,
        jac: Callable | None,
        xdata: jnp.ndarray | None,
        ydata: jnp.ndarray | None,
        transform: jnp.ndarray | None,
        x0: np.ndarray,
        args: tuple,
        kwargs: dict,
        jacobian_mode_selected: str = "fwd",
    ) -> tuple:
        """Setup residual and Jacobian functions.

        Returns
        -------
        rfunc : callable
            Residual function
        jac_func : callable
            Jacobian function
        """
        if xdata is not None and ydata is not None:
            # Check if fit function needs updating
            func_update = False
            try:
                if hasattr(self.f, "__code__") and hasattr(fun, "__code__"):
                    func_update = self.f.__code__.co_code != fun.__code__.co_code
                else:
                    func_update = self.f != fun
            except Exception:
                func_update = True

            # Update function if needed
            if func_update:
                self.update_function(fun)
                if jac is None:
                    self.autdiff_jac(jac, mode=jacobian_mode_selected)

            # Handle analytical Jacobian
            if jac is not None:
                if (
                    self.jac is None
                    or self.jac.__code__.co_code != jac.__code__.co_code
                ):
                    self.wrap_jac(jac)
            elif self.jac is not None and not func_update:
                self.autdiff_jac(jac, mode=jacobian_mode_selected)

            # Select appropriate residual function and Jacobian
            if transform is None:
                rfunc = self.func_none
                jac_func = self.jac_none
            elif transform.ndim == 1:
                rfunc = self.func_1d
                jac_func = self.jac_1d
            else:
                rfunc = self.func_2d
                jac_func = self.jac_2d
        else:
            # SciPy compatibility mode
            def wrap_func(fargs, xdata, ydata, data_mask, atransform):
                return jnp.atleast_1d(fun(fargs, *args, **kwargs))

            def wrap_jac(fargs, xdata, ydata, data_mask, atransform):
                return jnp.atleast_2d(jac(fargs, *args, **kwargs))

            rfunc = wrap_func
            if jac is None:
                adj = AutoDiffJacobian()
                jac_func = adj.create_ad_jacobian(
                    wrap_func, x0.size, masked=False, mode=jacobian_mode_selected
                )
            else:
                jac_func = wrap_jac

        return rfunc, jac_func

    def _evaluate_initial_residuals_and_jacobian(
        self,
        rfunc: Callable,
        jac_func: Callable,
        x0: np.ndarray,
        xdata: jnp.ndarray | None,
        ydata: jnp.ndarray | None,
        data_mask: jnp.ndarray | None,
        transform: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate initial residuals and Jacobian, with stability checks.

        Parameters
        ----------
        rfunc : Callable
            Residual function
        jac_func : Callable
            Jacobian function
        x0 : np.ndarray
            Initial parameters
        xdata : jnp.ndarray | None
            X data
        ydata : jnp.ndarray | None
            Y data
        data_mask : jnp.ndarray | None
            Data mask
        transform : jnp.ndarray | None
            Transform matrix

        Returns
        -------
        f0 : jnp.ndarray
            Initial residuals
        J0 : jnp.ndarray
            Initial Jacobian

        Raises
        ------
        ValueError
            If residuals are not 1-D or not finite
        """
        f0 = rfunc(x0, xdata, ydata, data_mask, transform)
        J0 = jac_func(x0, xdata, ydata, data_mask, transform)

        if f0.ndim != 1:
            raise ValueError(
                f"`fun` must return at most 1-d array_like. f0.shape: {f0.shape}"
            )

        if not np.all(np.isfinite(f0)):
            if self.enable_stability:
                self.logger.warning("Non-finite residuals detected, attempting to fix")
                f0 = self.stability_guard.safe_clip(f0, -1e10, 1e10)
                if not np.all(np.isfinite(f0)):
                    raise ValueError("Residuals are not finite after stabilization")
            else:
                raise ValueError("Residuals are not finite in the initial point.")

        return f0, J0

    def _check_and_fix_initial_jacobian(
        self, J0: jnp.ndarray, m: int, n: int
    ) -> jnp.ndarray:
        """Check and fix initial Jacobian if stability is enabled.

        Parameters
        ----------
        J0 : jnp.ndarray
            Initial Jacobian
        m : int
            Number of residuals
        n : int
            Number of parameters

        Returns
        -------
        J0 : jnp.ndarray
            Validated/fixed Jacobian

        Raises
        ------
        ValueError
            If Jacobian has wrong shape
        """
        # Check and fix Jacobian if stability is enabled
        if self.enable_stability and J0 is not None:
            J0_fixed, issues = self.stability_guard.check_and_fix_jacobian(J0)
            if issues:
                # Only warn if there's an actual problem, not just SVD skipped for performance
                has_problem = (
                    issues.get("has_nan")
                    or issues.get("has_inf")
                    or issues.get("is_ill_conditioned")
                    or issues.get("regularized")
                )
                if has_problem:
                    self.logger.warning(
                        "Jacobian issues detected and fixed", issues=issues
                    )
                elif issues.get("svd_skipped"):
                    self.logger.debug(
                        "SVD skipped for large Jacobian (expected for datasets > 10M points)",
                        issues=issues,
                    )
                J0 = J0_fixed

        if J0 is not None and J0.shape != (m, n):
            raise ValueError(
                f"The return value of `jac` has wrong shape: expected {(m, n)}, "
                f"actual {J0.shape}."
            )

        return J0

    def _compute_initial_cost(
        self,
        f0: jnp.ndarray,
        loss: str | Callable,
        loss_function: Callable | None,
        f_scale: float,
        data_mask: jnp.ndarray,
    ) -> float:
        """Compute initial cost from residuals and loss function.

        Parameters
        ----------
        f0 : jnp.ndarray
            Initial residuals
        loss : str | Callable
            Loss function name or callable
        loss_function : Callable | None
            Loss function implementation
        f_scale : float
            Loss function scale parameter
        data_mask : jnp.ndarray
            Data mask

        Returns
        -------
        initial_cost : float
            Initial cost value

        Raises
        ------
        ValueError
            If callable loss returns wrong shape
        """
        m = f0.size
        self.logger.debug("Computing initial cost", loss_type=loss, f_scale=f_scale)

        if callable(loss):
            rho = loss_function(f0, f_scale, data_mask=data_mask)
            if rho.shape != (3, m):
                raise ValueError("The return value of `loss` callable has wrong shape.")
            initial_cost_jnp = self.trf.calculate_cost(rho, data_mask)
        elif loss_function is not None:
            initial_cost_jnp = loss_function(
                f0, f_scale, data_mask=data_mask, cost_only=True
            )
        else:
            initial_cost_jnp = self.trf.default_loss_func(f0)

        return np.array(initial_cost_jnp)

    def _check_memory_and_adjust_solver(
        self, m: int, n: int, method: str, tr_solver: str | None
    ) -> str | None:
        """Check memory requirements and adjust solver if needed.

        Parameters
        ----------
        m : int
            Number of residuals
        n : int
            Number of parameters
        method : str
            Optimization method
        tr_solver : str | None
            Current trust region solver

        Returns
        -------
        tr_solver : str | None
            Adjusted trust region solver (or original if no adjustment needed)
        """
        if self.enable_stability:
            memory_required = self.memory_manager.predict_memory_requirement(
                m, n, method
            )
            is_available, msg = self.memory_manager.check_memory_availability(
                memory_required
            )
            if not is_available:
                self.logger.warning("Memory constraint detected", message=msg)
                # Switch to memory-efficient solver
                tr_solver = "lsmr"

        return tr_solver

    def _create_stable_wrappers(
        self, rfunc: Callable, jac_func: Callable
    ) -> tuple[Callable, Callable]:
        """Create stability wrapper functions for residuals and Jacobian.

        NOTE: Stability checks are only performed at initialization, not per-iteration.
        Per-iteration Jacobian modification was found to cause optimization divergence
        due to accumulated numerical perturbations and expensive SVD computations.

        The residual wrapper still checks for NaN/Inf at each evaluation since this
        is a cheap O(n) check that can catch numerical explosions early.

        Parameters
        ----------
        rfunc : Callable
            Original residual function
        jac_func : Callable
            Original Jacobian function

        Returns
        -------
        rfunc : Callable
            Wrapped residual function (with NaN/Inf checking)
        jac_func : Callable
            Original Jacobian function (NOT wrapped - stability checked at init only)
        """
        if self.enable_stability:
            original_rfunc = rfunc

            def stable_rfunc(x, xd, yd, dm, tf):
                result = original_rfunc(x, xd, yd, dm, tf)
                if not jnp.all(jnp.isfinite(result)):
                    result = self.stability_guard.safe_clip(result, -1e10, 1e10)
                return result

            # NOTE: Jacobian is NOT wrapped - stability checked only at initialization
            # via _check_and_fix_initial_jacobian(). Per-iteration Jacobian modification
            # causes optimization divergence due to SVD overhead and accumulated perturbations.
            return stable_rfunc, jac_func

        return rfunc, jac_func

    def _run_trf_optimization(
        self,
        rfunc: Callable,
        jac_func: Callable,
        xdata: jnp.ndarray | None,
        ydata: jnp.ndarray | None,
        data_mask: jnp.ndarray,
        transform: jnp.ndarray | None,
        x0: np.ndarray,
        f0: jnp.ndarray,
        J0: jnp.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: float | None,
        f_scale: float,
        x_scale: np.ndarray,
        loss_function: Callable | None,
        tr_options: dict,
        verbose: int,
        timeit: bool,
        tr_solver: str | None,
        method: str,
        loss: str,
        n: int,
        m: int,
        initial_cost: float,
        timeout_kwargs: dict,
        callback: Callable | None,
    ):
        """Run TRF optimization with diagnostics and logging.

        Returns
        -------
        result : OptimizeResult
            Optimization result
        """
        with self.logger.timer("optimization"):
            self.logger.debug("Calling TRF optimizer", initial_cost=initial_cost)

            # Initialize diagnostics if enabled
            if self.enable_diagnostics:
                self.diagnostics.start_optimization(
                    n_params=n, n_data=m, method=method, loss=loss
                )

            result = self.trf.trf(
                rfunc,
                xdata,
                ydata,
                jac_func,
                data_mask,
                transform,
                x0,
                f0,
                J0,
                lb,
                ub,
                ftol,
                xtol,
                gtol,
                max_nfev,
                f_scale,
                x_scale,
                loss_function,
                tr_options.copy(),
                verbose,
                timeit,
                solver=tr_solver if tr_solver else "exact",
                diagnostics=self.diagnostics if self.enable_diagnostics else None,
                callback=callback,
                **timeout_kwargs,
            )

        return result

    def _process_optimization_result(self, result, initial_cost: float, verbose: int):
        """Process optimization result and log convergence.

        Parameters
        ----------
        result : OptimizeResult
            Optimization result
        initial_cost : float
            Initial cost value
        verbose : int
            Verbosity level

        Returns
        -------
        result : OptimizeResult
            Processed result with message and success flag
        """
        result.message = TERMINATION_MESSAGES[result.status]
        result.success = result.status > 0

        # Log convergence
        self.logger.convergence(
            reason=result.message,
            iterations=getattr(result, "nit", None),
            final_cost=result.cost,
            time_elapsed=self.logger.timers.get("optimization", 0),
            final_gradient_norm=getattr(result, "optimality", None),
        )

        if verbose >= 1:
            self.logger.info(result.message)
            self.logger.info(
                f"Function evaluations {result.nfev}, initial cost {initial_cost:.4e}, final cost "
                f"{result.cost:.4e}, first-order optimality {result.optimality:.2e}."
            )

        return result

    def least_squares(
        self,
        fun: Callable,
        x0: ArrayLike,
        jac: Callable | None = None,
        bounds: BoundsTuple | tuple[float, float] = (-np.inf, np.inf),
        method: MethodLiteral = "trf",
        ftol: float = DEFAULT_FTOL,
        xtol: float = DEFAULT_XTOL,
        gtol: float = DEFAULT_GTOL,
        x_scale: Literal["jac"] | ArrayLike | float = 1.0,
        loss: str = "linear",
        f_scale: float = 1.0,
        diff_step: ArrayLike | None = None,
        tr_solver: Literal["exact", "lsmr"] | None = None,
        tr_options: dict[str, Any] | None = None,
        jac_sparsity: ArrayLike | None = None,
        max_nfev: float | None = None,
        verbose: int = 0,
        jacobian_mode: Literal["auto", "fwd", "rev"] | None = None,
        xdata: ArrayLike | None = None,
        ydata: ArrayLike | None = None,
        data_mask: ArrayLike | None = None,
        transform: ArrayLike | None = None,
        timeit: bool = False,
        callback: CallbackFunction | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        **timeout_kwargs: Any,
    ) -> dict[str, Any]:
        """Solve nonlinear least squares problem using JAX-accelerated algorithms.

        This method orchestrates the optimization process by calling focused
        helper methods for each major step: validation, function setup,
        initial evaluation, stability checks, and optimization execution.

        Parameters
        ----------
        fun : callable
            Residual function. Must use jax.numpy operations.
        x0 : array_like
            Initial parameter guess.
        jac : callable or None, optional
            Jacobian function. If None, uses JAX autodiff.

        bounds : 2-tuple, optional
            Parameter bounds as (lower, upper).
        method : str, optional
            Optimization algorithm ('trf').
        ftol, xtol, gtol : float, optional
            Convergence tolerances for function, parameters, and gradient.
        x_scale : str or array_like, optional
            Parameter scaling ('jac' for automatic).
        loss : str or callable, optional
            Robust loss function ('linear', 'huber', 'soft_l1', etc.).
        f_scale : float, optional
            Scale parameter for robust loss functions.
        max_nfev : int, optional
            Maximum function evaluations.
        verbose : int, optional
            Verbosity level (0, 1, or 2).
        jacobian_mode : {'auto', 'fwd', 'rev'}, optional
            Jacobian automatic differentiation mode. If None, uses configuration
            from environment variable, config file, or auto-default. Default is None.
            - 'auto': Automatically select based on problem dimensions
            - 'fwd': Force forward-mode AD (jacfwd)
            - 'rev': Force reverse-mode AD (jacrev)
        xdata, ydata : array_like, optional
            Data for curve fitting applications.
        data_mask : array_like, optional
            Boolean mask for data exclusion.
        transform : array_like, optional
            Transformation matrix for weighted fitting.
        timeit : bool, optional
            Enable detailed timing analysis.
        callback : callable or None, optional
            Callback function called after each optimization iteration with signature
            ``callback(iteration, cost, params, info)``. Useful for monitoring
            optimization progress, logging, or implementing custom stopping criteria.
            If None (default), no callback is invoked.
        args : tuple, optional
            Additional arguments for objective function.
        kwargs : dict, optional
            Additional optimization parameters.

        Returns
        -------
        result : OptimizeResult
            Optimization result with solution, convergence info, and statistics.
        """
        # Step 1: Initialize parameters and validate options
        if kwargs is None:
            kwargs = {}
        if tr_options is None:
            tr_options = {}
        if "options" in timeout_kwargs:
            raise TypeError("'options' is not a supported keyword argument")

        if data_mask is None and ydata is not None:
            data_mask = jnp.ones(len(ydata), dtype=bool)

        # Step 2: Validate inputs
        x0, lb, ub, ftol, xtol, gtol, x_scale = self._validate_least_squares_inputs(
            x0, bounds, method, jac, loss, verbose, max_nfev, ftol, xtol, gtol, x_scale
        )

        self.n = len(x0)
        n = x0.size

        # Step 2.5: Determine Jacobian mode with configuration precedence
        # Precedence: function parameter > env var > config file > auto-default
        if jacobian_mode is not None:
            # Function parameter has highest priority
            jacobian_mode_config = jacobian_mode
            jacobian_mode_source = "function parameter"
        else:
            # Get from environment/config/default
            from nlsq.config import get_jacobian_mode

            jacobian_mode_config, jacobian_mode_source = get_jacobian_mode()

        # Step 3: Log optimization setup
        self.logger.info(
            "Starting least squares optimization",
            method=method,
            n_params=self.n,
            loss=loss,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        # Step 4: Setup residual and Jacobian functions
        # First, do a preliminary evaluation to determine problem dimensions
        # for Jacobian mode selection (if using autodiff)
        if jac is None and xdata is not None and ydata is not None:
            # Quick dimension check for auto mode selection
            # We need to know m (number of residuals) for mode selection
            # Create a temporary residual function for dimension check
            # Note: Uses @jit (not cached_jit) because this closure captures
            # different fun/xdata/ydata each call, so caching wouldn't help
            @jit
            def temp_residual(args_temp):
                func_eval = fun(xdata, *args_temp) - ydata
                return jnp.where(
                    data_mask if data_mask is not None else True, func_eval, 0
                )

            f0_temp = temp_residual(x0)
            m_temp = f0_temp.size

            # Select Jacobian mode based on problem dimensions
            jacobian_mode_selected, jacobian_rationale = jacobian_mode_selector(
                n, m_temp, mode=jacobian_mode_config
            )

            # Log Jacobian mode selection in debug mode
            self.logger.debug(
                f"Jacobian mode: '{jacobian_mode_selected}' (from {jacobian_mode_source}). Rationale: {jacobian_rationale}"
            )
        else:
            # Analytical Jacobian or SciPy mode - use default forward mode
            jacobian_mode_selected = "fwd"
            jacobian_rationale = "analytical Jacobian or SciPy compatibility mode"
            self.logger.debug(
                f"Jacobian mode: '{jacobian_mode_selected}'. Rationale: {jacobian_rationale}"
            )

        rfunc, jac_func = self._setup_functions(
            fun, jac, xdata, ydata, transform, x0, args, kwargs, jacobian_mode_selected
        )

        # Step 5: Evaluate initial residuals and Jacobian
        f0, J0 = self._evaluate_initial_residuals_and_jacobian(
            rfunc, jac_func, x0, xdata, ydata, data_mask, transform
        )

        m = f0.size

        # Step 6: Check and fix initial Jacobian
        J0 = self._check_and_fix_initial_jacobian(J0, m, n)

        # Step 7: Setup data mask and loss function
        if data_mask is None:
            data_mask = jnp.ones(m)

        loss_function = self.ls.get_loss_function(loss)

        # Step 8: Compute initial cost
        initial_cost = self._compute_initial_cost(
            f0, loss, loss_function, f_scale, data_mask
        )

        # Step 8.5: Detect sparsity and auto-select sparse solver if beneficial
        # This happens AFTER initial Jacobian computation (so we have J0 available)
        # Auto-selection triggers when: sparsity >50% AND n_residuals >10K
        sparsity_ratio = 0.0
        is_sparse_problem = False
        sparse_solver_selected = False

        if xdata is not None and ydata is not None and fun is not None:
            # Import sparsity detection function
            from nlsq.core.sparse_jacobian import detect_sparsity_at_p0

            # Detect sparsity at p0 (uses sampling for efficiency)
            try:
                sparsity_ratio, is_sparse_problem = detect_sparsity_at_p0(
                    func=fun,
                    p0=x0,
                    xdata=xdata,
                    n_residuals=m,
                    threshold=0.01,
                    sample_size=min(100, m),
                )

                # Auto-selection logic:
                # 1. Must have high sparsity (>50%)
                # 2. Must have large problem size (>10K residuals)
                # 3. User has not explicitly set tr_solver (tr_solver is None)
                if is_sparse_problem and m > 10000 and tr_solver is None:
                    # Activate sparse solver
                    tr_solver = "sparse"
                    sparse_solver_selected = True
                    self.logger.info(
                        f"Sparse solver activated: sparsity={sparsity_ratio:.1%}, "
                        f"n_residuals={m}, n_params={n}"
                    )
                else:
                    # Use dense solver (default)
                    if tr_solver is None:
                        tr_solver = "exact"  # Default dense solver
                    reason = []
                    if not is_sparse_problem:
                        reason.append(f"low sparsity ({sparsity_ratio:.1%})")
                    if m <= 10000:
                        reason.append(f"small problem (n_residuals={m})")
                    if tr_solver is not None:
                        reason.append("user-specified tr_solver")

                    self.logger.debug(
                        f"Dense solver selected: {', '.join(reason) if reason else 'default'}"
                    )

            except Exception as e:
                # If sparsity detection fails, fall back to dense solver
                self.logger.warning(
                    f"Sparsity detection failed: {e}. Using dense solver."
                )
                if tr_solver is None:
                    tr_solver = "exact"

        # Step 9: Check memory and adjust solver if needed
        tr_solver = self._check_memory_and_adjust_solver(m, n, method, tr_solver)

        # Step 10: Create stable wrappers for residual and Jacobian functions
        rfunc, jac_func = self._create_stable_wrappers(rfunc, jac_func)

        # Step 11: Run TRF optimization
        result = self._run_trf_optimization(
            rfunc,
            jac_func,
            xdata,
            ydata,
            data_mask,
            transform,
            x0,
            f0,
            J0,
            lb,
            ub,
            ftol,
            xtol,
            gtol,
            max_nfev,
            f_scale,
            x_scale,
            loss_function,
            tr_options,
            verbose,
            timeit,
            tr_solver,
            method,
            loss,
            n,
            m,
            initial_cost,
            timeout_kwargs,
            callback,
        )

        # Step 12: Process optimization result
        result = self._process_optimization_result(result, initial_cost, verbose)

        # Step 13: Add sparsity diagnostics to result (Task 6.5)
        # This provides transparency about whether sparse solver was used
        result.sparsity_detected = {
            "detected": is_sparse_problem,
            "ratio": float(sparsity_ratio),
            "solver": "sparse" if sparse_solver_selected else "dense",
            "n_residuals": m,
            "n_params": n,
        }

        # Log sparsity info in debug mode
        self.logger.debug(
            f"Sparsity diagnostics: detected={is_sparse_problem}, "
            f"ratio={sparsity_ratio:.1%}, solver={'sparse' if sparse_solver_selected else 'dense'}"
        )

        return result

    def autdiff_jac(self, jac: None, mode: str = "fwd") -> None:
        """We do this for all three sigma transformed functions such
        that if sigma is changed from none to 1D to covariance sigma then no
        retracing is needed.

        Parameters
        ----------
        jac : None
            Passed in to maintain compatibility with the user defined Jacobian
            function.
        mode : str, optional
            Jacobian mode ('fwd' or 'rev'), by default 'fwd'
        """
        self.jac_none = self.adjn.create_ad_jacobian(self.func_none, self.n, mode=mode)
        self.jac_1d = self.adj1d.create_ad_jacobian(self.func_1d, self.n, mode=mode)
        self.jac_2d = self.adj2d.create_ad_jacobian(self.func_2d, self.n, mode=mode)
        # jac is
        self.jac = jac

    def update_function(self, func: Callable) -> None:
        """Wraps the given fit function to be a residual function using the
        data. The wrapped function is in a JAX JIT compatible format which
        is purely functional. This requires that both the data mask and the
        uncertainty transform are passed to the function. Even for the case
        where the data mask is all True and the uncertainty transform is None
        we still need to pass these arguments to the function due JAX's
        functional nature.

        Parameters
        ----------
        func : Callable
            The fit function to wrap.

        Returns
        -------
        None
        """

        # Note: Uses @jit (not cached_jit) because this closure captures 'func'
        # which changes each call, so caching based on source wouldn't work
        @jit
        def masked_residual_func(
            args: jnp.ndarray,
            xdata: jnp.ndarray,
            ydata: jnp.ndarray,
            data_mask: jnp.ndarray,
        ) -> jnp.ndarray:
            """Compute the residual of the function evaluated at `args` with
            respect to the data.

            This function computes the residual of the user fit function
            evaluated at `args` with respect to the data `(xdata, ydata)`,
            masked by `data_mask`. The residual is defined as the difference
            between the function evaluation and the data. The masked residual
            is obtained by setting the residual to 0 wherever the corresponding
            element of `data_mask` is 0.

            Parameters
            ----------
            args : jnp.ndarray
                The parameters of the function.
            xdata : jnp.ndarray
                The independent variable data.
            ydata : jnp.ndarray
                The dependent variable data.
            data_mask : jnp.ndarray
                The mask for the data.

            Returns
            -------
            jnp.ndarray
                The masked residual of the function evaluated at `args` with respect to the data.
            """
            # JAX 0.8.0+ handles tuple unpacking efficiently without TracerArrayConversionError
            # This replaces the previous 100-line if-elif chain (Optimization #2)
            # See: OPTIMIZATION_QUICK_REFERENCE.md for performance analysis
            func_eval = func(xdata, *args) - ydata
            return jnp.where(data_mask, func_eval, 0)

        # need to define a separate function for each of the different
        # sigma/covariance cases as the uncertainty transform is different
        # for each case. In future could remove the no transfore bit by setting
        # the uncertainty transform to all ones in the case where there is no
        # uncertainty transform.

        @jit
        def func_no_transform(
            args: jnp.ndarray,
            xdata: jnp.ndarray,
            ydata: jnp.ndarray,
            data_mask: jnp.ndarray,
            atransform: jnp.ndarray,
        ) -> jnp.ndarray:
            """The residual function when there is no uncertainty transform.
            The atranform argument is not used in this case, but is included
            for consistency with the other cases."""
            return masked_residual_func(args, xdata, ydata, data_mask)

        @jit
        def func_1d_transform(
            args: jnp.ndarray,
            xdata: jnp.ndarray,
            ydata: jnp.ndarray,
            data_mask: jnp.ndarray,
            atransform: jnp.ndarray,
        ) -> jnp.ndarray:
            """The residual function when there is a 1D uncertainty transform,
            that is when only the diagonal elements of the inverse covariance
            matrix are used."""
            return atransform * masked_residual_func(args, xdata, ydata, data_mask)

        @jit
        def func_2d_transform(
            args: jnp.ndarray,
            xdata: jnp.ndarray,
            ydata: jnp.ndarray,
            data_mask: jnp.ndarray,
            atransform: jnp.ndarray,
        ) -> jnp.ndarray:
            """The residual function when there is a 2D uncertainty transform,
            that is when the full covariance matrix is given."""
            f = masked_residual_func(args, xdata, ydata, data_mask)
            return jax_solve_triangular(atransform, f, lower=True)

        self.func_none = func_no_transform
        self.func_1d = func_1d_transform
        self.func_2d = func_2d_transform
        self.f = func

    def wrap_jac(self, jac: Callable) -> None:
        """Wraps an user defined Jacobian function to allow for data masking
        and uncertainty transforms. The wrapped function is in a JAX JIT
        compatible format which is purely functional. This requires that both
        the data mask and the uncertainty transform are passed to the function.

        Using an analytical Jacobian of the fit function is equivalent to
        the Jacobian of the residual function.

        Also note that the analytical Jacobian doesn't require the independent
        ydata, but we still need to pass it to the function to maintain
        compatibility with autdiff version which does require the ydata.

        Parameters
        ----------
        jac : Callable
            The Jacobian function to wrap.

        Returns
        -------
        jnp.ndarray
            The masked Jacobian of the function evaluated at `args` with respect to the data.
        """

        # Note: Uses @jit (not cached_jit) because these closures capture 'jac'
        # which changes each call, so caching based on source wouldn't work
        @jit
        def jac_func(coords: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
            # Create individual arguments from the array for JAX compatibility
            # This avoids the TracerArrayConversionError with dynamic unpacking
            if args.size == 1:
                jac_fwd = jac(coords, args[0])
            elif args.size == 2:
                jac_fwd = jac(coords, args[0], args[1])
            elif args.size == 3:
                jac_fwd = jac(coords, args[0], args[1], args[2])
            elif args.size == 4:
                jac_fwd = jac(coords, args[0], args[1], args[2], args[3])
            elif args.size == 5:
                jac_fwd = jac(coords, args[0], args[1], args[2], args[3], args[4])
            elif args.size == 6:
                jac_fwd = jac(
                    coords, args[0], args[1], args[2], args[3], args[4], args[5]
                )
            else:
                # For more parameters, use a more generic approach
                args_list = [args[i] for i in range(args.size)]
                jac_fwd = jac(coords, *args_list)
            return jnp.array(jac_fwd)

        @jit
        def masked_jac(
            coords: jnp.ndarray, args: jnp.ndarray, data_mask: jnp.ndarray
        ) -> jnp.ndarray:
            """Compute the wrapped Jacobian but masks out the padded elements
            with 0s"""
            Jt = jac_func(coords, args)
            return jnp.where(data_mask, Jt, 0).T

        @jit
        def jac_no_transform(
            args: jnp.ndarray,
            coords: jnp.ndarray,
            ydata: jnp.ndarray,
            data_mask: jnp.ndarray,
            atransform: jnp.ndarray,
        ) -> jnp.ndarray:
            """The wrapped Jacobian function when there is no
            uncertainty transform."""
            return jnp.atleast_2d(masked_jac(coords, args, data_mask))

        @jit
        def jac_1d_transform(
            args: jnp.ndarray,
            coords: jnp.ndarray,
            ydata: jnp.ndarray,
            data_mask: jnp.ndarray,
            atransform: jnp.ndarray,
        ) -> jnp.ndarray:
            """The wrapped Jacobian function when there is a 1D uncertainty
            transform, that is when only the diagonal elements of the inverse
            covariance matrix are used."""
            J = masked_jac(coords, args, data_mask)
            return jnp.atleast_2d(atransform[:, jnp.newaxis] * jnp.asarray(J))

        @jit
        def jac_2d_transform(
            args: jnp.ndarray,
            coords: jnp.ndarray,
            ydata: jnp.ndarray,
            data_mask: jnp.ndarray,
            atransform: jnp.ndarray,
        ) -> jnp.ndarray:
            """The wrapped Jacobian function when there is a 2D uncertainty
            transform, that is when the full covariance matrix is given."""

            J = masked_jac(coords, args, data_mask)
            return jnp.atleast_2d(
                jax_solve_triangular(atransform, jnp.asarray(J), lower=True)
            )

        # we need all three versions of the Jacobian function to allow for
        # changing the sigma transform from none to 1D to 2D without having
        # to retrace the function
        self.jac_none = jac_no_transform
        self.jac_1d = jac_1d_transform
        self.jac_2d = jac_2d_transform
        self.jac = jac
