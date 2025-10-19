"""Generic interface for least-squares minimization."""

from collections.abc import Callable, Sequence
from typing import Any, Literal
from warnings import warn

import numpy as np

# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

_jax_config = JAXConfig()
import jax.numpy as jnp
from jax import jacfwd, jit
from jax.scipy.linalg import solve_triangular as jax_solve_triangular

from nlsq.common_scipy import EPS, in_bounds, make_strictly_feasible
from nlsq.constants import DEFAULT_FTOL, DEFAULT_GTOL, DEFAULT_XTOL
from nlsq.diagnostics import OptimizationDiagnostics
from nlsq.logging import get_logger
from nlsq.loss_functions import LossFunctionsJIT
from nlsq.memory_manager import get_memory_manager
from nlsq.stability import NumericalStabilityGuard
from nlsq.trf import TrustRegionReflective
from nlsq.types import ArrayLike, BoundsTuple, CallbackFunction, MethodLiteral

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
    """Wraps the residual fit function such that a masked jacfwd is performed
    on it. thereby giving the autodiff Jacobian. This needs to be a class since
    we need to maintain in memory three different versions of the Jacobian.
    """

    def create_ad_jacobian(
        self, func: Callable, num_args: int, masked: bool = True
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

        Returns
        -------
        Callable
            The function that returns the autodiff jacobian of the given
            function.
        """

        # create a list of argument indices for the wrapped function which
        # will correspond to the arguments of the residual fit function and
        # will be past to JAX's jacfwd function.
        arg_list = [4 + i for i in range(num_args)]

        @jit
        def wrap_func(*all_args) -> jnp.ndarray:
            """Wraps the residual fit function such that it can be passed to the
            jacfwd function. Jacfwd requires the function to a single list
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
            Then calls the jacfwd function on the wrapped function with the
            the arglist of the arguments to differentiate with respect to which
            is only the arguments of the original fit function.
            """

            fixed_args = [xdata, ydata, data_mask, atransform]
            all_args = [*fixed_args, *args]
            jac_fwd = jacfwd(wrap_func, argnums=arg_list)(*all_args)
            return jnp.array(jac_fwd)

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
        self, enable_stability: bool = False, enable_diagnostics: bool = False
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
        """
        super().__init__()  # not sure if this is needed
        self.trf = TrustRegionReflective()
        self.ls = LossFunctionsJIT()
        self.logger = get_logger("least_squares")
        # initialize jacobian to None and f to a dummy function
        self.f = lambda x: None
        self.jac = None

        # need a separate instance of the autodiff class for each of the
        # the different sigma/covariance cases
        self.adjn = AutoDiffJacobian()
        self.adj1d = AutoDiffJacobian()
        self.adj2d = AutoDiffJacobian()

        # Stability and diagnostics systems
        self.enable_stability = enable_stability
        self.enable_diagnostics = enable_diagnostics

        if enable_stability:
            self.stability_guard = NumericalStabilityGuard()
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
                    self.autdiff_jac(jac)

            # Handle analytical Jacobian
            if jac is not None:
                if (
                    self.jac is None
                    or self.jac.__code__.co_code != jac.__code__.co_code
                ):
                    self.wrap_jac(jac)
            elif self.jac is not None and not func_update:
                self.autdiff_jac(jac)

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
                jac_func = adj.create_ad_jacobian(wrap_func, x0.size, masked=False)
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
                self.logger.warning("Jacobian issues detected and fixed", issues=issues)
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

        Parameters
        ----------
        rfunc : Callable
            Original residual function
        jac_func : Callable
            Original Jacobian function

        Returns
        -------
        rfunc : Callable
            Wrapped residual function
        jac_func : Callable
            Wrapped Jacobian function
        """
        if self.enable_stability:
            original_rfunc = rfunc
            original_jac_func = jac_func

            def stable_rfunc(x, xd, yd, dm, tf):
                result = original_rfunc(x, xd, yd, dm, tf)
                if not jnp.all(jnp.isfinite(result)):
                    result = self.stability_guard.safe_clip(result, -1e10, 1e10)
                return result

            def stable_jac_func(x, xd, yd, dm, tf):
                J = original_jac_func(x, xd, yd, dm, tf)
                J_fixed, _ = self.stability_guard.check_and_fix_jacobian(J)
                return J_fixed

            return stable_rfunc, stable_jac_func

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
        rfunc, jac_func = self._setup_functions(
            fun, jac, xdata, ydata, transform, x0, args, kwargs
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

        return result

    def autdiff_jac(self, jac: None) -> None:
        """We do this for all three sigma transformed functions such
        that if sigma is changed from none to 1D to covariance sigma then no
        retracing is needed.

        Parameters
        ----------
        jac : None
            Passed in to maintain compatibility with the user defined Jacobian
            function.
        """
        self.jac_none = self.adjn.create_ad_jacobian(self.func_none, self.n)
        self.jac_1d = self.adj1d.create_ad_jacobian(self.func_1d, self.n)
        self.jac_2d = self.adj2d.create_ad_jacobian(self.func_2d, self.n)
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
