"""Trust Region Reflective algorithm for least-squares optimization.
The algorithm is based on ideas from paper [STIR]_. The main idea is to
account for the presence of the bounds by appropriate scaling of the variables (or,
equivalently, changing a trust-region shape). Let's introduce a vector v::

           | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
    v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
           | 1,           otherwise

where g is the gradient of a cost function and lb, ub are the bounds. Its
components are distances to the bounds at which the anti-gradient points (if
this distance is finite). Define a scaling matrix D = diag(v**0.5).
First-order optimality conditions can be stated as::

    D^2 g(x) = 0.

Meaning that components of the gradient should be zero for strictly interior
variables, and components must point inside the feasible region for variables
on the bound.
Now consider this system of equations as a new optimization problem. If the
point x is strictly interior (not on the bound), then the left-hand side is
differentiable and the Newton step for it satisfies::

    (D^2 H + diag(g) Jv) p = -D^2 g

where H is the Hessian matrix (or its J^T J approximation in least squares),
Jv is the Jacobian matrix of v with components -1, 1 or 0, such that all
elements of matrix C = diag(g) Jv are non-negative. Introduce the change
of the variables x = D x_h (_h would be "hat" in LaTeX). In the new variables,
we have a Newton step satisfying::

    B_h p_h = -g_h,

where B_h = D H D + C, g_h = D g. In least squares B_h = J_h^T J_h, where
J_h = J D. Note that J_h and g_h are proper Jacobian and gradient with respect
to "hat" variables. To guarantee global convergence we formulate a
trust-region problem based on the Newton step in the new variables::

    0.5 * p_h^T B_h p + g_h^T p_h -> min, ||p_h|| <= Delta

In the original space B = H + D^{-1} C D^{-1}, and the equivalent trust-region
problem is::

    0.5 * p^T B p + g^T p -> min, ||D^{-1} p|| <= Delta

Here, the meaning of the matrix D becomes more clear: it alters the shape
of a trust-region, such that large steps towards the bounds are not allowed.
In the implementation, the trust-region problem is solved in "hat" space,
but handling of the bounds is done in the original space (see below and read
the code).
The introduction of the matrix D doesn't allow to ignore bounds, the algorithm
must keep iterates strictly feasible (to satisfy aforementioned
differentiability), the parameter theta controls step back from the boundary
(see the code for details).
The algorithm does another important trick. If the trust-region solution
doesn't fit into the bounds, then a reflected (from a firstly encountered
bound) search direction is considered. For motivation and analysis refer to
[STIR]_ paper (and other papers of the authors). In practice, it doesn't need
a lot of justifications, the algorithm simply chooses the best step among
three: a constrained trust-region step, a reflected step and a constrained
Cauchy step (a minimizer along -g_h in "hat" space, or -D^2 g in the original
space).
Another feature is that a trust-region radius control strategy is modified to
account for appearance of the diagonal C matrix (called diag_h in the code).
Note that all described peculiarities are completely gone as we consider
problems without bounds (the algorithm becomes a standard trust-region type
algorithm very similar to ones implemented in MINPACK).
The implementation supports two methods of solving the trust-region problem.
The first, called 'exact', applies SVD on Jacobian and then solves the problem
very accurately using the algorithm described in [JJMore]_. It is not
applicable to large problem. The second, called 'lsmr', uses the 2-D subspace
approach (sometimes called "indefinite dogleg"), where the problem is solved
in a subspace spanned by the gradient and the approximate Gauss-Newton step
found by ``scipy.sparse.linalg.lsmr``. A 2-D trust-region problem is
reformulated as a 4th order algebraic equation and solved very accurately by
``numpy.roots``. The subspace approach allows to solve very large problems
(up to couple of millions of residuals on a regular PC), provided the Jacobian
matrix is sufficiently sparse.
References
----------
.. [STIR] Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior,
      and Conjugate Gradient Method for Large-Scale Bound-Constrained
      Minimization Problems," SIAM Journal on Scientific Computing,
      Vol. 21, Number 1, pp 1-23, 1999.
.. [JJMore] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
    and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
"""

# mypy: ignore-errors

import time
import warnings
from collections.abc import Callable

import numpy as np

# REMOVED: from numpy.linalg import norm  # Use JAX norm (jnorm) instead
# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

__jax_config = JAXConfig()
import jax.numpy as jnp
from jax import debug, jit, lax
from jax.numpy.linalg import norm as jnorm
from jax.tree_util import tree_flatten

# Import safe SVD with fallback (full deterministic SVD only)
from nlsq.stability.svd_fallback import (
    compute_svd_with_fallback,
    initialize_gpu_safely,
)

# Setup logging
from nlsq.utils.logging import get_logger

logger = get_logger("trf")

# Initialize GPU settings safely
initialize_gpu_safely()

# Import dataclasses for SVDCache
from typing import NamedTuple

from nlsq.caching.unified_cache import get_global_cache
from nlsq.callbacks import StopOptimization
from nlsq.common_jax import CommonJIT, solve_lsq_trust_region_jax
from nlsq.common_scipy import (
    CL_scaling_vector,
    check_termination,
    find_active_constraints,
    in_bounds,
    intersect_trust_region,
    make_strictly_feasible,
    minimize_quadratic_1d,
    print_header_nonlinear,
    print_iteration_nonlinear,
    step_size_to_bound,
    update_tr_radius,
)
from nlsq.constants import (
    DEFAULT_MAX_NFEV_MULTIPLIER,
    INITIAL_LEVENBERG_MARQUARDT_LAMBDA,
    MAX_TRUST_RADIUS,
    MIN_TRUST_RADIUS,
)
from nlsq.core._optimize import OptimizeResult

# Logging support
# Optimizer base class
from nlsq.core.optimizer_base import TrustRegionOptimizerBase

# Profiling support
from nlsq.core.profiler import NullProfiler, TRFProfiler

# JIT-compiled helper functions (extracted for modularity)
from nlsq.core.trf_jit import TrustRegionJITFunctions

# Mixed precision support
from nlsq.precision.mixed_precision import (
    ConvergenceMetrics,
    MixedPrecisionConfig,
    MixedPrecisionManager,
    OptimizationState,
    PrecisionState,
)
from nlsq.stability.guard import NumericalStabilityGuard
from nlsq.utils.diagnostics import OptimizationDiagnostics


class SVDCache(NamedTuple):
    """Cache SVD decomposition across inner loop iterations when Jacobian unchanged.

    This cache stores the SVD components (U, s, V) along with the scaled Jacobian
    J_h to avoid redundant SVD computations during inner loop iterations where
    the step is rejected and parameters remain unchanged.

    Attributes
    ----------
    U : jnp.ndarray
        Left singular vectors (m × k), where m is residuals and k = min(m, n).
    s : jnp.ndarray
        Singular values (k,).
    V : jnp.ndarray
        Right singular vectors (n × k), where n is parameters.
    J_h : jnp.ndarray
        Scaled Jacobian in "hat" space (m × n).
    x_hash : int
        Hash of parameter vector for cache validation. Cache is valid only
        when the current parameter hash matches this value.

    Notes
    -----
    The cache is valid only when `x_hash` matches the current parameter vector's hash.
    When a step is rejected (actual_reduction <= 0), the parameters don't change,
    so the SVD can be reused. When a step is accepted, the cache must be invalidated.

    The expected speedup from SVD caching is 20-40% on problems with frequent step
    rejections, as SVD computation is O(mn²) and dominates iteration time.
    """

    U: jnp.ndarray
    s: jnp.ndarray
    V: jnp.ndarray
    J_h: jnp.ndarray
    x_hash: int


# Algorithm constants
# Trust region parameters
TR_REDUCTION_FACTOR = 0.25  # Factor to reduce trust region when numerical issues occur
TR_BOUNDARY_THRESHOLD = 0.95  # Threshold for checking if step is close to boundary
LOSS_FUNCTION_COEFF = 0.5  # Coefficient for loss function (0.5 * ||f||^2)
SQRT_EXPONENT = 0.5  # Exponent for square root in scaling (v**0.5)


class TrustRegionReflective(TrustRegionJITFunctions, TrustRegionOptimizerBase):
    """Trust Region Reflective algorithm for bounded least squares optimization.

    Implements the TRF algorithm with variable scaling to handle parameter bounds.
    Supports exact (SVD) and iterative (CG) solvers for trust region subproblems.
    """

    def __init__(self, enable_stability: bool = False):
        """Initialize the TrustRegionReflective optimizer.

        Creates JIT-compiled functions and sets up logging infrastructure.
        All optimization functions are compiled during initialization for
        maximum performance during solve operations.

        Parameters
        ----------
        enable_stability : bool, default False
            Enable numerical stability checks and fixes
        """
        TrustRegionJITFunctions.__init__(self)
        TrustRegionOptimizerBase.__init__(self, name="trf")
        self.cJIT = CommonJIT()

        # Initialize unified cache for JIT compilation tracking
        self.cache = get_global_cache()

        # Initialize stability system
        self.enable_stability = enable_stability
        if enable_stability:
            self.stability_guard = NumericalStabilityGuard()

    @staticmethod
    def _log_iteration_callback(
        iteration, nfev, cost, actual_reduction, step_norm, g_norm
    ):
        """Wrapper for logging callback that converts JAX arrays to Python scalars.

        This function is called by jax.debug.callback and ensures all arguments
        are converted from JAX arrays to Python scalars before logging.

        Parameters
        ----------
        iteration : int or jax.Array
            Iteration number
        nfev : int or jax.Array
            Number of function evaluations
        cost : float or jax.Array
            Current cost
        actual_reduction : float or jax.Array or None
            Actual cost reduction
        step_norm : float or jax.Array or None
            Step norm
        g_norm : float or jax.Array
            Gradient norm
        """
        # Convert JAX arrays to Python scalars
        iteration = int(iteration) if hasattr(iteration, "item") else iteration
        nfev = int(nfev) if hasattr(nfev, "item") else nfev
        cost = float(cost) if hasattr(cost, "item") else cost
        g_norm = float(g_norm) if hasattr(g_norm, "item") else g_norm

        # Handle optional values
        if actual_reduction is not None:
            actual_reduction = (
                float(actual_reduction)
                if hasattr(actual_reduction, "item")
                else actual_reduction
            )
        if step_norm is not None:
            step_norm = float(step_norm) if hasattr(step_norm, "item") else step_norm

        print_iteration_nonlinear(
            iteration, nfev, cost, actual_reduction, step_norm, g_norm
        )

    def trf(
        self,
        fun: Callable,
        xdata: jnp.ndarray | tuple[jnp.ndarray],
        ydata: jnp.ndarray,
        jac: Callable,
        data_mask: jnp.ndarray,
        transform: jnp.ndarray,
        x0: np.ndarray,
        f0: jnp.ndarray,
        J0: jnp.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: int,
        f_scale: float,
        x_scale: np.ndarray,
        loss_function: None | Callable,
        tr_options: dict,
        verbose: int,
        timeit: bool = False,
        solver: str = "exact",
        diagnostics: OptimizationDiagnostics | None = None,
        callback: Callable | None = None,
        **kwargs,
    ) -> dict:
        """Minimize a scalar function of one or more variables using the
        trust-region reflective algorithm. Although I think this is not good
        coding style, I maintained the original code format from SciPy such
        that the code is easier to compare with the original. See the note
        from the algorithms original author below.


        For efficiency, it makes sense to run
        the simplified version of the algorithm when no bounds are imposed.
        We decided to write the two separate functions. It violates the DRY
        principle, but the individual functions are kept the most readable.

        Parameters
        ----------
        fun : callable
            The residual function
        xdata : array_like or tuple of array_like
            The independent variable where the data is measured. If `xdata` is a
            tuple, then the input arguments to `fun` are assumed to be
            ``(xdata[0], xdata[1], ...)``.
        ydata : jnp.ndarray
            The dependent data
        jac : callable
            The Jacobian of `fun`.
        data_mask : jnp.ndarray
            The mask for the data.
        transform : jnp.ndarray
            The uncertainty transform for the data.
        x0 : jnp.ndarray
            Initial guess. Array of real elements of size (n,), where 'n' is the
            number of independent variables.
        f0 : jnp.ndarray
            Initial residuals. Array of real elements of size (m,), where 'm' is
            the number of data points.
        J0 : jnp.ndarray
            Initial Jacobian. Array of real elements of size (m, n), where 'm' is
            the number of data points and 'n' is the number of independent
            variables.
        lb : jnp.ndarray
            Lower bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ub : jnp.ndarray
            Upper bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ftol : float
            Tolerance for termination by the change of the cost function.
        xtol : float
            Tolerance for termination by the change of the independent variables.
        gtol : float
            Tolerance for termination by the norm of the gradient.
        max_nfev : int
            Maximum number of function evaluations.
        f_scale : float
            Cost function scalar
        x_scale : jnp.ndarray
            Scaling factors for independent variables.
        loss_function : callable, optional
            Loss function. If None, the standard least-squares problem is
            solved.
        tr_options : dict
            Options for the trust-region algorithm.
        verbose : int
            Level of algorithm's verbosity:

                * 0 (default) : work silently.
                * 1 : display a termination report.

        timeit : bool, optional
            If True, the time for each step is measured if the unbounded
            version is being ran. Default is False.
        """
        # bounded or unbounded version
        if np.all(lb == -np.inf) and np.all(ub == np.inf):
            # unbounded version as timed and untimed version
            if not timeit:
                return self.trf_no_bounds(
                    fun,
                    xdata,
                    ydata,
                    jac,
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
                    solver,
                    callback,
                    **kwargs,
                )
            else:
                return self.trf_no_bounds_timed(
                    fun,
                    xdata,
                    ydata,
                    jac,
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
                    solver,
                    callback,
                )
        else:
            return self.trf_bounds(
                fun,
                xdata,
                ydata,
                jac,
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
                solver,
                callback,
                **kwargs,
            )

    def _initialize_trf_state(
        self,
        x0: np.ndarray,
        f: jnp.ndarray,
        J: jnp.ndarray,
        loss_function: Callable | None,
        x_scale: np.ndarray | str,
        f_scale: float,
        data_mask: jnp.ndarray,
    ) -> dict:
        """Initialize optimization state for TRF algorithm.

        This helper extracts the initialization logic from trf_no_bounds,
        reducing complexity and improving testability.

        Parameters
        ----------
        x0 : np.ndarray
            Initial parameter guess
        f : jnp.ndarray
            Initial residuals
        J : jnp.ndarray
            Initial Jacobian matrix
        loss_function : Callable or None
            Loss function (None for standard least squares)
        x_scale : np.ndarray or str
            Parameter scaling factors or 'jac' for Jacobian-based scaling
        f_scale : float
            Residual scaling factor
        data_mask : jnp.ndarray
            Data masking array

        Returns
        -------
        dict
            Initial state containing x, f, J, cost, g, scale, Delta, etc.
        """
        m, n = J.shape
        state = {
            "x": x0.copy(),
            "f": f,
            "J": J,
            "nfev": 1,
            "njev": 1,
            "m": m,
            "n": n,
        }

        # Apply loss function if provided
        if loss_function is not None:
            rho = loss_function(f, f_scale)
            state["cost"] = self.calculate_cost(rho, data_mask)
            # Save original residuals before scaling (for res.fun)
            state["f_true"] = f
            state["J"], state["f"] = self.cJIT.scale_for_robust_loss_function(J, f, rho)
        else:
            state["cost"] = self.default_loss_func(f)
            # No scaling applied, so f is already the true residuals
            state["f_true"] = f

        # Compute gradient
        state["g"] = self.compute_grad(state["J"], state["f"])

        # Compute scaling factors
        jac_scale = isinstance(x_scale, str) and x_scale == "jac"
        if jac_scale:
            state["scale"], state["scale_inv"] = self.cJIT.compute_jac_scale(J)
            state["jac_scale"] = True
        else:
            state["scale"], state["scale_inv"] = x_scale, 1 / x_scale
            state["jac_scale"] = False

        # Initialize trust region radius
        Delta = jnorm(x0 * state["scale_inv"])  # Use JAX norm
        state["Delta"] = Delta if Delta > 0 else 1.0

        return state

    def _check_convergence_criteria(
        self,
        g: jnp.ndarray,
        gtol: float,
    ) -> int | None:
        """Check if gradient convergence criterion is met.

        This helper extracts convergence checking logic from trf_no_bounds,
        reducing complexity and improving readability.

        Parameters
        ----------
        g : jnp.ndarray
            Current gradient vector
        gtol : float
            Gradient tolerance for convergence

        Returns
        -------
        int or None
            Termination status: 1 if gradient tolerance satisfied, None otherwise
        """
        g_norm = jnorm(g, ord=jnp.inf)

        if g_norm < gtol:
            self.logger.debug(
                "Convergence: gradient tolerance satisfied",
                g_norm=float(g_norm),
                gtol=gtol,
            )
            return 1

        return None

    def _solve_trust_region_subproblem(
        self,
        J: jnp.ndarray,
        f: jnp.ndarray,
        g: jnp.ndarray,
        scale: np.ndarray,
        Delta: float,
        alpha: float,
        solver: str,
    ) -> dict:
        """Solve the trust region subproblem.

        This helper extracts the subproblem setup and solving logic,
        reducing complexity and improving readability.

        Parameters
        ----------
        J : jnp.ndarray
            Current Jacobian matrix
        f : jnp.ndarray
            Current residuals
        g : jnp.ndarray
            Current gradient
        scale : np.ndarray
            Parameter scaling factors
        Delta : float
            Current trust region radius
        alpha : float
            Levenberg-Marquardt parameter
        solver : str
            Solver type ('cg' or 'exact')

        Returns
        -------
        dict
            Subproblem solution containing:
            - J_h: Scaled Jacobian
            - g_h: Scaled gradient
            - d: Scaling vector
            - d_jnp: JAX scaling vector
            - step_h: Step in scaled space (for CG solver)
            - s, V, uf: SVD components (for exact solver)
        """
        # Setup scaled variables
        d = scale
        d_jnp = jnp.array(scale)
        g_h_jnp = self.compute_grad_hat(g, d_jnp)

        result = {
            "d": d,
            "d_jnp": d_jnp,
            "g_h": g_h_jnp,
        }

        # Solve trust region subproblem
        if solver == "cg":
            # Conjugate gradient solver
            J_h = J * d_jnp
            step_h = self.solve_tr_subproblem_cg(J, f, d_jnp, Delta, alpha)
            result.update(
                {
                    "J_h": J_h,
                    "step_h": step_h,
                    "s": None,
                    "V": None,
                    "uf": None,
                }
            )
        elif solver == "sparse":
            # Sparse solver path (Task 6.4: Sparse Activation)
            # TODO: Implement sparse SVD using JAX sparse operations
            # For now, fall back to dense exact solver to maintain correctness
            # Full sparse implementation would use:
            # - JAX sparse matrix operations for Jacobian
            # - Sparse QR or sparse SVD decomposition
            # - Iterative sparse linear solvers
            # Target: 3-10x speed, 5-50x memory reduction on sparse problems
            svd_output = self.svd_no_bounds(J, d_jnp, f)
            J_h = svd_output[0]
            s, V, uf = svd_output[2:]
            result.update(
                {
                    "J_h": J_h,
                    "step_h": None,
                    "s": s,
                    "V": V,
                    "uf": uf,
                }
            )
        else:
            # SVD-based exact solver (default dense)
            svd_output = self.svd_no_bounds(J, d_jnp, f)
            J_h = svd_output[0]
            # PERFORMANCE FIX: Keep arrays as JAX to avoid conversion overhead (8-12% gain)
            # JAX arrays work with NumPy operations through duck typing, eliminating
            # explicit array conversion reduces memory allocations and copies
            s, V, uf = svd_output[2:]  # Keep as JAX arrays instead of converting
            result.update(
                {
                    "J_h": J_h,
                    "step_h": None,  # Computed later in inner loop
                    "s": s,
                    "V": V,
                    "uf": uf,
                }
            )

        return result

    def _evaluate_step_acceptance(
        self,
        fun: Callable,
        jac: Callable,
        x: np.ndarray,
        f: jnp.ndarray,
        J: jnp.ndarray,
        J_h: jnp.ndarray,
        g_h_jnp: jnp.ndarray,
        cost: float,
        d: np.ndarray,
        d_jnp: jnp.ndarray,
        Delta: float,
        alpha: float,
        step_h: jnp.ndarray | None,
        s: np.ndarray | None,
        V: np.ndarray | None,
        uf: np.ndarray | None,
        xdata: np.ndarray,
        ydata: np.ndarray,
        data_mask: jnp.ndarray,
        transform: Callable | None,
        loss_function: Callable | None,
        f_scale: float,
        scale_inv: np.ndarray,
        jac_scale: bool,
        solver: str,
        ftol: float,
        xtol: float,
        max_nfev: int,
        nfev: int,
    ) -> dict:
        """Evaluate step acceptance through inner trust region loop.

        This method implements the inner loop of the TRF algorithm, which
        repeatedly solves the trust region subproblem and evaluates candidate
        steps until an acceptable step is found.

        Parameters
        ----------
        fun : Callable
            Function to evaluate residuals
        jac : Callable
            Function to evaluate Jacobian
        x : np.ndarray
            Current parameter values
        f : jnp.ndarray
            Current residuals (possibly scaled by loss function)
        J : jnp.ndarray
            Current Jacobian (possibly scaled by loss function)
        J_h : jnp.ndarray
            Scaled Jacobian for subproblem
        g_h_jnp : jnp.ndarray
            Scaled gradient for subproblem
        cost : float
            Current cost value
        d : np.ndarray
            Parameter scaling factors
        d_jnp : jnp.ndarray
            Parameter scaling factors (JAX array)
        Delta : float
            Trust region radius
        alpha : float
            Levenberg-Marquardt parameter
        step_h : jnp.ndarray | None
            Pre-computed step (for CG solver), None for exact solver
        s : np.ndarray | None
            SVD singular values (for exact solver), None for CG
        V : np.ndarray | None
            SVD V matrix (for exact solver), None for CG
        uf : np.ndarray | None
            SVD U^T @ f (for exact solver), None for CG
        xdata : np.ndarray
            Independent variable data
        ydata : np.ndarray
            Dependent variable data
        data_mask : jnp.ndarray
            Mask for valid data points
        transform : Callable | None
            Parameter transformation function
        loss_function : Callable | None
            Robust loss function
        f_scale : float
            Residual scale factor
        scale_inv : np.ndarray
            Inverse parameter scaling
        jac_scale : bool
            Whether using Jacobian-based scaling
        solver : str
            Trust region solver ('cg' or 'exact')
        ftol : float
            Cost function tolerance
        xtol : float
            Parameter tolerance
        max_nfev : int
            Maximum function evaluations
        nfev : int
            Current function evaluation count

        Returns
        -------
        dict
            Dictionary containing:
            - accepted : bool - Whether a step was accepted
            - x_new : np.ndarray - New parameter values (if accepted)
            - f_new : jnp.ndarray - New residuals (if accepted)
            - J_new : jnp.ndarray - New Jacobian (if accepted)
            - cost_new : float - New cost value (if accepted)
            - g_new : jnp.ndarray - New gradient (if accepted)
            - scale : np.ndarray - Updated parameter scaling (if accepted)
            - scale_inv : np.ndarray - Updated inverse scaling (if accepted)
            - actual_reduction : float - Actual cost reduction
            - step_norm : float - Step norm
            - Delta : float - Updated trust region radius
            - alpha : float - Updated Levenberg-Marquardt parameter
            - termination_status : int | None - Termination status code
            - nfev : int - Updated function evaluation count
            - njev : int - Jacobian evaluation count (1 if accepted, 0 otherwise)
        """
        n, m = len(x), len(f)
        actual_reduction = -1
        inner_loop_count = 0
        max_inner_iterations = 100
        termination_status = None
        step_norm = 0

        while (
            actual_reduction <= 0
            and nfev < max_nfev
            and inner_loop_count < max_inner_iterations
        ):
            inner_loop_count += 1

            # Solve subproblem (reuse step or compute new one)
            if solver == "cg":
                if inner_loop_count > 1:
                    step_h = self.solve_tr_subproblem_cg(J, f, d_jnp, Delta, alpha)
                _n_iter = 1  # Dummy value for compatibility
            else:
                step_h, alpha, _n_iter = solve_lsq_trust_region_jax(
                    n, m, uf, s, V, Delta, initial_alpha=alpha
                )

            # Compute predicted reduction
            predicted_reduction_jnp = -self.cJIT.evaluate_quadratic(
                J_h, g_h_jnp, step_h
            )
            predicted_reduction = predicted_reduction_jnp

            # Transform step and evaluate objective
            step = d * step_h
            x_new = x + step
            f_new = fun(x_new, xdata, ydata, data_mask, transform)
            nfev += 1
            step_h_norm = jnorm(step_h)

            # Check for numerical issues
            if not self.check_isfinite(f_new):
                Delta = TR_REDUCTION_FACTOR * step_h_norm
                continue

            # Compute actual reduction
            if loss_function is not None:
                cost_new_jnp = loss_function(f_new, f_scale, data_mask, cost_only=True)
            else:
                cost_new_jnp = self.default_loss_func(f_new)
            cost_new = cost_new_jnp
            actual_reduction = cost - cost_new

            # Update trust region radius
            Delta_new, ratio = update_tr_radius(
                Delta,
                actual_reduction,
                predicted_reduction,
                step_h_norm,
                step_h_norm > TR_BOUNDARY_THRESHOLD * Delta,
            )

            # Check termination criteria
            step_norm = jnorm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, jnorm(x), ratio, ftol, xtol
            )

            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

            # Exit inner loop if we have a successful step
            if actual_reduction > 0:
                break

        # Check if inner loop hit iteration limit
        if inner_loop_count >= max_inner_iterations:
            self.logger.warning(
                "Inner optimization loop hit iteration limit",
                inner_iterations=inner_loop_count,
                actual_reduction=actual_reduction,
            )
            termination_status = -3  # Inner loop limit exceeded

        # Prepare result
        result = {
            "accepted": actual_reduction > 0,
            "actual_reduction": max(0, actual_reduction),
            "step_norm": step_norm if actual_reduction > 0 else 0,
            "Delta": Delta,
            "alpha": alpha,
            "termination_status": termination_status,
            "nfev": nfev,
            "njev": 0,  # Will be set to 1 if step is accepted
        }

        # If step was accepted, compute new state
        if actual_reduction > 0:
            result.update(
                {
                    "x_new": x_new,
                    "f_new": f_new,
                    "cost_new": cost_new,
                    "njev": 1,
                }
            )

            # Compute new Jacobian
            J_new = jac(x_new, xdata, ydata, data_mask, transform)

            # Apply loss function if provided
            if loss_function is not None:
                rho = loss_function(f_new, f_scale)
                J_new, f_new_scaled = self.cJIT.scale_for_robust_loss_function(
                    J_new, f_new, rho
                )
                result["f_new"] = f_new_scaled  # Scaled residuals for optimization
                result["f_true_new"] = f_new  # Unscaled residuals for res.fun
            else:
                result["f_new"] = f_new
                result["f_true_new"] = f_new  # No scaling, so both are the same

            result["J_new"] = J_new

            # Compute new gradient
            g_new = self.compute_grad(J_new, result["f_new"])
            result["g_new"] = g_new

            # Update scaling if using Jacobian-based scaling
            if jac_scale:
                scale_new, scale_inv_new = self.cJIT.compute_jac_scale(J_new, scale_inv)
                result["scale"] = scale_new
                result["scale_inv"] = scale_inv_new

        return result

    def trf_no_bounds(
        self,
        fun: Callable,
        xdata: jnp.ndarray | tuple[jnp.ndarray],
        ydata: jnp.ndarray,
        jac: Callable,
        data_mask: jnp.ndarray,
        transform: jnp.ndarray,
        x0: np.ndarray,
        f: jnp.ndarray,
        J: jnp.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: int,
        f_scale: float,
        x_scale: np.ndarray,
        loss_function: None | Callable,
        tr_options: dict,
        verbose: int,
        solver: str = "exact",
        callback: Callable | None = None,
        profiler: TRFProfiler | NullProfiler | None = None,
        mixed_precision_manager: MixedPrecisionManager | None = None,
        mixed_precision_config: MixedPrecisionConfig | None = None,
        **kwargs,
    ) -> dict:
        """Unbounded version of the trust-region reflective algorithm.

        Parameters
        ----------
        fun : callable
            The residual function
        xdata : array_like or tuple of array_like
            The independent variable where the data is measured. If `xdata` is a
            tuple, then the input arguments to `fun` are assumed to be
            ``(xdata[0], xdata[1], ...)``.
        ydata : jnp.ndarray
            The dependent data
        jac : callable
            The Jacobian of `fun`.
        data_mask : jnp.ndarray
            The mask for the data.
        transform : jnp.ndarray
            The uncertainty transform for the data.
        x0 : jnp.ndarray
            Initial guess. Array of real elements of size (n,), where 'n' is the
            number of independent variables.
        f0 : jnp.ndarray
            Initial residuals. Array of real elements of size (m,), where 'm' is
            the number of data points.
        J0 : jnp.ndarray
            Initial Jacobian. Array of real elements of size (m, n), where 'm' is
            the number of data points and 'n' is the number of independent
            variables.
        lb : jnp.ndarray
            Lower bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ub : jnp.ndarray
            Upper bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ftol : float
            Tolerance for termination by the change of the cost function.
        xtol : float
            Tolerance for termination by the change of the independent variables.
        gtol : float
            Tolerance for termination by the norm of the gradient.
        max_nfev : int
            Maximum number of function evaluations.
        f_scale : float
            Cost function scalar
        x_scale : jnp.ndarray
            Scaling factors for independent variables.
        loss_function : callable, optional
            Loss function. If None, the standard least-squares problem is
            solved.
        tr_options : dict
            Options for the trust-region algorithm.
        verbose : int
            Level of algorithm's verbosity:

                * 0 (default) : work silently.
                * 1 : display a termination report.

        mixed_precision_manager : MixedPrecisionManager, optional
            Pre-initialized mixed precision manager. If provided, mixed_precision_config
            is ignored. Use when sharing manager across multiple optimizations.
        mixed_precision_config : MixedPrecisionConfig, optional
            Configuration for automatic mixed precision fallback. If provided and
            mixed_precision_manager is None, a new manager is created with this config.
            Default is None (mixed precision disabled).

        Returns
        -------
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

        profiler : TRFProfiler, NullProfiler, or None, optional
            Profiler for timing algorithm operations. If None, uses NullProfiler
            (zero overhead). Use TRFProfiler() for detailed performance analysis.
            Default is None.

        Notes
        -----
        The algorithm is described in [13]_.

        MAINTENANCE NOTE: There is a profiling-instrumented version of this function
        called `trf_no_bounds_timed()` used for performance analysis. If you modify
        this function, please apply equivalent changes there. See TRFProfiler classes
        above for future consolidation approach.

        """

        # Initialize profiler (NullProfiler if not provided for zero overhead)
        if profiler is None:
            profiler = NullProfiler()

        # Initialize mixed precision manager if configured
        if mixed_precision_manager is None and mixed_precision_config is not None:
            mixed_precision_manager = MixedPrecisionManager(
                mixed_precision_config, verbose=(verbose > 0)
            )

        # Store original tolerances for potential fallback
        original_tolerances = {"ftol": ftol, "xtol": xtol, "gtol": gtol}

        # Initialize optimization state using helper
        state = self._initialize_trf_state(
            x0=x0,
            f=f,
            J=J,
            loss_function=loss_function,
            x_scale=x_scale,
            f_scale=f_scale,
            data_mask=data_mask,
        )

        # Extract state variables
        x = state["x"]
        f = state["f"]
        J = state["J"]
        cost = state["cost"]
        g = state["g"]
        g_jnp = g  # Keep as JAX array for performance
        scale = state["scale"]
        scale_inv = state["scale_inv"]
        Delta = state["Delta"]
        nfev = state["nfev"]
        njev = state["njev"]
        m = state["m"]
        n = state["n"]
        jac_scale = state["jac_scale"]
        f_true = state["f_true"]  # Original unscaled residuals (for res.fun)

        # Log optimization start
        self.logger.info(
            "Starting TRF optimization (no bounds)",
            n_params=n,
            n_residuals=m,
            max_nfev=max_nfev,
        )

        # Set max_nfev if not provided
        if max_nfev is None:
            max_nfev = x0.size * DEFAULT_MAX_NFEV_MULTIPLIER

        alpha = INITIAL_LEVENBERG_MARQUARDT_LAMBDA  # "Levenberg-Marquardt" parameter

        termination_status = None
        iteration = 0
        step_norm = None
        actual_reduction = None

        if verbose == 2:
            print_header_nonlinear()

        # Trust region optimization loop
        with self.logger.timer("optimization", log_result=False):
            while True:
                # Check gradient convergence using helper (only if not already terminated)
                if termination_status is None:
                    termination_status = self._check_convergence_criteria(g, gtol)
                g_norm = jnorm(g, ord=jnp.inf)  # For logging/printing

                if verbose == 2:
                    # Use jax.debug.callback to avoid blocking host-device transfers
                    # Callback runs asynchronously and doesn't block JAX execution
                    debug.callback(
                        self._log_iteration_callback,
                        iteration,
                        nfev,
                        cost,
                        actual_reduction,
                        step_norm,
                        g_norm,
                    )

                if termination_status is not None or nfev == max_nfev:
                    if nfev == max_nfev:
                        self.logger.warning(
                            "Maximum number of function evaluations reached", nfev=nfev
                        )
                    break

                # Log iteration details
                self.logger.optimization_step(
                    iteration=iteration,
                    cost=cost,
                    gradient_norm=g_norm,
                    step_size=Delta if iteration > 0 else None,
                    nfev=nfev,
                )

                # Solve trust region subproblem using helper
                subproblem_result = self._solve_trust_region_subproblem(
                    J=J,
                    f=f,
                    g=g,
                    scale=scale,
                    Delta=Delta,
                    alpha=alpha,
                    solver=solver,
                )

                # Extract subproblem solution
                d = subproblem_result["d"]
                d_jnp = subproblem_result["d_jnp"]
                g_h_jnp = subproblem_result["g_h"]
                J_h = subproblem_result["J_h"]
                step_h = subproblem_result["step_h"]
                s = subproblem_result["s"]
                V = subproblem_result["V"]
                uf = subproblem_result["uf"]

                # Evaluate and potentially accept step using helper
                acceptance_result = self._evaluate_step_acceptance(
                    fun=fun,
                    jac=jac,
                    x=x,
                    f=f,
                    J=J,
                    J_h=J_h,
                    g_h_jnp=g_h_jnp,
                    cost=cost,
                    d=d,
                    d_jnp=d_jnp,
                    Delta=Delta,
                    alpha=alpha,
                    step_h=step_h,
                    s=s,
                    V=V,
                    uf=uf,
                    xdata=xdata,
                    ydata=ydata,
                    data_mask=data_mask,
                    transform=transform,
                    loss_function=loss_function,
                    f_scale=f_scale,
                    scale_inv=scale_inv,
                    jac_scale=jac_scale,
                    solver=solver,
                    ftol=ftol,
                    xtol=xtol,
                    max_nfev=max_nfev,
                    nfev=nfev,
                )

                # Update state from acceptance result
                if acceptance_result["accepted"]:
                    x = acceptance_result["x_new"]
                    f = acceptance_result["f_new"]  # Scaled residuals for optimization
                    f_true = acceptance_result[
                        "f_true_new"
                    ]  # Unscaled residuals for res.fun
                    J = acceptance_result["J_new"]
                    cost = acceptance_result["cost_new"]
                    g = acceptance_result["g_new"]
                    g_jnp = g
                    njev += acceptance_result["njev"]

                    if jac_scale and "scale" in acceptance_result:
                        scale = acceptance_result["scale"]
                        scale_inv = acceptance_result["scale_inv"]

                actual_reduction = acceptance_result["actual_reduction"]
                step_norm = acceptance_result["step_norm"]
                Delta = acceptance_result["Delta"]
                alpha = acceptance_result["alpha"]
                nfev = acceptance_result["nfev"]

                if acceptance_result["termination_status"] is not None:
                    termination_status = acceptance_result["termination_status"]

                iteration += 1

                # Mixed precision monitoring and upgrade
                if (
                    mixed_precision_manager is not None
                    and acceptance_result["accepted"]
                ):
                    # Compute parameter change for precision monitoring
                    param_change = jnorm(d_jnp) if step_norm is not None else 0.0

                    # Check for NaN/Inf in current state
                    has_nan_inf = bool(
                        jnp.isnan(f).any()
                        or jnp.isinf(f).any()
                        or jnp.isnan(J).any()
                        or jnp.isinf(J).any()
                        or jnp.isnan(g_jnp).any()
                        or jnp.isinf(g_jnp).any()
                    )

                    # Report metrics to manager
                    metrics = ConvergenceMetrics(
                        iteration=iteration,
                        residual_norm=float(jnorm(f)),
                        gradient_norm=float(g_norm),
                        parameter_change=float(param_change),
                        cost=float(cost),
                        trust_radius=float(Delta),
                        has_nan_inf=has_nan_inf,
                    )
                    mixed_precision_manager.report_metrics(metrics)

                    # Update best parameters
                    mixed_precision_manager.update_best(x, float(cost), iteration)

                    # Check if precision upgrade needed
                    if mixed_precision_manager.should_upgrade():
                        # Create optimization state for upgrade
                        opt_state = OptimizationState(
                            x=x,
                            f=f,
                            J=J,
                            g=g_jnp,
                            cost=float(cost),
                            trust_radius=float(Delta),
                            iteration=iteration,
                            dtype=x.dtype,
                            algorithm_specific={"alpha": alpha},
                        )

                        # Perform upgrade
                        upgraded_state = mixed_precision_manager.upgrade_precision(
                            opt_state
                        )

                        # Update optimization variables with upgraded state
                        x = upgraded_state.x
                        f = upgraded_state.f
                        J = upgraded_state.J
                        g = upgraded_state.g
                        g_jnp = g
                        cost = upgraded_state.cost
                        Delta = upgraded_state.trust_radius
                        iteration = upgraded_state.iteration
                        alpha = upgraded_state.algorithm_specific["alpha"]

                        # Continue optimization in float64
                        self.logger.info(
                            "Continuing optimization in float64",
                            iteration=iteration,
                            cost=float(cost),
                        )

                # Invoke user callback if provided
                if callback is not None:
                    try:
                        callback(
                            iteration=iteration,
                            cost=float(cost),  # JAX scalar → Python float
                            params=np.array(x),  # JAX array → NumPy array
                            info={
                                "gradient_norm": float(g_norm),
                                "nfev": nfev,
                                "step_norm": float(step_norm)
                                if step_norm is not None
                                else None,
                                "actual_reduction": float(actual_reduction)
                                if actual_reduction is not None
                                else None,
                            },
                        )
                    except StopOptimization:
                        termination_status = 2  # User-requested stop
                        self.logger.info(
                            "Optimization stopped by callback (StopOptimization)"
                        )
                        break
                    except Exception as e:
                        warnings.warn(
                            f"Callback raised exception: {e}. Continuing optimization.",
                            RuntimeWarning,
                        )

        if termination_status is None:
            termination_status = 0

        # Float64 failure fallback: If float64 optimization failed to converge,
        # fall back to relaxed float32 with best parameters from history
        if (
            mixed_precision_manager is not None
            and mixed_precision_manager.state == PrecisionState.FLOAT64_ACTIVE
            and termination_status == 0  # Max iterations reached without convergence
        ):
            self.logger.info(
                "Float64 optimization failed to converge, applying relaxed float32 fallback"
            )

            # Get best state from entire optimization history
            best_params = mixed_precision_manager.get_best_parameters()
            best_cost = mixed_precision_manager.tracker.get_best_cost()
            best_iteration = mixed_precision_manager.tracker.best_iteration

            # Create state with best parameters
            fallback_state = OptimizationState(
                x=best_params,
                f=f,  # Will be recomputed
                J=J,  # Will be recomputed
                g=g,  # Will be recomputed
                cost=best_cost,
                trust_radius=float(Delta),
                iteration=best_iteration,
                dtype=jnp.float64,
                algorithm_specific={"alpha": alpha},
            )

            # Apply relaxed fallback (converts to float32, relaxes tolerances)
            fallback_state, relaxed_tol = (
                mixed_precision_manager.apply_relaxed_fallback(
                    fallback_state, original_tolerances
                )
            )

            self.logger.info(
                f"Retrying with relaxed tolerances: "
                f"gtol={relaxed_tol['gtol']:.2e}, "
                f"ftol={relaxed_tol['ftol']:.2e}, "
                f"xtol={relaxed_tol['xtol']:.2e}"
            )

            # Retry optimization with relaxed criteria and half iteration budget
            retry_max_nfev = max(max_nfev // 2, 50)
            x = fallback_state.x
            Delta = fallback_state.trust_radius
            iteration = 0  # Reset iteration counter for retry

            # Recompute initial state for retry
            f, J, cost, g, g_norm, _ = self._compute_initial_state(
                fun, xdata, ydata, jac, x, loss_function, f_scale, data_mask
            )
            g_jnp = g

            # Retry loop with relaxed tolerances
            for retry_iter in range(retry_max_nfev):
                # Check relaxed convergence criteria
                if g_norm < relaxed_tol["gtol"]:
                    termination_status = 1  # Gradient tolerance satisfied
                    self.logger.info(
                        f"Fallback converged via gradient tolerance at iteration {retry_iter}"
                    )
                    break

                # Compute trust region step
                try:
                    step_result = self.compute_trust_region_step(
                        J=J,
                        g=g_jnp,
                        Delta=Delta,
                        lb_scaled=None,
                        ub_scaled=None,
                        theta=0.0,
                        solver=solver,
                        tr_options=tr_options,
                    )
                    d_jnp = step_result["step"]
                    step_norm = step_result.get("step_norm")
                except Exception as e:
                    self.logger.warning(f"Fallback step computation failed: {e}")
                    break

                # Evaluate step
                acceptance_result = self._evaluate_step(
                    fun=fun,
                    xdata=xdata,
                    ydata=ydata,
                    jac=jac,
                    x=x,
                    f=f,
                    cost=cost,
                    J=J,
                    g=g_jnp,
                    d=d_jnp,
                    Delta=Delta,
                    loss_function=loss_function,
                    f_scale=f_scale,
                    data_mask=data_mask,
                )

                if acceptance_result["accepted"]:
                    # Update state
                    x = acceptance_result["x_new"]
                    f = acceptance_result["f_new"]
                    cost = acceptance_result["cost_new"]
                    J = acceptance_result["J_new"]
                    g = acceptance_result["g_new"]
                    g_jnp = g
                    g_norm = acceptance_result["g_norm_new"]

                    # Update trust radius
                    if acceptance_result["ratio"] > 0.75:
                        Delta = min(Delta * 2.0, MAX_TRUST_RADIUS)
                    elif acceptance_result["ratio"] < 0.25:
                        Delta *= 0.5

                    # Check relaxed convergence
                    if (
                        acceptance_result.get("cost_reduction", 0)
                        < relaxed_tol["ftol"] * cost
                    ):
                        termination_status = 2  # Cost tolerance satisfied
                        self.logger.info(
                            f"Fallback converged via cost tolerance at iteration {retry_iter}"
                        )
                        break

                    if step_norm is not None and step_norm < relaxed_tol["xtol"]:
                        termination_status = 3  # Step tolerance satisfied
                        self.logger.info(
                            f"Fallback converged via step tolerance at iteration {retry_iter}"
                        )
                        break
                else:
                    # Reduce trust radius
                    Delta *= 0.5
                    if Delta < MIN_TRUST_RADIUS:
                        self.logger.info("Fallback trust radius too small, stopping")
                        break

            # Log final fallback result
            final_best_params = mixed_precision_manager.get_best_parameters()
            final_best_cost = mixed_precision_manager.tracker.get_best_cost()
            self.logger.info(
                f"Fallback complete. Best cost: {final_best_cost:.6e} "
                f"(status: {termination_status})"
            )

            # Use best parameters from entire history for final result
            x = final_best_params
            cost = final_best_cost

        active_mask = jnp.zeros_like(x)  # JAX zeros instead of NumPy

        # Convert JAX arrays to NumPy for final return
        return OptimizeResult(
            x=x,
            cost=float(cost),  # Convert JAX scalar to Python float
            fun=f_true,
            jac=J,
            grad=np.array(g),  # Convert JAX array to NumPy
            optimality=float(g_norm),  # Convert JAX scalar to Python float
            active_mask=active_mask,
            nfev=nfev,
            njev=njev,
            nit=iteration,  # Number of iterations performed
            status=termination_status,
            all_times={},
        )

    def trf_bounds(
        self,
        fun: Callable,
        xdata: jnp.ndarray | tuple[jnp.ndarray],
        ydata: jnp.ndarray,
        jac: Callable,
        data_mask: jnp.ndarray,
        transform: jnp.ndarray,
        x0: np.ndarray,
        f: jnp.ndarray,
        J: jnp.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: int,
        f_scale: float,
        x_scale: np.ndarray,
        loss_function: None | Callable,
        tr_options: dict,
        verbose: int,
        solver: str = "exact",
        callback: Callable | None = None,
        **kwargs,
    ) -> dict:
        """Bounded version of the trust-region reflective algorithm.

        Parameters
        ----------
        fun : callable
            The residual function
        xdata : array_like or tuple of array_like
            The independent variable where the data is measured. If `xdata` is a
            tuple, then the input arguments to `fun` are assumed to be
            ``(xdata[0], xdata[1], ...)``.
        ydata : jnp.ndarray
            The dependent data
        jac : callable
            The Jacobian of `fun`.
        data_mask : jnp.ndarray
            The mask for the data.
        transform : jnp.ndarray
            The uncertainty transform for the data.
        x0 : jnp.ndarray
            Initial guess. Array of real elements of size (n,), where 'n' is the
            number of independent variables.
        f0 : jnp.ndarray
            Initial residuals. Array of real elements of size (m,), where 'm' is
            the number of data points.
        J0 : jnp.ndarray
            Initial Jacobian. Array of real elements of size (m, n), where 'm' is
            the number of data points and 'n' is the number of independent
            variables.
        lb : jnp.ndarray
            Lower bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ub : jnp.ndarray
            Upper bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ftol : float
            Tolerance for termination by the change of the cost function.
        xtol : float
            Tolerance for termination by the change of the independent variables.
        gtol : float
            Tolerance for termination by the norm of the gradient.
        max_nfev : int
            Maximum number of function evaluations.
        f_scale : float
            Cost function scalar
        x_scale : jnp.ndarray
            Scaling factors for independent variables.
        loss_function : callable, optional
            Loss function. If None, the standard least-squares problem is
            solved.
        tr_options : dict
            Options for the trust-region algorithm.
        verbose : int
            Level of algorithm's verbosity:

                * 0 (default) : work silently.
                * 1 : display a termination report.

        Returns
        -------
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

        Notes
        -----
        The algorithm is described in [13]_.

        References
        ----------
        .. [13] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation and
                Theory," in Numerical Analysis, ed. G. A. Watson (1978), pp. 105-116.
                DOI: 10.1017/CBO9780511819595.006
        .. [2] T. F. Coleman and Y. Li, “An interior trust region approach for
                nonlinear minimization subject to bounds,” SIAM Journal on
                Optimization, vol. 6, no. 2, pp. 418–445, 1996.
        """

        x = x0.copy()
        f_true = f
        nfev = 1
        njev = 1
        m, n = J.shape

        if loss_function is not None:
            rho = loss_function(f, f_scale)
            cost_jnp = self.calculate_cost(rho, data_mask)
            J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)
        else:
            cost_jnp = self.default_loss_func(f)

        # Keep cost as JAX array for performance (convert only when needed)
        cost = cost_jnp

        g_jnp = self.compute_grad(J, f)
        # Keep gradient as JAX array for performance
        g = g_jnp

        jac_scale = isinstance(x_scale, str) and x_scale == "jac"
        if jac_scale:
            scale, scale_inv = self.cJIT.compute_jac_scale(J)
        else:
            scale, scale_inv = x_scale, 1 / x_scale

        v, dv = CL_scaling_vector(x, g, lb, ub)

        # Convert to JAX arrays and use .at[] syntax for immutable array updates
        v = jnp.array(v)
        dv = jnp.array(dv)
        mask = dv != 0
        v = v.at[mask].set(v[mask] * scale_inv[mask])
        Delta = jnorm(x0 * scale_inv / v**SQRT_EXPONENT)
        if Delta == 0:
            Delta = 1.0

        # Use JAX norm for gradient norm calculation
        g_norm = jnorm(g * v, ord=jnp.inf)

        if max_nfev is None:
            max_nfev = x0.size * DEFAULT_MAX_NFEV_MULTIPLIER

        alpha = INITIAL_LEVENBERG_MARQUARDT_LAMBDA  # "Levenberg-Marquardt" parameter

        termination_status = None
        iteration = 0
        step_norm = None
        actual_reduction = None

        if verbose == 2:
            print_header_nonlinear()

        f_zeros = jnp.zeros([n])
        while True:
            v, dv = CL_scaling_vector(x, g, lb, ub)
            # Convert to JAX arrays for later .at[] operations
            v = jnp.array(v)
            dv = jnp.array(dv)

            # Use JAX norm for gradient norm calculation
            g_norm = jnorm(g * v, ord=jnp.inf)
            if g_norm < gtol:
                termination_status = 1

            if verbose == 2:
                # Use jax.debug.callback to avoid blocking host-device transfers
                debug.callback(
                    self._log_iteration_callback,
                    iteration,
                    nfev,
                    cost,
                    actual_reduction,
                    step_norm,
                    g_norm,
                )

            if termination_status is not None or nfev == max_nfev:
                break

            # Now compute variables in "hat" space. Here, we also account for
            # scaling introduced by `x_scale` parameter. This part is a bit tricky,
            # you have to write down the formulas and see how the trust-region
            # problem is formulated when the two types of scaling are applied.
            # The idea is that first we apply `x_scale` and then apply Coleman-Li
            # approach in the new variables.

            # v is recomputed in the variables after applying `x_scale`, note that
            # components which were identically 1 not affected.
            # Use JAX .at[] syntax for immutable array updates
            mask = dv != 0
            v = v.at[mask].set(v[mask] * scale_inv[mask])

            # Here, we apply two types of scaling.
            d = v**SQRT_EXPONENT * scale

            # C = diag(g * scale) Jv
            diag_h = g * dv * scale

            # After all this has been done, we continue normally.

            # "hat" gradient.
            g_h = d * g
            J_diag = jnp.diag(diag_h**SQRT_EXPONENT)
            d_jnp = jnp.array(d)

            # Choose solver based on solver parameter
            if solver == "cg":
                # Use conjugate gradient solver
                J_h = J * d_jnp
                p_h = self.solve_tr_subproblem_cg_bounds(
                    J, f, d_jnp, J_diag, f_zeros, Delta, alpha
                )
                s, V, uf = None, None, None  # Not needed for CG path
            elif solver == "sparse":
                # Sparse solver path (Task 6.4: Sparse Activation)
                # TODO: Implement sparse SVD with bounds
                # For now, fall back to dense exact solver to maintain correctness
                output = self.svd_bounds(f, J, d_jnp, J_diag, f_zeros)
                J_h = output[0]
                s, V, uf = output[2:]
            else:
                # Use exact SVD solver (default)
                output = self.svd_bounds(f, J, d_jnp, J_diag, f_zeros)
                J_h = output[0]
                s, V, uf = output[2:]  # Keep as JAX arrays - no NumPy conversion

            # theta controls step back step ratio from the bounds.
            theta = max(0.995, 1 - g_norm)

            actual_reduction = -1
            inner_loop_count = 0
            max_inner_iterations = 100  # Prevent infinite loops
            while (
                actual_reduction <= 0
                and nfev < max_nfev
                and inner_loop_count < max_inner_iterations
            ):
                inner_loop_count += 1

                if solver == "cg":
                    # CG path: step already computed
                    # For subsequent iterations in inner loop, re-solve with updated alpha
                    if inner_loop_count > 1:
                        p_h = self.solve_tr_subproblem_cg_bounds(
                            J, f, d_jnp, J_diag, f_zeros, Delta, alpha
                        )
                    _n_iter = 1  # Dummy value for compatibility
                else:
                    # SVD path: use exact solver (JAX version for pure JAX execution)
                    p_h, alpha, _n_iter = solve_lsq_trust_region_jax(
                        n, m, uf, s, V, Delta, initial_alpha=alpha
                    )

                p = d * p_h  # Trust-region solution in the original space.
                step, step_h, predicted_reduction = self.select_step(
                    x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta
                )

                x_new = make_strictly_feasible(x + step.copy(), lb, ub, rstep=0)
                f_new = fun(x_new, xdata, ydata, data_mask, transform)

                nfev += 1

                step_h_norm = jnorm(step_h)
                if not self.check_isfinite(f_new):
                    Delta = 0.25 * step_h_norm
                    continue

                if loss_function is not None:
                    cost_new_jnp = loss_function(
                        f_new, f_scale, data_mask, cost_only=True
                    )
                else:
                    cost_new_jnp = self.default_loss_func(f_new)
                # Keep as JAX array for performance
                cost_new = cost_new_jnp

                actual_reduction = cost - cost_new
                Delta_new, ratio = update_tr_radius(
                    Delta,
                    actual_reduction,
                    predicted_reduction,
                    step_h_norm,
                    step_h_norm > 0.95 * Delta,
                )

                step_norm = jnorm(step)
                termination_status = check_termination(
                    actual_reduction, cost, step_norm, jnorm(x), ratio, ftol, xtol
                )
                if termination_status is not None:
                    break

                alpha *= Delta / Delta_new
                Delta = Delta_new

            # Check if inner loop hit iteration limit
            if inner_loop_count >= max_inner_iterations:
                self.logger.warning(
                    "Inner optimization loop hit iteration limit",
                    inner_iterations=inner_loop_count,
                    actual_reduction=actual_reduction,
                )
                termination_status = -3  # Inner loop limit exceeded

            if actual_reduction > 0:
                x = x_new
                f = f_new
                f_true = f

                cost = cost_new

                J = jac(x, xdata, ydata, data_mask, transform)

                njev += 1

                if loss_function is not None:
                    rho = loss_function(f, f_scale)
                    J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)

                g_jnp = self.compute_grad(J, f)
                if jac_scale:
                    scale, scale_inv = self.cJIT.compute_jac_scale(J, scale_inv)
                # Keep as JAX array for performance
                g = g_jnp
            else:
                step_norm = 0
                actual_reduction = 0

            iteration += 1

            # Invoke user callback if provided
            if callback is not None:
                try:
                    callback(
                        iteration=iteration,
                        cost=float(cost),  # JAX scalar → Python float
                        params=np.array(x),  # JAX array → NumPy array
                        info={
                            "gradient_norm": float(g_norm),
                            "nfev": nfev,
                            "step_norm": float(step_norm)
                            if step_norm is not None
                            else None,
                            "actual_reduction": float(actual_reduction)
                            if actual_reduction is not None
                            else None,
                        },
                    )
                except StopOptimization:
                    termination_status = 2  # User-requested stop
                    self.logger.info(
                        "Optimization stopped by callback (StopOptimization)"
                    )
                    break
                except Exception as e:
                    warnings.warn(
                        f"Callback raised exception: {e}. Continuing optimization.",
                        RuntimeWarning,
                    )

        if termination_status is None:
            termination_status = 0

        active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
        # Convert JAX arrays to NumPy for final return
        return OptimizeResult(
            x=x,
            cost=float(cost),  # Convert JAX scalar to Python float
            fun=f_true,
            jac=J,
            grad=np.array(g),  # Convert JAX array to NumPy
            optimality=float(g_norm),  # Convert JAX scalar to Python float
            active_mask=active_mask,
            nfev=nfev,
            njev=njev,
            nit=iteration,  # Number of iterations performed
            status=termination_status,
        )

    def select_step(
        self,
        x: np.ndarray,
        J_h: jnp.ndarray,
        diag_h: jnp.ndarray,
        g_h: jnp.ndarray,
        p: np.ndarray,
        p_h: np.ndarray,
        d: np.ndarray,
        Delta: float,
        lb: np.ndarray,
        ub: np.ndarray,
        theta: float,
    ):
        """Select the best step according to Trust Region Reflective algorithm.

        Parameters
        ----------
        x : np.ndarray
            Current set parameter vector.
        J_h : jnp.ndarray
            Jacobian matrix in the scaled 'hat' space.
        diag_h : jnp.ndarray
            Diagonal of the scaled matrix C = diag(g * scale) Jv?
        g_h : jnp.ndarray
            Gradient vector in the scaled 'hat' space.
        p : np.ndarray
            Trust-region step in the original space.
        p_h : np.ndarray
            Trust-region step in the scaled 'hat' space.
        d : np.ndarray
            Scaling vector.
        Delta : float
            Trust-region radius.
        lb : np.ndarray
            Lower bounds on variables.
        ub : np.ndarray
            Upper bounds on variables.
        theta : float
            Controls step back step ratio from the bounds.

        Returns
        -------
        step : np.ndarray
            Step in the original space.
        step_h : np.ndarray
            Step in the scaled 'hat' space.
        predicted_reduction : float
            Predicted reduction in the cost function.
        """
        if in_bounds(x + p, lb, ub):
            p_value = self.cJIT.evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
            return p, p_h, -p_value

        p_stride, hits = step_size_to_bound(x, p, lb, ub)

        # Compute the reflected direction.
        r_h = jnp.array(p_h)  # JAX copy instead of NumPy
        # Use JAX .at[] syntax for immutable array updates
        hits_mask = hits.astype(bool)
        r_h = r_h.at[hits_mask].set(r_h[hits_mask] * -1)
        r = d * r_h

        # Restrict trust-region step, such that it hits the bound.
        p *= p_stride
        p_h *= p_stride
        x_on_bound = x + p

        # Reflected direction will cross first either feasible region or trust
        # region boundary.
        _, to_tr = intersect_trust_region(p_h, r_h, Delta)
        to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)

        # Find lower and upper bounds on a step size along the reflected
        # direction, considering the strict feasibility requirement. There is no
        # single correct way to do that, the chosen approach seems to work best
        # on test problems.
        r_stride = min(to_bound, to_tr)
        if r_stride > 0:
            r_stride_l = (1 - theta) * p_stride / r_stride
            r_stride_u = theta * to_bound if r_stride == to_bound else to_tr
        else:
            r_stride_l = 0
            r_stride_u = -1

        # Check if reflection step is available.
        if r_stride_l <= r_stride_u:
            a, b, c = self.cJIT.build_quadratic_1d(J_h, g_h, r_h, s0=p_h, diag=diag_h)

            r_stride, r_value = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c=c)
            r_h *= r_stride
            r_h += p_h
            r = r_h * d
        else:
            r_value = jnp.inf  # JAX infinity instead of NumPy

        # Now correct p_h to make it strictly interior.
        p *= theta
        p_h *= theta
        p_value = self.cJIT.evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)

        ag_h = -g_h
        ag = d * ag_h

        to_tr = Delta / jnorm(ag_h)
        to_bound, _ = step_size_to_bound(x, ag, lb, ub)
        ag_stride = theta * to_bound if to_bound < to_tr else to_tr

        a, b = self.cJIT.build_quadratic_1d(J_h, g_h, ag_h, diag=diag_h)
        ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
        ag_h *= ag_stride
        ag *= ag_stride

        if p_value < r_value and p_value < ag_value:
            return p, p_h, -p_value
        elif r_value < p_value and r_value < ag_value:
            return r, r_h, -r_value
        else:
            return ag, ag_h, -ag_value

    def trf_no_bounds_timed(
        self,
        fun: Callable,
        xdata: jnp.ndarray | tuple[jnp.ndarray],
        ydata: jnp.ndarray,
        jac: Callable,
        data_mask: jnp.ndarray,
        transform: jnp.ndarray,
        x0: np.ndarray,
        f: jnp.ndarray,
        J: jnp.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: int,
        f_scale: float,
        x_scale: np.ndarray,
        loss_function: None | Callable,
        tr_options: dict,
        verbose: int,
        solver: str = "exact",
        callback: Callable | None = None,
    ) -> dict:
        """Trust Region Reflective algorithm with detailed profiling.

        MAINTENANCE NOTE
        ----------------
        This function is a profiling-instrumented version of `trf_no_bounds()`.
        It includes .block_until_ready() calls after every JAX operation to get
        accurate GPU timing, which adds overhead unsuitable for production use.

        **If you modify trf_no_bounds(), please apply equivalent changes here.**

        The two functions implement the same algorithm but differ in:
        - This version: Adds timing instrumentation via block_until_ready()
        - trf_no_bounds(): Uses helper methods (_initialize_trf_state, etc.)

        Future work: Consolidate using TRFProfiler abstraction (see classes above).

        This function records timing for each operation and returns them in the
        `all_times` field of the result. Used exclusively for performance analysis
        in benchmarks/profile_trf.py.

        Parameters
        ----------
        fun : callable
            The residual function
        xdata : array_like or tuple of array_like
            The independent variable where the data is measured. If `xdata` is a
            tuple, then the input arguments to `fun` are assumed to be
            ``(xdata[0], xdata[1], ...)``.
        ydata : jnp.ndarray
            The dependent data
        jac : callable
            The Jacobian of `fun`.
        data_mask : jnp.ndarray
            The mask for the data.
        transform : jnp.ndarray
            The uncertainty transform for the data.
        x0 : jnp.ndarray
            Initial guess. Array of real elements of size (n,), where 'n' is the
            number of independent variables.
        f0 : jnp.ndarray
            Initial residuals. Array of real elements of size (m,), where 'm' is
            the number of data points.
        J0 : jnp.ndarray
            Initial Jacobian. Array of real elements of size (m, n), where 'm' is
            the number of data points and 'n' is the number of independent
            variables.
        lb : jnp.ndarray
            Lower bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ub : jnp.ndarray
            Upper bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ftol : float
            Tolerance for termination by the change of the cost function.
        xtol : float
            Tolerance for termination by the change of the independent variables.
        gtol : float
            Tolerance for termination by the norm of the gradient.
        max_nfev : int
            Maximum number of function evaluations.
        f_scale : float
            Cost function scalar
        x_scale : jnp.ndarray
            Scaling factors for independent variables.
        loss_function : callable, optional
            Loss function. If None, the standard least-squares problem is
            solved.
        tr_options : dict
            Options for the trust-region algorithm.
        verbose : int
            Level of algorithm's verbosity:

                * 0 (default) : work silently.
                * 1 : display a termination report.

        Returns
        -------
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

        Notes
        -----
        The algorithm is described in [13]_.
        """

        ftimes = []
        jtimes = []
        svd_times = []
        ctimes = []
        gtimes = []
        gtimes2 = []
        ptimes = []

        svd_ctimes = []
        g_ctimes = []
        c_ctimes = []
        p_ctimes = []

        x = x0.copy()

        # NOTE: We avoid excessive .block_until_ready() calls to enable JAX async execution.
        # Sync only at critical decision points where Python needs actual values.
        st = time.time()
        f = fun(x, xdata, ydata, data_mask, transform)
        f.block_until_ready()  # Single sync for timing
        ftimes.append(time.time() - st)
        f_true = f
        nfev = 1

        st = time.time()
        J = jac(x, xdata, ydata, data_mask, transform)
        J.block_until_ready()  # Single sync for timing
        jtimes.append(time.time() - st)

        njev = 1
        m, n = J.shape

        if loss_function is not None:
            rho = loss_function(f, f_scale)
            cost_jnp = self.calculate_cost(rho, data_mask)
            J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)
        else:
            st1 = time.time()
            cost_jnp = self.default_loss_func(f)
            cost_jnp.block_until_ready()  # Sync for timing
            st2 = time.time()
        cost = cost_jnp  # Keep as JAX array - no NumPy conversion
        st3 = time.time()

        ctimes.append(st2 - st1)
        c_ctimes.append(st3 - st2)

        st1 = time.time()
        g_jnp = self.compute_grad(J, f)
        g_jnp.block_until_ready()  # Sync for timing
        st2 = time.time()
        g = g_jnp  # Keep as JAX array - no NumPy conversion
        st3 = time.time()

        gtimes.append(st2 - st1)
        g_ctimes.append(st3 - st2)

        jac_scale = isinstance(x_scale, str) and x_scale == "jac"
        if jac_scale:
            scale, scale_inv = self.cJIT.compute_jac_scale(J)
        else:
            scale, scale_inv = x_scale, 1 / x_scale

        Delta = jnorm(x0 * scale_inv)
        if Delta == 0:
            Delta = 1.0

        if max_nfev is None:
            max_nfev = x0.size * DEFAULT_MAX_NFEV_MULTIPLIER

        alpha = INITIAL_LEVENBERG_MARQUARDT_LAMBDA  # "Levenberg-Marquardt" parameter

        termination_status = None
        iteration = 0
        step_norm = None
        actual_reduction = None

        if verbose == 2:
            print_header_nonlinear()

        while True:
            g_norm = jnorm(g, ord=jnp.inf)  # Use JAX norm with JAX infinity
            if g_norm < gtol:
                termination_status = 1

            if verbose == 2:
                # Use jax.debug.callback to avoid blocking host-device transfers
                debug.callback(
                    self._log_iteration_callback,
                    iteration,
                    nfev,
                    cost,
                    actual_reduction,
                    step_norm,
                    g_norm,
                )

            if termination_status is not None or nfev == max_nfev:
                break

            d = scale
            d_jnp = jnp.array(scale)

            # g_h = d * g
            g_h_jnp = self.compute_grad_hat(g_jnp, d_jnp)

            # Choose solver based on solver parameter
            if solver == "cg":
                # Use conjugate gradient solver (timed)
                st = time.time()
                J_h = J * d_jnp
                step_h = self.solve_tr_subproblem_cg(J, f, d_jnp, Delta, alpha)
                step_h.block_until_ready()  # Single sync for timing
                svd_times.append(time.time() - st)

                st = time.time()
                s, V, uf = None, None, None  # Not needed for CG path
                svd_ctimes.append(time.time() - st)
            elif solver == "sparse":
                # Sparse solver path (Task 6.4: Sparse Activation)
                # TODO: Implement sparse SVD
                # For now, fall back to dense exact solver to maintain correctness
                st = time.time()
                svd_output = self.svd_no_bounds(J, d_jnp, f)
                tree_flatten(svd_output)[0][
                    0
                ].block_until_ready()  # Single sync for timing
                svd_times.append(time.time() - st)
                J_h = svd_output[0]

                st = time.time()
                s, V, uf = svd_output[2:]
                svd_ctimes.append(time.time() - st)
            else:
                # Use exact SVD solver (default)
                st = time.time()
                svd_output = self.svd_no_bounds(J, d_jnp, f)
                tree_flatten(svd_output)[0][
                    0
                ].block_until_ready()  # Single sync for timing
                svd_times.append(time.time() - st)
                J_h = svd_output[0]

                st = time.time()
                s, V, uf = svd_output[2:]  # Keep as JAX arrays - no NumPy conversion
                svd_ctimes.append(time.time() - st)

            actual_reduction = -1
            inner_loop_count = 0
            max_inner_iterations = 100  # Prevent infinite loops
            while (
                actual_reduction <= 0
                and nfev < max_nfev
                and inner_loop_count < max_inner_iterations
            ):
                inner_loop_count += 1

                if solver == "cg":
                    # CG path: step already computed
                    # For subsequent iterations in inner loop, re-solve with updated alpha
                    if inner_loop_count > 1:
                        step_h = self.solve_tr_subproblem_cg(J, f, d_jnp, Delta, alpha)
                        # No explicit sync needed - JAX async handles it
                    _n_iter = 1  # Dummy value for compatibility
                else:
                    # SVD path: use exact solver (JAX version for pure JAX execution)
                    step_h, alpha, _n_iter = solve_lsq_trust_region_jax(
                        n, m, uf, s, V, Delta, initial_alpha=alpha
                    )

                st1 = time.time()
                predicted_reduction_jnp = -self.cJIT.evaluate_quadratic(
                    J_h, g_h_jnp, step_h
                )
                predicted_reduction_jnp.block_until_ready()  # Single sync for timing
                st2 = time.time()
                predicted_reduction = predicted_reduction_jnp  # Keep as JAX array
                st3 = time.time()
                ptimes.append(st2 - st1)
                p_ctimes.append(st3 - st2)

                step = d * step_h
                x_new = x + step

                st = time.time()
                f_new = fun(x_new, xdata, ydata, data_mask, transform)
                f_new.block_until_ready()  # Single sync for timing
                ftimes.append(time.time() - st)

                nfev += 1

                step_h_norm = jnorm(step_h)

                if not self.check_isfinite(f_new):
                    Delta = 0.25 * step_h_norm
                    continue

                if loss_function is not None:
                    cost_new_jnp = loss_function(
                        f_new, f_scale, data_mask, cost_only=True
                    )
                else:
                    st1 = time.time()
                    cost_new_jnp = self.default_loss_func(f_new)
                    cost_new_jnp.block_until_ready()  # Single sync for timing
                    st2 = time.time()
                    cost_new = cost_new_jnp  # Keep as JAX array - no NumPy conversion
                    st3 = time.time()

                    ctimes.append(st2 - st1)
                    c_ctimes.append(st3 - st2)

                actual_reduction = cost - cost_new

                Delta_new, ratio = update_tr_radius(
                    Delta,
                    actual_reduction,
                    predicted_reduction,
                    step_h_norm,
                    step_h_norm > 0.95 * Delta,
                )

                step_norm = jnorm(step)
                termination_status = check_termination(
                    actual_reduction, cost, step_norm, jnorm(x), ratio, ftol, xtol
                )

                if termination_status is not None:
                    break

                alpha *= Delta / Delta_new
                Delta = Delta_new

            # Check if inner loop hit iteration limit
            if inner_loop_count >= max_inner_iterations:
                self.logger.warning(
                    "Inner optimization loop hit iteration limit",
                    inner_iterations=inner_loop_count,
                    actual_reduction=actual_reduction,
                )
                termination_status = -3  # Inner loop limit exceeded

            if actual_reduction > 0:
                x = x_new

                f = f_new
                f_true = f

                cost = cost_new

                st = time.time()
                J = jac(x, xdata, ydata, data_mask, transform)
                J.block_until_ready()  # Single sync for timing
                jtimes.append(time.time() - st)

                njev += 1

                if loss_function is not None:
                    rho = loss_function(f, f_scale)
                    J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)

                st1 = time.time()
                g_jnp = self.compute_grad(J, f)
                g_jnp.block_until_ready()  # Single sync for timing
                st2 = time.time()
                g = g_jnp  # Keep as JAX array - no NumPy conversion
                st3 = time.time()

                gtimes.append(st2 - st1)
                g_ctimes.append(st3 - st2)

                if jac_scale:
                    scale, scale_inv = self.cJIT.compute_jac_scale(J, scale_inv)

            else:
                step_norm = 0
                actual_reduction = 0

            iteration += 1

            # Invoke user callback if provided
            if callback is not None:
                try:
                    callback(
                        iteration=iteration,
                        cost=float(cost),  # JAX scalar → Python float
                        params=np.array(x),  # JAX array → NumPy array
                        info={
                            "gradient_norm": float(g_norm),
                            "nfev": nfev,
                            "step_norm": float(step_norm)
                            if step_norm is not None
                            else None,
                            "actual_reduction": float(actual_reduction)
                            if actual_reduction is not None
                            else None,
                        },
                    )
                except StopOptimization:
                    termination_status = 2  # User-requested stop
                    self.logger.info(
                        "Optimization stopped by callback (StopOptimization)"
                    )
                    break
                except Exception as e:
                    warnings.warn(
                        f"Callback raised exception: {e}. Continuing optimization.",
                        RuntimeWarning,
                    )

        if termination_status is None:
            termination_status = 0

        active_mask = jnp.zeros_like(x)  # JAX zeros instead of NumPy

        tlabels = [
            "ftimes",
            "jtimes",
            "svd_times",
            "ctimes",
            "gtimes",
            "ptimes",
            "g_ctimes",
            "c_ctimes",
            "svd_ctimes",
            "p_ctimes",
            "gtimes2",
        ]
        all_times = [
            ftimes,
            jtimes,
            svd_times,
            ctimes,
            gtimes,
            ptimes,
            g_ctimes,
            c_ctimes,
            svd_ctimes,
            p_ctimes,
            gtimes2,
        ]

        tdicts = dict(zip(tlabels, all_times, strict=False))
        return OptimizeResult(
            x=x,
            cost=cost,
            fun=f_true,
            jac=J,
            grad=g,
            optimality=g_norm,
            active_mask=active_mask,
            nfev=nfev,
            njev=njev,
            nit=iteration,  # Number of iterations performed
            status=termination_status,
            all_times=tdicts,
        )

    def optimize(
        self,
        fun: Callable,
        x0: np.ndarray,
        jac: Callable | None = None,
        bounds: tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
        **kwargs,
    ) -> OptimizeResult:
        """Perform optimization using trust region reflective algorithm.

        This method provides a simplified interface to the TRF algorithm.
        For full control and curve fitting applications, use the `trf` method directly.

        Parameters
        ----------
        fun : callable
            The objective function to minimize. Should return residuals.
        x0 : np.ndarray
            Initial guess for parameters
        jac : callable, optional
            Jacobian function. If None, uses automatic differentiation.
        bounds : tuple of arrays
            Lower and upper bounds for parameters
        **kwargs
            Additional optimization parameters

        Returns
        -------
        OptimizeResult
            The optimization result

        Raises
        ------
        NotImplementedError
            This simplified interface is not yet implemented.
            Use the `trf` method for full curve fitting functionality.
        """
        raise NotImplementedError(
            "The simplified optimize() interface is not yet implemented for TrustRegionReflective. "
            "This class is designed for curve fitting applications. "
            "Use the `trf()` method directly, or use the higher-level interfaces in "
            "`nlsq.curve_fit()` or `LeastSquares.least_squares()`."
        )
