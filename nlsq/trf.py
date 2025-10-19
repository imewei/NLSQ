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

import time
import warnings
from collections.abc import Callable

import numpy as np
from numpy.linalg import norm

# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

_jax_config = JAXConfig()
import jax.numpy as jnp
from jax import jit
from jax.numpy.linalg import norm as jnorm
from jax.tree_util import tree_flatten

# Setup logging
from nlsq.logging import get_logger

# Import safe SVD with fallback
from nlsq.svd_fallback import compute_svd_with_fallback, initialize_gpu_safely

logger = get_logger("trf")

# Initialize GPU settings safely
initialize_gpu_safely()

from nlsq._optimize import OptimizeResult
from nlsq.callbacks import StopOptimization
from nlsq.common_jax import CommonJIT
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
    solve_lsq_trust_region,
    step_size_to_bound,
    update_tr_radius,
)
from nlsq.constants import (
    DEFAULT_MAX_NFEV_MULTIPLIER,
    INITIAL_LEVENBERG_MARQUARDT_LAMBDA,
)
from nlsq.diagnostics import OptimizationDiagnostics

# Logging support
# Optimizer base class
from nlsq.optimizer_base import TrustRegionOptimizerBase
from nlsq.stability import NumericalStabilityGuard

# Algorithm constants
# Trust region parameters
TR_REDUCTION_FACTOR = 0.25  # Factor to reduce trust region when numerical issues occur
TR_BOUNDARY_THRESHOLD = 0.95  # Threshold for checking if step is close to boundary
LOSS_FUNCTION_COEFF = 0.5  # Coefficient for loss function (0.5 * ||f||^2)
SQRT_EXPONENT = 0.5  # Exponent for square root in scaling (v**0.5)

# Numerical stability thresholds
NUMERICAL_ZERO_THRESHOLD = 1e-14  # Threshold for values considered numerically zero
DEFAULT_TOLERANCE = 1e-6  # Default tolerance for iterative solvers


class TrustRegionJITFunctions:
    """JIT-compiled functions for Trust Region Reflective optimization algorithm.

    This class contains all JAX JIT-compiled functions required for the Trust Region
    Reflective algorithm. It provides optimized GPU/TPU-accelerated implementations
    of core mathematical operations including gradient computation, SVD decomposition,
    iterative solvers, and quadratic evaluations.

    Core Operations
    ---------------
    - **Gradient Computation**: JAX-accelerated gradient calculation using J^T * f
    - **SVD Decomposition**: Singular value decomposition for trust region subproblems
    - **Conjugate Gradient**: Iterative solver for large-scale problems
    - **Cost Function Evaluation**: Loss function computation with masking support
    - **Hat Space Transformation**: Scaled variable transformations for bounds handling

    JIT Compilation Benefits
    ------------------------
    - **GPU Acceleration**: All operations optimized for GPU/TPU hardware
    - **Memory Efficiency**: Reduced memory allocations through compilation
    - **Automatic Differentiation**: JAX autodiff for exact Jacobian computation
    - **XLA Optimization**: Advanced compiler optimizations for performance

    Algorithm Integration
    ---------------------
    The class implements two solution methods:
    1. **Exact SVD**: Uses singular value decomposition for small to medium problems
    2. **Conjugate Gradient**: Iterative method for large sparse problems

    Memory Management
    -----------------
    - Bounded Problems: Augmented system handling for constraint optimization
    - Unbounded Problems: Direct system solving for unconstrained optimization
    - Scaling Matrices: Efficient diagonal matrix operations in hat space

    Technical Implementation
    ------------------------
    All functions use JAX JIT compilation for performance. The class automatically
    creates optimized versions during initialization. Functions handle both bounded
    and unbounded optimization variants with appropriate augmentation strategies.

    Performance Characteristics
    ---------------------------
    - **Small Problems**: Direct SVD solution O(mn^2 + n^3)
    - **Large Problems**: CG iteration O(k*mn) where k is iteration count
    - **GPU Memory**: Optimized for batch processing and memory reuse
    - **Numerical Stability**: Double precision arithmetic with condition monitoring
    """

    def __init__(self):
        """Call all of the create functions which create the JAX/JIT functions
        that are members of the class."""
        self.create_grad_func()
        self.create_grad_hat()
        self.create_svd_funcs()
        self.create_iterative_solvers()
        self.create_default_loss_func()
        self.create_calculate_cost()
        self.create_check_isfinite()

    def create_default_loss_func(self):
        """Create the default loss function which is simply the sum of the
        squares of the residuals."""

        @jit
        def loss_function(f: jnp.ndarray) -> jnp.ndarray:
            """The default loss function is the sum of the squares of the
            residuals divided by two.

            Parameters
            ----------
            f : jnp.ndarray
                The residuals.

            Returns
            -------
            jnp.ndarray
                The loss function value.
            """
            return LOSS_FUNCTION_COEFF * jnp.dot(f, f)
            # return LOSS_FUNCTION_COEFF * jnp.sum(f**2)

        self.default_loss_func = loss_function

    def create_grad_func(self):
        """Create the function to compute the gradient of the loss function
        which is simply the function evaluation dotted with the Jacobian."""

        @jit
        def compute_grad(J: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
            """Compute the gradient of the loss function.

            Parameters
            ----------
            J : jnp.ndarray
                The Jacobian matrix.
            f : jnp.ndarray
                The residuals.

            Returns
            -------
            jnp.ndarray
                The gradient of the loss function.
            """
            return f.dot(J)

        self.compute_grad = compute_grad

    def create_grad_hat(self):
        """Calculate the gradient in the "hat" space, which is just multiplying
        the gradient by the diagonal matrix D. This is used in the trust region
        algorithm. Here we only use the diagonals of D, since D is diagonal.
        """

        @jit
        def compute_grad_hat(g: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
            """Compute the gradient in the "hat" space.

            Parameters
            ----------
            g : jnp.ndarray
            The gradient of the loss function.
            d : jnp.ndarray
            The diagonal of the diagonal matrix D.

            Returns
            -------
            jnp.ndarray
            The gradient in the "hat" space.
            """
            return d * g

        self.compute_grad_hat = compute_grad_hat

    def create_svd_funcs(self):
        """Create the functions to compute the SVD of the Jacobian matrix.
        There are two versions, one for problems with bounds and one for
        problems without bounds. The version for problems with bounds is
        slightly more complicated."""

        @jit
        def svd_no_bounds(
            J: jnp.ndarray, d: jnp.ndarray, f: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Compute the SVD of the Jacobian matrix, J, in the "hat" space.
            This is the version for problems without bounds.

            Parameters
            ----------
            J : jnp.ndarray
                The Jacobian matrix.
            d : jnp.ndarray
                The diagonal of the diagonal matrix D.
            f : jnp.ndarray
                The residuals.

            Returns
            -------
            J_h : jnp.ndarray
                  the Jacobian matrix in the "hat" space.
            U : jnp.ndarray
                the left singular vectors of the SVD of J_h.
            s : jnp.ndarray
                the singular values of the SVD of J_h.
            V : jnp.ndarray
                the right singular vectors of the SVD of J_h.
            uf : jnp.ndarray
                 the dot product of U.T and f.
            """
            J_h = J * d
            # Use safe SVD with fallback
            U, s, V = compute_svd_with_fallback(J_h, full_matrices=False)
            uf = U.T.dot(f)
            return J_h, U, s, V, uf

        @jit
        def svd_bounds(
            f: jnp.ndarray,
            J: jnp.ndarray,
            d: jnp.ndarray,
            J_diag: jnp.ndarray,
            f_zeros: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Compute the SVD of the Jacobian matrix, J, in the "hat" space.
            This is the version for problems with bounds.

            Parameters
            ----------
            J : jnp.ndarray
                The Jacobian matrix.
            d : jnp.ndarray
                The diagonal of the diagonal matrix D.
            f : jnp.ndarray
                The residuals.
            J_diag : jnp.ndarray
                    Added term to Jacobian matrix.
            f_zeros : jnp.ndarray
                    Empty residuals for the added term.


            Returns
            -------
            J_h : jnp.ndarray
                  the Jacobian matrix in the "hat" space.
            U : jnp.ndarray
                the left singular vectors of the SVD of J_h.
            s : jnp.ndarray
                the singular values of the SVD of J_h.
            V : jnp.ndarray
                the right singular vectors of the SVD of J_h.
            uf : jnp.ndarray
                 the dot product of U.T and f.
            """
            J_h = J * d
            J_augmented = jnp.concatenate([J_h, J_diag])
            f_augmented = jnp.concatenate([f, f_zeros])
            # Use safe SVD with fallback
            U, s, V = compute_svd_with_fallback(J_augmented, full_matrices=False)
            uf = U.T.dot(f_augmented)
            return J_h, U, s, V, uf

        self.svd_no_bounds = svd_no_bounds
        self.svd_bounds = svd_bounds

    def create_iterative_solvers(self):
        """Create iterative solvers for trust region subproblems as alternatives to SVD."""

        @jit
        def conjugate_gradient_solve(
            J: jnp.ndarray,
            f: jnp.ndarray,
            d: jnp.ndarray,
            alpha: float = 0.0,
            max_iter: int = None,
            tol: float = DEFAULT_TOLERANCE,
        ) -> tuple[jnp.ndarray, jnp.ndarray, int]:
            """Solve the normal equations using conjugate gradient method.

            Solves (J^T J + alpha*I) p = -J^T f using CG without forming J^T J explicitly.

            Parameters
            ----------
            J : jnp.ndarray
                Jacobian matrix (m x n)
            f : jnp.ndarray
                Residual vector (m,)
            d : jnp.ndarray
                Scaling diagonal (n,)
            alpha : float
                Regularization parameter
            max_iter : int, optional
                Maximum CG iterations (default: min(n, 100))
            tol : float
                Convergence tolerance

            Returns
            -------
            p : jnp.ndarray
                Solution vector (n,)
            residual_norm : jnp.ndarray
                Final residual norm
            n_iter : int
                Number of CG iterations
            """
            # Solve (J^T J + alpha I) x = -J^T f using conjugate gradient
            _m, n = J.shape
            if max_iter is None:
                max_iter = min(n, 100)

            # Scale Jacobian and setup RHS
            J_scaled = J * d[None, :]
            b = -J_scaled.T @ f

            # Initialize CG
            x = jnp.zeros(n)
            r = b
            p = r.copy()
            rsold = jnp.dot(r, r)

            for i in range(max_iter):
                # Matrix-vector product: (J^T J + alpha I) p
                Jp = J_scaled @ p
                JTJp = J_scaled.T @ Jp
                Ap = JTJp + alpha * p

                # Step size
                pAp = jnp.dot(p, Ap)
                pAp = jnp.where(
                    jnp.abs(pAp) < NUMERICAL_ZERO_THRESHOLD,
                    NUMERICAL_ZERO_THRESHOLD,
                    pAp,
                )
                alpha_cg = rsold / pAp

                # Update solution and residual
                x = x + alpha_cg * p
                r = r - alpha_cg * Ap
                rsnew = jnp.dot(r, r)

                # Check convergence
                if jnp.sqrt(rsnew) < tol:
                    return x, jnp.sqrt(rsnew), i + 1

                # Update search direction
                beta = rsnew / rsold
                p = r + beta * p
                rsold = rsnew

            return x, jnp.sqrt(rsold), max_iter

        @jit
        def solve_tr_subproblem_cg(
            J: jnp.ndarray,
            f: jnp.ndarray,
            d: jnp.ndarray,
            Delta: float,
            alpha: float = 0.0,
            max_iter: int = None,
        ) -> jnp.ndarray:
            """Solve trust region subproblem using conjugate gradient.

            This replaces the SVD-based solve_lsq_trust_region function.
            """
            # First try to solve without regularization (alpha=0)
            p_gn, _residual_norm, _n_iter = conjugate_gradient_solve(
                J, f, d, 0.0, max_iter
            )

            # Check if Gauss-Newton step is within trust region
            p_gn_norm = jnp.linalg.norm(p_gn)

            # If within trust region, return Gauss-Newton step
            if p_gn_norm <= Delta:
                return p_gn

            # Otherwise, need to find optimal alpha using regularized CG
            # Use simple approach: solve with current alpha
            p_reg, _, _ = conjugate_gradient_solve(J, f, d, alpha, max_iter)

            # Scale to trust region boundary
            p_reg_norm = jnp.linalg.norm(p_reg)
            p_reg_norm = jnp.where(
                p_reg_norm < NUMERICAL_ZERO_THRESHOLD,
                NUMERICAL_ZERO_THRESHOLD,
                p_reg_norm,
            )
            scaling = Delta / p_reg_norm

            return scaling * p_reg

        @jit
        def solve_tr_subproblem_cg_bounds(
            J: jnp.ndarray,
            f: jnp.ndarray,
            d: jnp.ndarray,
            J_diag: jnp.ndarray,
            f_zeros: jnp.ndarray,
            Delta: float,
            alpha: float = 0.0,
            max_iter: int = None,
        ) -> jnp.ndarray:
            """Solve trust region subproblem with bounds using conjugate gradient."""
            # Augment the system for bounds
            J_augmented = jnp.concatenate([J * d[None, :], J_diag])
            f_augmented = jnp.concatenate([f, f_zeros])
            d_augmented = jnp.ones(J_augmented.shape[1])  # Already scaled

            # First try to solve without regularization (alpha=0)
            p_gn, _residual_norm, _n_iter = conjugate_gradient_solve(
                J_augmented, f_augmented, d_augmented, 0.0, max_iter
            )

            # Check if Gauss-Newton step is within trust region
            p_gn_norm = jnp.linalg.norm(p_gn)

            # If within trust region, return Gauss-Newton step
            if p_gn_norm <= Delta:
                return p_gn

            # Otherwise, need to find optimal alpha using regularized CG
            # Use simple approach: solve with current alpha
            p_reg, _, _ = conjugate_gradient_solve(
                J_augmented, f_augmented, d_augmented, alpha, max_iter
            )

            # Scale to trust region boundary
            p_reg_norm = jnp.linalg.norm(p_reg)
            p_reg_norm = jnp.where(
                p_reg_norm < NUMERICAL_ZERO_THRESHOLD,
                NUMERICAL_ZERO_THRESHOLD,
                p_reg_norm,
            )
            scaling = Delta / p_reg_norm

            return scaling * p_reg

        # Store the iterative solver functions
        self.conjugate_gradient_solve = conjugate_gradient_solve
        self.solve_tr_subproblem_cg = solve_tr_subproblem_cg
        self.solve_tr_subproblem_cg_bounds = solve_tr_subproblem_cg_bounds

    def create_calculate_cost(self):
        """Create the function to calculate the cost function."""

        @jit
        def calculate_cost(rho, data_mask):
            """Calculate the cost function.

            Parameters
            ----------
            rho : jnp.ndarray
                The per element cost times two.
            data_mask : jnp.ndarray
                The mask for the data.

            Returns
            -------
            jnp.ndarray
                The cost function.
            """
            cost_array = jnp.where(data_mask, rho[0], 0)
            return LOSS_FUNCTION_COEFF * jnp.sum(cost_array)

        self.calculate_cost = calculate_cost

    def create_check_isfinite(self):
        """Create the function to check if the evaluated residuals are finite."""

        @jit
        def isfinite(f_new: jnp.ndarray) -> bool:
            """Check if the evaluated residuals are finite.

            Parameters
            ----------
            f_new : jnp.ndarray
                The evaluated residuals.

            Returns
            -------
            bool
                True if all residuals are finite, False otherwise.
            """
            return jnp.all(jnp.isfinite(f_new))

        self.check_isfinite = isfinite


# ============================================================================
# TRF Profiling Abstraction
# ============================================================================


class TRFProfiler:
    """Profiler for timing TRF algorithm operations.

    Records detailed timing information for each operation in the TRF algorithm,
    including GPU synchronization via block_until_ready() for accurate timings.

    This enables performance analysis without duplicating the entire algorithm.
    """

    def __init__(self):
        """Initialize profiler with empty timing arrays."""
        self.ftimes = []  # Function evaluations
        self.jtimes = []  # Jacobian evaluations
        self.svd_times = []  # SVD computations
        self.ctimes = []  # Cost computations (JAX)
        self.gtimes = []  # Gradient computations (JAX)
        self.gtimes2 = []  # Gradient norm computations
        self.ptimes = []  # Parameter updates

        # Conversion times (JAX → NumPy)
        self.svd_ctimes = []  # SVD conversion
        self.g_ctimes = []  # Gradient conversion
        self.c_ctimes = []  # Cost conversion
        self.p_ctimes = []  # Parameter conversion

    def time_operation(self, operation: str, jax_result):
        """Time a JAX operation with GPU synchronization.

        Parameters
        ----------
        operation : str
            Operation name ('fun', 'jac', 'svd', 'cost', 'grad', etc.)
        jax_result :
            JAX array result to synchronize

        Returns
        -------
        result
            The synchronized result (same as input)
        """
        import time

        st = time.time()
        result = jax_result.block_until_ready()
        elapsed = time.time() - st

        # Record timing
        if operation == "fun":
            self.ftimes.append(elapsed)
        elif operation == "jac":
            self.jtimes.append(elapsed)
        elif operation == "svd":
            self.svd_times.append(elapsed)
        elif operation == "cost":
            self.ctimes.append(elapsed)
        elif operation == "grad":
            self.gtimes.append(elapsed)
        elif operation == "grad_norm":
            self.gtimes2.append(elapsed)
        elif operation == "param_update":
            self.ptimes.append(elapsed)

        return result

    def time_conversion(self, operation: str, start_time: float):
        """Record timing for JAX → NumPy conversion.

        Parameters
        ----------
        operation : str
            Conversion operation ('svd_convert', 'grad_convert', 'cost_convert', 'param_convert')
        start_time : float
            Start time from time.time()
        """
        import time

        elapsed = time.time() - start_time

        if operation == "svd_convert":
            self.svd_ctimes.append(elapsed)
        elif operation == "grad_convert":
            self.g_ctimes.append(elapsed)
        elif operation == "cost_convert":
            self.c_ctimes.append(elapsed)
        elif operation == "param_convert":
            self.p_ctimes.append(elapsed)

    def get_timing_data(self) -> dict:
        """Get all recorded timing data.

        Returns
        -------
        dict
            Dictionary containing all timing arrays
        """
        return {
            "ftimes": self.ftimes,
            "jtimes": self.jtimes,
            "svd_times": self.svd_times,
            "ctimes": self.ctimes,
            "gtimes": self.gtimes,
            "gtimes2": self.gtimes2,
            "ptimes": self.ptimes,
            "svd_ctimes": self.svd_ctimes,
            "g_ctimes": self.g_ctimes,
            "c_ctimes": self.c_ctimes,
            "p_ctimes": self.p_ctimes,
        }


class NullProfiler:
    """Null object profiler with zero overhead.

    Provides same interface as TRFProfiler but does nothing,
    enabling profiling to be toggled with no performance impact.
    """

    def time_operation(self, operation: str, jax_result):
        """No-op timing - returns result unchanged."""
        return jax_result

    def time_conversion(self, operation: str, start_time: float):
        """No-op conversion timing."""
        pass

    def get_timing_data(self) -> dict:
        """Returns empty timing data."""
        return {
            "ftimes": [],
            "jtimes": [],
            "svd_times": [],
            "ctimes": [],
            "gtimes": [],
            "gtimes2": [],
            "ptimes": [],
            "svd_ctimes": [],
            "g_ctimes": [],
            "c_ctimes": [],
            "p_ctimes": [],
        }


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

        # Initialize stability system
        self.enable_stability = enable_stability
        if enable_stability:
            self.stability_guard = NumericalStabilityGuard()

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
        Delta = norm(x0 * state["scale_inv"])
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
        else:
            # SVD-based exact solver
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
                step_h, alpha, _n_iter = solve_lsq_trust_region(
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
            step_h_norm = norm(step_h)

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
            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol
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
                    print_iteration_nonlinear(
                        iteration, nfev, cost, actual_reduction, step_norm, g_norm
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

        active_mask = np.zeros_like(x)

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

        v[dv != 0] *= scale_inv[dv != 0]
        Delta = norm(x0 * scale_inv / v**SQRT_EXPONENT)
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

            # Use JAX norm for gradient norm calculation
            g_norm = jnorm(g * v, ord=jnp.inf)
            if g_norm < gtol:
                termination_status = 1

            # if verbose == 2:
            #     print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
            #                               step_norm, g_norm)

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
            v[dv != 0] *= scale_inv[dv != 0]

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
            else:
                # Use exact SVD solver (default)
                output = self.svd_bounds(f, J, d_jnp, J_diag, f_zeros)
                J_h = output[0]
                s, V, uf = (np.array(val) for val in output[2:])

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
                    # SVD path: use exact solver
                    p_h, alpha, _n_iter = solve_lsq_trust_region(
                        n, m, uf, s, V, Delta, initial_alpha=alpha
                    )

                p = d * p_h  # Trust-region solution in the original space.
                step, step_h, predicted_reduction = self.select_step(
                    x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta
                )

                x_new = make_strictly_feasible(x + step.copy(), lb, ub, rstep=0)
                f_new = fun(x_new, xdata, ydata, data_mask, transform)

                nfev += 1

                step_h_norm = norm(step_h)
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

                step_norm = norm(step)
                termination_status = check_termination(
                    actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol
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
        r_h = np.copy(p_h)
        r_h[hits.astype(bool)] *= -1
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
            r_value = np.inf

        # Now correct p_h to make it strictly interior.
        p *= theta
        p_h *= theta
        p_value = self.cJIT.evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)

        ag_h = -g_h
        ag = d * ag_h

        to_tr = Delta / norm(ag_h)
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
        in benchmark/profile_trf.py.

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

        st = time.time()
        f = fun(x, xdata, ydata, data_mask, transform).block_until_ready()
        ftimes.append(time.time() - st)
        f_true = f
        nfev = 1

        st = time.time()
        J = jac(x, xdata, ydata, data_mask, transform).block_until_ready()
        jtimes.append(time.time() - st)

        njev = 1
        m, n = J.shape

        if loss_function is not None:
            rho = loss_function(f, f_scale)
            cost_jnp = self.calculate_cost(rho, data_mask)
            J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)
        else:
            st1 = time.time()
            cost_jnp = self.default_loss_func(f).block_until_ready()
            st2 = time.time()
        cost = np.array(cost_jnp)
        st3 = time.time()

        ctimes.append(st2 - st1)
        c_ctimes.append(st3 - st2)

        st1 = time.time()
        g_jnp = self.compute_grad(J, f).block_until_ready()
        st2 = time.time()
        g = np.array(g_jnp)
        st3 = time.time()

        gtimes.append(st2 - st1)
        g_ctimes.append(st3 - st2)

        jac_scale = isinstance(x_scale, str) and x_scale == "jac"
        if jac_scale:
            scale, scale_inv = self.cJIT.compute_jac_scale(J)
        else:
            scale, scale_inv = x_scale, 1 / x_scale

        Delta = norm(x0 * scale_inv)
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
            g_norm = norm(g, ord=np.inf)
            if g_norm < gtol:
                termination_status = 1

            # if verbose == 2:
            #     print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
            #                               step_norm, g_norm)

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
                J_h = (J * d_jnp).block_until_ready()
                step_h = self.solve_tr_subproblem_cg(
                    J, f, d_jnp, Delta, alpha
                ).block_until_ready()
                svd_times.append(time.time() - st)

                st = time.time()
                s, V, uf = None, None, None  # Not needed for CG path
                svd_ctimes.append(time.time() - st)
            else:
                # Use exact SVD solver (default)
                st = time.time()
                svd_output = self.svd_no_bounds(J, d_jnp, f)
                tree_flatten(svd_output)[0][0].block_until_ready()
                svd_times.append(time.time() - st)
                J_h = svd_output[0]

                st = time.time()
                s, V, uf = (np.array(val) for val in svd_output[2:])
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
                        step_h = self.solve_tr_subproblem_cg(
                            J, f, d_jnp, Delta, alpha
                        ).block_until_ready()
                    _n_iter = 1  # Dummy value for compatibility
                else:
                    # SVD path: use exact solver
                    step_h, alpha, _n_iter = solve_lsq_trust_region(
                        n, m, uf, s, V, Delta, initial_alpha=alpha
                    )

                st1 = time.time()
                predicted_reduction_jnp = -self.cJIT.evaluate_quadratic(
                    J_h, g_h_jnp, step_h
                ).block_until_ready()
                st2 = time.time()
                predicted_reduction = np.array(predicted_reduction_jnp)
                st3 = time.time()
                ptimes.append(st2 - st1)
                p_ctimes.append(st3 - st2)

                step = d * step_h
                x_new = x + step

                st = time.time()
                f_new = fun(
                    x_new, xdata, ydata, data_mask, transform
                ).block_until_ready()
                ftimes.append(time.time() - st)

                nfev += 1

                step_h_norm = norm(step_h)

                if not self.check_isfinite(f_new):
                    Delta = 0.25 * step_h_norm
                    continue

                if loss_function is not None:
                    cost_new_jnp = loss_function(
                        f_new, f_scale, data_mask, cost_only=True
                    )
                else:
                    st1 = time.time()
                    cost_new_jnp = self.default_loss_func(f_new).block_until_ready()
                    st2 = time.time()
                    cost_new = np.array(cost_new_jnp)
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

                step_norm = norm(step)
                termination_status = check_termination(
                    actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol
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
                J = jac(x, xdata, ydata, data_mask, transform).block_until_ready()
                jtimes.append(time.time() - st)

                njev += 1

                if loss_function is not None:
                    rho = loss_function(f, f_scale)
                    J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)

                st1 = time.time()
                g_jnp = self.compute_grad(J, f).block_until_ready()
                st2 = time.time()
                g = np.array(g_jnp)
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

        active_mask = np.zeros_like(x)

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
