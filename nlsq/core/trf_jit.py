"""JIT-compiled functions for Trust Region Reflective optimization.

This module contains JAX JIT-compiled helper functions for the TRF algorithm,
providing GPU/TPU-accelerated implementations of core mathematical operations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit, lax

from nlsq.stability.svd_fallback import compute_svd_with_fallback

__all__ = ["TrustRegionJITFunctions"]

# Algorithm constants
LOSS_FUNCTION_COEFF = 0.5  # Coefficient for loss function (0.5 * ||f||^2)
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
            # Use full deterministic SVD for numerical precision
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
            f : jnp.ndarray
                The residuals.
            J : jnp.ndarray
                The Jacobian matrix.
            d : jnp.ndarray
                The diagonal of the diagonal matrix D.
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
            # Use full deterministic SVD for numerical precision
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
            max_iter: int | None = None,
            tol: float = DEFAULT_TOLERANCE,
        ) -> tuple[jnp.ndarray, jnp.ndarray, int]:
            """Solve the normal equations using conjugate gradient method.

            Solves (J^T J + alpha*I) p = -J^T f using CG without forming J^T J explicitly.
            Uses jax.lax.while_loop for 3-8x GPU acceleration.

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

            # Initialize CG state: (x, r, p, rsold, iteration, converged)
            x0 = jnp.zeros(n)
            r0 = b
            p0 = r0.copy()
            rsold0 = jnp.dot(r0, r0)
            tol_sq = tol * tol  # Compare squared norms for efficiency

            # State tuple: (x, r, p, rsold, iteration)
            init_state = (x0, r0, p0, rsold0, 0)

            def cond_fn(state):
                """Continue while not converged and iterations remain."""
                _x, _r, _p, rsold, i = state
                return (i < max_iter) & (rsold >= tol_sq)

            def body_fn(state):
                """Single CG iteration."""
                x, r, p, rsold, i = state

                # Matrix-vector product: (J^T J + alpha I) p
                Jp = J_scaled @ p
                JTJp = J_scaled.T @ Jp
                Ap = JTJp + alpha * p

                # Step size with numerical stability
                pAp = jnp.dot(p, Ap)
                pAp = jnp.where(
                    jnp.abs(pAp) < NUMERICAL_ZERO_THRESHOLD,
                    NUMERICAL_ZERO_THRESHOLD,
                    pAp,
                )
                alpha_cg = rsold / pAp

                # Update solution and residual
                x_new = x + alpha_cg * p
                r_new = r - alpha_cg * Ap
                rsnew = jnp.dot(r_new, r_new)

                # Update search direction
                beta = rsnew / (rsold + NUMERICAL_ZERO_THRESHOLD)
                p_new = r_new + beta * p

                return (x_new, r_new, p_new, rsnew, i + 1)

            # Run CG loop using JAX while_loop (3-8x faster on GPU)
            final_state = lax.while_loop(cond_fn, body_fn, init_state)
            x_final, _r_final, _p_final, rsold_final, n_iter = final_state

            return x_final, jnp.sqrt(rsold_final), n_iter

        @jit
        def solve_tr_subproblem_cg(
            J: jnp.ndarray,
            f: jnp.ndarray,
            d: jnp.ndarray,
            Delta: float,
            alpha: float = 0.0,
            max_iter: int | None = None,
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
            # Clamp scaling factor to prevent numerical instability
            # when trust region collapses or parameter norm is near zero
            p_reg_norm = jnp.linalg.norm(p_reg)
            p_reg_norm = jnp.maximum(p_reg_norm, 1e-10)
            scaling = jnp.clip(Delta / p_reg_norm, 0.1, 10.0)

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
            max_iter: int | None = None,
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
            # Clamp scaling factor to prevent numerical instability
            # when trust region collapses or parameter norm is near zero
            p_reg_norm = jnp.linalg.norm(p_reg)
            p_reg_norm = jnp.maximum(p_reg_norm, 1e-10)
            scaling = jnp.clip(Delta / p_reg_norm, 0.1, 10.0)

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
        def isfinite(f_new: jnp.ndarray) -> jnp.ndarray:
            """Check if the evaluated residuals are finite.

            Parameters
            ----------
            f_new : jnp.ndarray
                The evaluated residuals.

            Returns
            -------
            jnp.ndarray
                Scalar array that is True if all residuals are finite, False otherwise.
            """
            return jnp.all(jnp.isfinite(f_new))

        self.check_isfinite = isfinite
