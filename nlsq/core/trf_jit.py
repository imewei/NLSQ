"""JIT-compiled functions for Trust Region Reflective optimization.

This module contains JAX JIT-compiled helper functions for the TRF algorithm,
providing GPU/TPU-accelerated implementations of core mathematical operations.

XLA Memory Optimization
-----------------------
All JIT functions are defined at module level as singletons. This ensures each
function is compiled only once per unique input shape, regardless of how many
TrustRegionJITFunctions instances are created. Previously, each instance created
10+ new @jit closures, causing unbounded XLA compilation cache growth.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.linalg import svd as jax_svd

from nlsq.stability.svd_fallback import compute_svd_with_fallback, is_gpu_error

__all__ = ["TrustRegionJITFunctions"]

# Algorithm constants
LOSS_FUNCTION_COEFF = 0.5  # Coefficient for loss function (0.5 * ||f||^2)
NUMERICAL_ZERO_THRESHOLD = 1e-14  # Threshold for values considered numerically zero
DEFAULT_TOLERANCE = 1e-8  # Default tolerance for iterative solvers (matches outer ftol/gtol/xtol defaults)
# Fixed CG iteration limit — prevents shape-dependent recompilation.
# The while_loop convergence check provides early exit for small problems.
CG_MAX_ITERATIONS = 100


# ---------------------------------------------------------------------------
# Module-level JIT-compiled functions (singletons — compiled once per shape)
# ---------------------------------------------------------------------------


@jit
def _default_loss_func(f: jnp.ndarray) -> jnp.ndarray:
    """Default loss: 0.5 * ||f||^2."""
    return LOSS_FUNCTION_COEFF * jnp.dot(f, f)


@jit
def _compute_grad(J: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
    """Compute gradient of loss function: f^T J."""
    return f.dot(J)


@jit
def _compute_grad_hat(g: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Compute gradient in hat space: d * g."""
    return d * g


@jit
def _svd_no_bounds_jit(
    J: jnp.ndarray, d: jnp.ndarray, f: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    J_h = J * d
    U, s, Vt = jax_svd(J_h, full_matrices=False)
    uf = U.T.dot(f)
    return J_h, U, s, Vt.T, uf


@jit
def _svd_bounds_jit(
    f: jnp.ndarray,
    J: jnp.ndarray,
    d: jnp.ndarray,
    J_diag: jnp.ndarray,
    f_zeros: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    J_h = J * d
    J_augmented = jnp.concatenate([J_h, J_diag])
    f_augmented = jnp.concatenate([f, f_zeros])
    U, s, Vt = jax_svd(J_augmented, full_matrices=False)
    uf = U.T.dot(f_augmented)
    return J_h, U, s, Vt.T, uf


@jit
def _conjugate_gradient_solve(
    J: jnp.ndarray,
    f: jnp.ndarray,
    d: jnp.ndarray,
    alpha: float = 0.0,
    tol: float = DEFAULT_TOLERANCE,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Solve (J^T J + alpha*I) p = -J^T f using CG.

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

    Notes
    -----
    Uses fixed CG_MAX_ITERATIONS (100) to prevent shape-dependent XLA
    recompilation. The while_loop convergence check exits early for
    small problems (n < 100).
    """
    _m, n = J.shape

    J_scaled = J * d[None, :]
    b = -J_scaled.T @ f

    x0 = jnp.zeros(n, dtype=b.dtype)
    r0 = b
    p0 = r0
    rsold0 = jnp.dot(r0, r0)
    tol_sq = tol * tol

    init_state = (x0, r0, p0, rsold0, 0)

    def cond_fn(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]):
        _x, _r, _p, rsold, i = state
        return (i < CG_MAX_ITERATIONS) & (rsold >= tol_sq)

    def body_fn(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]):
        x, r, p, rsold, i = state

        Jp = J_scaled @ p
        JTJp = J_scaled.T @ Jp
        Ap = JTJp + alpha * p

        pAp = jnp.dot(p, Ap)
        safe_pAp = jnp.where(
            pAp > NUMERICAL_ZERO_THRESHOLD, pAp, NUMERICAL_ZERO_THRESHOLD
        )
        alpha_cg = rsold / safe_pAp

        x_new = x + alpha_cg * p
        r_new = r - alpha_cg * Ap
        rsnew = jnp.dot(r_new, r_new)

        safe_rsold = jnp.where(rsold > 1e-30, rsold, 1.0)
        beta = jnp.where(rsold > 1e-30, rsnew / safe_rsold, 0.0)
        p_new = r_new + beta * p

        return (x_new, r_new, p_new, rsnew, i + 1)

    final_state = lax.while_loop(cond_fn, body_fn, init_state)
    x_final, _r_final, _p_final, rsold_final, n_iter = final_state

    return x_final, jnp.sqrt(rsold_final), n_iter


@jit
def _solve_tr_subproblem_cg(
    J: jnp.ndarray,
    f: jnp.ndarray,
    d: jnp.ndarray,
    Delta: float,
    alpha: float = 0.0,
) -> jnp.ndarray:
    """Solve trust region subproblem using conjugate gradient."""
    p_gn, _residual_norm, _n_iter = _conjugate_gradient_solve(J, f, d, 0.0)

    p_gn_norm = jnp.linalg.norm(p_gn)

    def compute_regularized():
        p_reg, _, _ = _conjugate_gradient_solve(J, f, d, alpha)
        p_reg_norm = jnp.maximum(jnp.linalg.norm(p_reg), 1e-10)
        # If regularized step is within trust region, use it directly;
        # otherwise scale to trust region boundary (no arbitrary clamping)
        scale = jnp.where(p_reg_norm <= Delta, 1.0, Delta / p_reg_norm)
        return scale * p_reg

    return lax.cond(
        p_gn_norm <= Delta,
        lambda: p_gn,
        compute_regularized,
    )


@jit
def _solve_tr_subproblem_cg_bounds(
    J: jnp.ndarray,
    f: jnp.ndarray,
    d: jnp.ndarray,
    J_diag: jnp.ndarray,
    f_zeros: jnp.ndarray,
    Delta: float,
    alpha: float = 0.0,
) -> jnp.ndarray:
    """Solve trust region subproblem with bounds using conjugate gradient."""
    J_augmented = jnp.concatenate([J * d[None, :], J_diag])
    f_augmented = jnp.concatenate([f, f_zeros])
    d_augmented = jnp.ones(J_augmented.shape[1], dtype=J_augmented.dtype)

    p_gn, _residual_norm, _n_iter = _conjugate_gradient_solve(
        J_augmented, f_augmented, d_augmented, 0.0
    )

    p_gn_norm = jnp.linalg.norm(p_gn)

    def compute_regularized():
        p_reg, _, _ = _conjugate_gradient_solve(
            J_augmented, f_augmented, d_augmented, alpha
        )
        p_reg_norm = jnp.maximum(jnp.linalg.norm(p_reg), 1e-10)
        # If regularized step is within trust region, use it directly;
        # otherwise scale to trust region boundary (no arbitrary clamping)
        scale = jnp.where(p_reg_norm <= Delta, 1.0, Delta / p_reg_norm)
        return scale * p_reg

    return lax.cond(
        p_gn_norm <= Delta,
        lambda: p_gn,
        compute_regularized,
    )


@jit
def _calculate_cost(rho: jnp.ndarray, data_mask: jnp.ndarray) -> jnp.ndarray:
    """Calculate cost: 0.5 * sum(masked rho[0])."""
    cost_array = jnp.where(data_mask, rho[0], 0)
    return LOSS_FUNCTION_COEFF * jnp.sum(cost_array)


@jit
def _check_isfinite(f_new: jnp.ndarray) -> jnp.ndarray:
    """Check if all residuals are finite."""
    return jnp.all(jnp.isfinite(f_new))


# ---------------------------------------------------------------------------
# Python wrappers for SVD with GPU/CPU fallback (can't be fully JIT-compiled)
# ---------------------------------------------------------------------------


def _svd_no_bounds(
    J: jnp.ndarray, d: jnp.ndarray, f: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute SVD of J in hat space (unbounded variant) with GPU fallback."""
    try:
        return _svd_no_bounds_jit(J, d, f)
    except Exception as e:
        if is_gpu_error(e):
            J_h = J * d
            U, s, V = compute_svd_with_fallback(J_h, full_matrices=False)
            uf = U.T.dot(f)
            return J_h, U, s, V, uf
        raise


def _svd_bounds(
    f: jnp.ndarray,
    J: jnp.ndarray,
    d: jnp.ndarray,
    J_diag: jnp.ndarray,
    f_zeros: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute SVD of J in hat space (bounded variant) with GPU fallback."""
    try:
        return _svd_bounds_jit(f, J, d, J_diag, f_zeros)
    except Exception as e:
        if is_gpu_error(e):
            J_h = J * d
            J_augmented = jnp.concatenate([J_h, J_diag])
            f_augmented = jnp.concatenate([f, f_zeros])
            U, s, V = compute_svd_with_fallback(J_augmented, full_matrices=False)
            uf = U.T.dot(f_augmented)
            return J_h, U, s, V, uf
        raise


# ---------------------------------------------------------------------------
# Class interface (backward-compatible, delegates to module-level singletons)
# ---------------------------------------------------------------------------


class TrustRegionJITFunctions:
    """JIT-compiled functions for Trust Region Reflective optimization algorithm.

    All JIT functions are module-level singletons to prevent XLA compilation cache
    bloat. Each function is compiled once per unique input shape, shared across all
    TrustRegionJITFunctions instances.

    Core Operations
    ---------------
    - **Gradient Computation**: JAX-accelerated gradient calculation using J^T * f
    - **SVD Decomposition**: Singular value decomposition for trust region subproblems
    - **Conjugate Gradient**: Iterative solver for large-scale problems
    - **Cost Function Evaluation**: Loss function computation with masking support
    - **Hat Space Transformation**: Scaled variable transformations for bounds handling

    Performance Characteristics
    ---------------------------
    - **Small Problems**: Direct SVD solution O(mn^2 + n^3)
    - **Large Problems**: CG iteration O(k*mn) where k is iteration count
    - **GPU Memory**: Module-level singletons prevent per-instance recompilation
    - **Numerical Stability**: Double precision arithmetic with condition monitoring
    """

    def __init__(self):
        """Bind module-level JIT singletons to instance attributes."""
        self.default_loss_func = _default_loss_func
        self.compute_grad = _compute_grad
        self.compute_grad_hat = _compute_grad_hat
        self.svd_no_bounds = _svd_no_bounds
        self.svd_bounds = _svd_bounds
        self.conjugate_gradient_solve = _conjugate_gradient_solve
        self.solve_tr_subproblem_cg = _solve_tr_subproblem_cg
        self.solve_tr_subproblem_cg_bounds = _solve_tr_subproblem_cg_bounds
        self.calculate_cost = _calculate_cost
        self.check_isfinite = _check_isfinite
