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

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.linalg import svd as jax_svd

from nlsq.common_jax import check_termination_jax, update_tr_radius_jax
from nlsq.stability.svd_fallback import compute_svd_with_fallback, is_gpu_error

__all__ = ["TrustRegionJITFunctions"]

# Algorithm constants
LOSS_FUNCTION_COEFF = 0.5  # Coefficient for loss function (0.5 * ||f||^2)
NUMERICAL_ZERO_THRESHOLD = 1e-14  # Threshold for values considered numerically zero
_CG_TINY = 1e-300  # CG division guard (not a tolerance; keeps CG scale-invariant)
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
    J: jnp.ndarray,
    d: jnp.ndarray,
    f: jnp.ndarray,
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

    # Solve for a normalized RHS: otherwise the CG curvature p^T A p
    # squares the magnitude of b as well as J and can underflow even when
    # A, b and the true solution are all representable. Restore the solution
    # and residual norm to the original RHS scale after the iteration.
    b_scale = jnp.max(jnp.abs(b))
    b_scale = jnp.where(b_scale > 0, b_scale, 1.0)
    b = b / b_scale

    x0 = jnp.zeros(n, dtype=b.dtype)
    r0 = b
    p0 = r0
    rsold0 = jnp.dot(r0, r0)
    # The float64 floor underflows to zero for float32 inputs. Keep a
    # representable positive divisor without changing the float64 guard.
    cg_tiny = max(_CG_TINY, float(jnp.finfo(b.dtype).tiny))
    # Relative to ||J^T f||^2 so the stopping rule is invariant to the scale of f
    tol_sq = tol * tol * rsold0

    init_state = (x0, r0, p0, rsold0, 0)

    def cond_fn(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]):
        _x, _r, _p, rsold, i = state
        return (i < CG_MAX_ITERATIONS) & (rsold > tol_sq)

    def body_fn(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]):
        x, r, p, rsold, i = state

        Jp = J_scaled @ p
        JTJp = J_scaled.T @ Jp
        Ap = JTJp + alpha * p

        pAp = jnp.dot(p, Ap)
        # Positivity guard only: an absolute floor would be scale dependent
        safe_pAp = jnp.where(pAp > cg_tiny, pAp, cg_tiny)
        alpha_cg = rsold / safe_pAp

        x_new = x + alpha_cg * p
        r_new = r - alpha_cg * Ap
        rsnew = jnp.dot(r_new, r_new)

        safe_rsold = jnp.where(rsold > cg_tiny, rsold, 1.0)
        beta = jnp.where(rsold > cg_tiny, rsnew / safe_rsold, 0.0)
        p_new = r_new + beta * p

        return (x_new, r_new, p_new, rsnew, i + 1)

    final_state = lax.while_loop(cond_fn, body_fn, init_state)
    x_final, _r_final, _p_final, rsold_final, n_iter = final_state

    return x_final * b_scale, jnp.sqrt(rsold_final) * b_scale, n_iter


@jit
def _seed_lm_alpha(g_norm: jnp.ndarray, alpha: float, Delta: float) -> jnp.ndarray:
    """Standard LM initial damping estimate (||J^T f|| / Delta, per More 1978).

    Used only when alpha is not already positive -- see the CG/lsmr trust
    region solvers below for why alpha reaches them still at its unset
    value of 0.0 on every call.
    """
    seed_alpha = jnp.where(Delta > 0, g_norm / Delta, 0.0)
    return jnp.where(alpha > 0, alpha, seed_alpha)


@jit
def _solve_tr_subproblem_cg(
    J: jnp.ndarray,
    f: jnp.ndarray,
    d: jnp.ndarray,
    Delta: float,
    alpha: float = 0.0,
    tol: float = DEFAULT_TOLERANCE,
) -> jnp.ndarray:
    """Solve trust region subproblem using conjugate gradient."""
    p_gn, _residual_norm, _n_iter = _conjugate_gradient_solve(J, f, d, 0.0, tol)

    p_gn_norm = jnp.linalg.norm(p_gn)

    def compute_regularized():
        # alpha is meant to carry the LM damping parameter forward across
        # outer iterations via a multiplicative rescale (trf.py), but the
        # value this CG solver seeds internally is never written back to
        # that outer-loop alpha, so alpha reaches here still at its unset
        # value of 0.0 on every call, and 0 * anything is still 0 --
        # without a nonzero seed here, this branch always regularizes with
        # alpha=0, silently degrading to the unregularized Gauss-Newton
        # solution merely rescaled to the trust region boundary.
        g_norm = jnp.linalg.norm((J * d[None, :]).T @ f)
        effective_alpha = _seed_lm_alpha(g_norm, alpha, Delta)
        p_reg, _, _ = _conjugate_gradient_solve(J, f, d, effective_alpha, tol)
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
    tol: float = DEFAULT_TOLERANCE,
) -> jnp.ndarray:
    """Solve trust region subproblem with bounds using conjugate gradient."""
    J_augmented = jnp.concatenate([J * d[None, :], J_diag])
    f_augmented = jnp.concatenate([f, f_zeros])
    d_augmented = jnp.ones(J_augmented.shape[1], dtype=J_augmented.dtype)

    p_gn, _residual_norm, _n_iter = _conjugate_gradient_solve(
        J_augmented,
        f_augmented,
        d_augmented,
        0.0,
        tol,
    )

    p_gn_norm = jnp.linalg.norm(p_gn)

    def compute_regularized():
        # See _solve_tr_subproblem_cg for why this seed is needed on every
        # call (alpha reaches here still at its unset value of 0.0).
        g_norm = jnp.linalg.norm(J_augmented.T @ f_augmented)
        effective_alpha = _seed_lm_alpha(g_norm, alpha, Delta)
        p_reg, _, _ = _conjugate_gradient_solve(
            J_augmented,
            f_augmented,
            d_augmented,
            effective_alpha,
            tol,
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


# Fused step-acceptance scalars (A2). Positions in the vector returned by
# `accept_step_scalars`; the host reads it with a single device-to-host transfer.
ACC_FINITE = 0
ACC_REDUCTION = 1
ACC_TERM = 2
ACC_DELTA = 3
ACC_ALPHA = 4
ACC_STEP_H_NORM = 5
ACC_STEP_NORM = 6
TR_BOUNDARY_THRESHOLD = 0.95  # step counts as "on the boundary" above this fraction


@jit
def accept_step_scalars(
    f_new: jnp.ndarray,
    cost_new: jnp.ndarray,
    cost: jnp.ndarray,
    Delta: jnp.ndarray,
    alpha: jnp.ndarray,
    predicted_reduction: jnp.ndarray,
    step_h: jnp.ndarray,
    step: jnp.ndarray,
    x: jnp.ndarray,
    ftol: float,
    xtol: float,
) -> jnp.ndarray:
    """All scalar math of one trust-region attempt, as one float64 vector.

    Same formulas as the sequential path (``update_tr_radius_jax``,
    ``check_termination_jax``, alpha rescaling). ``cost_new`` is ignored by the
    host when the residuals are not finite (entry ``ACC_FINITE`` is 0), so the
    loss may be evaluated before the finiteness check.
    """
    finite = jnp.all(jnp.isfinite(f_new))
    actual_reduction = cost - cost_new
    step_h_norm = jnp.linalg.norm(step_h)
    step_norm = jnp.linalg.norm(step)
    Delta_new, ratio = update_tr_radius_jax(
        Delta,
        actual_reduction,
        predicted_reduction,
        step_h_norm,
        step_h_norm > TR_BOUNDARY_THRESHOLD * Delta,
    )
    term = check_termination_jax(
        actual_reduction, cost, step_norm, jnp.linalg.norm(x), ratio, ftol, xtol
    )
    raw_alpha = alpha * Delta / Delta_new
    alpha_new = jnp.where(
        Delta_new > 0,
        jnp.minimum(jnp.where(jnp.isfinite(raw_alpha), raw_alpha, 1e30), 1e30),
        alpha,
    )
    return jnp.stack(
        [
            finite,
            actual_reduction,
            term,
            Delta_new,
            alpha_new,
            step_h_norm,
            step_norm,
        ]
    ).astype(jnp.float64)


# ---------------------------------------------------------------------------
# Python wrappers for SVD with GPU/CPU fallback (can't be fully JIT-compiled)
# ---------------------------------------------------------------------------


def _svd_no_bounds(
    J: jnp.ndarray,
    d: jnp.ndarray,
    f: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute SVD of J in hat space (unbounded variant) with GPU fallback."""
    try:
        result = _svd_no_bounds_jit(J, d, f)
        # Force materialization so an async GPU/cuSolver failure is raised HERE,
        # inside this try, rather than later at a downstream blocking op where it
        # would bypass the CPU fallback below.
        jax.block_until_ready(result)
        return result
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
        result = _svd_bounds_jit(f, J, d, J_diag, f_zeros)
        # Force materialization so an async GPU/cuSolver failure is raised HERE,
        # inside this try, rather than later at a downstream blocking op where it
        # would bypass the CPU fallback below.
        jax.block_until_ready(result)
        return result
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
