"""Phase 2: Streaming Gauss-Newton optimization.

This module contains the GaussNewtonPhase class that encapsulates
the streaming Gauss-Newton logic for the AdaptiveHybridStreamingOptimizer.

The Gauss-Newton phase streams over the full dataset in chunks,
accumulating JtJ and Jtr implicitly, and solving normal equations via CG.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from nlsq.utils.logging import get_logger

if TYPE_CHECKING:
    from jax import Array

    from nlsq.precision.parameter_normalizer import NormalizedModelWrapper
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

_logger = get_logger("gauss_newton_phase")


@dataclass(frozen=True, slots=True)
class GNResult:
    """Result from streaming Gauss-Newton phase.

    Attributes:
        params: Final optimized parameters.
        cost: Final cost value.
        iterations: Number of GN iterations.
        converged: Whether GN converged.
        jacobian: Final Jacobian matrix (optional).
        cov: Parameter covariance matrix (optional).
    """

    params: Array
    cost: float
    iterations: int
    converged: bool
    jacobian: Array | None = None
    cov: Array | None = None


class GaussNewtonPhase:
    """Phase 2: Streaming Gauss-Newton with implicit JtJ.

    This class encapsulates the streaming Gauss-Newton logic for large
    dataset optimization. It streams over the full dataset in chunks,
    accumulating J^T J and J^T r, then solving the normal equations.

    Parameters
    ----------
    config : HybridStreamingConfig
        Configuration for streaming optimization.
    normalized_model : NormalizedModelWrapper
        Model wrapper operating in normalized parameter space.
    normalized_bounds : tuple of array_like or None
        Parameter bounds in normalized space.

    Attributes
    ----------
    config : HybridStreamingConfig
        Configuration object.
    normalized_model : NormalizedModelWrapper
        Normalized model wrapper.
    normalized_bounds : tuple of array_like or None
        Bounds in normalized space.

    Notes
    -----
    The Gauss-Newton method iteratively solves::

        (J^T J) @ step = J^T r

    where J is the Jacobian and r is the residual vector.
    """

    def __init__(
        self,
        config: HybridStreamingConfig,
        normalized_model: NormalizedModelWrapper,
        normalized_bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> None:
        """Initialize GaussNewtonPhase.

        Parameters
        ----------
        config : HybridStreamingConfig
            Configuration for streaming optimization.
        normalized_model : NormalizedModelWrapper
            Model wrapper operating in normalized parameter space.
        normalized_bounds : tuple of array_like or None
            Parameter bounds in normalized space.
        """
        self.config = config
        self.normalized_model = normalized_model
        self.normalized_bounds = normalized_bounds

        # Pre-compiled functions (set externally or lazily initialized)
        self._jacobian_fn_compiled: Callable | None = None
        self._cost_fn_compiled: Callable | None = None

        # Accumulators for checkpointing
        self.phase2_JTJ_accumulator: jnp.ndarray | None = None
        self.phase2_JTr_accumulator: jnp.ndarray | None = None

        # Stall detection
        self._consecutive_rejections: int = 0

    def set_jacobian_fn(self, fn: Callable | None) -> None:
        """Set pre-compiled Jacobian function."""
        self._jacobian_fn_compiled = fn

    def set_cost_fn(self, fn: Callable | None) -> None:
        """Set pre-compiled cost function."""
        self._cost_fn_compiled = fn

    def run(
        self,
        data_source: tuple[jnp.ndarray, jnp.ndarray],
        initial_params: jnp.ndarray,
        phase_history: list[dict[str, Any]],
        best_tracker: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Phase 2 streaming Gauss-Newton optimization.

        Parameters
        ----------
        data_source : tuple of array_like
            Full dataset as (x_data, y_data).
        initial_params : array_like
            Starting parameters in normalized space (from Phase 1).
        phase_history : list
            Phase history list to append records to.
        best_tracker : dict
            Dictionary tracking best_params_global and best_cost_global.

        Returns
        -------
        result : dict
            Phase 2 optimization result with keys:
            - 'final_params': Final parameters in normalized space
            - 'best_params': Best parameters found
            - 'best_cost': Best cost achieved
            - 'final_cost': Final cost value
            - 'iterations': Number of Gauss-Newton iterations
            - 'convergence_reason': Why optimization stopped
            - 'gradient_norm': Final gradient norm
            - 'JTJ_final': Final accumulated J^T J matrix (for Phase 3)
            - 'residual_sum_sq': Total residual sum of squares (for Phase 3)
            - 'gn_result': GNResult dataclass instance
        """
        current_params = initial_params
        trust_radius = self.config.trust_region_initial

        # Track best parameters
        best_params = current_params
        best_cost = jnp.inf
        prev_cost = jnp.inf

        x_data, y_data = data_source
        n_params = len(current_params)
        chunk_size = self.config.chunk_size
        n_points = len(x_data)

        final_JTJ = jnp.zeros((n_params, n_params))
        final_JTr = jnp.zeros(n_params)
        final_residual_sum_sq = 0.0

        verbose = getattr(self.config, "verbose", 1)
        log_frequency = getattr(self.config, "log_frequency", 1)

        # Compute initial JTJ with progress reporting
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        init_start_time = time.time()
        if verbose >= 1:
            print(
                f"  Computing initial JTJ ({n_chunks} chunks, {n_points:,} points)..."
            )

        for chunk_idx, i in enumerate(range(0, n_points, chunk_size)):
            x_chunk = x_data[i : i + chunk_size]
            y_chunk = y_data[i : i + chunk_size]

            final_JTJ, final_JTr, res_sq = self._accumulate_jtj_jtr(
                x_chunk, y_chunk, current_params, final_JTJ, final_JTr
            )
            final_residual_sum_sq += res_sq

            if verbose >= 1 and (
                (chunk_idx + 1) % max(1, n_chunks // 10) == 0
                or (chunk_idx + 1) == n_chunks
            ):
                elapsed = time.time() - init_start_time
                pct = (chunk_idx + 1) / n_chunks * 100
                print(
                    f"  Initial JTJ: {chunk_idx + 1}/{n_chunks} chunks "
                    f"({pct:.0f}%), elapsed={elapsed:.1f}s"
                )

        if verbose >= 1:
            init_elapsed = time.time() - init_start_time
            print(
                f"  Initial JTJ complete: cost={final_residual_sum_sq:.6e}, "
                f"time={init_elapsed:.1f}s"
            )

        # Initialize stall detection
        self._consecutive_rejections = 0
        gradient_norm = 0.0

        # Gauss-Newton loop
        for iteration in range(self.config.gauss_newton_max_iterations):
            iter_start_time = time.time()

            # Perform one Gauss-Newton iteration
            iter_result = self._gauss_newton_iteration(
                data_source, current_params, trust_radius
            )

            new_params = iter_result["new_params"]
            new_cost = iter_result["new_cost"]
            gradient_norm = iter_result["gradient_norm"]
            actual_reduction = iter_result["actual_reduction"]
            trust_radius = iter_result["trust_radius"]
            iter_time = time.time() - iter_start_time

            # Progress logging
            if verbose >= 1 and (iteration + 1) % log_frequency == 0:
                max_iter = self.config.gauss_newton_max_iterations
                print(
                    f"  GN iter {iteration + 1}/{max_iter}: "
                    f"cost={new_cost:.6e}, grad={gradient_norm:.6e}, "
                    f"red={actual_reduction:.6e}, Δ={trust_radius:.4f}, "
                    f"time={iter_time:.1f}s"
                )

            # Update best parameters
            if new_cost < best_cost:
                best_cost = new_cost
                best_params = new_params

            # Track global best
            if new_cost < best_tracker.get("best_cost_global", float("inf")):
                best_tracker["best_cost_global"] = new_cost
                best_tracker["best_params_global"] = new_params

            # Accept step if cost decreased
            cost_before_step = prev_cost if jnp.isfinite(prev_cost) else new_cost

            if actual_reduction > 0:
                current_params = new_params
                cost_before_step = (
                    prev_cost
                    if jnp.isfinite(prev_cost)
                    else new_cost + actual_reduction
                )
                prev_cost = new_cost
                self._consecutive_rejections = 0

                # Recompute J^T J at new params for Phase 3
                JTJ = jnp.zeros((n_params, n_params))
                JTr = jnp.zeros(n_params)
                residual_sum_sq = 0.0

                for i in range(0, n_points, chunk_size):
                    x_chunk = x_data[i : i + chunk_size]
                    y_chunk = y_data[i : i + chunk_size]

                    JTJ, JTr, res_sq = self._accumulate_jtj_jtr(
                        x_chunk, y_chunk, current_params, JTJ, JTr
                    )
                    residual_sum_sq += res_sq

                final_JTJ = JTJ
                final_residual_sum_sq = residual_sum_sq
            else:
                self._consecutive_rejections += 1

                # Stall detection
                if self._consecutive_rejections >= 10 and gradient_norm > 1e-4:
                    trust_radius = self.config.trust_region_initial
                    self._consecutive_rejections = 0
                    if verbose >= 1:
                        print(
                            f"  Stall detected: resetting trust radius to "
                            f"{trust_radius:.4f}"
                        )

            # Check convergence: gradient norm
            if gradient_norm < self.config.gauss_newton_tol:
                phase_record = {
                    "phase": 2,
                    "name": "gauss_newton",
                    "iterations": iteration + 1,
                    "final_cost": new_cost,
                    "best_cost": best_cost,
                    "convergence_reason": "Gradient norm below tolerance",
                    "gradient_norm": gradient_norm,
                    "timestamp": time.time(),
                }
                phase_history.append(phase_record)

                gn_result = GNResult(
                    params=new_params,
                    cost=new_cost,
                    iterations=iteration + 1,
                    converged=True,
                )

                return {
                    "final_params": new_params,
                    "best_params": best_params,
                    "best_cost": best_cost,
                    "final_cost": new_cost,
                    "iterations": iteration + 1,
                    "convergence_reason": "Gradient norm below tolerance",
                    "gradient_norm": gradient_norm,
                    "JTJ_final": final_JTJ,
                    "residual_sum_sq": final_residual_sum_sq,
                    "gn_result": gn_result,
                }

            # Check convergence: cost change
            cost_change = abs(cost_before_step - new_cost)
            relative_change = cost_change / (abs(cost_before_step) + 1e-10)

            if relative_change < self.config.gauss_newton_tol:
                phase_record = {
                    "phase": 2,
                    "name": "gauss_newton",
                    "iterations": iteration + 1,
                    "final_cost": new_cost,
                    "best_cost": best_cost,
                    "convergence_reason": "Cost change below tolerance",
                    "gradient_norm": gradient_norm,
                    "timestamp": time.time(),
                }
                phase_history.append(phase_record)

                gn_result = GNResult(
                    params=new_params,
                    cost=new_cost,
                    iterations=iteration + 1,
                    converged=True,
                )

                return {
                    "final_params": new_params,
                    "best_params": best_params,
                    "best_cost": best_cost,
                    "final_cost": new_cost,
                    "iterations": iteration + 1,
                    "convergence_reason": "Cost change below tolerance",
                    "gradient_norm": gradient_norm,
                    "JTJ_final": final_JTJ,
                    "residual_sum_sq": final_residual_sum_sq,
                    "gn_result": gn_result,
                }

        # Maximum iterations reached
        phase_record = {
            "phase": 2,
            "name": "gauss_newton",
            "iterations": self.config.gauss_newton_max_iterations,
            "final_cost": prev_cost,
            "best_cost": best_cost,
            "convergence_reason": "Maximum iterations reached",
            "gradient_norm": gradient_norm,
            "timestamp": time.time(),
        }
        phase_history.append(phase_record)

        gn_result = GNResult(
            params=best_params,
            cost=best_cost,
            iterations=self.config.gauss_newton_max_iterations,
            converged=False,
        )

        return {
            "final_params": best_params,
            "best_params": best_params,
            "best_cost": best_cost,
            "final_cost": prev_cost,
            "iterations": self.config.gauss_newton_max_iterations,
            "convergence_reason": "Maximum iterations reached",
            "gradient_norm": gradient_norm,
            "JTJ_final": final_JTJ,
            "residual_sum_sq": final_residual_sum_sq,
            "gn_result": gn_result,
        }

    def _gauss_newton_iteration(
        self,
        data_source: tuple[jnp.ndarray, jnp.ndarray],
        current_params: jnp.ndarray,
        trust_radius: float,
    ) -> dict[str, Any]:
        """Perform one complete Gauss-Newton iteration.

        Parameters
        ----------
        data_source : tuple of array_like
            Full dataset as (x_data, y_data).
        current_params : array_like
            Current parameters in normalized space.
        trust_radius : float
            Current trust region radius.

        Returns
        -------
        result : dict
            Iteration result with keys: new_params, new_cost, step,
            actual_reduction, predicted_reduction, trust_radius, gradient_norm.
        """
        x_data, y_data = data_source
        n_params = len(current_params)
        chunk_size = self.config.chunk_size
        n_points = len(x_data)

        # Accumulate J^T J and J^T r
        JTJ = jnp.zeros((n_params, n_params))
        JTr = jnp.zeros(n_params)
        total_cost = 0.0

        for i in range(0, n_points, chunk_size):
            x_chunk = x_data[i : i + chunk_size]
            y_chunk = y_data[i : i + chunk_size]
            JTJ, JTr, chunk_cost = self._accumulate_jtj_jtr(
                x_chunk, y_chunk, current_params, JTJ, JTr
            )
            total_cost += chunk_cost

        # Add group variance regularization if enabled
        if (
            self.config.enable_group_variance_regularization
            and self.config.group_variance_indices
        ):
            var_lambda = self.config.group_variance_lambda
            for start, end in self.config.group_variance_indices:
                group_params = current_params[start:end]
                n_group = end - start
                group_mean = jnp.mean(group_params)

                grad_var = (2.0 / n_group) * (group_params - group_mean)
                JTr = JTr.at[start:end].add(-var_lambda * grad_var)

                diag_term = (2.0 / n_group) * jnp.eye(n_group)
                off_diag_term = (2.0 / (n_group * n_group)) * jnp.ones(
                    (n_group, n_group)
                )
                H_var = diag_term - off_diag_term
                JTJ = JTJ.at[start:end, start:end].add(var_lambda * H_var)

                group_var = jnp.var(group_params)
                total_cost += var_lambda * float(group_var) * n_points

        gradient_norm = float(jnp.linalg.norm(JTr))

        # Solve for Gauss-Newton step
        step, predicted_reduction = self._solve_gauss_newton_step(
            JTJ, JTr, trust_radius
        )

        # Apply step
        new_params = current_params + step

        # Clip to bounds if available
        if self.normalized_bounds is not None:
            lb, ub = self.normalized_bounds
            new_params = jnp.clip(new_params, lb, ub)

        # Evaluate cost at new parameters
        new_cost = self._compute_cost(new_params, x_data, y_data)

        # Compute actual reduction
        actual_reduction = total_cost - new_cost

        # Update trust region
        if predicted_reduction > 0:
            reduction_ratio = actual_reduction / predicted_reduction
        else:
            reduction_ratio = 0.0

        min_trust_radius = getattr(self.config, "min_trust_radius", 1e-8)
        max_trust_radius = getattr(self.config, "max_trust_radius", 1000.0)
        step_norm = float(jnp.linalg.norm(step))

        if reduction_ratio < 0.25:
            new_trust_radius = trust_radius * 0.5
            if new_trust_radius < min_trust_radius and gradient_norm > 1e-4:
                scaled_grad = 0.1 * gradient_norm / max(1.0, gradient_norm)
                new_trust_radius = min(scaled_grad, 1.0)
        elif reduction_ratio > 0.75 and step_norm >= 0.9 * trust_radius:
            new_trust_radius = min(trust_radius * 2.0, max_trust_radius)
        else:
            new_trust_radius = trust_radius

        new_trust_radius = max(new_trust_radius, min_trust_radius)

        return {
            "new_params": new_params,
            "new_cost": new_cost,
            "step": step,
            "actual_reduction": actual_reduction,
            "predicted_reduction": predicted_reduction,
            "trust_radius": new_trust_radius,
            "gradient_norm": gradient_norm,
        }

    def _accumulate_jtj_jtr(
        self,
        x_chunk: jnp.ndarray,
        y_chunk: jnp.ndarray,
        params: jnp.ndarray,
        JTJ_prev: jnp.ndarray,
        JTr_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, float]:
        """Accumulate J^T J and J^T r for a data chunk.

        Parameters
        ----------
        x_chunk : array_like
            Independent variable chunk.
        y_chunk : array_like
            Dependent variable chunk.
        params : array_like
            Current parameters in normalized space.
        JTJ_prev : array_like
            Previous accumulated J^T J.
        JTr_prev : array_like
            Previous accumulated J^T r.

        Returns
        -------
        JTJ_new : array_like
            Updated J^T J accumulation.
        JTr_new : array_like
            Updated J^T r accumulation.
        chunk_cost : float
            Sum of squared residuals for this chunk.
        """
        # Compute predictions and residuals
        predictions = self.normalized_model(x_chunk, *params)
        residuals = y_chunk - predictions

        # Compute Jacobian for this chunk
        J_chunk = self._compute_jacobian_chunk(x_chunk, params)

        # Accumulate
        JTJ_new = JTJ_prev + J_chunk.T @ J_chunk
        JTr_new = JTr_prev + J_chunk.T @ residuals
        chunk_cost = float(jnp.sum(residuals**2))

        # Store accumulators for checkpointing
        self.phase2_JTJ_accumulator = JTJ_new
        self.phase2_JTr_accumulator = JTr_new

        return JTJ_new, JTr_new, chunk_cost

    def _compute_jacobian_chunk(
        self,
        x_chunk: jnp.ndarray,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute exact Jacobian for a data chunk.

        Parameters
        ----------
        x_chunk : array_like
            Independent variable chunk.
        params : array_like
            Parameters in normalized space.

        Returns
        -------
        J_chunk : array_like
            Jacobian matrix of shape (n_points, n_params).
        """
        if self._jacobian_fn_compiled is not None:
            return self._jacobian_fn_compiled(params, x_chunk)

        normalized_model = self.normalized_model

        def model_at_x(p, x_single):
            return normalized_model(x_single, *p)

        jac_fn = jax.vmap(lambda x: jax.jacrev(model_at_x, argnums=0)(params, x))
        return jac_fn(x_chunk)

    def _solve_gauss_newton_step(
        self,
        JTJ: jnp.ndarray,
        JTr: jnp.ndarray,
        trust_radius: float,
        regularization: float = 1e-10,
    ) -> tuple[jnp.ndarray, float]:
        """Solve Gauss-Newton step using SVD.

        Parameters
        ----------
        JTJ : array_like
            Accumulated J^T J matrix.
        JTr : array_like
            Accumulated J^T r vector.
        trust_radius : float
            Trust region radius.
        regularization : float
            Tikhonov regularization parameter.

        Returns
        -------
        step : array_like
            Gauss-Newton step.
        predicted_reduction : float
            Predicted reduction in cost function.
        """
        n_params = JTJ.shape[0]

        # Add Tikhonov regularization
        JTJ_reg = JTJ + regularization * jnp.eye(n_params)

        # Compute SVD
        U, s, Vt = jnp.linalg.svd(JTJ_reg, full_matrices=False)

        # Solve using SVD
        UTb = U.T @ JTr
        s_threshold = jnp.max(s) * 1e-10
        s_safe = jnp.where(s > s_threshold, s, s_threshold)
        step_hat = UTb / s_safe
        step = Vt.T @ step_hat

        # Apply trust region constraint
        step_norm = jnp.linalg.norm(step)
        if step_norm > trust_radius:
            step = step * (trust_radius / step_norm)

        # Compute predicted reduction
        pred_red = jnp.dot(JTr, step) - 0.5 * jnp.dot(step, JTJ @ step)
        predicted_reduction = float(max(float(pred_red), 0.0))

        return step, predicted_reduction

    def _compute_cost(
        self,
        params: jnp.ndarray,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
    ) -> float:
        """Compute total cost (sum of squared residuals).

        Parameters
        ----------
        params : array_like
            Parameters in normalized space.
        x_data : array_like
            Full x data.
        y_data : array_like
            Full y data.

        Returns
        -------
        total_cost : float
            Total sum of squared residuals.
        """
        chunk_size = self.config.chunk_size
        n_points = len(x_data)
        total_cost = 0.0

        for i in range(0, n_points, chunk_size):
            x_chunk = x_data[i : i + chunk_size]
            y_chunk = y_data[i : i + chunk_size]

            if self._cost_fn_compiled is not None:
                chunk_cost = self._cost_fn_compiled(params, x_chunk, y_chunk)
            else:
                predictions = self.normalized_model(x_chunk, *params)
                residuals = y_chunk - predictions
                chunk_cost = float(jnp.sum(residuals**2))

            total_cost += chunk_cost

        # Add group variance regularization if enabled
        if (
            self.config.enable_group_variance_regularization
            and self.config.group_variance_indices
        ):
            var_lambda = self.config.group_variance_lambda
            for start, end in self.config.group_variance_indices:
                group_params = params[start:end]
                group_var = jnp.var(group_params)
                total_cost += var_lambda * float(group_var) * n_points

        return total_cost


__all__ = ["GNResult", "GaussNewtonPhase"]
