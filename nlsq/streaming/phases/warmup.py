"""Phase 1: L-BFGS Warmup optimization.

This module contains the WarmupPhase class that encapsulates
the L-BFGS warmup logic for the AdaptiveHybridStreamingOptimizer.

The warmup phase runs L-BFGS on sampled data chunks to provide
warm-started parameters for the streaming Gauss-Newton phase.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import optax  # type: ignore[import-not-found,import-untyped]

from nlsq.streaming.telemetry import get_defense_telemetry
from nlsq.utils.logging import get_logger

if TYPE_CHECKING:
    from jax import Array

    from nlsq.precision.parameter_normalizer import NormalizedModelWrapper
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

_logger = get_logger("warmup_phase")


@dataclass(frozen=True, slots=True)
class WarmupResult:
    """Result from L-BFGS warmup phase.

    Attributes:
        params: Optimized parameters after warmup.
        cost: Final cost after warmup.
        iterations: Number of warmup iterations performed.
        converged: Whether warmup converged.
        cost_history: Cost at each iteration.
    """

    params: Array
    cost: float
    iterations: int
    converged: bool
    cost_history: list[float]


class WarmupPhase:
    """Phase 1: L-BFGS warmup for initial convergence.

    This class encapsulates the L-BFGS warmup logic that provides
    warm-started parameters for the streaming Gauss-Newton phase.

    L-BFGS provides 5-10x faster convergence to the basin of attraction
    compared to first-order warmup by using approximate second-order
    (Hessian) information.

    Parameters
    ----------
    config : HybridStreamingConfig
        Configuration for streaming optimization.
    normalized_model : NormalizedModelWrapper
        Model wrapper operating in normalized parameter space.

    Attributes
    ----------
    config : HybridStreamingConfig
        Configuration object.
    normalized_model : NormalizedModelWrapper
        Normalized model wrapper.

    Notes
    -----
    The 4-Layer Defense Strategy is implemented to prevent warmup divergence:
    - Layer 1: Warm start detection (skip if already near optimum)
    - Layer 2: Adaptive initial step size based on relative loss
    - Layer 3: Cost-increase guard (abort if loss increases beyond tolerance)
    - Layer 4: Trust region constraint (clip update magnitude)
    """

    def __init__(
        self,
        config: HybridStreamingConfig,
        normalized_model: NormalizedModelWrapper,
    ) -> None:
        """Initialize WarmupPhase.

        Parameters
        ----------
        config : HybridStreamingConfig
            Configuration for streaming optimization.
        normalized_model : NormalizedModelWrapper
            Model wrapper operating in normalized parameter space.
        """
        self.config = config
        self.normalized_model = normalized_model

        # State for 4-layer defense strategy
        self._initial_loss: float | None = None
        self._relative_loss: float | None = None
        self._lr_mode: str | None = None
        self._clip_count: int = 0

        # Residual weighting state (optional)
        self._residual_weights: jnp.ndarray | None = None

    def set_residual_weights(self, weights: jnp.ndarray | None) -> None:
        """Set per-group residual weights for weighted least squares.

        Parameters
        ----------
        weights : array_like or None
            Per-group weights for weighted MSE computation.
        """
        self._residual_weights = weights

    def run(
        self,
        data_source: tuple[jnp.ndarray, jnp.ndarray],
        initial_params: jnp.ndarray,
        phase_history: list[dict[str, Any]],
        best_tracker: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Phase 1 L-BFGS warmup.

        Parameters
        ----------
        data_source : tuple of array_like
            Data source as (x_data, y_data).
        initial_params : array_like
            Initial parameters in normalized space.
        phase_history : list
            Phase history list to append records to.
        best_tracker : dict
            Dictionary tracking best_params_global and best_cost_global.

        Returns
        -------
        result : dict
            Phase 1 result with keys:
            - 'final_params': Final parameters in normalized space
            - 'best_params': Best parameters found during warmup
            - 'best_loss': Best loss value
            - 'final_loss': Final loss value
            - 'iterations': Number of iterations performed
            - 'switch_reason': Reason for switching to Phase 2
            - 'warmup_result': WarmupResult dataclass instance
        """
        # Extract data
        x_data, y_data = data_source
        x_data = jnp.asarray(x_data, dtype=jnp.float64)
        y_data = jnp.asarray(y_data, dtype=jnp.float64)

        current_params = initial_params

        # Create loss function
        loss_fn = self._create_loss_fn()

        # Record telemetry for warmup start
        telemetry = get_defense_telemetry()
        telemetry.record_warmup_start()

        # =====================================================
        # LAYER 1: Warm Start Detection
        # =====================================================
        initial_loss = float(loss_fn(current_params, x_data, y_data))
        y_variance = float(jnp.var(y_data))
        relative_loss = initial_loss / (y_variance + 1e-10)

        self._initial_loss = initial_loss
        self._relative_loss = relative_loss

        if self.config.verbose >= 2:
            _logger.debug(
                f"Phase 1 initial assessment: loss={initial_loss:.6e}, "
                f"y_var={y_variance:.6e}, relative_loss={relative_loss:.6e}"
            )

        # Check warm start threshold
        if (
            self.config.enable_warm_start_detection
            and relative_loss < self.config.warm_start_threshold
        ):
            telemetry.record_layer1_trigger(
                relative_loss=relative_loss, threshold=self.config.warm_start_threshold
            )

            phase_record = {
                "phase": 1,
                "name": "lbfgs_warmup",
                "iterations": 0,
                "final_loss": initial_loss,
                "best_loss": initial_loss,
                "switch_reason": (
                    f"Warm start detected (relative_loss={relative_loss:.4e} "
                    f"< {self.config.warm_start_threshold})"
                ),
                "timestamp": time.time(),
                "skipped": True,
                "warm_start": True,
                "relative_loss": relative_loss,
            }
            phase_history.append(phase_record)

            if self.config.verbose >= 1:
                _logger.info(
                    f"Phase 1: Skipping L-BFGS warmup - warm start detected "
                    f"(relative_loss={relative_loss:.4e})"
                )

            warmup_result = WarmupResult(
                params=current_params,
                cost=initial_loss,
                iterations=0,
                converged=True,
                cost_history=[initial_loss],
            )

            return {
                "final_params": current_params,
                "best_params": current_params,
                "best_loss": initial_loss,
                "final_loss": initial_loss,
                "iterations": 0,
                "switch_reason": "Warm start detected - skipping L-BFGS warmup",
                "warm_start": True,
                "relative_loss": relative_loss,
                "warmup_result": warmup_result,
            }

        # =====================================================
        # LAYER 2: Adaptive Initial Step Size Selection
        # =====================================================
        initial_step, lr_mode = self._select_initial_step_size(relative_loss, telemetry)
        self._lr_mode = lr_mode

        # Create L-BFGS optimizer
        optimizer, opt_state = self._create_lbfgs_optimizer(
            current_params, initial_step
        )

        # Best parameter tracking
        best_params = current_params
        best_loss = initial_loss
        cost_history = [initial_loss]

        prev_loss = initial_loss
        self._clip_count = 0

        # Warmup loop
        for iteration in range(self.config.max_warmup_iterations):
            current_params, loss_value, grad_norm, opt_state, _line_search_failed = (
                self._lbfgs_step(
                    params=current_params,
                    opt_state=opt_state,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    x_batch=x_data,
                    y_batch=y_data,
                    iteration=iteration,
                    best_tracker=best_tracker,
                )
            )

            cost_history.append(loss_value)

            # Track best parameters
            if loss_value < best_loss:
                best_loss = loss_value
                best_params = current_params

            # =====================================================
            # LAYER 3: Cost-Increase Guard
            # =====================================================
            if self.config.enable_cost_guard and iteration > 0:
                result = self._check_cost_guard(
                    iteration=iteration,
                    loss_value=loss_value,
                    best_params=best_params,
                    best_loss=best_loss,
                    lr_mode=lr_mode,
                    relative_loss=relative_loss,
                    phase_history=phase_history,
                    telemetry=telemetry,
                    cost_history=cost_history,
                )
                if result is not None:
                    return result

            # Check switch criteria after minimum warmup iterations
            if iteration >= self.config.warmup_iterations:
                should_switch, reason = self._check_switch_criteria(
                    iteration=iteration,
                    current_loss=loss_value,
                    prev_loss=prev_loss,
                    grad_norm=grad_norm,
                )

                if should_switch:
                    phase_record = {
                        "phase": 1,
                        "name": "lbfgs_warmup",
                        "iterations": iteration + 1,
                        "final_loss": loss_value,
                        "best_loss": best_loss,
                        "switch_reason": reason,
                        "timestamp": time.time(),
                        "lr_mode": lr_mode,
                        "relative_loss": relative_loss,
                    }
                    phase_history.append(phase_record)

                    warmup_result = WarmupResult(
                        params=current_params,
                        cost=loss_value,
                        iterations=iteration + 1,
                        converged=True,
                        cost_history=cost_history,
                    )

                    return {
                        "final_params": current_params,
                        "best_params": best_params,
                        "best_loss": best_loss,
                        "final_loss": loss_value,
                        "iterations": iteration + 1,
                        "switch_reason": reason,
                        "lr_mode": lr_mode,
                        "relative_loss": relative_loss,
                        "warmup_result": warmup_result,
                    }

            prev_loss = loss_value

        # Maximum iterations reached
        phase_record = {
            "phase": 1,
            "name": "lbfgs_warmup",
            "iterations": self.config.max_warmup_iterations,
            "final_loss": loss_value,
            "best_loss": best_loss,
            "switch_reason": "Maximum iterations reached",
            "timestamp": time.time(),
            "lr_mode": lr_mode,
            "relative_loss": relative_loss,
        }
        phase_history.append(phase_record)

        warmup_result = WarmupResult(
            params=current_params,
            cost=loss_value,
            iterations=self.config.max_warmup_iterations,
            converged=False,
            cost_history=cost_history,
        )

        return {
            "final_params": current_params,
            "best_params": best_params,
            "best_loss": best_loss,
            "final_loss": loss_value,
            "iterations": self.config.max_warmup_iterations,
            "switch_reason": "Maximum iterations reached",
            "lr_mode": lr_mode,
            "relative_loss": relative_loss,
            "warmup_result": warmup_result,
        }

    def _create_loss_fn(self) -> Callable:
        """Create loss function for warmup phase.

        Returns
        -------
        loss_fn : callable
            Loss function: loss_fn(params, x_batch, y_batch) -> scalar.
        """
        normalized_model = self.normalized_model

        enable_var_reg = self.config.enable_group_variance_regularization
        var_lambda = self.config.group_variance_lambda
        var_indices = self.config.group_variance_indices

        enable_weighting = (
            self.config.enable_residual_weighting and self._residual_weights is not None
        )
        residual_weights = self._residual_weights

        if enable_var_reg and var_indices and enable_weighting:
            group_slices = [(start, end) for start, end in var_indices]

            @jax.jit
            def loss_fn(
                params: jnp.ndarray, x_batch: jnp.ndarray, y_batch: jnp.ndarray
            ) -> jnp.ndarray:
                predictions = normalized_model(x_batch, *params)
                residuals = y_batch - predictions
                group_idx = x_batch[:, 0].astype(jnp.int32)
                assert residual_weights is not None
                weights = residual_weights[group_idx]
                wmse = jnp.sum(weights * residuals**2) / jnp.sum(weights)

                variance_penalty: jnp.ndarray = jnp.array(0.0)
                for start, end in group_slices:
                    group_params = params[start:end]
                    group_var = jnp.var(group_params)
                    variance_penalty = variance_penalty + group_var

                return wmse + var_lambda * variance_penalty

        elif enable_var_reg and var_indices:
            group_slices = [(start, end) for start, end in var_indices]

            @jax.jit
            def loss_fn(
                params: jnp.ndarray, x_batch: jnp.ndarray, y_batch: jnp.ndarray
            ) -> jnp.ndarray:
                predictions = normalized_model(x_batch, *params)
                residuals = y_batch - predictions
                mse = jnp.mean(residuals**2)

                variance_penalty: jnp.ndarray = jnp.array(0.0)
                for start, end in group_slices:
                    group_params = params[start:end]
                    group_var = jnp.var(group_params)
                    variance_penalty = variance_penalty + group_var

                return mse + var_lambda * variance_penalty

        elif enable_weighting:

            @jax.jit
            def loss_fn(
                params: jnp.ndarray, x_batch: jnp.ndarray, y_batch: jnp.ndarray
            ) -> jnp.ndarray:
                predictions = normalized_model(x_batch, *params)
                residuals = y_batch - predictions
                group_idx = x_batch[:, 0].astype(jnp.int32)
                assert residual_weights is not None
                weights = residual_weights[group_idx]
                wmse = jnp.sum(weights * residuals**2) / jnp.sum(weights)
                return wmse

        else:

            @jax.jit
            def loss_fn(
                params: jnp.ndarray, x_batch: jnp.ndarray, y_batch: jnp.ndarray
            ) -> jnp.ndarray:
                predictions = normalized_model(x_batch, *params)
                residuals = y_batch - predictions
                return jnp.mean(residuals**2)

        return loss_fn

    def _select_initial_step_size(
        self, relative_loss: float, telemetry: Any
    ) -> tuple[float, str]:
        """Select initial step size based on relative loss (Layer 2).

        Parameters
        ----------
        relative_loss : float
            Initial loss relative to data variance.
        telemetry : DefenseLayerTelemetry
            Telemetry collector.

        Returns
        -------
        initial_step : float
            Selected initial step size.
        lr_mode : str
            Mode name for logging.
        """
        if self.config.enable_adaptive_warmup_lr:
            if relative_loss < 0.1:
                initial_step = self.config.lbfgs_refinement_step_size
                lr_mode = "refinement"
            elif relative_loss < 1.0:
                initial_step = 0.5
                lr_mode = "careful"
            else:
                initial_step = self.config.lbfgs_exploration_step_size
                lr_mode = "exploration"

            telemetry.record_layer2_lr_mode(mode=lr_mode, relative_loss=relative_loss)

            if self.config.verbose >= 2:
                _logger.debug(
                    f"L-BFGS adaptive: mode={lr_mode}, step={initial_step:.2f}, "
                    f"rel_loss={relative_loss:.4e}"
                )
        else:
            initial_step = self.config.lbfgs_initial_step_size
            lr_mode = "fixed"
            telemetry.record_layer2_lr_mode(mode=lr_mode, relative_loss=relative_loss)

        return initial_step, lr_mode

    def _create_lbfgs_optimizer(
        self, params: jnp.ndarray, initial_step_size: float
    ) -> tuple[optax.GradientTransformationExtraArgs, optax.OptState]:
        """Create L-BFGS optimizer with optax.

        Parameters
        ----------
        params : array_like
            Initial parameters in normalized space.
        initial_step_size : float
            Initial step size for L-BFGS line search.

        Returns
        -------
        optimizer : optax.GradientTransformationExtraArgs
            L-BFGS optimizer instance.
        opt_state : optax.OptState
            Initial optimizer state.
        """
        line_search_type = self.config.lbfgs_line_search
        if line_search_type == "backtracking":
            linesearch = optax.scale_by_backtracking_linesearch(
                max_backtracking_steps=20,
                slope_rtol=1e-4,
                decrease_factor=0.8,
                increase_factor=1.5,
                max_learning_rate=initial_step_size,
            )
        else:
            linesearch = optax.scale_by_zoom_linesearch(
                max_linesearch_steps=20,
                initial_guess_strategy="one",
            )

        optimizer = optax.lbfgs(
            learning_rate=initial_step_size,
            memory_size=self.config.lbfgs_history_size,
            scale_init_precond=True,
            linesearch=linesearch,
        )

        if self.config.gradient_clip_value is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.gradient_clip_value),
                optimizer,
            )

        opt_state = optimizer.init(params)
        return optimizer, opt_state

    def _lbfgs_step(
        self,
        params: jnp.ndarray,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformationExtraArgs,
        loss_fn: Callable,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
        iteration: int,
        best_tracker: dict[str, Any],
    ) -> tuple[jnp.ndarray, float, float, optax.OptState, bool]:
        """Perform single L-BFGS optimization step.

        Parameters
        ----------
        params : array_like
            Current parameters in normalized space.
        opt_state : optax.OptState
            Current optimizer state.
        optimizer : optax.GradientTransformationExtraArgs
            L-BFGS optimizer instance.
        loss_fn : callable
            Loss function.
        x_batch : array_like
            Independent variable batch.
        y_batch : array_like
            Dependent variable batch.
        iteration : int
            Current iteration number.
        best_tracker : dict
            Dictionary tracking best_params_global and best_cost_global.

        Returns
        -------
        new_params : array_like
            Updated parameters.
        loss : float
            Loss value before update.
        grad_norm : float
            L2 norm of gradient.
        new_opt_state : optax.OptState
            Updated optimizer state.
        line_search_failed : bool
            True if line search failed.
        """
        # Validate input parameters
        if not self._validate_numerics(params, context="at L-BFGS step input"):
            if self.config.enable_fault_tolerance:
                return params, float("inf"), float("inf"), opt_state, True
            raise ValueError("Numerical issues detected in L-BFGS step input")

        # Compute loss and gradient
        loss_value, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)

        # Validate loss and gradients
        if not self._validate_numerics(
            params, loss=float(loss_value), gradients=grads, context="in L-BFGS step"
        ):
            if self.config.enable_fault_tolerance:
                return params, float("inf"), float("inf"), opt_state, True
            raise ValueError("Numerical issues detected in L-BFGS step")

        grad_norm = jnp.linalg.norm(grads)

        def value_fn(p):
            return loss_fn(p, x_batch, y_batch)

        try:
            updates, new_opt_state = optimizer.update(
                grads,
                opt_state,
                params,
                value=loss_value,
                grad=grads,
                value_fn=value_fn,
            )
            line_search_failed = False
        except Exception as e:
            if self.config.verbose >= 2:
                _logger.warning(f"L-BFGS line search failed: {e}")
            updates = -self.config.lbfgs_initial_step_size * grads
            new_opt_state = opt_state
            line_search_failed = True

            telemetry = get_defense_telemetry()
            telemetry.record_lbfgs_line_search_failure(iteration, str(e))

        # Layer 4: Trust Region Constraint
        if self.config.enable_step_clipping:
            original_update_norm = float(jnp.linalg.norm(updates))
            max_norm = self.config.max_warmup_step_size
            updates = self._clip_update_norm(updates, max_norm)

            if original_update_norm > max_norm:
                self._clip_count += 1
                telemetry = get_defense_telemetry()
                telemetry.record_layer4_clip(
                    original_norm=original_update_norm, max_norm=max_norm
                )

        new_params = optax.apply_updates(params, updates)

        # Validate updated parameters
        if not self._validate_numerics(new_params, context="after L-BFGS update"):
            if self.config.enable_fault_tolerance:
                return params, float(loss_value), float(grad_norm), opt_state, True
            raise ValueError("NaN/Inf in parameters after L-BFGS update")

        # Track best parameters globally
        if float(loss_value) < best_tracker.get("best_cost_global", float("inf")):
            best_tracker["best_cost_global"] = float(loss_value)
            best_tracker["best_params_global"] = new_params

        # Record history buffer fill event
        if iteration == self.config.lbfgs_history_size:
            telemetry = get_defense_telemetry()
            telemetry.record_lbfgs_history_fill(iteration)

        return (
            new_params,
            float(loss_value),
            float(grad_norm),
            new_opt_state,
            line_search_failed,
        )

    @staticmethod
    def _clip_update_norm(updates: jnp.ndarray, max_norm: float) -> jnp.ndarray:
        """Clip parameter update vector to maximum L2 norm (JIT-compatible).

        Parameters
        ----------
        updates : array_like
            Parameter updates from optimizer.
        max_norm : float
            Maximum allowed L2 norm.

        Returns
        -------
        clipped_updates : array_like
            Updates with L2 norm <= max_norm.
        """
        update_norm = jnp.linalg.norm(updates)
        scale = jnp.minimum(1.0, max_norm / (update_norm + 1e-10))
        return updates * scale

    def _validate_numerics(
        self,
        params: jnp.ndarray,
        loss: float | None = None,
        gradients: jnp.ndarray | None = None,
        context: str = "",
    ) -> bool:
        """Validate numerical stability.

        Parameters
        ----------
        params : array_like
            Parameters to validate.
        loss : float, optional
            Loss value to validate.
        gradients : array_like, optional
            Gradient values to validate.
        context : str, optional
            Context string for error messages.

        Returns
        -------
        is_valid : bool
            True if all values are finite.
        """
        if not getattr(self.config, "validate_numerics", False):
            return True

        if not jnp.all(jnp.isfinite(params)):
            return False

        if loss is not None and not jnp.isfinite(loss):
            return False

        return gradients is None or bool(jnp.all(jnp.isfinite(gradients)))

    def _check_switch_criteria(
        self,
        iteration: int,
        current_loss: float,
        prev_loss: float,
        grad_norm: float,
    ) -> tuple[bool, str]:
        """Check if Phase 1 should switch to Phase 2.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        current_loss : float
            Current loss value.
        prev_loss : float
            Previous loss value.
        grad_norm : float
            Current gradient norm.

        Returns
        -------
        should_switch : bool
            Whether to switch to Phase 2.
        reason : str
            Reason for switching.
        """
        active_criteria = self.config.active_switching_criteria or []

        if "max_iter" in active_criteria:
            if iteration >= self.config.max_warmup_iterations:
                return True, "Maximum warmup iterations reached"

        if "gradient" in active_criteria:
            if grad_norm < self.config.gradient_norm_threshold:
                return True, "Gradient norm below threshold"

        if "plateau" in active_criteria:
            eps = jnp.finfo(jnp.float64).eps
            relative_change = jnp.abs(current_loss - prev_loss) / (
                jnp.abs(prev_loss) + eps
            )
            if relative_change < self.config.loss_plateau_threshold:
                return True, "Loss plateau detected"

        return False, ""

    def _check_cost_guard(
        self,
        iteration: int,
        loss_value: float,
        best_params: jnp.ndarray,
        best_loss: float,
        lr_mode: str,
        relative_loss: float,
        phase_history: list[dict[str, Any]],
        telemetry: Any,
        cost_history: list[float],
    ) -> dict[str, Any] | None:
        """Check cost-increase guard (Layer 3).

        Parameters
        ----------
        iteration : int
            Current iteration.
        loss_value : float
            Current loss.
        best_params : array_like
            Best parameters found.
        best_loss : float
            Best loss found.
        lr_mode : str
            Learning rate mode.
        relative_loss : float
            Initial relative loss.
        phase_history : list
            Phase history list.
        telemetry : DefenseLayerTelemetry
            Telemetry collector.
        cost_history : list
            Cost history.

        Returns
        -------
        result : dict or None
            Result dict if guard triggered, None otherwise.
        """
        assert self._initial_loss is not None
        cost_increase_ratio = loss_value / self._initial_loss
        cost_threshold = 1.0 + self.config.cost_increase_tolerance

        if cost_increase_ratio > cost_threshold:
            telemetry.record_layer3_trigger(
                cost_ratio=cost_increase_ratio,
                tolerance=self.config.cost_increase_tolerance,
                iteration=iteration,
            )

            if self.config.verbose >= 1:
                _logger.warning(
                    f"Phase 1: Cost increase guard triggered at iteration "
                    f"{iteration + 1}. Loss {loss_value:.6e} > "
                    f"{self._initial_loss:.6e} * {cost_threshold:.2f}. "
                    f"Reverting to best params (loss={best_loss:.6e})."
                )

            phase_record = {
                "phase": 1,
                "name": "lbfgs_warmup",
                "iterations": iteration + 1,
                "final_loss": loss_value,
                "best_loss": best_loss,
                "switch_reason": (
                    f"Cost increase guard triggered (ratio={cost_increase_ratio:.4f})"
                ),
                "timestamp": time.time(),
                "cost_guard_triggered": True,
                "lr_mode": lr_mode,
                "relative_loss": relative_loss,
            }
            phase_history.append(phase_record)

            warmup_result = WarmupResult(
                params=best_params,
                cost=best_loss,
                iterations=iteration + 1,
                converged=False,
                cost_history=cost_history,
            )

            return {
                "final_params": best_params,
                "best_params": best_params,
                "best_loss": best_loss,
                "final_loss": loss_value,
                "iterations": iteration + 1,
                "switch_reason": "Cost increase guard triggered",
                "cost_guard_triggered": True,
                "cost_increase_ratio": cost_increase_ratio,
                "lr_mode": lr_mode,
                "relative_loss": relative_loss,
                "warmup_result": warmup_result,
            }

        return None


__all__ = ["WarmupPhase", "WarmupResult"]
