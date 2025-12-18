"""Adaptive Hybrid Streaming Optimizer with Parameter Normalization.

This module implements a four-phase hybrid optimizer that solves three fundamental
issues in streaming optimization:

1. Weak gradient signals from parameter scale imbalance (via normalization)
2. Slow convergence near optimum (via Gauss-Newton)
3. Crude covariance estimation (via exact J^T J accumulation)

The optimizer operates in four phases:
- **Phase 0**: Parameter normalization setup
- **Phase 1**: Adam warmup with adaptive switching
- **Phase 2**: Streaming Gauss-Newton with exact J^T J accumulation
- **Phase 3**: Denormalization and covariance transform

This implementation focuses on Phase 0 setup logic and phase tracking infrastructure.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Callable, Tuple

import h5py
import jax
import jax.numpy as jnp
import optax

from .hybrid_streaming_config import HybridStreamingConfig
from .parameter_normalizer import NormalizedModelWrapper, ParameterNormalizer
from .stability import NumericalStabilityGuard

__all__ = ["AdaptiveHybridStreamingOptimizer"]


class AdaptiveHybridStreamingOptimizer:
    """Adaptive hybrid streaming optimizer with four-phase optimization.

    This optimizer combines parameter normalization, Adam warmup, streaming
    Gauss-Newton, and exact covariance computation to provide:

    - Fast convergence for parameters with different scales
    - Accurate uncertainty estimates on large datasets
    - Memory-efficient streaming for unlimited dataset sizes
    - Production-ready fault tolerance

    The optimization proceeds through four phases:

    - **Phase 0**: Setup parameter normalization and bounds transformation
    - **Phase 1**: Adam warmup with adaptive switching to Phase 2
    - **Phase 2**: Streaming Gauss-Newton with exact J^T J accumulation
    - **Phase 3**: Denormalize parameters and transform covariance matrix

    Parameters
    ----------
    config : HybridStreamingConfig, optional
        Configuration for all phases of optimization. If None, uses default
        configuration. See HybridStreamingConfig for details.

    Attributes
    ----------
    config : HybridStreamingConfig
        Configuration object controlling all phases
    current_phase : int
        Current optimization phase (0, 1, 2, or 3)
    phase_history : list
        History of phase transitions with timing information
    phase_start_time : float or None
        Start time of current phase (seconds since epoch)
    normalized_params : jax.Array or None
        Current parameters in normalized space
    normalizer : ParameterNormalizer or None
        Parameter normalizer instance (created in Phase 0)
    normalized_model : NormalizedModelWrapper or None
        Wrapped model function operating in normalized space
    normalized_bounds : tuple of jax.Array or None
        Bounds transformed to normalized space
    normalization_jacobian : jax.Array or None
        Denormalization Jacobian for covariance transform

    Examples
    --------
    Basic usage with default configuration:

    >>> from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig
    >>> import jax.numpy as jnp
    >>> config = HybridStreamingConfig()
    >>> optimizer = AdaptiveHybridStreamingOptimizer(config)

    With bounds-based normalization:

    >>> config = HybridStreamingConfig(
    ...     normalize=True,
    ...     normalization_strategy='bounds'
    ... )
    >>> optimizer = AdaptiveHybridStreamingOptimizer(config)

    With custom warmup settings:

    >>> config = HybridStreamingConfig(
    ...     warmup_iterations=300,
    ...     warmup_learning_rate=0.01,
    ...     gauss_newton_tol=1e-10
    ... )
    >>> optimizer = AdaptiveHybridStreamingOptimizer(config)

    See Also
    --------
    HybridStreamingConfig : Configuration for all phases
    ParameterNormalizer : Parameter normalization implementation
    curve_fit : High-level interface with method='hybrid_streaming'

    Notes
    -----
    Based on Adaptive Hybrid Streaming Optimizer specification:
    ``agent-os/specs/2025-12-18-adaptive-hybrid-streaming-optimizer/spec.md``
    """

    def __init__(self, config: HybridStreamingConfig | None = None):
        """Initialize adaptive hybrid streaming optimizer.

        Parameters
        ----------
        config : HybridStreamingConfig, optional
            Configuration for all phases. If None, uses default configuration.
        """
        # Store configuration
        self.config = config if config is not None else HybridStreamingConfig()

        # Phase tracking infrastructure
        self.current_phase: int = 0
        self.phase_history: list[dict[str, Any]] = []
        self.phase_start_time: float | None = None
        self.normalized_params: jnp.ndarray | None = None

        # Phase 0: Normalization components (created during setup)
        self.normalizer: ParameterNormalizer | None = None
        self.normalized_model: NormalizedModelWrapper | None = None
        self.normalized_bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None
        self.normalization_jacobian: jnp.ndarray | None = None

        # Store original model and parameters (for reference)
        self.original_model: callable | None = None
        self.original_p0: jnp.ndarray | None = None
        self.original_bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None

        # Fault tolerance components
        self.stability_guard = NumericalStabilityGuard()
        self.best_params_global: jnp.ndarray | None = None
        self.best_cost_global: float = jnp.inf
        self.checkpoint_counter: int = 0
        self.retry_count: int = 0

        # Phase-specific state for checkpointing
        self.phase1_optimizer_state: optax.OptState | None = None
        self.phase2_JTJ_accumulator: jnp.ndarray | None = None
        self.phase2_JTr_accumulator: jnp.ndarray | None = None

        # Multi-device support
        self.device_info: dict[str, Any] | None = None
        self.multi_device_config: dict[str, Any] | None = None

        # Mixed precision support
        self.current_precision: jnp.dtype = jnp.float64  # Default to float64
        self.phase_precisions: dict[int, jnp.dtype] = {}  # Precision per phase
        self.precision_upgrade_triggered: bool = False  # Track if upgrade occurred

    def _setup_normalization(
        self,
        model: callable,
        p0: jnp.ndarray,
        bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> None:
        """Setup parameter normalization (Phase 0).

        This method determines the normalization strategy based on config and
        inputs, creates the ParameterNormalizer instance, wraps the model
        function, transforms bounds to normalized space, and stores the
        normalization Jacobian for Phase 3 covariance transform.

        Parameters
        ----------
        model : callable
            User model function with signature: ``model(x, *params) -> predictions``
        p0 : array_like
            Initial parameter guess of shape (n_params,)
        bounds : tuple of array_like, optional
            Parameter bounds as (lb, ub) where lb and ub are arrays of shape
            (n_params,). If None, no bounds are applied.

        Notes
        -----
        This method sets up the following attributes:
        - self.normalizer: ParameterNormalizer instance
        - self.normalized_model: NormalizedModelWrapper instance
        - self.normalized_bounds: Transformed bounds in normalized space
        - self.normalization_jacobian: Jacobian for covariance transform
        - self.normalized_params: Initial parameters in normalized space
        - self.original_model, self.original_p0, self.original_bounds: References

        The normalization strategy is determined as follows:
        - If config.normalize is False: Use 'none' strategy (identity)
        - Otherwise: Use config.normalization_strategy ('auto', 'bounds', 'p0', 'none')
        - 'auto' selects 'bounds' if bounds provided, else 'p0'
        """
        # Store original inputs
        self.original_model = model
        self.original_p0 = jnp.asarray(p0, dtype=jnp.float64)
        self.original_bounds = bounds

        # Determine normalization strategy
        if not self.config.normalize:
            # Normalization disabled: use identity transform
            strategy = 'none'
        else:
            # Use configured strategy
            strategy = self.config.normalization_strategy

        # Create ParameterNormalizer
        self.normalizer = ParameterNormalizer(
            p0=self.original_p0,
            bounds=bounds,
            strategy=strategy
        )

        # Create NormalizedModelWrapper
        self.normalized_model = NormalizedModelWrapper(
            model_fn=model,
            normalizer=self.normalizer
        )

        # Transform bounds to normalized space
        if bounds is not None:
            lb_normalized, ub_normalized = self.normalizer.transform_bounds()
            self.normalized_bounds = (lb_normalized, ub_normalized)
        else:
            self.normalized_bounds = None

        # Store normalization Jacobian for Phase 3 covariance transform
        self.normalization_jacobian = self.normalizer.normalization_jacobian

        # Initialize normalized parameters
        self.normalized_params = self.normalizer.normalize(self.original_p0)

        # Record Phase 0 completion in history
        phase_record = {
            'phase': 0,
            'name': 'normalization_setup',
            'strategy': self.normalizer.strategy,
            'timestamp': time.time(),
            'normalized_params_shape': self.normalized_params.shape,
            'has_bounds': bounds is not None,
        }
        self.phase_history.append(phase_record)

    def _create_adam_optimizer(
        self,
        params: jnp.ndarray,
    ) -> Tuple[optax.GradientTransformation, optax.OptState]:
        """Create Adam optimizer with optax, optionally with learning rate schedule and gradient clipping.

        Parameters
        ----------
        params : array_like
            Initial parameters in normalized space

        Returns
        -------
        optimizer : optax.GradientTransformation
            Adam optimizer instance (potentially chained with gradient clipping)
        opt_state : optax.OptState
            Initial optimizer state

        Notes
        -----
        Uses learning rate from config.warmup_learning_rate.
        If config.use_learning_rate_schedule is True, uses optax.warmup_cosine_decay_schedule.
        If config.gradient_clip_value is not None, chains optax.clip_by_global_norm with Adam.

        Examples
        --------
        Basic Adam (no schedule, no clipping):
        >>> config = HybridStreamingConfig(warmup_learning_rate=0.001)
        >>> optimizer = AdaptiveHybridStreamingOptimizer(config)
        >>> opt, state = optimizer._create_adam_optimizer(params)

        Adam with learning rate schedule:
        >>> config = HybridStreamingConfig(
        ...     use_learning_rate_schedule=True,
        ...     warmup_learning_rate=0.001,
        ...     lr_schedule_warmup_steps=50,
        ...     lr_schedule_decay_steps=450
        ... )
        >>> optimizer = AdaptiveHybridStreamingOptimizer(config)
        >>> opt, state = optimizer._create_adam_optimizer(params)

        Adam with gradient clipping:
        >>> config = HybridStreamingConfig(gradient_clip_value=1.0)
        >>> optimizer = AdaptiveHybridStreamingOptimizer(config)
        >>> opt, state = optimizer._create_adam_optimizer(params)
        """
        # Determine learning rate (fixed or schedule)
        if self.config.use_learning_rate_schedule:
            # Use warmup + cosine decay schedule
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.config.warmup_learning_rate,
                warmup_steps=self.config.lr_schedule_warmup_steps,
                decay_steps=self.config.lr_schedule_decay_steps,
                end_value=self.config.lr_schedule_end_value,
            )
        else:
            # Use fixed learning rate
            learning_rate = self.config.warmup_learning_rate

        # Build optimizer chain
        optimizer_components = []

        # Add gradient clipping if configured
        if self.config.gradient_clip_value is not None:
            optimizer_components.append(
                optax.clip_by_global_norm(self.config.gradient_clip_value)
            )

        # Add Adam optimizer
        optimizer_components.append(optax.adam(learning_rate=learning_rate))

        # Chain components if multiple, else use single Adam
        if len(optimizer_components) > 1:
            optimizer = optax.chain(*optimizer_components)
        else:
            optimizer = optimizer_components[0]

        # Initialize optimizer state
        opt_state = optimizer.init(params)

        return optimizer, opt_state

    def _create_warmup_loss_fn(self) -> Callable:
        """Create loss function for warmup phase.

        Returns
        -------
        loss_fn : callable
            Loss function with signature: loss_fn(params, x_batch, y_batch) -> scalar_loss
            Operates in normalized parameter space and returns mean squared residuals.

        Notes
        -----
        The loss function is JIT-compiled for performance.
        """
        # Use normalized model wrapper
        normalized_model = self.normalized_model

        @jax.jit
        def loss_fn(params: jnp.ndarray, x_batch: jnp.ndarray, y_batch: jnp.ndarray) -> float:
            """Compute mean squared residuals in normalized space.

            Parameters
            ----------
            params : array_like
                Parameters in normalized space
            x_batch : array_like
                Independent variable batch
            y_batch : array_like
                Dependent variable batch

            Returns
            -------
            loss : float
                Mean squared residuals
            """
            # Predict using normalized model (automatically denormalizes params)
            predictions = normalized_model(x_batch, *params)

            # Compute residuals
            residuals = y_batch - predictions

            # Return mean squared residuals
            return jnp.mean(residuals ** 2)

        return loss_fn

    def _adam_step(
        self,
        params: jnp.ndarray,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, float, float, optax.OptState]:
        """Perform single Adam optimization step with numerical validation.

        Parameters
        ----------
        params : array_like
            Current parameters in normalized space
        opt_state : optax.OptState
            Current optimizer state
        optimizer : optax.GradientTransformation
            Adam optimizer instance
        loss_fn : callable
            Loss function
        x_batch : array_like
            Independent variable batch
        y_batch : array_like
            Dependent variable batch

        Returns
        -------
        new_params : array_like
            Updated parameters in normalized space
        loss : float
            Loss value before update
        grad_norm : float
            L2 norm of gradient
        new_opt_state : optax.OptState
            Updated optimizer state

        Notes
        -----
        Uses jax.value_and_grad for efficient loss and gradient computation.
        Includes NaN/Inf validation if enabled in config.
        """
        # Validate input parameters
        self._validate_numerics(params, context="at Adam step input")

        # Compute loss and gradient
        loss_value, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)

        # Validate loss and gradients
        if not self._validate_numerics(params, loss=float(loss_value), gradients=grads, context="in Adam step"):
            # Handle numerical issues
            if hasattr(self.config, 'enable_fault_tolerance') and self.config.enable_fault_tolerance:
                # Return current params unchanged (fallback)
                return params, float('inf'), float('inf'), opt_state
            else:
                raise ValueError("Numerical issues detected in Adam step")

        # Compute gradient norm
        grad_norm = jnp.linalg.norm(grads)

        # Apply optimizer updates
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        # Validate updated parameters
        if not self._validate_numerics(new_params, context="after Adam update"):
            # Fallback: keep old parameters
            if hasattr(self.config, 'enable_fault_tolerance') and self.config.enable_fault_tolerance:
                return params, float(loss_value), float(grad_norm), opt_state
            else:
                raise ValueError("NaN/Inf in parameters after Adam update")

        # Track best parameters globally
        if float(loss_value) < self.best_cost_global:
            self.best_cost_global = float(loss_value)
            self.best_params_global = new_params

        # Store optimizer state for checkpointing
        self.phase1_optimizer_state = new_opt_state

        return new_params, float(loss_value), float(grad_norm), new_opt_state

    def _check_phase1_switch_criteria(
        self,
        iteration: int,
        current_loss: float,
        prev_loss: float,
        grad_norm: float,
    ) -> Tuple[bool, str]:
        """Check if Phase 1 should switch to Phase 2.

        Parameters
        ----------
        iteration : int
            Current iteration number
        current_loss : float
            Current loss value
        prev_loss : float
            Previous loss value
        grad_norm : float
            Current gradient norm

        Returns
        -------
        should_switch : bool
            Whether to switch to Phase 2
        reason : str
            Reason for switching (or empty if not switching)

        Notes
        -----
        Checks criteria specified in config.active_switching_criteria:
        - 'plateau': Loss plateau detection
        - 'gradient': Gradient norm below threshold
        - 'max_iter': Maximum iterations reached
        """
        active_criteria = self.config.active_switching_criteria

        # Check max iterations criterion
        if 'max_iter' in active_criteria:
            if iteration >= self.config.max_warmup_iterations:
                return True, "Maximum warmup iterations reached"

        # Check gradient norm criterion
        if 'gradient' in active_criteria:
            if grad_norm < self.config.gradient_norm_threshold:
                return True, "Gradient norm below threshold"

        # Check loss plateau criterion
        if 'plateau' in active_criteria:
            # Compute relative loss change
            eps = jnp.finfo(jnp.float64).eps
            relative_change = jnp.abs(current_loss - prev_loss) / (jnp.abs(prev_loss) + eps)

            if relative_change < self.config.loss_plateau_threshold:
                return True, "Loss plateau detected"

        # No switch criteria met
        return False, ""

    def _run_phase1_warmup(
        self,
        data_source: Any,
        model: Callable,
        p0: jnp.ndarray,
        bounds: Tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run Phase 1 Adam warmup.

        Parameters
        ----------
        data_source : various types
            Data source (tuple of arrays for now)
        model : callable
            User model function
        p0 : array_like
            Initial parameter guess
        bounds : tuple of array_like, optional
            Parameter bounds

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

        Notes
        -----
        Operates entirely in normalized parameter space.
        Tracks best parameters throughout warmup.
        """
        # Setup normalization if not already done
        if self.normalizer is None:
            self._setup_normalization(model, p0, bounds)

        # Extract data from source (simple tuple for now)
        if isinstance(data_source, tuple) and len(data_source) == 2:
            x_data, y_data = data_source
            x_data = jnp.asarray(x_data, dtype=jnp.float64)
            y_data = jnp.asarray(y_data, dtype=jnp.float64)
        else:
            raise NotImplementedError(
                "Only tuple data sources (x_data, y_data) supported in Phase 1 warmup"
            )

        # Initialize parameters in normalized space
        current_params = self.normalized_params

        # Create Adam optimizer
        optimizer, opt_state = self._create_adam_optimizer(current_params)

        # Create loss function
        loss_fn = self._create_warmup_loss_fn()

        # Best parameter tracking
        best_params = current_params
        best_loss = float('inf')

        # Initialize previous loss
        prev_loss = float('inf')

        # Warmup loop
        for iteration in range(self.config.max_warmup_iterations):
            # Perform Adam step (using full data for now)
            current_params, loss_value, grad_norm, opt_state = self._adam_step(
                params=current_params,
                opt_state=opt_state,
                optimizer=optimizer,
                loss_fn=loss_fn,
                x_batch=x_data,
                y_batch=y_data,
            )

            # Track best parameters
            if loss_value < best_loss:
                best_loss = loss_value
                best_params = current_params

            # Check for precision upgrade if NaN/Inf detected
            self._upgrade_precision_if_needed(
                params=current_params,
                loss=loss_value,
                gradients=None,  # gradient not directly available
            )

            # Save checkpoint periodically if enabled
            if (hasattr(self.config, 'enable_checkpoints') and self.config.enable_checkpoints and
                hasattr(self.config, 'checkpoint_frequency') and
                (iteration + 1) % self.config.checkpoint_frequency == 0):

                if hasattr(self.config, 'checkpoint_dir') and self.config.checkpoint_dir:
                    checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_phase1_iter{iteration+1}.h5"
                    self.current_phase = 1
                    self.normalized_params = current_params
                    self._save_checkpoint(checkpoint_path)

            # Check switch criteria after minimum warmup iterations
            if iteration >= self.config.warmup_iterations:
                should_switch, reason = self._check_phase1_switch_criteria(
                    iteration=iteration,
                    current_loss=loss_value,
                    prev_loss=prev_loss,
                    grad_norm=grad_norm,
                )

                if should_switch:
                    # Record Phase 1 completion
                    phase_record = {
                        'phase': 1,
                        'name': 'adam_warmup',
                        'iterations': iteration + 1,
                        'final_loss': loss_value,
                        'best_loss': best_loss,
                        'switch_reason': reason,
                        'timestamp': time.time(),
                    }
                    self.phase_history.append(phase_record)

                    return {
                        'final_params': current_params,
                        'best_params': best_params,
                        'best_loss': best_loss,
                        'final_loss': loss_value,
                        'iterations': iteration + 1,
                        'switch_reason': reason,
                    }

            # Update previous loss
            prev_loss = loss_value

        # Maximum iterations reached (this shouldn't happen if max_iter criterion active)
        phase_record = {
            'phase': 1,
            'name': 'adam_warmup',
            'iterations': self.config.max_warmup_iterations,
            'final_loss': loss_value,
            'best_loss': best_loss,
            'switch_reason': 'Maximum iterations reached',
            'timestamp': time.time(),
        }
        self.phase_history.append(phase_record)

        return {
            'final_params': current_params,
            'best_params': best_params,
            'best_loss': best_loss,
            'final_loss': loss_value,
            'iterations': self.config.max_warmup_iterations,
            'switch_reason': 'Maximum iterations reached',
        }

    def _compute_jacobian_chunk(
        self,
        x_chunk: jnp.ndarray,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute exact Jacobian for a data chunk using vmap+grad.

        This method computes the Jacobian matrix J where J[i, j] = ∂f_i/∂p_j
        for all data points in the chunk. Uses JAX automatic differentiation
        for exact gradients (no finite differences).

        Parameters
        ----------
        x_chunk : array_like
            Independent variable chunk of shape (n_points,) or (n_points, n_features)
        params : array_like
            Parameters in normalized space of shape (n_params,)

        Returns
        -------
        J_chunk : array_like
            Jacobian matrix of shape (n_points, n_params)

        Notes
        -----
        Uses jax.jacrev with vmap for efficient per-point gradient computation.
        The normalized model wrapper automatically handles parameter denormalization.
        """
        # Define function to compute model output for a single x value
        def model_at_x(p, x_single):
            # Call normalized model (which handles denormalization internally)
            return self.normalized_model(x_single, *p)

        # Compute Jacobian using jacrev (reverse-mode AD)
        # jacrev computes gradient w.r.t. first argument (params)
        # vmap over x_chunk (second argument) to get Jacobian for all points
        jac_fn = jax.vmap(lambda x: jax.jacrev(model_at_x, argnums=0)(params, x))
        J_chunk = jac_fn(x_chunk)

        return J_chunk

    def _accumulate_jtj_jtr(
        self,
        x_chunk: jnp.ndarray,
        y_chunk: jnp.ndarray,
        params: jnp.ndarray,
        JTJ_prev: jnp.ndarray,
        JTr_prev: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Accumulate J^T J and J^T r across chunks for memory-efficient Gauss-Newton.

        This is the key method that enables streaming optimization. Instead of storing
        the full Jacobian (n_points × n_params), we only accumulate the products:
        - J^T J: (n_params × n_params) - Gauss-Newton Hessian approximation
        - J^T r: (n_params,) - Gradient vector

        Memory: O(p^2) instead of O(np) where n >> p

        Parameters
        ----------
        x_chunk : array_like
            Independent variable chunk of shape (n_points_chunk,)
        y_chunk : array_like
            Observed data chunk of shape (n_points_chunk,)
        params : array_like
            Current parameters in normalized space of shape (n_params,)
        JTJ_prev : array_like
            Previous accumulated J^T J of shape (n_params, n_params)
        JTr_prev : array_like
            Previous accumulated J^T r of shape (n_params,)

        Returns
        -------
        JTJ_new : array_like
            Updated J^T J accumulation of shape (n_params, n_params)
        JTr_new : array_like
            Updated J^T r accumulation of shape (n_params,)
        residual_sum_sq : float
            Sum of squared residuals for this chunk

        Notes
        -----
        Accumulation formulas:
        - J^T J_new = J^T J_prev + J_chunk^T @ J_chunk
        - J^T r_new = J^T r_prev + J_chunk^T @ r_chunk
        where r_chunk = y_chunk - f(x_chunk, params)
        """
        # Compute predictions for chunk
        predictions = self.normalized_model(x_chunk, *params)

        # Compute residuals
        residuals = y_chunk - predictions

        # Compute Jacobian for chunk
        J_chunk = self._compute_jacobian_chunk(x_chunk, params)

        # Accumulate J^T J: (n_params, n_params)
        JTJ_new = JTJ_prev + J_chunk.T @ J_chunk

        # Accumulate J^T r: (n_params,)
        JTr_new = JTr_prev + J_chunk.T @ residuals

        # Compute residual sum of squares for this chunk
        residual_sum_sq = float(jnp.sum(residuals ** 2))

        # Store accumulators for checkpointing
        self.phase2_JTJ_accumulator = JTJ_new
        self.phase2_JTr_accumulator = JTr_new

        return JTJ_new, JTr_new, residual_sum_sq

    def _solve_gauss_newton_step(
        self,
        JTJ: jnp.ndarray,
        JTr: jnp.ndarray,
        trust_radius: float,
        regularization: float = 1e-10,
    ) -> Tuple[jnp.ndarray, float]:
        """Solve Gauss-Newton step using SVD following trf.py patterns.

        Solves the trust region subproblem:
            minimize: 0.5 * p^T (J^T J) p + (J^T r)^T p
            subject to: ||p|| <= trust_radius

        Uses SVD decomposition for numerical stability and handles rank-deficient
        matrices through regularization.

        Parameters
        ----------
        JTJ : array_like
            Accumulated J^T J matrix of shape (n_params, n_params)
        JTr : array_like
            Accumulated J^T r vector of shape (n_params,)
        trust_radius : float
            Trust region radius (maximum allowed step norm)
        regularization : float, default=1e-10
            Tikhonov regularization parameter for numerical stability

        Returns
        -------
        step : array_like
            Gauss-Newton step of shape (n_params,)
        predicted_reduction : float
            Predicted reduction in cost function: -g^T p - 0.5 p^T H p

        Notes
        -----
        Follows the SVD-based trust region solver pattern from trf.py.
        Uses regularization to handle rank-deficient or ill-conditioned J^T J.

        Algorithm:
        1. Compute SVD: J^T J = U S V^T
        2. Solve regularized system: (J^T J + λI)^-1 (-J^T r)
        3. Scale step if norm exceeds trust radius
        """
        n_params = JTJ.shape[0]

        # Add Tikhonov regularization for numerical stability
        JTJ_reg = JTJ + regularization * jnp.eye(n_params)

        # Compute SVD of regularized J^T J
        U, s, Vt = jnp.linalg.svd(JTJ_reg, full_matrices=False)

        # Solve for Gauss-Newton step using SVD
        # (U S V^T) p = -J^T r
        # p = V S^-1 U^T (-J^T r)

        # Compute U^T @ (-JTr)
        UTb = U.T @ (-JTr)

        # Solve diagonal system with regularization
        # Filter out small singular values
        s_threshold = jnp.max(s) * 1e-10
        s_safe = jnp.where(s > s_threshold, s, s_threshold)
        step_hat = UTb / s_safe

        # Transform back to parameter space
        step = Vt.T @ step_hat

        # Apply trust region constraint
        step_norm = jnp.linalg.norm(step)

        if step_norm > trust_radius:
            # Scale step to trust region boundary
            step = step * (trust_radius / step_norm)

        # Compute predicted reduction: -g^T p - 0.5 p^T H p
        # where g = J^T r and H = J^T J
        predicted_reduction = -jnp.dot(JTr, step) - 0.5 * jnp.dot(step, JTJ @ step)
        predicted_reduction = float(jnp.maximum(predicted_reduction, 0.0))

        return step, predicted_reduction

    def _apply_trust_region(
        self,
        step: jnp.ndarray,
        trust_radius: float,
    ) -> jnp.ndarray:
        """Apply trust region constraint by scaling step if necessary.

        If the step norm exceeds the trust radius, scale it to lie on the
        trust region boundary.

        Parameters
        ----------
        step : array_like
            Proposed parameter step of shape (n_params,)
        trust_radius : float
            Trust region radius

        Returns
        -------
        scaled_step : array_like
            Step scaled to satisfy ||step|| <= trust_radius

        Notes
        -----
        This is a simple trust region implementation. More sophisticated
        approaches (e.g., dogleg, 2D subspace) are used in trf.py but
        require the full Jacobian.
        """
        step_norm = jnp.linalg.norm(step)

        if step_norm <= trust_radius:
            # Step is within trust region
            return step
        else:
            # Scale step to trust region boundary
            return step * (trust_radius / step_norm)

    def _gauss_newton_iteration(
        self,
        data_source: tuple[jnp.ndarray, jnp.ndarray],
        current_params: jnp.ndarray,
        trust_radius: float,
    ) -> dict[str, Any]:
        """Perform one complete Gauss-Newton iteration with J^T J accumulation.

        This method:
        1. Accumulates J^T J and J^T r across all data chunks
        2. Solves the trust region subproblem for the step
        3. Evaluates the new parameters and cost
        4. Updates trust region based on actual vs predicted reduction

        Parameters
        ----------
        data_source : tuple of array_like
            Full dataset as (x_data, y_data)
        current_params : array_like
            Current parameters in normalized space of shape (n_params,)
        trust_radius : float
            Current trust region radius

        Returns
        -------
        result : dict
            Iteration result with keys:
            - 'new_params': Updated parameters
            - 'new_cost': New cost value
            - 'step': Parameter step taken
            - 'actual_reduction': Actual cost reduction
            - 'predicted_reduction': Predicted cost reduction
            - 'trust_radius': Updated trust region radius
            - 'gradient_norm': Gradient norm ||J^T r||

        Notes
        -----
        Uses chunk-based accumulation for memory efficiency.
        """
        x_data, y_data = data_source
        n_params = len(current_params)
        chunk_size = self.config.chunk_size

        # Initialize accumulators
        JTJ = jnp.zeros((n_params, n_params))
        JTr = jnp.zeros(n_params)
        total_cost = 0.0

        # Accumulate J^T J and J^T r across chunks
        n_points = len(x_data)
        for i in range(0, n_points, chunk_size):
            x_chunk = x_data[i:i+chunk_size]
            y_chunk = y_data[i:i+chunk_size]

            JTJ, JTr, res_sq = self._accumulate_jtj_jtr(
                x_chunk, y_chunk, current_params, JTJ, JTr
            )
            total_cost += res_sq

        # Compute gradient norm for convergence check
        gradient_norm = float(jnp.linalg.norm(JTr))

        # Solve for Gauss-Newton step
        step, predicted_reduction = self._solve_gauss_newton_step(
            JTJ, JTr, trust_radius
        )

        # Apply step to get new parameters
        new_params = current_params + step

        # Evaluate cost at new parameters
        new_cost = 0.0
        for i in range(0, n_points, chunk_size):
            x_chunk = x_data[i:i+chunk_size]
            y_chunk = y_data[i:i+chunk_size]

            predictions = self.normalized_model(x_chunk, *new_params)
            residuals = y_chunk - predictions
            new_cost += float(jnp.sum(residuals ** 2))

        # Compute actual reduction
        actual_reduction = total_cost - new_cost

        # Update trust region based on reduction ratio
        if predicted_reduction > 0:
            reduction_ratio = actual_reduction / predicted_reduction
        else:
            reduction_ratio = 0.0

        # Trust region update logic (from trf.py patterns)
        if reduction_ratio < 0.25:
            # Poor agreement: shrink trust region
            new_trust_radius = trust_radius * 0.25
        elif reduction_ratio > 0.75 and jnp.linalg.norm(step) >= 0.9 * trust_radius:
            # Good agreement and step at boundary: expand trust region
            new_trust_radius = min(trust_radius * 2.0, 1000.0)
        else:
            # Acceptable agreement: keep trust region
            new_trust_radius = trust_radius

        return {
            'new_params': new_params,
            'new_cost': new_cost,
            'step': step,
            'actual_reduction': actual_reduction,
            'predicted_reduction': predicted_reduction,
            'trust_radius': new_trust_radius,
            'gradient_norm': gradient_norm,
        }

    def _run_phase2_gauss_newton(
        self,
        data_source: tuple[jnp.ndarray, jnp.ndarray],
        initial_params: jnp.ndarray,
    ) -> dict[str, Any]:
        """Run Phase 2 streaming Gauss-Newton optimization.

        This is the main Phase 2 loop that iterates Gauss-Newton steps until
        convergence or maximum iterations.

        Parameters
        ----------
        data_source : tuple of array_like
            Full dataset as (x_data, y_data)
        initial_params : array_like
            Starting parameters in normalized space (from Phase 1)

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

        Notes
        -----
        Convergence criteria:
        - Gradient norm < gauss_newton_tol
        - Cost change < gauss_newton_tol
        - Maximum iterations reached
        """
        current_params = initial_params
        trust_radius = self.config.trust_region_initial

        # Track best parameters
        best_params = current_params
        best_cost = jnp.inf
        prev_cost = jnp.inf

        # Store final J^T J and residual sum for Phase 3
        # Initialize with JTJ at initial params (in case no steps accepted)
        x_data, y_data = data_source
        n_params = len(current_params)
        chunk_size = self.config.chunk_size
        n_points = len(x_data)

        final_JTJ = jnp.zeros((n_params, n_params))
        final_JTr = jnp.zeros(n_params)
        final_residual_sum_sq = 0.0

        # Compute initial JTJ
        for i in range(0, n_points, chunk_size):
            x_chunk = x_data[i:i+chunk_size]
            y_chunk = y_data[i:i+chunk_size]

            final_JTJ, final_JTr, res_sq = self._accumulate_jtj_jtr(
                x_chunk, y_chunk, current_params, final_JTJ, final_JTr
            )
            final_residual_sum_sq += res_sq

        # Gauss-Newton loop
        for iteration in range(self.config.gauss_newton_max_iterations):
            # Perform one Gauss-Newton iteration with retry logic
            max_retries = getattr(self.config, 'max_retries_per_batch', 0)
            retry_attempt = 0

            while retry_attempt <= max_retries:
                try:
                    iter_result = self._gauss_newton_iteration(
                        data_source, current_params, trust_radius
                    )

                    # Validate results
                    new_params = iter_result['new_params']
                    new_cost = iter_result['new_cost']

                    if not jnp.all(jnp.isfinite(new_params)) or not jnp.isfinite(new_cost):
                        if retry_attempt < max_retries:
                            # Retry with reduced trust region
                            trust_radius *= 0.5
                            retry_attempt += 1
                            continue
                        else:
                            # Max retries reached, use best known params
                            new_params = best_params
                            new_cost = best_cost
                            iter_result['gradient_norm'] = 0.0
                            iter_result['actual_reduction'] = 0.0
                            break

                    # Success
                    break

                except Exception as e:
                    if retry_attempt < max_retries:
                        # Retry with reduced trust region
                        trust_radius *= 0.5
                        retry_attempt += 1
                    else:
                        # Max retries reached, use fallback
                        if hasattr(self.config, 'enable_fault_tolerance') and self.config.enable_fault_tolerance:
                            # Use best parameters as fallback
                            new_params = best_params
                            new_cost = best_cost
                            iter_result = {
                                'new_params': new_params,
                                'new_cost': new_cost,
                                'gradient_norm': 0.0,
                                'actual_reduction': 0.0,
                                'trust_radius': trust_radius
                            }
                            break
                        else:
                            raise

            # Extract results
            new_params = iter_result['new_params']
            new_cost = iter_result['new_cost']
            gradient_norm = iter_result['gradient_norm']
            actual_reduction = iter_result['actual_reduction']
            trust_radius = iter_result['trust_radius']

            # Update best parameters
            if new_cost < best_cost:
                best_cost = new_cost
                best_params = new_params

            # Track global best
            if new_cost < self.best_cost_global:
                self.best_cost_global = new_cost
                self.best_params_global = new_params

            # Check for precision upgrade if NaN/Inf detected
            self._upgrade_precision_if_needed(
                params=new_params,
                loss=new_cost,
                gradients=iter_result.get('gradient'),
            )

            # Save checkpoint periodically if enabled
            if (hasattr(self.config, 'enable_checkpoints') and self.config.enable_checkpoints and
                hasattr(self.config, 'checkpoint_frequency') and
                (iteration + 1) % self.config.checkpoint_frequency == 0):

                if hasattr(self.config, 'checkpoint_dir') and self.config.checkpoint_dir:
                    checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_phase2_iter{iteration+1}.h5"
                    self.current_phase = 2
                    self.normalized_params = current_params
                    self._save_checkpoint(checkpoint_path)

            # Accept step if cost decreased
            if actual_reduction > 0:
                current_params = new_params
                prev_cost = new_cost

                # Recompute J^T J at new params for Phase 3
                # This ensures we have J^T J at the final parameters
                x_data, y_data = data_source
                n_params = len(current_params)
                chunk_size = self.config.chunk_size

                JTJ = jnp.zeros((n_params, n_params))
                JTr = jnp.zeros(n_params)
                residual_sum_sq = 0.0

                n_points = len(x_data)
                for i in range(0, n_points, chunk_size):
                    x_chunk = x_data[i:i+chunk_size]
                    y_chunk = y_data[i:i+chunk_size]

                    JTJ, JTr, res_sq = self._accumulate_jtj_jtr(
                        x_chunk, y_chunk, current_params, JTJ, JTr
                    )
                    residual_sum_sq += res_sq

                final_JTJ = JTJ
                final_residual_sum_sq = residual_sum_sq
            else:
                # Reject step, shrink trust region
                trust_radius *= 0.25

            # Check convergence: gradient norm
            if gradient_norm < self.config.gauss_newton_tol:
                # Record Phase 2 completion
                phase_record = {
                    'phase': 2,
                    'name': 'gauss_newton',
                    'iterations': iteration + 1,
                    'final_cost': new_cost,
                    'best_cost': best_cost,
                    'convergence_reason': 'Gradient norm below tolerance',
                    'gradient_norm': gradient_norm,
                    'timestamp': time.time(),
                }
                self.phase_history.append(phase_record)

                return {
                    'final_params': new_params,
                    'best_params': best_params,
                    'best_cost': best_cost,
                    'final_cost': new_cost,
                    'iterations': iteration + 1,
                    'convergence_reason': 'Gradient norm below tolerance',
                    'gradient_norm': gradient_norm,
                    'JTJ_final': final_JTJ,
                    'residual_sum_sq': final_residual_sum_sq,
                }

            # Check convergence: cost change
            cost_change = abs(prev_cost - new_cost)
            relative_change = cost_change / (abs(prev_cost) + 1e-10)

            if relative_change < self.config.gauss_newton_tol:
                phase_record = {
                    'phase': 2,
                    'name': 'gauss_newton',
                    'iterations': iteration + 1,
                    'final_cost': new_cost,
                    'best_cost': best_cost,
                    'convergence_reason': 'Cost change below tolerance',
                    'gradient_norm': gradient_norm,
                    'timestamp': time.time(),
                }
                self.phase_history.append(phase_record)

                return {
                    'final_params': new_params,
                    'best_params': best_params,
                    'best_cost': best_cost,
                    'final_cost': new_cost,
                    'iterations': iteration + 1,
                    'convergence_reason': 'Cost change below tolerance',
                    'gradient_norm': gradient_norm,
                    'JTJ_final': final_JTJ,
                    'residual_sum_sq': final_residual_sum_sq,
                }

        # Maximum iterations reached
        phase_record = {
            'phase': 2,
            'name': 'gauss_newton',
            'iterations': self.config.gauss_newton_max_iterations,
            'final_cost': prev_cost,
            'best_cost': best_cost,
            'convergence_reason': 'Maximum iterations reached',
            'gradient_norm': gradient_norm,
            'timestamp': time.time(),
        }
        self.phase_history.append(phase_record)

        return {
            'final_params': best_params,  # Use best, not current
            'best_params': best_params,
            'best_cost': best_cost,
            'final_cost': prev_cost,
            'iterations': self.config.gauss_newton_max_iterations,
            'convergence_reason': 'Maximum iterations reached',
            'gradient_norm': gradient_norm,
            'JTJ_final': final_JTJ,
            'residual_sum_sq': final_residual_sum_sq,
        }

    def _denormalize_params(self, normalized_params: jnp.ndarray) -> jnp.ndarray:
        """Denormalize parameters back to original space.

        This method uses the stored normalizer to transform parameters from
        normalized space (used during optimization) back to the original
        parameter space for final results.

        Parameters
        ----------
        normalized_params : array_like
            Parameters in normalized space of shape (n_params,)

        Returns
        -------
        params_original : array_like
            Parameters in original space of shape (n_params,)

        Notes
        -----
        Uses the normalizer.denormalize() method which implements the exact
        inverse of the normalization transform applied in Phase 0.

        Examples
        --------
        >>> # After optimization completes
        >>> normalized_result = jnp.array([0.6, 0.7])
        >>> original_result = optimizer._denormalize_params(normalized_result)
        """
        if self.normalizer is None:
            raise RuntimeError("Normalizer not initialized. Call _setup_normalization first.")

        return self.normalizer.denormalize(normalized_params)

    def _compute_normalized_covariance(self, JTJ: jnp.ndarray) -> jnp.ndarray:
        """Compute covariance matrix in normalized space from J^T J.

        The covariance is the inverse of the Hessian approximation:
        Cov_norm = (J^T J)^(-1)

        Uses pseudo-inverse for numerical stability and to handle rank-deficient
        or ill-conditioned matrices gracefully.

        Parameters
        ----------
        JTJ : array_like
            Accumulated J^T J matrix from Phase 2 of shape (n_params, n_params)

        Returns
        -------
        cov_norm : array_like
            Covariance matrix in normalized space of shape (n_params, n_params)

        Notes
        -----
        Uses jnp.linalg.pinv (pseudo-inverse) which is numerically stable and
        handles singular matrices via SVD. The result is guaranteed to be
        symmetric positive semi-definite.

        The pseudo-inverse uses SVD: J^T J = U S V^T
        Then: (J^T J)^(-1) = V S^(-1) U^T where small singular values are zeroed.
        """
        # Compute pseudo-inverse for numerical stability
        # This handles rank-deficient and ill-conditioned matrices
        cov_norm = jnp.linalg.pinv(JTJ)

        # Ensure symmetry (should already be symmetric, but enforce for numerical reasons)
        cov_norm = 0.5 * (cov_norm + cov_norm.T)

        return cov_norm

    def _transform_covariance(self, cov_norm: jnp.ndarray) -> jnp.ndarray:
        """Transform covariance from normalized to original parameter space.

        Uses the chain rule for covariance transformation:
        Cov_orig = D @ Cov_norm @ D^T

        where D is the denormalization Jacobian (stored in Phase 0).

        For our diagonal normalization (element-wise scaling), D is diagonal:
        D = diag(scales)

        This simplifies to:
        Cov_orig[i,j] = scale_i * scale_j * Cov_norm[i,j]

        Parameters
        ----------
        cov_norm : array_like
            Covariance matrix in normalized space of shape (n_params, n_params)

        Returns
        -------
        cov_orig : array_like
            Covariance matrix in original space of shape (n_params, n_params)

        Notes
        -----
        The transformation preserves:
        - Symmetry: Cov_orig is symmetric if Cov_norm is symmetric
        - Positive semi-definiteness: Eigenvalues remain non-negative
        - Shape: (n_params, n_params)

        Mathematical derivation:
        Let p_orig = denormalize(p_norm) = D @ p_norm + offset
        Then Jacobian of denormalization is D (constant, diagonal)
        By chain rule: Cov(p_orig) = D @ Cov(p_norm) @ D^T
        """
        if self.normalization_jacobian is None:
            raise RuntimeError("Normalization Jacobian not available. Call _setup_normalization first.")

        # Get denormalization Jacobian (diagonal matrix with scales)
        D = self.normalization_jacobian

        # Transform covariance: Cov_orig = D @ Cov_norm @ D^T
        cov_orig = D @ cov_norm @ D.T

        # Ensure symmetry (should be preserved, but enforce for numerical stability)
        cov_orig = 0.5 * (cov_orig + cov_orig.T)

        return cov_orig

    def _apply_residual_variance(
        self,
        cov_orig: jnp.ndarray,
        residual_sum_sq: float,
        n_points: int,
    ) -> Tuple[jnp.ndarray, float]:
        """Apply residual variance scaling to covariance matrix.

        Scales the covariance by the residual variance estimate:
        sigma^2 = residual_sum_sq / (n - p)
        Cov_final = sigma^2 * Cov_orig

        This provides the final covariance estimate accounting for the
        goodness-of-fit of the model to the data.

        Parameters
        ----------
        cov_orig : array_like
            Covariance matrix in original space (before variance scaling)
            of shape (n_params, n_params)
        residual_sum_sq : float
            Total sum of squared residuals from Phase 2
        n_points : int
            Total number of data points

        Returns
        -------
        cov_final : array_like
            Final scaled covariance matrix of shape (n_params, n_params)
        sigma_sq : float
            Residual variance estimate (sigma^2)

        Notes
        -----
        The residual variance is computed as:
        sigma^2 = sum(residuals^2) / (n - p)

        where n is the number of data points and p is the number of parameters.
        This is an unbiased estimator of the error variance.

        Special cases:
        - If n <= p: sigma^2 set to infinity (covariance undefined)
        - If residual_sum_sq = 0: Perfect fit, sigma^2 = 0

        This matches scipy.optimize.curve_fit behavior when absolute_sigma=False.
        """
        n_params = cov_orig.shape[0]

        # Compute degrees of freedom
        dof = n_points - n_params

        if dof <= 0:
            # Not enough data points to estimate variance
            # Set to infinity (covariance undefined)
            sigma_sq = jnp.inf
            cov_final = jnp.full_like(cov_orig, jnp.inf)
        else:
            # Compute residual variance
            sigma_sq = residual_sum_sq / dof

            # Scale covariance by residual variance
            cov_final = sigma_sq * cov_orig

        return cov_final, float(sigma_sq)

    def _compute_standard_errors(self, pcov: jnp.ndarray) -> jnp.ndarray:
        """Compute standard errors from covariance matrix diagonal.

        Standard errors are the square root of the diagonal elements of
        the covariance matrix:
        perr[i] = sqrt(pcov[i, i])

        Parameters
        ----------
        pcov : array_like
            Covariance matrix of shape (n_params, n_params)

        Returns
        -------
        perr : array_like
            Standard errors of shape (n_params,)

        Notes
        -----
        Standard errors represent the uncertainty in each parameter estimate
        (1-sigma confidence interval).

        Special cases:
        - If pcov[i,i] < 0: Sets perr[i] = NaN (indicates numerical issues)
        - If pcov[i,i] = inf: Sets perr[i] = inf (undefined uncertainty)

        The standard errors can be used to compute confidence intervals:
        95% CI: param ± 1.96 * perr
        """
        # Extract diagonal elements
        variances = jnp.diag(pcov)

        # Compute standard errors (sqrt of variances)
        # Handle negative variances by setting to NaN
        perr = jnp.where(
            variances >= 0,
            jnp.sqrt(variances),
            jnp.nan
        )

        return perr

    def _run_phase3_finalize(
        self,
        optimized_params_normalized: jnp.ndarray,
        JTJ_final: jnp.ndarray,
        residual_sum_sq: float,
        n_points: int,
    ) -> dict[str, Any]:
        """Run Phase 3: Denormalization and covariance transform.

        This is the final phase that:
        1. Denormalizes optimized parameters to original space
        2. Computes covariance in normalized space from J^T J
        3. Transforms covariance to original space
        4. Applies residual variance scaling
        5. Computes standard errors

        Parameters
        ----------
        optimized_params_normalized : array_like
            Optimized parameters in normalized space from Phase 2
        JTJ_final : array_like
            Final accumulated J^T J matrix from Phase 2
        residual_sum_sq : float
            Total sum of squared residuals
        n_points : int
            Total number of data points

        Returns
        -------
        result : dict
            Phase 3 result with keys:
            - 'popt': Optimized parameters in original space
            - 'pcov': Full covariance matrix in original space
            - 'perr': Standard errors (1-sigma)
            - 'sigma_sq': Residual variance estimate
            - 'diagnostics': Phase 3 diagnostics

        Notes
        -----
        This method orchestrates all Phase 3 operations to produce
        final results compatible with scipy.optimize.curve_fit:
        - popt: Optimized parameters
        - pcov: Covariance matrix for uncertainty analysis
        - perr: Standard errors (not returned by scipy but useful)

        The covariance transformation follows:
        1. Cov_norm = (J^T J)^(-1)  [in normalized space]
        2. Cov_orig = D @ Cov_norm @ D^T  [transform to original space]
        3. Cov_final = sigma^2 * Cov_orig  [scale by residual variance]
        """
        phase_start = time.time()

        # Step 1: Denormalize parameters to original space
        popt = self._denormalize_params(optimized_params_normalized)

        # Step 2: Compute covariance in normalized space
        cov_norm = self._compute_normalized_covariance(JTJ_final)

        # Step 3: Transform covariance to original space
        cov_orig = self._transform_covariance(cov_norm)

        # Step 4: Apply residual variance scaling
        pcov, sigma_sq = self._apply_residual_variance(
            cov_orig, residual_sum_sq, n_points
        )

        # Step 5: Compute standard errors
        perr = self._compute_standard_errors(pcov)

        # Record Phase 3 completion
        phase_duration = time.time() - phase_start
        phase_record = {
            'phase': 3,
            'name': 'denormalization_covariance',
            'duration': phase_duration,
            'sigma_sq': sigma_sq,
            'cov_condition': float(jnp.linalg.cond(pcov)) if jnp.isfinite(pcov).all() else jnp.inf,
            'timestamp': time.time(),
        }
        self.phase_history.append(phase_record)

        return {
            'popt': popt,
            'pcov': pcov,
            'perr': perr,
            'sigma_sq': sigma_sq,
            'diagnostics': phase_record,
        }

    def _validate_numerics(
        self,
        params: jnp.ndarray,
        loss: float | None = None,
        gradients: jnp.ndarray | None = None,
        context: str = "",
    ) -> bool:
        """Validate numerical stability of parameters, loss, and gradients.

        Parameters
        ----------
        params : array_like
            Parameters to validate
        loss : float, optional
            Loss value to validate
        gradients : array_like, optional
            Gradient values to validate
        context : str, optional
            Context string for error messages

        Returns
        -------
        is_valid : bool
            True if all values are finite, False otherwise

        Raises
        ------
        ValueError
            If validate_numerics is enabled and non-finite values detected
        """
        if not hasattr(self.config, 'validate_numerics') or not self.config.validate_numerics:
            return True

        # Check parameters
        if not jnp.all(jnp.isfinite(params)):
            if hasattr(self.config, 'enable_fault_tolerance') and self.config.enable_fault_tolerance:
                # Log warning but continue
                return False
            else:
                raise ValueError(f"NaN/Inf detected in parameters {context}")

        # Check loss
        if loss is not None and not jnp.isfinite(loss):
            if hasattr(self.config, 'enable_fault_tolerance') and self.config.enable_fault_tolerance:
                return False
            else:
                raise ValueError(f"NaN/Inf detected in loss {context}")

        # Check gradients
        if gradients is not None and not jnp.all(jnp.isfinite(gradients)):
            if hasattr(self.config, 'enable_fault_tolerance') and self.config.enable_fault_tolerance:
                return False
            else:
                raise ValueError(f"NaN/Inf detected in gradients {context}")

        return True

    def _save_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Save checkpoint with phase-specific state to HDF5 file.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file (.h5)

        Notes
        -----
        Checkpoint format version 3.0 includes:
        - current_phase: Current phase number
        - normalized_params: Parameters in normalized space
        - phase1_optimizer_state: Optax Adam state (mu, nu, count)
        - phase2_jtj_accumulator: Accumulated J^T J matrix
        - phase2_jtr_accumulator: Accumulated J^T r vector
        - best_params_global: Best parameters found globally
        - best_cost_global: Best cost value globally
        - phase_history: Complete phase history
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(checkpoint_path, 'w') as f:
            # Version metadata
            f.attrs['version'] = '3.0'
            f.attrs['timestamp'] = time.time()

            # Create phase_state group
            phase_state = f.create_group('phase_state')

            # Save current phase and parameters
            phase_state.create_dataset('current_phase', data=self.current_phase)
            if self.normalized_params is not None:
                phase_state.create_dataset('normalized_params', data=self.normalized_params)

            # Save Phase 1 optimizer state (Optax Adam)
            if self.phase1_optimizer_state is not None:
                opt_state_group = phase_state.create_group('phase1_optimizer_state')
                # Optax Adam state has structure: (ScaleByAdamState, EmptyState)
                # ScaleByAdamState has: count, mu, nu
                adam_state = self.phase1_optimizer_state[0]
                opt_state_group.create_dataset('count', data=int(adam_state.count))
                opt_state_group.create_dataset('mu', data=adam_state.mu)
                opt_state_group.create_dataset('nu', data=adam_state.nu)

            # Save Phase 2 accumulators
            if self.phase2_JTJ_accumulator is not None:
                phase_state.create_dataset('phase2_jtj_accumulator', data=self.phase2_JTJ_accumulator)
            if self.phase2_JTr_accumulator is not None:
                phase_state.create_dataset('phase2_jtr_accumulator', data=self.phase2_JTr_accumulator)

            # Save best parameters tracking
            if self.best_params_global is not None:
                phase_state.create_dataset('best_params_global', data=self.best_params_global)
            phase_state.create_dataset('best_cost_global', data=self.best_cost_global)

            # Save phase history as pickle (easier for complex structures)
            if self.phase_history:
                import numpy as np
                phase_history_bytes = pickle.dumps(self.phase_history)
                phase_state.create_dataset('phase_history', data=np.void(phase_history_bytes))

            # Save normalization state (if exists)
            if self.normalizer is not None:
                norm_group = phase_state.create_group('normalizer_state')
                norm_group.attrs['strategy'] = self.normalizer.strategy
                if self.normalizer.scales is not None:
                    norm_group.create_dataset('scales', data=self.normalizer.scales)
                if self.normalizer.offsets is not None:
                    norm_group.create_dataset('offsets', data=self.normalizer.offsets)

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load checkpoint and restore phase-specific state.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file (.h5)

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist
        ValueError
            If checkpoint version is incompatible
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with h5py.File(checkpoint_path, 'r') as f:
            # Check version
            version = f.attrs.get('version', '1.0')
            if not version.startswith('3.'):
                raise ValueError(f"Incompatible checkpoint version: {version} (expected 3.x)")

            phase_state = f['phase_state']

            # Restore current phase
            self.current_phase = int(phase_state['current_phase'][()])

            # Restore normalized parameters
            if 'normalized_params' in phase_state:
                self.normalized_params = jnp.array(phase_state['normalized_params'])

            # Restore Phase 1 optimizer state
            if 'phase1_optimizer_state' in phase_state:
                opt_state_group = phase_state['phase1_optimizer_state']
                count = int(opt_state_group['count'][()])
                mu = jnp.array(opt_state_group['mu'])
                nu = jnp.array(opt_state_group['nu'])

                # Reconstruct Optax Adam state structure
                # Note: This is a simplified reconstruction
                # Full state may require creating optimizer instance
                from optax._src.transform import ScaleByAdamState
                adam_state = ScaleByAdamState(count=count, mu=mu, nu=nu)
                self.phase1_optimizer_state = (adam_state, optax.EmptyState())

            # Restore Phase 2 accumulators
            if 'phase2_jtj_accumulator' in phase_state:
                self.phase2_JTJ_accumulator = jnp.array(phase_state['phase2_jtj_accumulator'])
            if 'phase2_jtr_accumulator' in phase_state:
                self.phase2_JTr_accumulator = jnp.array(phase_state['phase2_jtr_accumulator'])

            # Restore best parameters tracking
            if 'best_params_global' in phase_state:
                self.best_params_global = jnp.array(phase_state['best_params_global'])
            self.best_cost_global = float(phase_state['best_cost_global'][()])

            # Restore phase history
            if 'phase_history' in phase_state:
                phase_history_bytes = bytes(phase_state['phase_history'][()])
                self.phase_history = pickle.loads(phase_history_bytes)

            # Restore normalizer state (if exists)
            if 'normalizer_state' in phase_state:
                norm_group = phase_state['normalizer_state']
                strategy = norm_group.attrs['strategy']
                scales = jnp.array(norm_group['scales']) if 'scales' in norm_group else None
                offsets = jnp.array(norm_group['offsets']) if 'offsets' in norm_group else None

                # Note: Full normalizer reconstruction requires bounds/p0
                # This is a partial restore - full normalizer created in _setup_normalization

    def _detect_available_devices(self) -> dict[str, Any]:
        """Detect available GPU/TPU devices using JAX.

        Returns
        -------
        device_info : dict
            Dictionary with keys:
            - 'device_count': Number of available devices (int)
            - 'device_type': Type of devices ('cpu', 'gpu', 'tpu')
            - 'devices': List of JAX device objects

        Notes
        -----
        Uses jax.devices() to detect all available devices.
        Device type is determined from the first device's platform.

        Examples
        --------
        >>> optimizer = AdaptiveHybridStreamingOptimizer()
        >>> info = optimizer._detect_available_devices()
        >>> print(info['device_count'])
        1
        >>> print(info['device_type'])
        'cpu'
        """
        # Get all available devices
        devices = jax.devices()
        device_count = len(devices)

        # Determine device type from first device
        if device_count > 0:
            platform = devices[0].platform
            # Map JAX platform names to our device types
            if platform == 'cpu':
                device_type = 'cpu'
            elif platform in ['gpu', 'cuda', 'rocm']:
                device_type = 'gpu'
            elif platform == 'tpu':
                device_type = 'tpu'
            else:
                device_type = 'cpu'  # Default fallback
        else:
            device_type = 'cpu'

        device_info = {
            'device_count': device_count,
            'device_type': device_type,
            'devices': devices,
        }

        # Store for later use
        self.device_info = device_info

        return device_info

    def _should_use_multi_device(self, device_info: dict[str, Any]) -> bool:
        """Determine if multi-device should be used based on config and availability.

        Parameters
        ----------
        device_info : dict
            Device information from _detect_available_devices()

        Returns
        -------
        should_use : bool
            True if multi-device should be used, False otherwise

        Notes
        -----
        Multi-device is used only if:
        1. config.enable_multi_device is True
        2. More than 1 device is available
        3. Devices are not CPU (CPU multi-device not beneficial)
        """
        # Check if enabled in config
        if not self.config.enable_multi_device:
            return False

        # Check if multiple devices available
        if device_info['device_count'] <= 1:
            return False

        # Don't use multi-device for CPU (no benefit)
        if device_info['device_type'] == 'cpu':
            return False

        return True

    def _setup_multi_device(self, device_info: dict[str, Any]) -> dict[str, Any]:
        """Setup multi-device configuration for data-parallel computation.

        Parameters
        ----------
        device_info : dict
            Device information from _detect_available_devices()

        Returns
        -------
        multi_device_config : dict
            Configuration with keys:
            - 'use_multi_device': Whether multi-device is enabled (bool)
            - 'device_count': Number of devices to use (int)
            - 'devices': List of JAX device objects
            - 'axis_name': Axis name for pmap/psum ('devices')

        Notes
        -----
        If multi-device cannot be used, returns configuration for single device.
        Sets up axis_name='devices' for pmap and psum coordination.

        Examples
        --------
        >>> optimizer = AdaptiveHybridStreamingOptimizer()
        >>> device_info = optimizer._detect_available_devices()
        >>> config = optimizer._setup_multi_device(device_info)
        >>> print(config['use_multi_device'])
        False  # On single-device system
        """
        should_use = self._should_use_multi_device(device_info)

        if should_use:
            # Multi-device configuration
            multi_device_config = {
                'use_multi_device': True,
                'device_count': device_info['device_count'],
                'devices': device_info['devices'],
                'axis_name': 'devices',
            }
        else:
            # Single-device fallback
            multi_device_config = {
                'use_multi_device': False,
                'device_count': 1,
                'devices': device_info['devices'][:1] if device_info['devices'] else [],
                'axis_name': None,
            }

        # Store for later use
        self.multi_device_config = multi_device_config

        return multi_device_config

    def _pmap_jacobian_computation(
        self,
        x_chunks: list[jnp.ndarray],
        params: jnp.ndarray,
    ) -> list[jnp.ndarray]:
        """Compute Jacobian across multiple devices using pmap (NOT IMPLEMENTED).

        This method would use jax.pmap for data-parallel Jacobian computation
        across multiple GPUs/TPUs. However, due to the complexity of properly
        sharding data and handling device communication, we defer this to
        future work.

        Parameters
        ----------
        x_chunks : list of array_like
            Data chunks to distribute across devices
        params : array_like
            Parameters in normalized space

        Returns
        -------
        J_chunks : list of array_like
            Jacobian matrices computed on each device

        Notes
        -----
        This is a placeholder for future multi-device support.
        Currently falls back to single-device computation.

        For proper pmap implementation, we would need:
        1. Data sharding strategy matching device count
        2. Replicated parameters across devices
        3. pmap-compatible Jacobian function
        4. Device mesh configuration

        Raises
        ------
        NotImplementedError
            Always raised - pmap not yet implemented
        """
        raise NotImplementedError(
            "pmap Jacobian computation not yet implemented. "
            "Use single-device computation via _compute_jacobian_chunk()."
        )

    def _aggregate_jtj_across_devices(self, JTJ_local: jnp.ndarray) -> jnp.ndarray:
        """Aggregate J^T J matrix across devices using psum.

        On single device, this is a no-op and returns the input matrix unchanged.
        On multi-device systems, this would use jax.lax.psum to sum matrices
        across all devices.

        Parameters
        ----------
        JTJ_local : array_like
            Local J^T J matrix of shape (n_params, n_params)

        Returns
        -------
        JTJ_global : array_like
            Aggregated J^T J matrix of shape (n_params, n_params)

        Notes
        -----
        Single-device case: Returns input unchanged
        Multi-device case (future): Would use jax.lax.psum with axis_name='devices'

        Mathematical operation:
        JTJ_global = sum_over_devices(JTJ_local)

        For proper multi-device aggregation:
        ```python
        if self.multi_device_config['use_multi_device']:
            JTJ_global = jax.lax.psum(JTJ_local, axis_name='devices')
        else:
            JTJ_global = JTJ_local
        ```

        Examples
        --------
        >>> optimizer = AdaptiveHybridStreamingOptimizer()
        >>> JTJ = jnp.array([[10.0, 2.0], [2.0, 8.0]])
        >>> JTJ_agg = optimizer._aggregate_jtj_across_devices(JTJ)
        >>> assert jnp.allclose(JTJ_agg, JTJ)  # Same on single device
        """
        # Check if multi-device is configured and enabled
        if (self.multi_device_config is not None and
            self.multi_device_config.get('use_multi_device', False)):

            # Multi-device aggregation would use psum
            # For now, we just return the local matrix (single-device fallback)
            # Future implementation:
            # return jax.lax.psum(JTJ_local, axis_name='devices')

            # Log warning about fallback
            import warnings
            warnings.warn(
                "Multi-device aggregation not yet fully implemented. "
                "Falling back to single-device computation.",
                UserWarning
            )

        # Single-device case or fallback: return unchanged
        return JTJ_local

    def _setup_precision(self) -> None:
        """Setup precision strategy based on config.

        Determines precision for each phase:
        - precision='auto': Phase 0 float64, Phase 1 float32, Phase 2+ float64
        - precision='float32': float32 throughout
        - precision='float64': float64 throughout

        Notes
        -----
        Phase 0 (normalization) always uses float64 for accuracy.
        Phase 1 (Adam warmup) can use float32 for memory efficiency with 'auto'.
        Phase 2+ (Gauss-Newton, covariance) use float64 for numerical stability.

        Sets self.current_precision and self.phase_precisions.
        Immediately transitions to Phase 1 precision for 'auto' mode after setup.
        """
        if self.config.precision == 'auto':
            # Auto mode: float32 for Phase 1, float64 for others
            self.phase_precisions = {
                0: jnp.float64,  # Normalization: always float64
                1: jnp.float32,  # Warmup: float32 for memory
                2: jnp.float64,  # Gauss-Newton: float64 for stability
                3: jnp.float64,  # Covariance: float64 for accuracy
            }
            # Phase 0 is just setup, start in Phase 1 precision (float32)
            self.current_precision = jnp.float32

        elif self.config.precision == 'float32':
            # User forces float32 throughout
            self.phase_precisions = {
                0: jnp.float32,
                1: jnp.float32,
                2: jnp.float32,
                3: jnp.float64,  # Covariance always float64 for accuracy
            }
            self.current_precision = jnp.float32

        elif self.config.precision == 'float64':
            # User forces float64 throughout
            self.phase_precisions = {
                0: jnp.float64,
                1: jnp.float64,
                2: jnp.float64,
                3: jnp.float64,
            }
            self.current_precision = jnp.float64

        else:
            raise ValueError(
                f"Invalid precision: {self.config.precision}. "
                "Must be 'auto', 'float32', or 'float64'."
            )

    def _convert_precision(self, target_dtype: jnp.dtype) -> None:
        """Convert arrays and optimizer state to target precision.

        Parameters
        ----------
        target_dtype : jnp.dtype
            Target data type (jnp.float32 or jnp.float64)

        Notes
        -----
        Converts:
        - self.normalized_params
        - self.phase1_optimizer_state (if exists)
        - self.phase2_JTJ_accumulator (if exists)
        - self.phase2_JTr_accumulator (if exists)
        - self.best_params_global (if exists)

        Uses JAX device_put for zero-copy conversion where possible.
        """
        if self.current_precision == target_dtype:
            # Already at target precision
            return

        # Convert normalized parameters
        if self.normalized_params is not None:
            self.normalized_params = self.normalized_params.astype(target_dtype)

        # Convert Phase 1 optimizer state (Optax Adam)
        if self.phase1_optimizer_state is not None:
            try:
                # Optax Adam state structure: (ScaleByAdamState, EmptyState)
                adam_state = self.phase1_optimizer_state[0]
                from optax._src.transform import ScaleByAdamState

                # Convert mu and nu to target precision
                new_adam_state = ScaleByAdamState(
                    count=adam_state.count,
                    mu=adam_state.mu.astype(target_dtype),
                    nu=adam_state.nu.astype(target_dtype),
                )
                self.phase1_optimizer_state = (new_adam_state, optax.EmptyState())
            except Exception:
                # If conversion fails, reset optimizer state
                self.phase1_optimizer_state = None

        # Convert Phase 2 accumulators
        if self.phase2_JTJ_accumulator is not None:
            self.phase2_JTJ_accumulator = self.phase2_JTJ_accumulator.astype(target_dtype)

        if self.phase2_JTr_accumulator is not None:
            self.phase2_JTr_accumulator = self.phase2_JTr_accumulator.astype(target_dtype)

        # Convert best parameters tracking
        if self.best_params_global is not None:
            self.best_params_global = self.best_params_global.astype(target_dtype)

        # Update current precision
        self.current_precision = target_dtype

    def _check_precision_upgrade_needed(
        self,
        params: jnp.ndarray | None = None,
        loss: float | None = None,
        gradients: jnp.ndarray | None = None,
    ) -> bool:
        """Check if precision upgrade is needed due to numerical issues.

        Parameters
        ----------
        params : array_like, optional
            Current parameters to check
        loss : float, optional
            Current loss value to check
        gradients : array_like, optional
            Current gradients to check

        Returns
        -------
        needs_upgrade : bool
            True if precision upgrade recommended

        Notes
        -----
        Checks for:
        - NaN/Inf in parameters, loss, or gradients
        - Only triggers upgrade if current precision is float32
        - Only triggers upgrade once (self.precision_upgrade_triggered)
        """
        # Only upgrade from float32 to float64
        if self.current_precision != jnp.float32:
            return False

        # Only upgrade once
        if self.precision_upgrade_triggered:
            return False

        # Check parameters for NaN/Inf
        if params is not None and not jnp.all(jnp.isfinite(params)):
            return True

        # Check loss for NaN/Inf
        if loss is not None and not jnp.isfinite(loss):
            return True

        # Check gradients for NaN/Inf
        if gradients is not None and not jnp.all(jnp.isfinite(gradients)):
            return True

        return False

    def _upgrade_precision_if_needed(
        self,
        params: jnp.ndarray | None = None,
        loss: float | None = None,
        gradients: jnp.ndarray | None = None,
    ) -> None:
        """Upgrade precision if numerical issues detected.

        Parameters
        ----------
        params : array_like, optional
            Current parameters to check
        loss : float, optional
            Current loss value to check
        gradients : array_like, optional
            Current gradients to check

        Notes
        -----
        If upgrade is needed, converts all state to float64 and marks
        precision_upgrade_triggered to prevent repeated upgrades.
        """
        if self._check_precision_upgrade_needed(params, loss, gradients):
            # Log upgrade
            if hasattr(self.config, 'verbose') and self.config.verbose:
                print("WARNING: Numerical issues detected in float32. Upgrading to float64.")

            # Convert to float64
            self._convert_precision(jnp.float64)

            # Mark upgrade triggered
            self.precision_upgrade_triggered = True

    def _handle_phase_transition_precision(self, new_phase: int) -> None:
        """Handle precision changes at phase transitions.

        Parameters
        ----------
        new_phase : int
            Phase number transitioning to (1, 2, or 3)

        Notes
        -----
        For precision='auto':
        - Phase 0 -> Phase 1: Convert to float32 (memory efficiency)
        - Phase 1 -> Phase 2: Convert to float64 (numerical stability)
        - Phase 2 -> Phase 3: Already float64

        For user-specified precision, respects user choice.
        """
        # Get target precision for new phase
        target_precision = self.phase_precisions.get(new_phase, jnp.float64)

        # Convert if different from current
        if target_precision != self.current_precision:
            self._convert_precision(target_precision)

    def fit(
        self,
        data_source: Any,
        func: callable,
        p0: jnp.ndarray,
        bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        sigma: jnp.ndarray | None = None,
        absolute_sigma: bool = False,
        callback: callable | None = None,
        verbose: int = 1,
    ) -> dict[str, Any]:
        """Fit model parameters using four-phase hybrid optimization.

        This method orchestrates all four phases:
        - Phase 0: Setup normalization
        - Phase 1: Adam warmup
        - Phase 2: Streaming Gauss-Newton
        - Phase 3: Denormalization and covariance

        Parameters
        ----------
        data_source : various types
            Data source for optimization. Can be:
            - Tuple of arrays: (x_data, y_data)
            - Generator yielding (x_batch, y_batch)
            - HDF5 file path with datasets
        func : callable
            Model function with signature: ``func(x, *params) -> predictions``
        p0 : array_like
            Initial parameter guess of shape (n_params,)
        bounds : tuple of array_like, optional
            Parameter bounds as (lb, ub)
        sigma : array_like, optional
            Uncertainties in y_data for weighted least squares
        absolute_sigma : bool, default=False
            If True, sigma is used in absolute sense (pcov not scaled)
        callback : callable, optional
            Callback with signature callback(params, loss, iteration)
            Called every config.callback_frequency iterations
        verbose : int, default=1
            Verbosity level (0=silent, 1=progress, 2=debug)

        Returns
        -------
        result : dict
            Optimization result dictionary with keys:
            - 'x': Optimized parameters in original space
            - 'success': Boolean indicating success
            - 'message': Status message
            - 'fun': Final residuals
            - 'pcov': Covariance matrix (Phase 3)
            - 'perr': Standard errors (Phase 3)
            - 'streaming_diagnostics': Phase information, timing, etc.

        Notes
        -----
        The result dictionary is compatible with scipy.optimize.curve_fit
        and can be used interchangeably.
        """
        # Track total optimization time
        total_start_time = time.time()

        # Phase timing storage
        phase_timings = {}
        phase_iterations = {}

        # Convert p0 to JAX array
        p0_array = jnp.asarray(p0, dtype=jnp.float64)

        # Extract data from source (currently only supports tuple)
        if isinstance(data_source, tuple) and len(data_source) == 2:
            x_data, y_data = data_source
            x_data = jnp.asarray(x_data, dtype=jnp.float64)
            y_data = jnp.asarray(y_data, dtype=jnp.float64)
            n_points = len(x_data)
        else:
            raise NotImplementedError(
                "Only tuple data sources (x_data, y_data) currently supported"
            )

        # ============================================================
        # Phase 0: Setup Normalization and Precision
        # ============================================================
        if verbose >= 1:
            print("=" * 60)
            print("Adaptive Hybrid Streaming Optimizer")
            print("=" * 60)
            print(f"Dataset size: {n_points:,} points")
            print(f"Parameters: {len(p0_array)}")
            print(f"Normalization: {self.config.normalization_strategy}")
            print(f"Precision: {self.config.precision}")
            print()

        phase0_start = time.time()
        self._setup_normalization(func, p0_array, bounds)
        self._setup_precision()  # Setup precision strategy
        phase0_duration = time.time() - phase0_start
        phase_timings['phase0_normalization'] = phase0_duration

        if verbose >= 1:
            print(f"Phase 0: Normalization setup complete ({phase0_duration:.3f}s)")
            print(f"  Strategy: {self.normalizer.strategy}")
            print(f"  Precision: {self.current_precision}")
            print()

        # ============================================================
        # Phase 1: Adam Warmup
        # ============================================================
        # Handle precision transition to Phase 1
        self._handle_phase_transition_precision(1)

        if verbose >= 1:
            print("Phase 1: Adam warmup...")
            print(f"  Precision: {self.current_precision}")

        phase1_start = time.time()
        phase1_result = self._run_phase1_warmup(
            data_source=(x_data, y_data),
            model=func,
            p0=p0_array,
            bounds=bounds,
        )
        phase1_duration = time.time() - phase1_start
        phase_timings['phase1_warmup'] = phase1_duration
        phase_iterations['phase1'] = phase1_result['iterations']

        if verbose >= 1:
            print(f"Phase 1 complete: {phase1_result['iterations']} iterations ({phase1_duration:.3f}s)")
            print(f"  Best loss: {phase1_result['best_loss']:.6e}")
            print(f"  Switch reason: {phase1_result['switch_reason']}")
            print()

        # Use best parameters from Phase 1 as starting point for Phase 2
        warmup_params = phase1_result['best_params']

        # ============================================================
        # Phase 2: Streaming Gauss-Newton
        # ============================================================
        # Handle precision transition to Phase 2
        self._handle_phase_transition_precision(2)

        if verbose >= 1:
            print("Phase 2: Streaming Gauss-Newton...")
            print(f"  Precision: {self.current_precision}")

        phase2_start = time.time()
        phase2_result = self._run_phase2_gauss_newton(
            data_source=(x_data, y_data),
            initial_params=warmup_params,
        )
        phase2_duration = time.time() - phase2_start
        phase_timings['phase2_gauss_newton'] = phase2_duration
        phase_iterations['phase2'] = phase2_result['iterations']

        if verbose >= 1:
            print(f"Phase 2 complete: {phase2_result['iterations']} iterations ({phase2_duration:.3f}s)")
            print(f"  Final cost: {phase2_result['final_cost']:.6e}")
            print(f"  Convergence: {phase2_result['convergence_reason']}")
            print(f"  Gradient norm: {phase2_result['gradient_norm']:.6e}")
            print()

        # ============================================================
        # Phase 3: Denormalization and Covariance
        # ============================================================
        # Handle precision transition to Phase 3 (always float64 for covariance)
        self._handle_phase_transition_precision(3)

        if verbose >= 1:
            print("Phase 3: Computing covariance...")
            print(f"  Precision: {self.current_precision}")

        phase3_start = time.time()
        phase3_result = self._run_phase3_finalize(
            optimized_params_normalized=phase2_result['final_params'],
            JTJ_final=phase2_result['JTJ_final'],
            residual_sum_sq=phase2_result['residual_sum_sq'],
            n_points=n_points,
        )
        phase3_duration = time.time() - phase3_start
        phase_timings['phase3_finalize'] = phase3_duration

        if verbose >= 1:
            print(f"Phase 3 complete ({phase3_duration:.3f}s)")
            print(f"  Residual variance (σ²): {phase3_result['sigma_sq']:.6e}")
            print()

        # ============================================================
        # Assemble Final Result
        # ============================================================
        total_duration = time.time() - total_start_time

        # Compute final residuals for 'fun' field
        final_predictions = func(x_data, *phase3_result['popt'])
        final_residuals = y_data - final_predictions

        # Build streaming diagnostics
        streaming_diagnostics = {
            'phase_timings': phase_timings,
            'phase_iterations': phase_iterations,
            'total_time': total_duration,
            'warmup_diagnostics': {
                'best_loss': phase1_result['best_loss'],
                'final_loss': phase1_result['final_loss'],
                'switch_reason': phase1_result['switch_reason'],
            },
            'gauss_newton_diagnostics': {
                'best_cost': phase2_result['best_cost'],
                'final_cost': phase2_result['final_cost'],
                'gradient_norm': phase2_result['gradient_norm'],
                'convergence_reason': phase2_result['convergence_reason'],
            },
            'phase_history': self.phase_history,
        }

        # Format result dictionary (scipy-compatible + NLSQ extensions)
        result = {
            'x': phase3_result['popt'],           # Optimized parameters
            'success': True,                       # Always True if we reach here
            'message': phase2_result['convergence_reason'],
            'fun': final_residuals,                # Final residuals
            'pcov': phase3_result['pcov'],         # Covariance matrix
            'perr': phase3_result['perr'],         # Standard errors (NLSQ extension)
            'streaming_diagnostics': streaming_diagnostics,  # NLSQ extension
        }

        if verbose >= 1:
            print("=" * 60)
            print("Optimization Complete")
            print("=" * 60)
            print(f"Total time: {total_duration:.3f}s")
            print(f"Final parameters: {result['x']}")
            print(f"Parameter std errors: {result['perr']}")
            print("=" * 60)

        return result

    @property
    def phase_status(self) -> dict[str, Any]:
        """Get current phase status and history.

        Returns
        -------
        status : dict
            Phase status dictionary with keys:
            - 'current_phase': Current phase number
            - 'phase_name': Name of current phase
            - 'phase_history': List of completed phases with timing
            - 'total_phases': Total number of phases (4)

        Examples
        --------
        >>> config = HybridStreamingConfig()
        >>> optimizer = AdaptiveHybridStreamingOptimizer(config)
        >>> status = optimizer.phase_status
        >>> print(status['current_phase'])
        0
        >>> print(status['phase_name'])
        Phase 0: Normalization Setup
        """
        phase_names = {
            0: "Phase 0: Normalization Setup",
            1: "Phase 1: Adam Warmup",
            2: "Phase 2: Streaming Gauss-Newton",
            3: "Phase 3: Denormalization and Covariance",
        }

        return {
            'current_phase': self.current_phase,
            'phase_name': phase_names.get(self.current_phase, "Unknown"),
            'phase_history': self.phase_history,
            'total_phases': 4,
        }
