"""Adaptive Hybrid Streaming Optimizer with Parameter Normalization.

This module implements a four-phase hybrid optimizer that solves three fundamental
issues in streaming optimization:

1. Weak gradient signals from parameter scale imbalance (via normalization)
2. Slow convergence near optimum (via Gauss-Newton)
3. Crude covariance estimation (via exact J^T J accumulation)

The optimizer operates in four phases:
- **Phase 0**: Parameter normalization setup
- **Phase 1**: L-BFGS warmup with adaptive switching
- **Phase 2**: Streaming Gauss-Newton with exact J^T J accumulation
- **Phase 3**: Denormalization and covariance transform

This implementation focuses on Phase 0 setup logic and phase tracking infrastructure.
"""

# mypy: disable-error-code="assignment,misc,return-value,valid-type,union-attr,operator,import-untyped,attr-defined,arg-type"
# Note: mypy errors are mostly assignment/return-value/union-attr issues from
# complex streaming state management. These require deeper refactoring.

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import-not-found]

from nlsq.global_optimization.config import GlobalOptimizationConfig
from nlsq.global_optimization.sampling import (
    get_sampler,
)
from nlsq.global_optimization.tournament import TournamentSelector
from nlsq.precision.parameter_normalizer import (
    NormalizedModelWrapper,
    ParameterNormalizer,
)
from nlsq.stability.guard import NumericalStabilityGuard
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.telemetry import (
    DefenseLayerTelemetry,
    get_defense_telemetry,
    reset_defense_telemetry,
)
from nlsq.utils.logging import get_logger

# Lazy import cache for CheckpointManager to avoid circular imports
# Import happens at first checkpoint operation
_lazy_imports: dict = {}

# Module-level logger for warmup defense diagnostics
_logger = get_logger("adaptive_hybrid_streaming")

# Re-export telemetry classes for backwards compatibility
__all__ = [
    "AdaptiveHybridStreamingOptimizer",
    "DefenseLayerTelemetry",
    "get_defense_telemetry",
    "reset_defense_telemetry",
]


def _fill_padded_rows(x_chunk: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Replace padded rows (mask == 0) with x_chunk[0] before model evaluation.

    Padded rows are x=0 (jnp.pad's default) and only ever occur in the LAST
    chunk of a lax.scan accumulation (pad_size < chunk_size by construction,
    so index 0 is always real data). For models singular/undefined at 0
    (e.g. a/x, log(x)), evaluating them there produces NaN/Inf, and
    NaN * 0 == NaN, so masking the residual/Jacobian AFTER evaluation would
    not zero it out and would poison the accumulated JTJ/JTr for the whole
    scan. x_chunk[0] is used rather than an arbitrary sentinel (e.g. 1.0)
    because it's guaranteed to be a value the model is actually defined at.
    """
    mask_expanded = mask.reshape((-1,) + (1,) * (x_chunk.ndim - 1)) > 0
    return jnp.where(mask_expanded, x_chunk, x_chunk[0])


class AdaptiveHybridStreamingOptimizer:
    """Adaptive hybrid streaming optimizer with four-phase optimization.

    This optimizer combines parameter normalization, L-BFGS warmup, streaming
    Gauss-Newton, and exact covariance computation to provide:

    - Fast convergence for parameters with different scales
    - Accurate uncertainty estimates on large datasets
    - Memory-efficient streaming for unlimited dataset sizes
    - Production-ready fault tolerance

    The optimization proceeds through four phases:

    - **Phase 0**: Setup parameter normalization and bounds transformation
    - **Phase 1**: L-BFGS warmup with adaptive switching to Phase 2
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
    ...     lbfgs_initial_step_size=0.5,
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
        self.phase_history: deque[dict[str, Any]] = deque(maxlen=100)
        self.phase_start_time: float | None = None
        self.normalized_params: jnp.ndarray | None = None

        # Phase 0: Normalization components (created during setup)
        self.normalizer: ParameterNormalizer | None = None
        self.normalized_model: NormalizedModelWrapper | None = None
        self.normalized_bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None
        self.normalization_jacobian: jnp.ndarray | None = None

        # Store original model and parameters (for reference)
        self.original_model: Callable | None = None
        self.original_p0: jnp.ndarray | None = None
        self.original_bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None

        # Fault tolerance components
        self.stability_guard = NumericalStabilityGuard()
        self.best_params_global: jnp.ndarray | None = None
        self.best_cost_global: float = float("inf")
        self.checkpoint_counter: int = 0
        self.retry_count: int = 0

        # Phase-specific state for checkpointing
        self.phase1_optimizer_state: optax.OptState | None = None
        self.phase2_JTJ_accumulator: jnp.ndarray | None = None
        self.phase2_JTr_accumulator: jnp.ndarray | None = None

        # Checkpoint manager (lazy initialized on first use)
        self._checkpoint_manager = None

        # Multi-start optimization with tournament selection
        self.multistart_candidates: jnp.ndarray | None = None
        self.tournament_selector = None
        self.multistart_best_candidate: jnp.ndarray | None = None
        self.multistart_diagnostics: dict[str, Any] | None = None

        # 4-Layer Defense Strategy state for warmup divergence prevention
        self._warmup_initial_loss: float | None = None
        self._warmup_relative_loss: float | None = None
        self._warmup_lr_mode: str | None = None
        self._warmup_clip_count: int = 0

        # Residual weighting state for weighted least squares
        # Allows domain-specific weighting of residuals during optimization
        self._residual_weights_jax: jnp.ndarray | None = None  # Per-group weights

        # Pre-compiled Jacobian function (set up in _setup_jacobian_fn)
        # This avoids recompilation overhead in _compute_jacobian_chunk
        self._jacobian_fn_compiled: Callable | None = None

        # User progress callback set by fit(); invoked every
        # config.callback_frequency iterations in Phase 1 and Phase 2.
        self._fit_callback: Callable | None = None

        # Pre-compiled cost-only function (set up in _setup_cost_fn)
        # Used for efficient new_cost evaluation in Gauss-Newton iterations
        self._cost_fn_compiled: Callable | None = None

        # Pre-compiled scan functions (set up in _setup_scan_functions)
        # Used for efficient chunk-based accumulation with JAX lax.scan
        self._jtj_scan_body_compiled: bool = False
        self._cost_scan_body_compiled: bool = False

    def _invoke_fit_callback(
        self,
        params: jnp.ndarray,
        loss: float,
        iteration: int,
    ) -> None:
        """Call the user progress callback, if any, at the configured frequency.

        A callback exception must not abort the fit -- it's user code we
        don't control, invoked from inside a hot loop.
        """
        if self._fit_callback is None:
            return
        if (iteration + 1) % max(1, self.config.callback_frequency) != 0:
            return
        try:
            self._fit_callback(params, loss, iteration)
        except Exception as exc:
            # exc_info=True preserves the traceback pointing into the user's
            # callback -- str(exc) alone (the previous behavior) discarded
            # exactly the information needed to debug a broken callback
            # (e.g. a typo in its signature, an exception in the user's own
            # plotting code). NLSQLogger has no .exception() method, only
            # .warning(exc_info=...).
            _logger.warning(
                f"fit() callback raised at iteration {iteration}: {exc}",
                exc_info=True,
            )

    def _update_global_best(
        self,
        loss_value: float,
        params: jnp.ndarray,
        n_points: int,
    ) -> None:
        """Update self.best_cost_global/best_params_global from a Phase 1 loss.

        Converts Phase 1's MSE to an SSR-equivalent (* n_points) so it stays
        comparable with Phase 2's SSR-based cost tracking in the SAME
        best_cost_global (see _run_phase2_gauss_newton) -- without this,
        Phase 2's `new_cost < best_cost_global` could essentially never fire
        once Phase 1 has run, silently freezing best_params_global at
        wherever Phase 1 left it.

        Skipped entirely when residual weighting is enabled: Phase 1's
        weighted loss (sum(w*r^2)/sum(w)) is not convertible to unweighted
        SSR by a simple n_points scale, and Phase 2 never applies residual
        weighting (it's a Phase-1-only feature), so a weighted Phase 1 loss
        is not comparable to Phase 2's cost at all. Silently storing an
        incomparable number here would be worse than leaving the tracker
        alone for Phase 2 (or a prior fit's None/inf) to set.
        """
        if self.config.enable_residual_weighting:
            return
        ssr_equivalent = float(loss_value) * n_points
        if ssr_equivalent < self.best_cost_global:
            self.best_cost_global = ssr_equivalent
            self.best_params_global = params

    def _setup_normalization(
        self,
        model: Callable,
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
        model : Callable
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
            strategy = "none"
        else:
            # Use configured strategy
            strategy = self.config.normalization_strategy

        # Create ParameterNormalizer
        self.normalizer = ParameterNormalizer(
            p0=self.original_p0,
            bounds=bounds,
            strategy=strategy,
        )

        # Create NormalizedModelWrapper
        self.normalized_model = NormalizedModelWrapper(
            model_fn=model,
            normalizer=self.normalizer,
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

        # Generate multi-start candidates if enabled
        if self.config.enable_multistart:
            self._generate_multistart_candidates(bounds)
        # Record Phase 0 completion in history
        phase_record = {
            "phase": 0,
            "name": "normalization_setup",
            "strategy": self.normalizer.strategy,
            "timestamp": time.time(),
            "normalized_params_shape": self.normalized_params.shape,
            "has_bounds": bounds is not None,
        }
        self.phase_history.append(phase_record)

        # Pre-compile Jacobian function for Phase 2 performance
        self._setup_jacobian_fn()

        # Initialize residual weights from config if enabled
        self._setup_residual_weights()

    def _setup_residual_weights(self) -> None:
        """Initialize residual weights from config.

        This method sets up per-group weights for weighted least squares
        optimization. When enabled, residuals are weighted during loss
        computation, allowing users to assign different importance to
        different groups of data points.

        The weights are stored as JAX arrays for efficient lookup during
        loss computation.

        Notes
        -----
        Residual weighting is useful for:
        - Heteroscedastic data (varying noise levels)
        - Emphasizing certain regions of the data
        - Domain-specific weighting schemes (e.g., XPCS shear-sensitivity)

        Scope limitation: weighting is only applied in the L-BFGS warmup
        loss (``_create_warmup_loss_fn``). Phase 2 Gauss-Newton refinement
        (``_accumulate_jtj_jtr`` / ``_accumulate_jtj_jtr_scan``) does not
        currently incorporate these weights, so the effective optimization
        objective is weighted during warmup and unweighted afterward.
        """
        if not self.config.enable_residual_weighting:
            return

        if self.config.residual_weights is None:
            _logger.warning(
                "Residual weighting enabled but no weights provided. "
                "Residual weighting will be disabled.",
            )
            return

        # Convert to JAX arrays for efficient computation
        self._residual_weights_jax = jnp.asarray(
            self.config.residual_weights,
            dtype=jnp.float64,
        )

        _logger.info(
            f"Residual weighting enabled: n_weights={len(self._residual_weights_jax)}, "
            f"weight_range=[{float(self._residual_weights_jax.min()):.3f}, "
            f"{float(self._residual_weights_jax.max()):.3f}]. "
            "Note: weighting applies only during the L-BFGS warmup phase, "
            "not Phase 2 Gauss-Newton refinement -- see docstring above.",
        )

    def set_residual_weights(self, weights: np.ndarray) -> None:
        """Set residual weights for weighted least squares optimization.

        This method allows updating weights during optimization, for example
        when weights need to be recomputed based on current parameter estimates.

        Parameters
        ----------
        weights : np.ndarray
            Per-group weights of shape (n_groups,). Higher weights give more
            importance to residuals in that group. The group index for each
            data point is determined by the first column of x_data.

        Notes
        -----
        Weights must be positive. The weighted MSE is computed as:
            wMSE = sum(w[group_idx] * residuals^2) / sum(w[group_idx])
        """
        self._residual_weights_jax = jnp.asarray(weights, dtype=jnp.float64)

        _logger.debug(
            f"Updated residual weights: range=[{float(self._residual_weights_jax.min()):.3f}, "
            f"{float(self._residual_weights_jax.max()):.3f}]",
        )

    def _setup_jacobian_fn(self) -> None:
        """Pre-compile the Jacobian function for efficient Phase 2 computation.

        This method creates a JIT-compiled Jacobian function that avoids
        recompilation overhead in _compute_jacobian_chunk. The function is
        stored in self._jacobian_fn_compiled and reuses the normalized_model.

        The Jacobian is computed using reverse-mode AD (jacrev) vectorized
        over data points (vmap). Pre-compilation provides 15-25% speedup
        in Phase 2 by avoiding repeated function tracing.

        Notes
        -----
        Must be called after _setup_normalization sets self.normalized_model.
        The compiled function has signature: (params, x_chunk) -> J_chunk
        where J_chunk has shape (n_points, n_params).
        """
        if self.normalized_model is None:
            raise RuntimeError(
                "_setup_jacobian_fn must be called after _setup_normalization",
            )

        # Capture normalized_model in closure at setup time
        normalized_model = self.normalized_model

        def compute_jacobian_core(
            params: jnp.ndarray,
            x_chunk: jnp.ndarray,
        ) -> jnp.ndarray:
            """Core Jacobian computation using vmap + jacrev.

            Parameters
            ----------
            params : array_like
                Parameters in normalized space of shape (n_params,)
            x_chunk : array_like
                Data chunk of shape (n_points,) or (n_points, n_features)

            Returns
            -------
            J_chunk : array_like
                Jacobian matrix of shape (n_points, n_params)
            """

            def model_at_point(p, x_single):
                return normalized_model(x_single, *p)

            # jacrev computes gradient w.r.t. first argument (params)
            # vmap over x_chunk to get Jacobian row for each point
            return jax.vmap(lambda x: jax.jacrev(model_at_point, argnums=0)(params, x))(
                x_chunk,
            )

        # JIT compile the Jacobian function
        # Note: We don't use static_argnums here since params shape is fixed
        # but values change. The function traces once per unique shape combination.
        self._jacobian_fn_compiled = jax.jit(compute_jacobian_core)

        # Optionally warm up the compiled function with a small test
        # This triggers compilation eagerly rather than on first use
        if self.config.verbose >= 2:
            _logger.debug("Pre-compiled Jacobian function for Phase 2")

        # Also set up the cost-only function
        self._setup_cost_fn()

    def _setup_cost_fn(self) -> None:
        """Pre-compile the cost-only function for efficient Gauss-Newton iterations.

        This method creates a JIT-compiled function that computes only the sum of
        squared residuals (cost) without computing the Jacobian. This is used in
        `_gauss_newton_iteration` to evaluate the cost at new parameters after
        taking a step.

        The cost function is significantly faster than re-computing the full
        Jacobian, providing 20-30% speedup in Phase 2 by avoiding redundant
        model evaluations.

        Notes
        -----
        Must be called after _setup_normalization sets self.normalized_model.
        The compiled function has signature:
            (params, x_data, y_data, chunk_size) -> total_cost
        """
        if self.normalized_model is None:
            raise RuntimeError(
                "_setup_cost_fn must be called after _setup_normalization",
            )

        # Capture normalized_model in closure at setup time
        normalized_model = self.normalized_model

        def compute_chunk_cost(
            params: jnp.ndarray,
            x_chunk: jnp.ndarray,
            y_chunk: jnp.ndarray,
        ) -> float:
            """Compute cost for a single chunk.

            Parameters
            ----------
            params : array_like
                Parameters in normalized space of shape (n_params,)
            x_chunk : array_like
                Data chunk of shape (n_points,) or (n_points, n_features)
            y_chunk : array_like
                Target chunk of shape (n_points,)

            Returns
            -------
            cost : float
                Sum of squared residuals for this chunk
            """
            predictions = normalized_model(x_chunk, *params)
            residuals = y_chunk - predictions
            return jnp.sum(residuals**2)

        # JIT compile the chunk cost function
        self._cost_fn_compiled = jax.jit(compute_chunk_cost)

        if self.config.verbose >= 2:
            _logger.debug("Pre-compiled cost function for Phase 2")

        # Also set up scan functions for efficient loop-free accumulation
        self._setup_scan_functions()

    def _setup_scan_functions(self) -> None:
        """Initialize scan function infrastructure for efficient chunk accumulation.

        This method prepares the optimizer to use JAX lax.scan for chunk-based
        operations, eliminating Python loop overhead and enabling XLA fusion.

        The scan approach provides 10-15% speedup by:
        - Eliminating Python interpreter overhead in hot loops
        - Enabling XLA to fuse operations across chunks
        - Reducing memory allocation overhead

        Notes
        -----
        The actual scan body functions are created inline in _accumulate_jtj_jtr_scan
        and _compute_cost_scan, as they need to capture the current params in a closure.
        This setup method primarily validates that prerequisites are met.
        """
        if self.normalized_model is None:
            raise RuntimeError(
                "_setup_scan_functions must be called after _setup_normalization",
            )

        # Mark scan functions as available (bodies created inline in each method)
        self._jtj_scan_body_compiled = True  # Flag indicating scan is available
        self._cost_scan_body_compiled = True

        if self.config.verbose >= 2:
            _logger.debug("Scan-based accumulation enabled for Phase 2")

    def _use_scan_for_accumulation(self) -> bool:
        """Determine whether to use JAX scan or Python loops for chunk accumulation.

        Returns True if JAX lax.scan should be used, False for Python loops.

        The decision is based on:
        - config.loop_strategy: 'auto', 'scan', or 'loop'
        - For 'auto': GPU/TPU use scan (better XLA fusion), CPU uses loops
          (lower tracing overhead)

        Returns
        -------
        use_scan : bool
            True to use jax.lax.scan, False to use Python for loops

        Notes
        -----
        Benchmarks show:
        - CPU: Python loops are ~10x faster due to scan tracing overhead
        - GPU: Scan is faster due to kernel launch overhead in Python loops
        """
        strategy = self.config.loop_strategy

        if strategy == "scan":
            return True
        if strategy == "loop":
            return False
        # 'auto'
        # Detect backend from first device
        devices = jax.devices()
        if devices:
            platform = devices[0].platform
            # Use scan on GPU/TPU, loops on CPU
            return platform in ("gpu", "cuda", "rocm", "tpu")
        return False  # Default to loops if no devices detected

    def _prepare_chunked_data(
        self,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        chunk_size: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """Prepare data for scan by padding and reshaping into fixed-size chunks.

        Parameters
        ----------
        x_data : array_like
            Full x data of shape (n_points,) or (n_points, n_features)
        y_data : array_like
            Full y data of shape (n_points,)
        chunk_size : int
            Size of each chunk

        Returns
        -------
        x_chunks : array_like
            Reshaped x data of shape (n_chunks, chunk_size, ...)
        y_chunks : array_like
            Reshaped y data of shape (n_chunks, chunk_size)
        mask_chunks : array_like
            Validity mask of shape (n_chunks, chunk_size), 1.0 for valid points
        n_valid_points : int
            Number of valid (non-padded) points

        Notes
        -----
        Pads data with zeros to make n_points evenly divisible by chunk_size.
        Uses a mask to ensure padded points don't contribute to cost or gradients.
        """
        n_points = x_data.shape[0]
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        padded_size = n_chunks * chunk_size
        pad_size = padded_size - n_points

        # Create validity mask (1.0 for valid, 0.0 for padded)
        mask = jnp.ones(n_points, dtype=jnp.float64)

        if pad_size > 0:
            # Pad x_data with zeros
            if x_data.ndim == 1:
                x_padded = jnp.pad(
                    x_data,
                    (0, pad_size),
                    mode="constant",
                    constant_values=0,
                )
            else:
                x_padded = jnp.pad(
                    x_data,
                    ((0, pad_size), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            # Pad y_data with zeros (mask will exclude these)
            y_padded = jnp.pad(
                y_data,
                (0, pad_size),
                mode="constant",
                constant_values=0,
            )

            # Pad mask with zeros (invalid)
            mask = jnp.pad(mask, (0, pad_size), mode="constant", constant_values=0)
        else:
            x_padded = x_data
            y_padded = y_data

        # Reshape into chunks
        if x_padded.ndim == 1:
            x_chunks = x_padded.reshape(n_chunks, chunk_size)
        else:
            n_features = x_padded.shape[1]
            x_chunks = x_padded.reshape(n_chunks, chunk_size, n_features)

        y_chunks = y_padded.reshape(n_chunks, chunk_size)
        mask_chunks = mask.reshape(n_chunks, chunk_size)

        return x_chunks, y_chunks, mask_chunks, n_points

    @staticmethod
    def _content_fingerprint(data: jnp.ndarray) -> tuple:
        """Cheap content fingerprint used to invalidate _padded_cache.

        Not a full hash (would need an O(n) host transfer on every call for
        100M+-point datasets), but shape + first/last/sum/sum-of-squares
        together make an accidental collision between two distinct real
        datasets vanishingly unlikely -- sum-of-squares is nearly free here
        since sum() already forces an O(n) device reduction.
        """
        flat = data.ravel()
        if flat.size == 0:
            return (data.shape, 0.0, 0.0, 0.0, 0.0)
        return (
            data.shape,
            float(flat[0]),
            float(flat[-1]),
            float(jnp.sum(data)),
            # vdot fuses the multiply+reduce into one pass without
            # materializing a full-size (data * data) temporary.
            float(jnp.vdot(flat, flat)),
        )

    def _get_padded_data(
        self,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        chunk_size: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """Get padded flat arrays for dynamic_slice-based scan, with caching.

        B006: Avoids pre-stacking data into (n_chunks, chunk_size, ...) arrays.
        Instead, pads flat arrays minimally and caches them for reuse across
        iterations (x/y data is constant, only params change).

        Returns
        -------
        x_padded : array_like
            Padded x data, flat (padded_size,) or (padded_size, n_features)
        y_padded : array_like
            Padded y data, flat (padded_size,)
        n_points : int
            Number of valid (non-padded) points
        n_chunks : int
            Number of chunks
        """
        # Check cache validity using data content hash (not id() which can be
        # recycled by GC -- see clear_cache()/_padded_cache history for why
        # identity-based invalidation was rejected).
        cache = getattr(self, "_padded_cache", None)
        if cache is not None:
            c_hash_x, c_hash_y, c_cs, c_xp, c_yp, c_np, c_nc = cache
            x_hash = self._content_fingerprint(x_data)
            y_hash = self._content_fingerprint(y_data)
            if c_hash_x == x_hash and c_hash_y == y_hash and c_cs == chunk_size:
                return c_xp, c_yp, c_np, c_nc

        n_points = x_data.shape[0]
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        pad_size = n_chunks * chunk_size - n_points

        if pad_size > 0:
            if x_data.ndim == 1:
                x_padded = jnp.pad(x_data, (0, pad_size))
            else:
                x_padded = jnp.pad(x_data, ((0, pad_size), (0, 0)))
            y_padded = jnp.pad(y_data, (0, pad_size))
        else:
            x_padded = x_data
            y_padded = y_data

        x_hash = self._content_fingerprint(x_data)
        y_hash = self._content_fingerprint(y_data)
        self._padded_cache = (
            x_hash,
            y_hash,
            chunk_size,
            x_padded,
            y_padded,
            n_points,
            n_chunks,
        )
        return x_padded, y_padded, n_points, n_chunks

    def clear_cache(self) -> None:
        """Release cached padded arrays to free memory.

        Call this after optimization completes or when reusing the optimizer
        with different data. The cache is automatically invalidated when
        data identity changes, but explicit clearing frees memory sooner.
        """
        self._padded_cache = None

    def _accumulate_jtj_jtr_scan(
        self,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, float]:
        """Accumulate J^T J and J^T r using JAX lax.scan for efficiency.

        This is the scan-based version of the chunk accumulation loop,
        providing 10-15% speedup by eliminating Python loop overhead.

        Parameters
        ----------
        x_data : array_like
            Full x data of shape (n_points,) or (n_points, n_features)
        y_data : array_like
            Full y data of shape (n_points,)
        params : array_like
            Current parameters in normalized space of shape (n_params,)

        Returns
        -------
        JTJ : array_like
            Accumulated J^T J of shape (n_params, n_params)
        JTr : array_like
            Accumulated J^T r of shape (n_params,)
        total_cost : float
            Total sum of squared residuals

        Notes
        -----
        Uses jax.lax.scan instead of Python for loop, enabling XLA fusion
        across chunks and eliminating interpreter overhead.
        Uses masking to handle padding correctly for non-divisible data sizes.
        """
        chunk_size = self.config.chunk_size
        n_params = len(params)

        # B006: Use cached padded flat arrays + dynamic_slice instead of
        # pre-stacking into (n_chunks, chunk_size, ...) arrays.
        x_padded, y_padded, n_points, n_chunks = self._get_padded_data(
            x_data,
            y_data,
            chunk_size,
        )

        # Capture functions and data for scan body closure
        normalized_model = self.normalized_model
        jacobian_fn = self._jacobian_fn_compiled
        x_ndim = x_padded.ndim

        chunk_indices = jnp.arange(n_chunks)

        def scan_body(carry, chunk_idx):
            """Scan body using dynamic_slice from flat padded arrays."""
            JTJ, JTr, total_cost = carry
            start = chunk_idx * chunk_size

            # Dynamic slice from flat padded arrays (no pre-stacking needed)
            if x_ndim == 1:
                x_chunk = jax.lax.dynamic_slice(x_padded, (start,), (chunk_size,))
            else:
                x_chunk = jax.lax.dynamic_slice(
                    x_padded,
                    (start, 0),
                    (chunk_size, x_padded.shape[1]),
                )
            y_chunk = jax.lax.dynamic_slice(y_padded, (start,), (chunk_size,))

            # Compute mask on-the-fly (tiny chunk_size array, not full n_points)
            mask = jnp.where(start + jnp.arange(chunk_size) < n_points, 1.0, 0.0)

            # See _fill_padded_rows: guards NaN/Inf from evaluating the model
            # at padded rows before masking. The mask below still excludes
            # padded rows from the accumulated sums either way.
            safe_x_chunk = _fill_padded_rows(x_chunk, mask)

            # Compute predictions and residuals
            predictions = normalized_model(safe_x_chunk, *params)
            residuals = y_chunk - predictions

            # Apply mask to residuals (zero out padded points)
            masked_residuals = residuals * mask

            # Compute Jacobian
            if jacobian_fn is not None:
                J_chunk = jacobian_fn(params, safe_x_chunk)
            else:

                def model_at_x(p, x_single):
                    return normalized_model(x_single, *p)

                J_chunk = jax.vmap(
                    lambda x: jax.jacrev(model_at_x, argnums=0)(params, x),
                )(safe_x_chunk)

            # Apply mask to Jacobian rows (zero out padded point gradients)
            masked_J = J_chunk * mask[:, None]

            # Accumulate with masked values
            JTJ_new = JTJ + masked_J.T @ masked_J
            JTr_new = JTr + masked_J.T @ masked_residuals
            cost_new = total_cost + jnp.sum(masked_residuals**2)

            return (JTJ_new, JTr_new, cost_new), None

        # Initialize carry
        init_carry = (
            jnp.zeros((n_params, n_params)),
            jnp.zeros(n_params),
            jnp.array(0.0),
        )

        # Run scan over chunk indices (not pre-stacked data arrays)
        (JTJ, JTr, total_cost), _ = jax.lax.scan(
            scan_body,
            init_carry,
            chunk_indices,
        )

        # Store accumulators for checkpointing
        self.phase2_JTJ_accumulator = JTJ
        self.phase2_JTr_accumulator = JTr

        return JTJ, JTr, float(total_cost)

    def _compute_cost_scan(
        self,
        params: jnp.ndarray,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
    ) -> float:
        """Compute total cost using JAX lax.scan for efficiency.

        This is the scan-based version of cost computation, providing
        10-15% speedup by eliminating Python loop overhead.

        Parameters
        ----------
        params : array_like
            Parameters in normalized space of shape (n_params,)
        x_data : array_like
            Full x data of shape (n_points,) or (n_points, n_features)
        y_data : array_like
            Full y data of shape (n_points,)

        Returns
        -------
        total_cost : float
            Total sum of squared residuals

        Notes
        -----
        Uses masking to handle padding correctly for non-divisible data sizes.
        """
        chunk_size = self.config.chunk_size

        # B006: Use cached padded flat arrays + dynamic_slice
        x_padded, y_padded, n_points, n_chunks = self._get_padded_data(
            x_data,
            y_data,
            chunk_size,
        )

        # Capture model and data for scan body closure
        normalized_model = self.normalized_model
        x_ndim = x_padded.ndim

        chunk_indices = jnp.arange(n_chunks)

        def scan_body(carry, chunk_idx):
            """Scan body for masked cost computation using dynamic_slice."""
            start = chunk_idx * chunk_size

            if x_ndim == 1:
                x_chunk = jax.lax.dynamic_slice(x_padded, (start,), (chunk_size,))
            else:
                x_chunk = jax.lax.dynamic_slice(
                    x_padded,
                    (start, 0),
                    (chunk_size, x_padded.shape[1]),
                )
            y_chunk = jax.lax.dynamic_slice(y_padded, (start,), (chunk_size,))

            mask = jnp.where(start + jnp.arange(chunk_size) < n_points, 1.0, 0.0)

            # See _fill_padded_rows: guards NaN/Inf from evaluating the model
            # at padded rows before masking.
            safe_x_chunk = _fill_padded_rows(x_chunk, mask)
            predictions = normalized_model(safe_x_chunk, *params)
            residuals = y_chunk - predictions
            masked_residuals = residuals * mask
            return carry + jnp.sum(masked_residuals**2), None

        # Initialize carry
        init_carry = jnp.array(0.0)

        # Run scan over chunk indices
        total_cost, _ = jax.lax.scan(
            scan_body,
            init_carry,
            chunk_indices,
        )

        return float(total_cost)

    def _compute_cost_only(
        self,
        params: jnp.ndarray,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
    ) -> float:
        """Compute total cost (sum of squared residuals) without Jacobian.

        This method efficiently computes only the cost at the given parameters,
        without computing the Jacobian. Used in Gauss-Newton iterations to
        evaluate the cost at new parameters after taking a step.

        Parameters
        ----------
        params : array_like
            Parameters in normalized space of shape (n_params,)
        x_data : array_like
            Full x data of shape (n_points,) or (n_points, n_features)
        y_data : array_like
            Full y data of shape (n_points,)

        Returns
        -------
        total_cost : float
            Total sum of squared residuals

        Performance
        -----------
        Dispatches between JAX scan (GPU/TPU) and Python loops (CPU) based on
        config.loop_strategy for optimal performance on each backend.
        """
        # Dispatch based on backend: scan for GPU/TPU, loops for CPU
        if self._use_scan_for_accumulation():
            # Use JAX scan for GPU/TPU (better XLA fusion, reduced kernel launches)
            return self._compute_cost_scan(params, x_data, y_data)

        # Use Python loops for CPU (lower tracing overhead)
        chunk_size = self.config.chunk_size
        n_points = len(x_data)
        total_cost = 0.0

        for i in range(0, n_points, chunk_size):
            x_chunk = x_data[i : i + chunk_size]
            y_chunk = y_data[i : i + chunk_size]

            # Use pre-compiled cost function if available
            if self._cost_fn_compiled is not None:
                chunk_cost = self._cost_fn_compiled(params, x_chunk, y_chunk)
            else:
                # Fallback to inline computation
                predictions = self.normalized_model(x_chunk, *params)
                residuals = y_chunk - predictions
                chunk_cost = float(jnp.sum(residuals**2))

            total_cost += chunk_cost

        return total_cost

    def _compute_cost_with_variance_regularization(
        self,
        params: jnp.ndarray,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
    ) -> float:
        """Compute total cost including group variance regularization.

        This is a convenience method that computes the MSE cost plus any
        group variance regularization penalty.

        Parameters
        ----------
        params : array_like
            Parameters in normalized space of shape (n_params,)
        x_data : array_like
            Full x data of shape (n_points,) or (n_points, n_features)
        y_data : array_like
            Full y data of shape (n_points,)

        Returns
        -------
        total_cost : float
            Total cost including MSE and variance regularization
        """
        # Base MSE cost
        total_cost = self._compute_cost_only(params, x_data, y_data)

        # Add group variance regularization if enabled
        if (
            self.config.enable_group_variance_regularization
            and self.config.group_variance_indices
        ):
            n_points = len(x_data)
            var_lambda = self.config.group_variance_lambda
            for start, end in self.config.group_variance_indices:
                group_params = params[start:end]
                group_var = jnp.var(group_params)
                total_cost += (
                    var_lambda
                    * float(jnp.where(jnp.isfinite(group_var), group_var, 0.0))
                    * n_points
                )

        return total_cost

    def _generate_multistart_candidates(
        self,
        bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> None:
        """Generate multi-start candidates using LHS or other sampling methods.

        Parameters
        ----------
        bounds : tuple of array_like, optional
            Parameter bounds as (lb, ub).

        Notes
        -----
        Generates n_starts candidates using the configured sampler (LHS, Sobol, Halton).
        If center_on_p0 is True, centers samples around the initial guess p0.
        Stores candidates in self.multistart_candidates.
        """
        import jax
        import numpy as np

        n_params = len(self.original_p0)
        n_starts = self.config.n_starts

        # Get sampler function
        sampler = get_sampler(self.config.multistart_sampler)

        # Generate samples in [0, 1] hypercube
        rng_key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        samples = sampler(n_starts, n_params, rng_key=rng_key)
        samples = np.asarray(samples)

        # Scale samples to bounds or around p0
        if bounds is not None and self.config.center_on_p0:
            # Center samples around p0 within bounds
            lb, ub = np.asarray(bounds[0]), np.asarray(bounds[1])
            p0 = np.asarray(self.original_p0)

            # Scale factor controls how much of the bounds to explore
            scale = self.config.scale_factor

            # Center around p0 with scaled range
            range_half = (ub - lb) * scale / 2
            center_lb = np.maximum(lb, p0 - range_half)
            center_ub = np.minimum(ub, p0 + range_half)

            # Scale samples to centered bounds
            candidates = center_lb + samples * (center_ub - center_lb)

        elif bounds is not None:
            # Scale samples to full bounds
            lb, ub = np.asarray(bounds[0]), np.asarray(bounds[1])
            candidates = lb + samples * (ub - lb)

        else:
            # No bounds: scale around p0 with heuristic range
            p0 = np.asarray(self.original_p0)
            scale_factor = self.config.scale_factor

            # Use p0 magnitude as scale (avoid zero)
            p0_scale = np.abs(p0) + 0.1
            range_half = p0_scale * scale_factor

            # Center samples around p0
            candidates = p0 + (samples - 0.5) * 2 * range_half

        self.multistart_candidates = jnp.asarray(candidates)

    def _run_tournament_selection(
        self,
        data_source: tuple[jnp.ndarray, jnp.ndarray],
        model: Callable,
    ) -> jnp.ndarray:
        """Run tournament selection to find the best starting candidate.

        Parameters
        ----------
        data_source : tuple
            Data as (x_data, y_data).
        model : Callable
            Model function.

        Returns
        -------
        best_params : array_like
            Best starting parameters in normalized space.
        """
        import numpy as np

        # Create GlobalOptimizationConfig from HybridStreamingConfig
        global_config = GlobalOptimizationConfig(
            n_starts=self.config.n_starts,
            elimination_rounds=self.config.elimination_rounds,
            elimination_fraction=self.config.elimination_fraction,
            batches_per_round=self.config.batches_per_round,
        )

        # Convert candidates to normalized space
        normalized_candidates = np.array(
            [
                np.asarray(self.normalizer.normalize(jnp.asarray(c)))
                for c in self.multistart_candidates
            ],
        )

        # Create tournament selector
        self.tournament_selector = TournamentSelector(
            candidates=normalized_candidates,
            config=global_config,
        )

        # Create data batch generator
        x_data, y_data = data_source
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        chunk_size = self.config.chunk_size
        n_points = len(x_data)

        def data_batch_generator():
            # Shuffle indices for each epoch
            indices = np.arange(n_points)
            np.random.shuffle(indices)

            # Yield chunks
            for i in range(0, n_points, chunk_size):
                batch_idx = indices[i : i + chunk_size]
                yield x_data[batch_idx], y_data[batch_idx]

            # Repeat if needed for more rounds
            for _ in range(
                self.config.elimination_rounds * self.config.batches_per_round,
            ):
                np.random.shuffle(indices)
                for i in range(0, n_points, chunk_size):
                    batch_idx = indices[i : i + chunk_size]
                    yield x_data[batch_idx], y_data[batch_idx]

        # Run tournament with normalized model
        try:
            best_candidates = self.tournament_selector.run_tournament(
                data_batch_iterator=data_batch_generator(),
                model=self.normalized_model,
                top_m=1,
            )

            # Store diagnostics
            self.multistart_diagnostics = self.tournament_selector.get_diagnostics()

            # Return best candidate
            best_normalized = best_candidates[0]
            self.multistart_best_candidate = best_normalized

            return jnp.asarray(best_normalized)

        except Exception as e:
            # Fallback: use original p0
            import warnings

            warnings.warn(
                f"Tournament selection failed: {e}. Using p0 as starting point.",
            )
            self.multistart_diagnostics = {"error": str(e), "fallback": True}
            return self.normalized_params

    def _create_lbfgs_optimizer(
        self,
        params: jnp.ndarray,
        initial_step_size: float | None = None,
    ) -> tuple[optax.GradientTransformationExtraArgs, optax.OptState]:
        """Create L-BFGS optimizer with optax for Phase 1 warmup.

        L-BFGS provides 5-10x faster convergence to the basin of attraction
        compared to first-order warmup by using approximate Hessian information.

        Parameters
        ----------
        params : array_like
            Initial parameters in normalized space
        initial_step_size : float, optional
            Override initial step size for L-BFGS line search.
            If None, uses config.lbfgs_initial_step_size.

        Returns
        -------
        optimizer : optax.GradientTransformationExtraArgs
            L-BFGS optimizer instance with line search
        opt_state : optax.OptState
            Initial optimizer state

        Notes
        -----
        Uses optax.lbfgs with backtracking line search for step acceptance.
        The history size is configured via config.lbfgs_history_size (default 10).

        Cold start scaffolding: During the first m iterations (where m is the
        history size), the Hessian approximation is poor (starts as identity).
        The initial_step_size parameter controls how conservative the first
        steps are before the history buffer fills.

        Examples
        --------
        Basic L-BFGS with default settings:
        >>> optimizer, state = self._create_lbfgs_optimizer(params)

        L-BFGS with small initial step (exploration mode):
        >>> optimizer, state = self._create_lbfgs_optimizer(params, initial_step_size=0.1)

        L-BFGS with large initial step (refinement mode):
        >>> optimizer, state = self._create_lbfgs_optimizer(params, initial_step_size=1.0)
        """
        # Determine initial step size (learning rate) for line search
        step_size = (
            initial_step_size
            if initial_step_size is not None
            else self.config.lbfgs_initial_step_size
        )

        # Configure line search based on config
        line_search_type = self.config.lbfgs_line_search
        if line_search_type == "backtracking":
            # Backtracking line search with Armijo condition
            linesearch = optax.scale_by_backtracking_linesearch(
                max_backtracking_steps=20,
                slope_rtol=1e-4,  # Armijo condition parameter
                decrease_factor=0.8,
                increase_factor=1.5,
                max_learning_rate=step_size,
            )
        else:
            # Default to zoom linesearch (Wolfe conditions)
            # This is the default optax.lbfgs linesearch
            linesearch = optax.scale_by_zoom_linesearch(
                max_linesearch_steps=20,
                initial_guess_strategy="one",
            )

        # Create L-BFGS optimizer
        # Note: optax.lbfgs uses the learning_rate to scale the initial step guess
        optimizer = optax.lbfgs(
            learning_rate=step_size,
            memory_size=self.config.lbfgs_history_size,
            scale_init_precond=True,  # Use scaled identity for cold start
            linesearch=linesearch,
        )

        # Chain with gradient clipping if configured
        if self.config.gradient_clip_value is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.gradient_clip_value),
                optimizer,
            )

        # Initialize optimizer state
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
    ) -> tuple[jnp.ndarray, float, float, optax.OptState, bool]:
        """Perform single L-BFGS optimization step with cold start scaffolding.

        Parameters
        ----------
        params : array_like
            Current parameters in normalized space
        opt_state : optax.OptState
            Current optimizer state
        optimizer : optax.GradientTransformationExtraArgs
            L-BFGS optimizer instance
        loss_fn : Callable
            Loss function
        x_batch : array_like
            Independent variable batch
        y_batch : array_like
            Dependent variable batch
        iteration : int
            Current iteration number (for cold start detection)

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
        line_search_failed : bool
            True if line search failed to find acceptable step

        Notes
        -----
        Uses jax.value_and_grad for efficient loss and gradient computation.
        Includes NaN/Inf validation if enabled in config.

        Cold start scaffolding: During the first m iterations (history_size),
        the step is scaled by lbfgs_initial_step_size to prevent overshooting
        when the Hessian approximation is poor.
        """
        # Validate input parameters
        self._validate_numerics(params, context="at L-BFGS step input")

        # Compute loss and gradient
        loss_value, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)

        # Validate loss and gradients
        if not self._validate_numerics(
            params,
            loss=float(loss_value),
            gradients=grads,
            context="in L-BFGS step",
        ):
            # Handle numerical issues
            if (
                hasattr(self.config, "enable_fault_tolerance")
                and self.config.enable_fault_tolerance
            ):
                # Return current params unchanged (fallback)
                return params, float("inf"), float("inf"), opt_state, True
            raise ValueError("Numerical issues detected in L-BFGS step")

        # Compute gradient norm
        grad_norm = jnp.linalg.norm(grads)

        # L-BFGS requires a value_fn for line search
        def value_fn(p):
            return loss_fn(p, x_batch, y_batch)

        # Apply optimizer updates (L-BFGS with line search)
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
            # Line search can fail in some cases
            if self.config.verbose >= 2:
                _logger.warning(f"L-BFGS line search failed: {e}")
            # Fall back to gradient descent step
            updates = -self.config.lbfgs_initial_step_size * grads
            new_opt_state = opt_state
            line_search_failed = True

            # Record line search failure in telemetry
            telemetry = get_defense_telemetry()
            telemetry.record_lbfgs_line_search_failure(iteration, str(e))

        # Layer 4: Trust Region Constraint - clip update magnitude (JIT-compatible)
        if self.config.enable_step_clipping:
            # Track original norm for telemetry before clipping
            original_update_norm = float(jnp.linalg.norm(updates))
            max_norm = self.config.max_warmup_step_size

            updates = self._clip_update_norm(updates, max_norm)

            # Record Layer 4 telemetry if clipping occurred
            if original_update_norm > max_norm:
                telemetry = get_defense_telemetry()
                telemetry.record_layer4_clip(
                    original_norm=original_update_norm,
                    max_norm=max_norm,
                )

        new_params = optax.apply_updates(params, updates)

        # Clip to bounds if available. Phase 2's Gauss-Newton step already
        # does this (see _gauss_newton_iteration); L-BFGS was missing it,
        # letting unconstrained descent drift outside the user's bounds in
        # normalized space -- exposed once best-parameter tracking above was
        # fixed to correctly pair loss with the point it was measured at.
        if self.normalized_bounds is not None:
            lb, ub = self.normalized_bounds
            new_params = jnp.clip(new_params, lb, ub)

        # Validate updated parameters
        if not self._validate_numerics(new_params, context="after L-BFGS update"):
            # Fallback: keep old parameters
            if (
                hasattr(self.config, "enable_fault_tolerance")
                and self.config.enable_fault_tolerance
            ):
                return params, float(loss_value), float(grad_norm), opt_state, True
            raise ValueError("NaN/Inf in parameters after L-BFGS update")

        # Track best parameters globally. loss_value was computed at the INPUT
        # params (see jax.value_and_grad(loss_fn)(params, ...) above), not at
        # new_params -- pairing it with new_params here mislabeled the just-
        # taken (possibly divergent) step with the previous step's loss.
        # See _update_global_best for the MSE->SSR unit conversion this needs.
        self._update_global_best(float(loss_value), params, len(x_batch))

        # Store optimizer state for checkpointing
        self.phase1_optimizer_state = new_opt_state

        # Record history buffer fill event (once when history is fully populated)
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

    def _create_warmup_loss_fn(self) -> Callable:
        """Create loss function for warmup phase.

        Returns
        -------
        loss_fn : Callable
            Loss function with signature: loss_fn(params, x_batch, y_batch) -> scalar_loss
            Operates in normalized parameter space and returns mean squared residuals.

        Notes
        -----
        The loss function is JIT-compiled for performance.

        When `enable_group_variance_regularization=True`, the loss becomes:
            L = MSE + group_variance_lambda * sum(Var(group_i))
        where each group_i is defined by `group_variance_indices`.
        This prevents per-angle parameters from absorbing angle-dependent physical signals.

        When `enable_residual_weighting=True`, the MSE becomes weighted MSE:
            wMSE = sum(w[group_idx] * residuals^2) / sum(w[group_idx])
        where w[group_idx] are per-group weights. The group index is determined
        by the first column of x_data.
        """
        # Use normalized model wrapper
        normalized_model = self.normalized_model

        # Capture config values for closure
        enable_var_reg = self.config.enable_group_variance_regularization
        var_lambda = self.config.group_variance_lambda
        var_indices = self.config.group_variance_indices

        # Capture residual weighting state for closure
        # Note: We capture the current state; if weights are updated via set_residual_weights,
        # a new loss function should be created by calling _create_warmup_loss_fn again
        enable_weighting = (
            self.config.enable_residual_weighting
            and self._residual_weights_jax is not None
        )
        residual_weights = self._residual_weights_jax

        # Pre-convert group indices to JAX arrays for lax.fori_loop.
        # Each group is extracted via dynamic_slice with a fixed max_group_size,
        # then masked to the actual group size for correct variance computation.
        if enable_var_reg and var_indices:
            group_starts = jnp.array([s for s, _e in var_indices], dtype=jnp.int32)
            group_sizes = jnp.array([e - s for s, e in var_indices], dtype=jnp.int32)
            n_groups = len(var_indices)
            max_group_size = max(e - s for s, e in var_indices)
        else:
            group_starts = jnp.zeros(0, dtype=jnp.int32)
            group_sizes = jnp.zeros(0, dtype=jnp.int32)
            n_groups = 0
            max_group_size = 0

        @jax.jit
        def loss_fn(
            params: jnp.ndarray,
            x_batch: jnp.ndarray,
            y_batch: jnp.ndarray,
        ) -> jnp.ndarray:
            predictions = normalized_model(x_batch, *params)
            residuals = y_batch - predictions

            # Base loss: weighted or unweighted MSE
            if enable_weighting:
                # group_idx comes from column 0 of x_batch (see docstring);
                # x_batch.ndim is static at trace time, so this check runs
                # once per distinct input shape, not per element. Without
                # it, 1D xdata (the common case) hits a cryptic
                # "too many indices for array" deep inside this jit'd
                # function instead of a clear, actionable error.
                if x_batch.ndim < 2:
                    raise ValueError(
                        "enable_residual_weighting=True requires xdata "
                        "shaped (N, k) with k>=1 -- column 0 must hold "
                        f"each point's group index. Got 1D xdata of shape "
                        f"{x_batch.shape}. Disable residual weighting or "
                        "provide xdata as a 2D array.",
                    )
                group_idx = x_batch[:, 0].astype(jnp.int32)
                assert residual_weights is not None
                weights = residual_weights[group_idx]
                base_loss = jnp.sum(weights * residuals**2) / jnp.sum(weights)
            else:
                base_loss = jnp.mean(residuals**2)

            # Variance regularization via lax.fori_loop (fixed XLA trace
            # regardless of number of groups)
            if enable_var_reg and var_indices:
                # dynamic_slice clamps its start index to keep a fixed-size
                # window in bounds. For a group whose (start + max_group_size)
                # overruns len(params) -- e.g. the trailing group when group
                # sizes are unequal -- that silently shifts the window to
                # include elements from the PREVIOUS group instead of raising.
                # Pad params so every max_group_size window fits without ever
                # needing to clamp; the mask below already ignores the padding.
                padded_params = jnp.concatenate(
                    [params, jnp.zeros(max_group_size, dtype=params.dtype)],
                )

                def var_body(i, penalty):
                    start = group_starts[i]
                    size = group_sizes[i]
                    # Extract with fixed max_group_size, mask to actual size
                    group_params = jax.lax.dynamic_slice(
                        padded_params,
                        (start,),
                        (max_group_size,),
                    )
                    mask = jnp.arange(max_group_size) < size
                    # Masked variance: Var = E[x^2] - E[x]^2 over valid elements
                    n = jnp.maximum(size, 1)  # avoid division by zero
                    mean_val = jnp.sum(jnp.where(mask, group_params, 0.0)) / n
                    sq_diff = jnp.where(mask, (group_params - mean_val) ** 2, 0.0)
                    group_var = jnp.sum(sq_diff) / n
                    return penalty + group_var

                variance_penalty = jax.lax.fori_loop(
                    0,
                    n_groups,
                    var_body,
                    jnp.array(0.0),
                )
                return base_loss + var_lambda * variance_penalty

            return base_loss

        return loss_fn

    @staticmethod
    def _clip_update_norm(updates: jnp.ndarray, max_norm: float) -> jnp.ndarray:
        """Clip parameter update vector to maximum L2 norm (JIT-compatible).

        This is Layer 4 of the 4-layer defense strategy for warmup divergence
        prevention. It limits the magnitude of warmup updates to prevent
        large steps that could destabilize optimization when near an optimum.

        Parameters
        ----------
        updates : array_like
            Parameter updates from optimizer
        max_norm : float
            Maximum allowed L2 norm for the update vector

        Returns
        -------
        clipped_updates : array_like
            Updates with L2 norm <= max_norm. If original norm <= max_norm,
            returns updates unchanged. Otherwise scales updates to have
            exactly max_norm.

        Notes
        -----
        Uses jnp.minimum for JIT compatibility - no Python conditionals.
        Small epsilon (1e-10) added to denominator to prevent division by zero.
        """
        update_norm = jnp.linalg.norm(updates)
        scale = jnp.minimum(1.0, max_norm / (update_norm + 1e-10))
        return updates * scale

    def _check_phase1_switch_criteria(
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

        # Check max iterations criterion. Unreachable via the normal Phase 1
        # driver loop (`for iteration in range(config.max_warmup_iterations)`
        # never yields iteration == max_warmup_iterations); the post-loop
        # fallback in _run_phase1_warmup covers that exit instead. Left as a
        # defensive check in case this method is ever called from elsewhere.
        if "max_iter" in active_criteria:
            if iteration >= self.config.max_warmup_iterations:
                return True, "Maximum warmup iterations reached"

        # Check gradient norm criterion
        if "gradient" in active_criteria:
            if grad_norm < self.config.gradient_norm_threshold:
                return True, "Gradient norm below threshold"

        # Check loss plateau criterion
        if "plateau" in active_criteria:
            # Compute relative loss change
            eps = jnp.finfo(jnp.float64).eps
            relative_change = jnp.abs(current_loss - prev_loss) / (
                jnp.abs(prev_loss) + eps
            )

            if relative_change < self.config.loss_plateau_threshold:
                return True, "Loss plateau detected"

        # No switch criteria met
        return False, ""

    # =========================================================================
    # Phase 1 Helper Methods (extracted for complexity reduction)
    # =========================================================================

    def _check_warm_start_and_record(
        self,
        current_params: jnp.ndarray,
        loss_fn: Callable,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        telemetry: Any,
    ) -> tuple[bool, dict[str, Any] | None, float, float]:
        """Check warm start detection (Layer 1 defense).

        Computes initial loss and relative loss, checks if we're already
        near optimal and can skip L-BFGS warmup.

        Parameters
        ----------
        current_params : jnp.ndarray
            Current parameters in normalized space.
        loss_fn : Callable
            Loss function.
        x_data : jnp.ndarray
            Input data.
        y_data : jnp.ndarray
            Output data.
        telemetry : DefenseLayerTelemetry
            Telemetry recorder.

        Returns
        -------
        should_exit : bool
            True if warm start detected and should skip warmup.
        result : dict or None
            If should_exit, the result dict to return. Otherwise None.
        initial_loss : float
            Computed initial loss value.
        relative_loss : float
            Computed relative loss (loss / y_variance).
        """
        initial_loss = float(loss_fn(current_params, x_data, y_data))
        y_variance = float(jnp.var(y_data))
        relative_loss = initial_loss / (y_variance + 1e-10)

        # Store for Layer 3 cost-increase guard
        self._warmup_initial_loss = initial_loss
        self._warmup_relative_loss = relative_loss

        # Log diagnostic info
        if self.config.verbose >= 2:
            _logger.debug(
                f"Phase 1 initial assessment: loss={initial_loss:.6e}, "
                f"y_var={y_variance:.6e}, relative_loss={relative_loss:.6e}",
            )

        # Check warm start threshold
        if (
            self.config.enable_warm_start_detection
            and relative_loss < self.config.warm_start_threshold
        ):
            # Record Layer 1 telemetry
            telemetry.record_layer1_trigger(
                relative_loss=relative_loss,
                threshold=self.config.warm_start_threshold,
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
            self.phase_history.append(phase_record)

            if self.config.verbose >= 1:
                _logger.info(
                    f"Phase 1: Skipping L-BFGS warmup - warm start detected "
                    f"(relative_loss={relative_loss:.4e})",
                )

            result = {
                "final_params": current_params,
                "best_params": current_params,
                "best_loss": initial_loss,
                "final_loss": initial_loss,
                "iterations": 0,
                "switch_reason": "Warm start detected - skipping L-BFGS warmup",
                "warm_start": True,
                "relative_loss": relative_loss,
            }
            return True, result, initial_loss, relative_loss

        return False, None, initial_loss, relative_loss

    def _select_adaptive_step_size(
        self,
        relative_loss: float,
        telemetry: Any,
    ) -> tuple[float, str]:
        """Select adaptive initial step size for L-BFGS (Layer 2 defense).

        Parameters
        ----------
        relative_loss : float
            Relative loss (loss / y_variance).
        telemetry : DefenseLayerTelemetry
            Telemetry recorder.

        Returns
        -------
        initial_step : float
            Selected initial step size.
        lr_mode : str
            Mode name ('refinement', 'careful', 'exploration', 'fixed').
        """
        if self.config.enable_adaptive_warmup_lr:
            if relative_loss < 0.1:
                # Refinement mode: near optimal, use large step for Newton-like speed
                initial_step = self.config.lbfgs_refinement_step_size
                lr_mode = "refinement"
            elif relative_loss < 1.0:
                # Careful mode: reasonable starting point
                initial_step = self.config.lbfgs_careful_step_size
                lr_mode = "careful"
            else:
                # Exploration mode: far from optimal, use small step to prevent overshoot
                initial_step = self.config.lbfgs_exploration_step_size
                lr_mode = "exploration"

            self._warmup_lr_mode = lr_mode

            # Record Layer 2 telemetry
            telemetry.record_layer2_lr_mode(mode=lr_mode, relative_loss=relative_loss)

            if self.config.verbose >= 2:
                _logger.debug(
                    f"Phase 1 L-BFGS adaptive step: mode={lr_mode}, step={initial_step:.2f}, "
                    f"relative_loss={relative_loss:.4e}",
                )
        else:
            initial_step = self.config.lbfgs_initial_step_size
            lr_mode = "fixed"
            self._warmup_lr_mode = lr_mode
            # Record fixed mode telemetry
            telemetry.record_layer2_lr_mode(mode=lr_mode, relative_loss=relative_loss)

        return initial_step, lr_mode

    def _check_cost_increase_guard(
        self,
        iteration: int,
        loss_value: float,
        best_loss: float,
        best_params: jnp.ndarray,
        lr_mode: str,
        relative_loss: float,
        telemetry: Any,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Check cost increase guard (Layer 3 defense).

        Parameters
        ----------
        iteration : int
            Current iteration number.
        loss_value : float
            Current loss value.
        best_loss : float
            Best loss seen so far.
        best_params : jnp.ndarray
            Best parameters seen so far.
        lr_mode : str
            Learning rate mode.
        relative_loss : float
            Initial relative loss.
        telemetry : DefenseLayerTelemetry
            Telemetry recorder.

        Returns
        -------
        should_exit : bool
            True if cost guard triggered.
        result : dict or None
            If should_exit, the result dict to return. Otherwise None.
        """
        if not self.config.enable_cost_guard or iteration <= 0:
            return False, None

        if self._warmup_initial_loss == 0:
            return False, None

        cost_increase_ratio = loss_value / self._warmup_initial_loss
        cost_threshold = 1.0 + self.config.cost_increase_tolerance

        if cost_increase_ratio > cost_threshold:
            # Record Layer 3 telemetry
            telemetry.record_layer3_trigger(
                cost_ratio=cost_increase_ratio,
                tolerance=self.config.cost_increase_tolerance,
                iteration=iteration,
            )

            # Loss increased beyond tolerance - abort and return best
            if self.config.verbose >= 1:
                _logger.warning(
                    f"Phase 1: Cost increase guard triggered at iteration "
                    f"{iteration + 1}. Loss {loss_value:.6e} > "
                    f"{self._warmup_initial_loss:.6e} * {cost_threshold:.2f}. "
                    f"Reverting to best params (loss={best_loss:.6e}).",
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
            self.phase_history.append(phase_record)

            result = {
                "final_params": best_params,  # Return BEST, not current
                "best_params": best_params,
                "best_loss": best_loss,
                "final_loss": loss_value,
                "iterations": iteration + 1,
                "switch_reason": "Cost increase guard triggered",
                "cost_guard_triggered": True,
                "cost_increase_ratio": cost_increase_ratio,
                "lr_mode": lr_mode,
                "relative_loss": relative_loss,
            }
            return True, result

        return False, None

    def _build_phase1_result(
        self,
        final_params: jnp.ndarray,
        best_params: jnp.ndarray,
        best_loss: float,
        final_loss: float,
        iterations: int,
        switch_reason: str,
        lr_mode: str,
        relative_loss: float,
        record_history: bool = True,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """Build Phase 1 result dict and optionally record to phase history.

        Parameters
        ----------
        final_params : jnp.ndarray
            Final parameters.
        best_params : jnp.ndarray
            Best parameters found.
        best_loss : float
            Best loss value.
        final_loss : float
            Final loss value.
        iterations : int
            Number of iterations performed.
        switch_reason : str
            Reason for switching/ending.
        lr_mode : str
            Learning rate mode used.
        relative_loss : float
            Initial relative loss.
        record_history : bool, optional
            Whether to append to phase_history.
        **extra_fields : Any
            Additional fields to include in result.

        Returns
        -------
        result : dict
            Phase 1 result dictionary.
        """
        if record_history:
            phase_record = {
                "phase": 1,
                "name": "lbfgs_warmup",
                "iterations": iterations,
                "final_loss": final_loss,
                "best_loss": best_loss,
                "switch_reason": switch_reason,
                "timestamp": time.time(),
                "lr_mode": lr_mode,
                "relative_loss": relative_loss,
            }
            self.phase_history.append(phase_record)

        result = {
            "final_params": final_params,
            "best_params": best_params,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "iterations": iterations,
            "switch_reason": switch_reason,
            "lr_mode": lr_mode,
            "relative_loss": relative_loss,
        }
        result.update(extra_fields)
        return result

    def _run_phase1_warmup(
        self,
        data_source: Any,
        model: Callable,
        p0: jnp.ndarray,
        bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run Phase 1 L-BFGS warmup.

        L-BFGS provides 5-10x faster convergence to the basin of attraction
        compared to first-order warmup by using approximate second-order
        (Hessian) information.

        Parameters
        ----------
        data_source : various types
            Data source (tuple of arrays for now)
        model : Callable
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
                "Only tuple data sources (x_data, y_data) supported in Phase 1 warmup",
            )

        # Initialize parameters in normalized space
        # Run tournament selection if multi-start enabled
        if self.config.enable_multistart and self.multistart_candidates is not None:
            current_params = self._run_tournament_selection(data_source, model)
        else:
            current_params = self.normalized_params

        # Create loss function FIRST (needed for warm start detection)
        loss_fn = self._create_warmup_loss_fn()

        # Record telemetry for warmup start
        telemetry = get_defense_telemetry()
        telemetry.record_warmup_start()

        # LAYER 1: Warm Start Detection
        should_exit, result, initial_loss, relative_loss = (
            self._check_warm_start_and_record(
                current_params=current_params,
                loss_fn=loss_fn,
                x_data=x_data,
                y_data=y_data,
                telemetry=telemetry,
            )
        )
        if should_exit:
            return result

        # LAYER 2: Adaptive Initial Step Size Selection for L-BFGS
        initial_step, lr_mode = self._select_adaptive_step_size(
            relative_loss=relative_loss,
            telemetry=telemetry,
        )

        # Create L-BFGS optimizer with selected initial step size
        optimizer, opt_state = self._create_lbfgs_optimizer(
            current_params,
            initial_step_size=initial_step,
        )

        # Best parameter tracking
        best_params = current_params
        best_loss = initial_loss  # Initialize with computed initial loss

        # Initialize previous loss
        prev_loss = initial_loss

        # Reset clip counter for diagnostics
        self._warmup_clip_count = 0

        # Warmup loop using L-BFGS
        for iteration in range(self.config.max_warmup_iterations):
            # _lbfgs_step returns loss_value computed at the params BEFORE the
            # step (see its docstring: "loss: Loss value before update"), while
            # the returned params are AFTER the step. Keep the pre-step point
            # so best-tracking pairs loss_value with the params it was actually
            # computed at, not with the just-taken step.
            params_before_step = current_params

            # Perform L-BFGS step (using full data for now)
            current_params, loss_value, grad_norm, opt_state, _line_search_failed = (
                self._lbfgs_step(
                    params=current_params,
                    opt_state=opt_state,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    x_batch=x_data,
                    y_batch=y_data,
                    iteration=iteration,
                )
            )

            # If line search failed, we may want to be more conservative
            # The _lbfgs_step already handles fallback behavior

            # Deferred line-search-failure detection: optax signals a real
            # line-search failure via a `failed` flag INSIDE the returned
            # optimizer state, not by raising, so the try/except in
            # _lbfgs_step never sees it and the telemetry never fires. Catch
            # it here instead, one iteration late and with no extra
            # evaluation: this iteration's loss_value is the loss AT the
            # params the PREVIOUS step produced, so comparing it to the
            # previous iteration's pre-step loss (prev_loss) tells us
            # whether that step actually decreased the objective.
            if iteration > 0 and loss_value > prev_loss:
                telemetry.record_lbfgs_line_search_failure(
                    iteration - 1,
                    f"step did not decrease loss: {prev_loss:.6e} -> {loss_value:.6e}",
                )

            # Track best parameters. fit() seeds Phase 2 from this best_params
            # (see phase1_result["best_params"]), so pairing loss_value with
            # the wrong params here previously handed Phase 2 a mislabeled
            # starting point.
            if loss_value < best_loss:
                best_loss = loss_value
                best_params = params_before_step

            # LAYER 3: Cost-Increase Guard
            cost_guard_exit, cost_guard_result = self._check_cost_increase_guard(
                iteration=iteration,
                loss_value=loss_value,
                best_loss=best_loss,
                best_params=best_params,
                lr_mode=lr_mode,
                relative_loss=relative_loss,
                telemetry=telemetry,
            )
            if cost_guard_exit:
                return cost_guard_result

            # Save checkpoint periodically if enabled
            if (
                hasattr(self.config, "enable_checkpoints")
                and self.config.enable_checkpoints
                and hasattr(self.config, "checkpoint_frequency")
                and (iteration + 1) % self.config.checkpoint_frequency == 0
            ):
                if (
                    hasattr(self.config, "checkpoint_dir")
                    and self.config.checkpoint_dir
                ):
                    checkpoint_path = (
                        Path(self.config.checkpoint_dir)
                        / f"checkpoint_phase1_iter{iteration + 1}.h5"
                    )
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
                    # loss_value is the loss at params_before_step (the point
                    # BEFORE the just-taken step), not at current_params (the
                    # step's result) -- see the pairing-fix comment above.
                    # That means the final step's own improvement was never
                    # compared against best_loss (there's no next iteration
                    # to do it), so best_params could silently lag one step
                    # behind current_params. Evaluate once, here, to close
                    # that gap and to report the correct final_loss for
                    # final_params (loss_value would mislabel it too).
                    final_loss_value = float(loss_fn(current_params, x_data, y_data))
                    if final_loss_value < best_loss:
                        best_loss = final_loss_value
                        best_params = current_params
                    self._update_global_best(
                        final_loss_value,
                        current_params,
                        len(x_data),
                    )
                    return self._build_phase1_result(
                        final_params=current_params,
                        best_params=best_params,
                        best_loss=best_loss,
                        final_loss=final_loss_value,
                        iterations=iteration + 1,
                        switch_reason=reason,
                        lr_mode=lr_mode,
                        relative_loss=relative_loss,
                    )

            # Progress callback (no-op if fit() wasn't given one). loss_value
            # is the loss AT params_before_step (see docstring of
            # _lbfgs_step), so pair it with that point, not the post-step
            # current_params, to keep callback telemetry internally consistent.
            self._invoke_fit_callback(params_before_step, loss_value, iteration)

            # Update previous loss
            prev_loss = loss_value

        # Maximum iterations reached (this shouldn't happen if max_iter criterion active)
        # Same final-step evaluation gap as the should_switch exit above.
        final_loss_value = float(loss_fn(current_params, x_data, y_data))
        if final_loss_value < best_loss:
            best_loss = final_loss_value
            best_params = current_params
        self._update_global_best(final_loss_value, current_params, len(x_data))
        return self._build_phase1_result(
            final_params=current_params,
            best_params=best_params,
            best_loss=best_loss,
            final_loss=final_loss_value,
            iterations=self.config.max_warmup_iterations,
            switch_reason="Maximum iterations reached",
            lr_mode=lr_mode,
            relative_loss=relative_loss,
        )

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

        Performance
        -----------
        Uses pre-compiled Jacobian function (self._jacobian_fn_compiled) when
        available, providing 15-25% speedup by avoiding repeated JIT tracing.
        Falls back to inline compilation for backwards compatibility.
        """
        # Use pre-compiled function if available (set up in _setup_jacobian_fn)
        if self._jacobian_fn_compiled is not None:
            return self._jacobian_fn_compiled(params, x_chunk)

        # Fallback: inline compilation (for backwards compatibility)
        # This path is slower due to repeated function tracing
        def model_at_x(p, x_single):
            return self.normalized_model(x_single, *p)

        jac_fn = jax.vmap(lambda x: jax.jacrev(model_at_x, argnums=0)(params, x))
        return jac_fn(x_chunk)

    def _accumulate_jtj_jtr(
        self,
        x_chunk: jnp.ndarray,
        y_chunk: jnp.ndarray,
        params: jnp.ndarray,
        JTJ_prev: jnp.ndarray,
        JTr_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, float]:
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
        residual_sum_sq = float(jnp.sum(residuals**2))

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
    ) -> tuple[jnp.ndarray, float]:
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
        JTJ_reg = JTJ + regularization * jnp.eye(n_params, dtype=jnp.float64)

        # Compute SVD of regularized J^T J
        U, s, Vt = jnp.linalg.svd(JTJ_reg, full_matrices=False)

        # Solve for Gauss-Newton step using SVD
        # The Gauss-Newton step minimizes the linearized problem:
        # Cost: C = 0.5 * ||r||^2 where r = y - f(x, p)
        # Gradient: g = -J^T r (since ∂r/∂p = -J)
        # The GN step solves: J^T J δ = J^T r
        # Therefore: δ = (J^T J)^{-1} J^T r  [NO negative sign!]
        #
        # Using SVD: J^T J = U S V^T, we have:
        # δ = V S^{-1} U^T (J^T r)

        # Compute U^T @ JTr (positive, not negative!)
        UTb = U.T @ JTr

        # Solve diagonal system with regularization
        # Filter out small singular values via truncated pseudo-inverse:
        # zero their contribution (1/s -> 0) rather than flooring s to
        # s_threshold, which would instead divide by a near-zero floor and
        # amplify null-space noise components by up to ~1/threshold.
        s_threshold = jnp.max(s) * 1e-10
        # Guard the denominator itself (not just its result) so 1.0/s never
        # evaluates 1/0 -> inf for an exactly-zero singular value before
        # `where` discards it -- transient inf/nan can otherwise trip
        # NLSQ's own stability guard even though it never reaches step_hat.
        s_safe = jnp.where(s > s_threshold, s, 1.0)
        s_inv = jnp.where(s > s_threshold, 1.0 / s_safe, 0.0)
        step_hat = UTb * s_inv

        # Transform back to parameter space
        step = Vt.T @ step_hat

        # Apply trust region constraint
        step_norm = jnp.linalg.norm(step)
        step = step * jnp.where(step_norm > trust_radius, trust_radius / step_norm, 1.0)

        # Compute predicted reduction: -g^T δ - 0.5 δ^T H δ
        # where g = -J^T r (gradient) and H = J^T J (Hessian approx)
        # Since g = -JTr, we have: -g^T δ = JTr^T δ
        # predicted_reduction = JTr^T δ - 0.5 δ^T (J^T J) δ
        predicted_reduction = jnp.maximum(
            jnp.dot(JTr, step) - 0.5 * jnp.dot(step, JTJ @ step),
            0.0,
        )

        return step, predicted_reduction

    # =========================================================================
    # CG-based Gauss-Newton Solver Methods (Task Group 3)
    # =========================================================================

    def _select_gn_solver(self, n_params: int) -> str:
        """Select which Gauss-Newton solver a given parameter count calls for.

        NOTE: this selection is NOT actually wired into the live Phase 2
        loop. `_gauss_newton_iteration` always calls the materialized
        SVD-based `_solve_gauss_newton_step`, regardless of what this method
        returns -- the CG-based implicit solve (`_solve_gauss_newton_step_cg`,
        `_cg_solve_implicit`, `_implicit_jtj_matvec`) is implemented and unit
        tested (see test_cg_solver.py) but never auto-selected. fit() raises
        NotImplementedError instead of silently running the O(p^2)-memory
        materialized solver when n_params >= cg_param_threshold; see fit().

        Parameters
        ----------
        n_params : int
            Number of parameters in the optimization problem

        Returns
        -------
        solver_type : str
            Either 'materialized' (for small p) or 'cg' (for large p) --
            advisory only, not consumed by the Phase 2 driver.

        Notes
        -----
        Threshold logic:
        - p < cg_param_threshold: Materialize J^T J, use SVD solve
        - p >= cg_param_threshold: Use CG with implicit matvec

        For p < 2000, O(p^3) SVD solve is fast and SVD provides better
        conditioning information. For p >= 2000, CG avoids O(p^2) memory
        for J^T J storage.
        """
        threshold = self.config.cg_param_threshold

        if n_params < threshold:
            solver_type = "materialized"
        else:
            solver_type = "cg"

        if self.config.verbose >= 2:
            _logger.debug(
                f"GN solver selection: p={n_params}, threshold={threshold}, "
                f"selected={solver_type}",
            )

        return solver_type

    def _implicit_jtj_matvec(
        self,
        v: jnp.ndarray,
        params: jnp.ndarray,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute (J^T J) @ v without materializing J^T J.

        Implements implicit matrix-vector product for CG solver:
            result = J^T @ (J @ v)

        This avoids O(p^2) storage for J^T J, enabling optimization with
        large parameter counts.

        Parameters
        ----------
        v : array_like
            Vector to multiply, shape (n_params,)
        params : array_like
            Current parameters in normalized space, shape (n_params,)
        x_data : array_like
            Full x data, shape (n_points,) or (n_points, n_features)
        y_data : array_like
            Full y data, shape (n_points,) (unused, kept for API consistency)

        Returns
        -------
        result : array_like
            Result of (J^T J) @ v, shape (n_params,)

        Notes
        -----
        Memory complexity: O(n_chunk * p) per chunk instead of O(p^2).
        Operates on chunks to avoid memory explosion for large datasets.

        The computation is:
        1. For each chunk: compute J_chunk @ v (forward pass)
        2. For each chunk: compute J_chunk^T @ (J_chunk @ v) (backward pass)
        3. Sum across chunks
        """
        chunk_size = self.config.chunk_size
        n_points = len(x_data)
        n_params = len(v)

        # Initialize result accumulator
        result = jnp.zeros(n_params)

        # Process in chunks to limit memory
        for i in range(0, n_points, chunk_size):
            x_chunk = x_data[i : i + chunk_size]

            # Compute Jacobian for this chunk: (chunk_size, n_params)
            J_chunk = self._compute_jacobian_chunk(x_chunk, params)

            # Forward: J @ v -> (chunk_size,)
            Jv = J_chunk @ v

            # Backward: J^T @ (J @ v) -> (n_params,)
            JTJv_chunk = J_chunk.T @ Jv

            # Accumulate
            result = result + JTJv_chunk

        return result

    def _compute_jacobi_preconditioner(
        self,
        params: jnp.ndarray,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Jacobi (diagonal) preconditioner for CG solver.

        Computes the diagonal of J^T J efficiently via chunked accumulation:
            diag_JTJ[j] = sum_i (J[i, j])^2

        Parameters
        ----------
        params : array_like
            Current parameters in normalized space, shape (n_params,)
        x_data : array_like
            Full x data, shape (n_points,) or (n_points, n_features)
        y_data : array_like
            Full y data, shape (n_points,) (unused, kept for API consistency)

        Returns
        -------
        diag_JTJ : array_like
            Diagonal of J^T J, shape (n_params,). Values are always positive.

        Notes
        -----
        The Jacobi preconditioner M = diag(J^T J) is applied as M^{-1} r
        in preconditioned CG. This scales the residual by inverse column
        norms, improving convergence for poorly-scaled problems.
        """
        chunk_size = self.config.chunk_size
        n_points = len(x_data)
        n_params = len(params)

        # Initialize diagonal accumulator
        diag_JTJ = jnp.zeros(n_params)

        # Accumulate column norms squared across chunks
        for i in range(0, n_points, chunk_size):
            x_chunk = x_data[i : i + chunk_size]

            # Compute Jacobian for this chunk: (chunk_size, n_params)
            J_chunk = self._compute_jacobian_chunk(x_chunk, params)

            # Accumulate squared column values: diag_JTJ[j] += sum_i J[i,j]^2
            diag_JTJ = diag_JTJ + jnp.sum(J_chunk**2, axis=0)

        # Ensure positive (add small regularization to avoid division by zero)
        diag_JTJ = jnp.maximum(diag_JTJ, self.config.regularization_factor)

        return diag_JTJ

    def _cg_solve_implicit(
        self,
        JTr: jnp.ndarray,
        params: jnp.ndarray,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        trust_radius: float,
    ) -> tuple[jnp.ndarray, int, bool]:
        """Solve (J^T J) @ step = J^T r using Conjugate Gradient with implicit matvec.

        Uses CG iteration with implicit J^T J matvec to avoid O(p^2) storage.
        Follows the pattern from trf.py:conjugate_gradient_solve().

        Parameters
        ----------
        JTr : array_like
            Right-hand side vector J^T r, shape (n_params,)
        params : array_like
            Current parameters in normalized space, shape (n_params,)
        x_data : array_like
            Full x data
        y_data : array_like
            Full y data
        trust_radius : float
            Trust region radius (for step constraint)

        Returns
        -------
        step : array_like
            Approximate solution, shape (n_params,)
        iterations : int
            Number of CG iterations performed
        converged : bool
            True if CG converged within tolerance

        Notes
        -----
        Implements Inexact Newton via CG with tolerance ||r_k|| < rtol * ||r_0||.
        On non-convergence, returns incomplete solution (still a descent direction).

        Uses jax.lax.while_loop for GPU acceleration.
        """
        n_params = len(JTr)
        max_iter = self.config.cg_max_iterations
        rtol = self.config.cg_relative_tolerance
        atol = self.config.cg_absolute_tolerance

        # Initial guess: zero
        step = jnp.zeros(n_params)

        # Initial residual: r = b - A @ x = JTr - (J^T J) @ 0 = JTr
        r = JTr
        r_norm_initial = jnp.linalg.norm(r)

        # Convergence threshold (Inexact Newton strategy)
        tol = jnp.maximum(rtol * r_norm_initial, atol)

        # Initialize CG vectors
        p = r  # Search direction
        r_dot_r = jnp.dot(r, r)

        # CG iteration state: (step, r, p, r_dot_r, iteration, converged)
        def cg_body(state):
            step, r, p, r_dot_r, iteration, _ = state

            # Compute A @ p = (J^T J) @ p implicitly
            Ap = self._implicit_jtj_matvec(p, params, x_data, y_data)

            # Compute step length
            pAp = jnp.dot(p, Ap)
            # Safeguard against negative curvature or zero
            pAp_safe = jnp.maximum(pAp, 1e-15)
            alpha = r_dot_r / pAp_safe

            # Update solution
            step_new = step + alpha * p

            # Update residual
            r_new = r - alpha * Ap

            # Compute new residual norm squared
            r_dot_r_new = jnp.dot(r_new, r_new)

            # Check convergence
            r_norm = jnp.sqrt(r_dot_r_new)
            converged = r_norm < tol

            # Compute beta for next search direction
            beta = r_dot_r_new / (r_dot_r + 1e-15)

            # Update search direction
            p_new = r_new + beta * p

            return (step_new, r_new, p_new, r_dot_r_new, iteration + 1, converged)

        def cg_cond(state):
            _, _, _, _, iteration, converged = state
            return jnp.logical_and(iteration < max_iter, jnp.logical_not(converged))

        # Run CG iterations using while_loop for GPU efficiency.
        # Seeding converged=True when r_norm_initial < atol makes cg_cond return
        # False on the first check, so the loop body never executes — avoids a
        # Python `if` on a traced value which would raise ConcretizationTypeError.
        init_state = (step, r, p, r_dot_r, jnp.array(0), r_norm_initial < atol)
        final_state = jax.lax.while_loop(cg_cond, cg_body, init_state)

        step_final, _r_final, _, _, iterations, converged = final_state

        # Apply trust region constraint
        step_norm = jnp.linalg.norm(step_final)
        if step_norm > trust_radius:
            step_final = step_final * (trust_radius / step_norm)

        return step_final, int(iterations), bool(converged)

    def _solve_gauss_newton_step_cg(
        self,
        JTJ: jnp.ndarray,
        JTr: jnp.ndarray,
        trust_radius: float,
        params: jnp.ndarray,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
    ) -> tuple[jnp.ndarray, float]:
        """Solve Gauss-Newton step using CG with implicit J^T J matvec.

        Alternative to SVD-based _solve_gauss_newton_step() for large parameter
        counts. Uses CG iteration with implicit matvec to avoid O(p^2) storage.

        Parameters
        ----------
        JTJ : array_like
            Accumulated J^T J matrix (may not be used if using implicit matvec)
        JTr : array_like
            Accumulated J^T r vector, shape (n_params,)
        trust_radius : float
            Trust region radius
        params : array_like
            Current parameters in normalized space
        x_data : array_like
            Full x data
        y_data : array_like
            Full y data

        Returns
        -------
        step : array_like
            Gauss-Newton step, shape (n_params,)
        predicted_reduction : float
            Predicted reduction in cost function

        Notes
        -----
        On CG non-convergence, returns incomplete solution which is typically
        still a descent direction (useful for trust region methods).
        """
        # Solve using CG with implicit matvec
        step, cg_iterations, converged = self._cg_solve_implicit(
            JTr,
            params,
            x_data,
            y_data,
            trust_radius,
        )

        # Log CG diagnostics
        if self.config.verbose >= 2:
            status = "converged" if converged else "incomplete"
            _logger.debug(f"CG solver: {cg_iterations} iterations, {status}")

        # If CG didn't converge, the incomplete solution is still usable
        # as a descent direction. Optionally apply Jacobi preconditioner
        # and re-solve if configured.
        if not converged and cg_iterations >= self.config.cg_max_iterations * 0.9:
            # CG struggled - this is logged for diagnostics but we use the
            # incomplete solution which is typically still a descent direction
            if self.config.verbose >= 1:
                _logger.warning(
                    f"CG solver hit iteration limit ({cg_iterations}). "
                    "Using incomplete solution as descent direction.",
                )

        # Compute predicted reduction: JTr^T @ step - 0.5 * step^T @ (J^T J) @ step
        # Use implicit matvec for the JTJ @ step term
        JTJ_step = self._implicit_jtj_matvec(step, params, x_data, y_data)
        predicted_reduction = jnp.dot(JTr, step) - 0.5 * jnp.dot(step, JTJ_step)
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
        Dispatches between JAX scan (GPU/TPU) and Python loops (CPU) based on
        config.loop_strategy for optimal performance on each backend.
        """
        x_data, y_data = data_source
        n_params = len(current_params)
        n_points = len(x_data)
        chunk_size = self.config.chunk_size

        # Dispatch based on backend: scan for GPU/TPU, loops for CPU
        if self._use_scan_for_accumulation():
            # Use JAX scan for GPU/TPU (better XLA fusion, reduced kernel launches)
            JTJ, JTr, total_cost = self._accumulate_jtj_jtr_scan(
                x_data,
                y_data,
                current_params,
            )
        else:
            # Use Python loops for CPU (lower tracing overhead)
            JTJ = jnp.zeros((n_params, n_params))
            JTr = jnp.zeros(n_params)
            total_cost = 0.0

            for i in range(0, n_points, chunk_size):
                x_chunk = x_data[i : i + chunk_size]
                y_chunk = y_data[i : i + chunk_size]
                JTJ, JTr, chunk_cost = self._accumulate_jtj_jtr(
                    x_chunk,
                    y_chunk,
                    current_params,
                    JTJ,
                    JTr,
                )
                total_cost += chunk_cost

        # Add group variance regularization if enabled
        # This prevents per-angle parameters from absorbing angle-dependent signals
        if (
            self.config.enable_group_variance_regularization
            and self.config.group_variance_indices
        ):
            var_lambda = self.config.group_variance_lambda
            for start, end in self.config.group_variance_indices:
                group_params = current_params[start:end]
                n_group = end - start
                group_mean = jnp.mean(group_params)

                # Gradient of variance: ∂Var/∂p_i = (2/n) * (p_i - mean)
                grad_var = (2.0 / n_group) * (group_params - group_mean)

                # JTr/JTJ are the gradient/Hessian of the HALF-cost 0.5*SSR
                # model (see _solve_gauss_newton_step: step solves JTJ@d=JTr
                # with JTr = -grad(0.5*SSR); predicted_reduction is likewise
                # in 0.5*SSR units, and the caller halves the full-SSR
                # actual_reduction before comparing). The regularization term
                # added to total_cost below is R = var_lambda*group_var*
                # n_points in FULL-SSR units, so its contribution here must
                # be halved to land in the same 0.5*SSR units as the rest of
                # JTr/JTJ -- otherwise the regularizer's effect on the
                # trust-region model is 2x too strong relative to the data
                # term, biasing accept/reject and trust-radius decisions.
                JTr = JTr.at[start:end].add(-0.5 * var_lambda * grad_var * n_points)

                # Hessian of variance: H = (2/n) * (I - (1/n)*11^T)
                # This is a dense (n_group x n_group) matrix
                diag_term = (2.0 / n_group) * jnp.eye(n_group, dtype=jnp.float64)
                off_diag_term = (2.0 / (n_group * n_group)) * jnp.ones(
                    (n_group, n_group),
                    dtype=jnp.float64,
                )
                H_var = diag_term - off_diag_term

                # Add to JTJ for the group block (n_points- and 0.5-scaled, see above)
                JTJ = JTJ.at[start:end, start:end].add(
                    0.5 * var_lambda * H_var * n_points,
                )

                # Add variance cost to total
                group_var = jnp.var(group_params)
                total_cost += (
                    var_lambda
                    * float(jnp.where(jnp.isfinite(group_var), group_var, 0.0))
                    * n_points
                )

        # Compute gradient norm for convergence check
        gradient_norm = float(jnp.linalg.norm(JTr))

        # Solve for Gauss-Newton step
        step, predicted_reduction = self._solve_gauss_newton_step(
            JTJ,
            JTr,
            trust_radius,
        )

        # Apply step to get new parameters
        new_params = current_params + step

        # Clip to bounds if available (important for constrained optimization)
        if self.normalized_bounds is not None:
            lb, ub = self.normalized_bounds
            new_params = jnp.clip(new_params, lb, ub)

        # Evaluate cost at new parameters using optimized pre-compiled function
        # This provides 20-30% speedup compared to inline computation
        new_cost = self._compute_cost_with_variance_regularization(
            new_params,
            x_data,
            y_data,
        )

        # Compute actual reduction
        actual_reduction = total_cost - new_cost

        # Update trust region based on reduction ratio.
        # actual_reduction is a full sum-of-squares reduction (||r||^2), while
        # predicted_reduction is for the 0.5*||r||^2 Gauss-Newton model. Halve
        # the actual side here so the ratio is ~1 for a good step (previously ~2,
        # which over-expanded / under-shrank the trust radius). Only the ratio is
        # rescaled; the stored actual_reduction stays full-SSR because the outer
        # loop reconstructs cost_before_step from it.
        if predicted_reduction > 0:
            reduction_ratio = (0.5 * actual_reduction) / predicted_reduction
        else:
            reduction_ratio = 0.0

        # Trust region update logic with recovery mechanism
        # Minimum and maximum trust radius bounds
        min_trust_radius = getattr(self.config, "min_trust_radius", 1e-8)
        max_trust_radius = getattr(self.config, "max_trust_radius", 1000.0)

        step_norm = float(jnp.linalg.norm(step))

        if reduction_ratio < 0.25:
            # Poor agreement: shrink trust region (use 0.5 instead of 0.25 for
            # less aggressive shrinkage)
            new_trust_radius = trust_radius * 0.5

            # Recovery mechanism: if trust radius is very small but gradient is
            # large, the optimizer may be stuck. Reset to allow exploration.
            if new_trust_radius < min_trust_radius and gradient_norm > 1e-4:
                # Reset to gradient-scaled value for recovery
                new_trust_radius = min(
                    0.1 * gradient_norm / max(1.0, gradient_norm),
                    1.0,
                )
        elif reduction_ratio > 0.75 and step_norm >= 0.9 * trust_radius:
            # Good agreement and step at boundary: expand trust region
            new_trust_radius = min(trust_radius * 2.0, max_trust_radius)
        else:
            # Acceptable agreement: keep trust region
            new_trust_radius = trust_radius

        # Enforce minimum trust radius to prevent complete collapse
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

    def _gn_iteration_with_retry(
        self,
        data_source: tuple[jnp.ndarray, jnp.ndarray],
        current_params: jnp.ndarray,
        trust_radius: float,
        best_params: jnp.ndarray,
        best_cost: float,
    ) -> tuple[dict[str, Any], float]:
        """Execute Gauss-Newton iteration with retry logic.

        Parameters
        ----------
        data_source : tuple of array_like
            Full dataset as (x_data, y_data).
        current_params : array_like
            Current parameters.
        trust_radius : float
            Current trust region radius.
        best_params : array_like
            Best parameters found so far (fallback).
        best_cost : float
            Best cost found so far (fallback).

        Returns
        -------
        iter_result : dict
            Iteration result dictionary.
        trust_radius : float
            Updated trust radius.
        """
        max_retries = getattr(self.config, "max_retries_per_batch", 0)

        for retry_attempt in range(max_retries + 1):
            try:
                iter_result = self._gauss_newton_iteration(
                    data_source,
                    current_params,
                    trust_radius,
                )
                new_params = iter_result["new_params"]
                new_cost = iter_result["new_cost"]

                # Validate results - if finite, return success
                if jnp.all(jnp.isfinite(new_params)) and jnp.isfinite(new_cost):
                    return iter_result, trust_radius

                # Non-finite results: retry or use fallback
                if retry_attempt < max_retries:
                    trust_radius *= 0.5
                    continue

                # Max retries exhausted: use best known params
                iter_result["new_params"] = best_params
                iter_result["new_cost"] = best_cost
                iter_result["gradient_norm"] = 0.0
                iter_result["actual_reduction"] = 0.0
                iter_result["retry_exhausted"] = True
                return iter_result, trust_radius

            except Exception as exc:
                if retry_attempt < max_retries:
                    trust_radius *= 0.5
                    continue

                # Max retries exhausted: check fault tolerance
                if not (
                    hasattr(self.config, "enable_fault_tolerance")
                    and self.config.enable_fault_tolerance
                ):
                    raise

                # Use best parameters as fallback. Log unconditionally: this
                # silently substituted the last-known-good point for a real
                # Gauss-Newton step with zero diagnostic trace before -- with
                # fault tolerance on (a real, documented opt-in), a genuine
                # bug anywhere in _gauss_newton_iteration would otherwise
                # produce a stalled/degraded fit with no indication why.
                _logger.warning(
                    f"_gauss_newton_iteration raised {type(exc).__name__}: "
                    f"{exc} after {max_retries} retries; falling back to "
                    "best known parameters for this iteration "
                    "(config.enable_fault_tolerance=True).",
                )
                iter_result = {
                    "new_params": best_params,
                    "new_cost": best_cost,
                    "gradient_norm": 0.0,
                    "actual_reduction": 0.0,
                    "trust_radius": trust_radius,
                    "retry_exhausted": True,
                }
                return iter_result, trust_radius

        # Should not reach here, but provide fallback
        iter_result = {
            "new_params": best_params,
            "new_cost": best_cost,
            "gradient_norm": 0.0,
            "actual_reduction": 0.0,
            "trust_radius": trust_radius,
            "retry_exhausted": True,
        }
        return iter_result, trust_radius

    def _should_save_checkpoint(self, iteration: int) -> bool:
        """Check if checkpoint should be saved at this iteration.

        Parameters
        ----------
        iteration : int
            Current iteration number (0-indexed).

        Returns
        -------
        should_save : bool
            True if checkpoint should be saved.
        """
        if not getattr(self.config, "enable_checkpoints", False):
            return False
        if not getattr(self.config, "checkpoint_dir", None):
            return False
        frequency = getattr(self.config, "checkpoint_frequency", 0)
        if frequency <= 0:
            return False
        return (iteration + 1) % frequency == 0

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

        # Track best parameters. best_cost is seeded with the actual initial
        # cost once it's computed below (not left at inf) so that a
        # retry-exhaustion fallback on iteration 0 -- before any GN step is
        # ever accepted -- reports the real starting cost instead of inf.
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

        # Get verbosity from config or default to 1 for progress output
        verbose = getattr(self.config, "verbose", 1)
        log_frequency = getattr(
            self.config,
            "log_frequency",
            1,
        )  # Log every N iterations

        # Compute initial JTJ with progress reporting
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        init_start_time = time.time()
        if verbose >= 1:
            print(
                f"  Computing initial JTJ ({n_chunks} chunks, {n_points:,} points)...",
            )

        for chunk_idx, i in enumerate(range(0, n_points, chunk_size)):
            x_chunk = x_data[i : i + chunk_size]
            y_chunk = y_data[i : i + chunk_size]

            final_JTJ, final_JTr, res_sq = self._accumulate_jtj_jtr(
                x_chunk,
                y_chunk,
                current_params,
                final_JTJ,
                final_JTr,
            )
            final_residual_sum_sq += res_sq

            # Progress for initial JTJ computation (every 10% or every 50 chunks)
            if verbose >= 1 and (
                (chunk_idx + 1) % max(1, n_chunks // 10) == 0
                or (chunk_idx + 1) == n_chunks
            ):
                elapsed = time.time() - init_start_time
                pct = (chunk_idx + 1) / n_chunks * 100
                print(
                    f"  Initial JTJ: {chunk_idx + 1}/{n_chunks} chunks "
                    f"({pct:.0f}%), elapsed={elapsed:.1f}s",
                )

        if verbose >= 1:
            init_elapsed = time.time() - init_start_time
            print(
                f"  Initial JTJ complete: cost={final_residual_sum_sq:.6e}, time={init_elapsed:.1f}s",
            )

        # Seed best_cost from the initial cost (see comment above best_cost's
        # declaration) now that it's been computed. Must use the SAME cost
        # definition new_cost uses below (_compute_cost_with_variance_regularization,
        # i.e. SSR + group-variance penalty when enabled) -- seeding from the
        # data-only final_residual_sum_sq left the penalty term out, making
        # the seed artificially low whenever enable_group_variance_regularization
        # is set, so `new_cost < best_cost` could never fire and Phase 2 would
        # silently return the unoptimized starting point while still reporting
        # convergence.
        best_cost = (
            float(
                self._compute_cost_with_variance_regularization(
                    current_params,
                    x_data,
                    y_data,
                ),
            )
            if self.config.enable_group_variance_regularization
            else float(final_residual_sum_sq)
        )

        # Gauss-Newton loop
        # Initialize stall detection counter
        self._consecutive_rejections = 0

        for iteration in range(self.config.gauss_newton_max_iterations):
            iter_start_time = time.time()

            # Perform one Gauss-Newton iteration with retry logic
            iter_result, retry_trust_radius = self._gn_iteration_with_retry(
                data_source,
                current_params,
                trust_radius,
                best_params,
                best_cost,
            )

            # Extract results
            new_params = iter_result["new_params"]
            new_cost = iter_result["new_cost"]
            gradient_norm = iter_result["gradient_norm"]
            actual_reduction = iter_result["actual_reduction"]
            # retry_exhausted marks a numerical-failure fallback: gradient_norm
            # and actual_reduction are sentinel 0.0 values, not real measurements,
            # so they must not be read as "converged" below.
            retry_exhausted = iter_result.get("retry_exhausted", False)
            # On retry exhaustion, iter_result["trust_radius"] is whatever the
            # LAST FAILED _gauss_newton_iteration call computed internally
            # (its recovery-reset logic can bump the radius back up in
            # response to a non-finite step). retry_trust_radius is the
            # correctly halved-per-retry value _gn_iteration_with_retry
            # tracked; use it instead so a numerical failure can't widen the
            # trust radius right before the next attempt.
            trust_radius = (
                retry_trust_radius if retry_exhausted else iter_result["trust_radius"]
            )
            iter_time = time.time() - iter_start_time

            # Progress logging for Phase 2 iterations
            if verbose >= 1 and (iteration + 1) % log_frequency == 0:
                print(
                    f"  GN iter {iteration + 1}/{self.config.gauss_newton_max_iterations}: "
                    f"cost={new_cost:.6e}, grad_norm={gradient_norm:.6e}, "
                    f"reduction={actual_reduction:.6e}, Δ={trust_radius:.4f}, "
                    f"time={iter_time:.1f}s",
                )

            # Update best parameters
            if new_cost < best_cost:
                best_cost = new_cost
                best_params = new_params

            # Track global best
            if new_cost < self.best_cost_global:
                self.best_cost_global = new_cost
                self.best_params_global = new_params

            # Progress callback (no-op if fit() wasn't given one)
            self._invoke_fit_callback(new_params, new_cost, iteration)

            # Save checkpoint periodically if enabled
            if self._should_save_checkpoint(iteration):
                checkpoint_path = (
                    Path(self.config.checkpoint_dir)
                    / f"checkpoint_phase2_iter{iteration + 1}.h5"
                )
                self.current_phase = 2
                self.normalized_params = current_params
                self._save_checkpoint(checkpoint_path)

            # Accept step if cost decreased
            # Save cost before step for convergence check. actual_reduction is
            # cost(current_params) - new_cost for every iteration (accepted or
            # rejected), so this recovers the pre-step cost even before the
            # first accepted step (when prev_cost is still inf). Using new_cost
            # alone here made a rejected first iteration compare new_cost
            # against itself, reporting a false "Cost change below tolerance".
            cost_before_step = (
                prev_cost if jnp.isfinite(prev_cost) else new_cost + actual_reduction
            )

            if actual_reduction > 0:
                current_params = new_params
                prev_cost = new_cost
                self._consecutive_rejections = 0  # Reset rejection counter on success

                # Recompute J^T J at new params for Phase 3
                # This ensures we have J^T J at the final parameters.
                # Dispatch between JAX scan (GPU/TPU) and Python loops (CPU),
                # same as the initial accumulation above and the Phase 2
                # failure fallback below -- the unconditional Python-loop
                # path here previously ran on every accepted GN iteration
                # even on GPU/TPU, defeating the scan path's XLA fusion.
                x_data, y_data = data_source
                n_params = len(current_params)

                if self._use_scan_for_accumulation():
                    JTJ, JTr, residual_sum_sq = self._accumulate_jtj_jtr_scan(
                        x_data,
                        y_data,
                        current_params,
                    )
                else:
                    chunk_size = self.config.chunk_size
                    n_points = len(x_data)
                    JTJ = jnp.zeros((n_params, n_params))
                    JTr = jnp.zeros(n_params)
                    residual_sum_sq = 0.0
                    for i in range(0, n_points, chunk_size):
                        x_chunk = x_data[i : i + chunk_size]
                        y_chunk = y_data[i : i + chunk_size]

                        JTJ, JTr, res_sq = self._accumulate_jtj_jtr(
                            x_chunk,
                            y_chunk,
                            current_params,
                            JTJ,
                            JTr,
                        )
                        residual_sum_sq += res_sq

                final_JTJ = JTJ
                final_residual_sum_sq = residual_sum_sq
            else:
                # Step rejected - trust radius is already updated in
                # _gauss_newton_iteration, no need to shrink again here.
                # Track consecutive rejections for stall detection.
                consecutive_rejections = getattr(self, "_consecutive_rejections", 0) + 1
                self._consecutive_rejections = consecutive_rejections

                # Stall detection: if many consecutive rejections with large
                # gradient, the optimizer is stuck. Reset trust radius.
                # On a retry_exhausted iteration, gradient_norm is a 0.0
                # sentinel (not a real measurement, see the convergence-check
                # comment below), so `gradient_norm > 1e-4` can never be true
                # and this reset could never fire for repeated numerical
                # failures -- consecutive_rejections would grow without bound
                # and the trust radius would stay permanently locked. Skip
                # the (meaningless, in that case) gradient check instead.
                should_reset_for_stall = consecutive_rejections >= 10 and (
                    retry_exhausted or gradient_norm > 1e-4
                )
                if should_reset_for_stall:
                    trust_radius = self.config.trust_region_initial
                    self._consecutive_rejections = 0
                    if verbose >= 1:
                        print(
                            f"  Stall detected: resetting trust radius to "
                            f"{trust_radius:.4f}",
                        )

            # Check convergence: gradient norm (skip on a retry-exhaustion
            # fallback, where gradient_norm is a 0.0 sentinel, not a real value)
            if not retry_exhausted and gradient_norm < self.config.gauss_newton_tol:
                # Record Phase 2 completion
                phase_record = {
                    "phase": 2,
                    "name": "gauss_newton",
                    "iterations": iteration + 1,
                    "final_cost": best_cost,
                    "best_cost": best_cost,
                    "convergence_reason": "Gradient norm below tolerance",
                    "gradient_norm": gradient_norm,
                    "timestamp": time.time(),
                }
                self.phase_history.append(phase_record)

                # Return the best accepted point, not the last trial point: a
                # rejected final step leaves new_params worse than best_params.
                return {
                    "final_params": best_params,
                    "best_params": best_params,
                    "best_cost": best_cost,
                    "final_cost": best_cost,
                    "iterations": iteration + 1,
                    "convergence_reason": "Gradient norm below tolerance",
                    "gradient_norm": gradient_norm,
                    "JTJ_final": final_JTJ,
                    "residual_sum_sq": final_residual_sum_sq,
                }

            # Check convergence: cost change (compare to cost before this step)
            cost_change = abs(cost_before_step - new_cost)
            relative_change = cost_change / (abs(cost_before_step) + 1e-10)

            if not retry_exhausted and relative_change < self.config.gauss_newton_tol:
                phase_record = {
                    "phase": 2,
                    "name": "gauss_newton",
                    "iterations": iteration + 1,
                    "final_cost": best_cost,
                    "best_cost": best_cost,
                    "convergence_reason": "Cost change below tolerance",
                    "gradient_norm": gradient_norm,
                    "timestamp": time.time(),
                }
                self.phase_history.append(phase_record)

                # Return the best accepted point, not the last trial point: a
                # rejected final step leaves new_params worse than best_params.
                return {
                    "final_params": best_params,
                    "best_params": best_params,
                    "best_cost": best_cost,
                    "final_cost": best_cost,
                    "iterations": iteration + 1,
                    "convergence_reason": "Cost change below tolerance",
                    "gradient_norm": gradient_norm,
                    "JTJ_final": final_JTJ,
                    "residual_sum_sq": final_residual_sum_sq,
                }

        # Maximum iterations reached.
        # final_params is best_params, so final_cost must match best_cost. Using
        # prev_cost here returned inf when every step was rejected (prev_cost is
        # only assigned on an accepted step); fall back to the finite initial
        # residual sum of squares in the degenerate all-rejected case.
        reported_final_cost = (
            best_cost if jnp.isfinite(best_cost) else final_residual_sum_sq
        )
        phase_record = {
            "phase": 2,
            "name": "gauss_newton",
            "iterations": self.config.gauss_newton_max_iterations,
            "final_cost": reported_final_cost,
            "best_cost": best_cost,
            "convergence_reason": "Maximum iterations reached",
            "gradient_norm": gradient_norm,
            "timestamp": time.time(),
        }
        self.phase_history.append(phase_record)

        return {
            "final_params": best_params,  # Use best, not current
            "best_params": best_params,
            "best_cost": best_cost,
            "final_cost": reported_final_cost,
            "iterations": self.config.gauss_newton_max_iterations,
            "convergence_reason": "Maximum iterations reached",
            "gradient_norm": gradient_norm,
            "JTJ_final": final_JTJ,
            "residual_sum_sq": final_residual_sum_sq,
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
            raise RuntimeError(
                "Normalizer not initialized. Call _setup_normalization first.",
            )

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
            raise RuntimeError(
                "Normalization Jacobian not available. Call _setup_normalization first.",
            )

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
    ) -> tuple[jnp.ndarray, float]:
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
        perr = jnp.where(variances >= 0, jnp.sqrt(variances), jnp.nan)

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
            cov_orig,
            residual_sum_sq,
            n_points,
        )

        # Step 5: Compute standard errors
        perr = self._compute_standard_errors(pcov)

        # Record Phase 3 completion
        phase_duration = time.time() - phase_start
        phase_record = {
            "phase": 3,
            "name": "denormalization_covariance",
            "duration": phase_duration,
            "sigma_sq": sigma_sq,
            "cov_condition": float(jnp.linalg.cond(pcov))
            if jnp.isfinite(pcov).all()
            else jnp.inf,
            "timestamp": time.time(),
        }
        self.phase_history.append(phase_record)

        return {
            "popt": popt,
            "pcov": pcov,
            "perr": perr,
            "sigma_sq": sigma_sq,
            "diagnostics": phase_record,
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
        if (
            not hasattr(self.config, "validate_numerics")
            or not self.config.validate_numerics
        ):
            return True

        # Check parameters
        if not jnp.all(jnp.isfinite(params)):
            if (
                hasattr(self.config, "enable_fault_tolerance")
                and self.config.enable_fault_tolerance
            ):
                # Log warning but continue
                return False
            raise ValueError(f"NaN/Inf detected in parameters {context}")

        # Check loss
        if loss is not None and not jnp.isfinite(loss):
            if (
                hasattr(self.config, "enable_fault_tolerance")
                and self.config.enable_fault_tolerance
            ):
                return False
            raise ValueError(f"NaN/Inf detected in loss {context}")

        # Check gradients
        if gradients is not None and not jnp.all(jnp.isfinite(gradients)):
            if (
                hasattr(self.config, "enable_fault_tolerance")
                and self.config.enable_fault_tolerance
            ):
                return False
            raise ValueError(f"NaN/Inf detected in gradients {context}")

        return True

    def _get_checkpoint_manager(self):
        """Get or create the checkpoint manager (lazy initialization).

        Returns
        -------
        CheckpointManager
            The checkpoint manager instance.
        """
        if self._checkpoint_manager is None:
            if "CheckpointManager" not in _lazy_imports:
                from nlsq.streaming.phases.checkpoint import CheckpointManager

                _lazy_imports["CheckpointManager"] = CheckpointManager
            self._checkpoint_manager = _lazy_imports["CheckpointManager"](self.config)
        return self._checkpoint_manager

    def _create_checkpoint_state(self):
        """Create a CheckpointState from current optimizer state.

        Returns
        -------
        CheckpointState
            State container with all optimizer state for checkpointing.
        """
        from nlsq.streaming.phases.checkpoint import CheckpointState

        return CheckpointState(
            current_phase=self.current_phase,
            normalized_params=self.normalized_params,
            phase1_optimizer_state=self.phase1_optimizer_state,
            phase2_JTJ_accumulator=self.phase2_JTJ_accumulator,
            phase2_JTr_accumulator=self.phase2_JTr_accumulator,
            best_params_global=self.best_params_global,
            best_cost_global=self.best_cost_global,
            phase_history=self.phase_history,
            normalizer=self.normalizer,
            tournament_selector=self.tournament_selector,
            multistart_candidates=self.multistart_candidates,
        )

    def _restore_from_checkpoint_state(self, state) -> None:
        """Restore optimizer state from a CheckpointState.

        Parameters
        ----------
        state : CheckpointState
            State container to restore from.
        """
        self.current_phase = state.current_phase
        self.normalized_params = state.normalized_params
        self.phase1_optimizer_state = state.phase1_optimizer_state
        self.phase2_JTJ_accumulator = state.phase2_JTJ_accumulator
        self.phase2_JTr_accumulator = state.phase2_JTr_accumulator
        self.best_params_global = state.best_params_global
        self.best_cost_global = state.best_cost_global
        self.phase_history = state.phase_history
        # Note: normalizer is NOT restored from checkpoint - recreated in _setup_normalization
        if state.tournament_selector is not None:
            self.tournament_selector = state.tournament_selector
        if state.multistart_candidates is not None:
            self.multistart_candidates = state.multistart_candidates

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
        - phase1_optimizer_state: Optax L-BFGS state (history + params)
        - phase2_jtj_accumulator: Accumulated J^T J matrix
        - phase2_jtr_accumulator: Accumulated J^T r vector
        - best_params_global: Best parameters found globally
        - best_cost_global: Best cost value globally
        - phase_history: Complete phase history
        """
        manager = self._get_checkpoint_manager()
        state = self._create_checkpoint_state()
        manager.save(checkpoint_path, state)

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
        manager = self._get_checkpoint_manager()

        # Create GlobalOptimizationConfig for tournament reconstruction if needed
        global_config = None
        if self.config.enable_multistart:
            global_config = GlobalOptimizationConfig(
                n_starts=self.config.n_starts,
                elimination_rounds=self.config.elimination_rounds,
                elimination_fraction=self.config.elimination_fraction,
                batches_per_round=self.config.batches_per_round,
            )

        state = manager.load(checkpoint_path, global_config)
        self._restore_from_checkpoint_state(state)

    def fit(
        self,
        data_source: Any,
        func: Callable,
        p0: jnp.ndarray,
        bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        sigma: jnp.ndarray | None = None,
        absolute_sigma: bool = False,
        callback: Callable | None = None,
        verbose: int = 1,
    ) -> dict[str, Any]:
        """Fit model parameters using four-phase hybrid optimization.

        This method orchestrates all four phases:
        - Phase 0: Setup normalization
        - Phase 1: L-BFGS warmup
        - Phase 2: Streaming Gauss-Newton
        - Phase 3: Denormalization and covariance

        Parameters
        ----------
        data_source : various types
            Data source for optimization. Can be:
            - Tuple of arrays: (x_data, y_data)
            - Generator yielding (x_batch, y_batch)
            - HDF5 file path with datasets
        func : Callable
            Model function with signature: ``func(x, *params) -> predictions``
        p0 : array_like
            Initial parameter guess of shape (n_params,)
        bounds : tuple of array_like, optional
            Parameter bounds as (lb, ub)
        sigma : array_like, optional
            Uncertainties in y_data for weighted least squares
        absolute_sigma : bool, default=False
            If True, sigma is used in absolute sense (pcov not scaled)
        callback : Callable, optional
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
                "Only tuple data sources (x_data, y_data) currently supported",
            )

        # sigma/absolute_sigma are accepted for API compatibility but no phase
        # actually applies sigma weighting to the loss or Jacobian; silently
        # ignoring it would return an unweighted fit while claiming otherwise.
        # Fail loudly instead of returning a silently-wrong result.
        if sigma is not None:
            raise NotImplementedError(
                "method='hybrid_streaming' does not support the 'sigma' "
                "parameter (weighted least squares) yet. Use a different "
                "method (e.g. 'trf') for weighted fits, or omit sigma.",
            )

        # config.cg_param_threshold documents auto-selecting a CG-based
        # implicit solver for p >= threshold to avoid O(p^2) JTJ memory
        # (_select_gn_solver). That selection is never consulted by the live
        # Phase 2 loop, which always materializes JTJ. Fail loudly rather
        # than silently attempt an O(p^2) allocation the docs say we'd avoid.
        if len(p0_array) >= self.config.cg_param_threshold:
            raise NotImplementedError(
                f"{len(p0_array)} parameters >= config.cg_param_threshold "
                f"({self.config.cg_param_threshold}): the documented CG "
                "auto-selection for large parameter counts is not wired "
                "into Phase 2, which always materializes an O(p^2) JTJ "
                "matrix. Raise cg_param_threshold if you accept that memory "
                "cost, or reduce the parameter count.",
            )

        # Reset per-fit optimization state now that all validation above has
        # passed. Without this, calling fit() more than once on the same
        # optimizer instance (a supported pattern -- nothing in __init__
        # forbids it) leaks best_params_global/best_cost_global from a PRIOR
        # fit's p0/bounds/model into this one: that state is stored in
        # normalized coordinates whose scale/offset depend on the previous
        # call's setup, and it now also feeds the Phase 1/2 exception-fallback
        # path below, so a stale value there could silently hand the wrong
        # coordinate system to a different model/parameter count instead of
        # erroring. Resetting AFTER validation (not before) means a second
        # fit() call with invalid arguments raises without first wiping the
        # first call's successful best_params_global/best_cost_global --
        # e.g. _save_checkpoint() serializes them and could otherwise silently
        # lose the prior fit's state on an unrelated validation failure.
        self.best_params_global = None
        self.best_cost_global = float("inf")

        # Progress callback, invoked from Phase 1/2 loops via
        # _invoke_fit_callback() every config.callback_frequency iterations.
        # Reset alongside best_params_global/best_cost_global above (after
        # validation, not before) for the same reason.
        self._fit_callback = callback

        # ============================================================
        # Phase 0: Setup Normalization
        # ============================================================
        if verbose >= 1:
            print("=" * 60)
            print("Adaptive Hybrid Streaming Optimizer")
            print("=" * 60)
            print(f"Dataset size: {n_points:,} points")
            print(f"Parameters: {len(p0_array)}")
            print(f"Normalization: {self.config.normalization_strategy}")
            print()

        phase0_start = time.time()
        self._setup_normalization(func, p0_array, bounds)
        phase0_duration = time.time() - phase0_start
        phase_timings["phase0_normalization"] = phase0_duration

        if verbose >= 1:
            print(f"Phase 0: Normalization setup complete ({phase0_duration:.3f}s)")
            print(f"  Strategy: {self.normalizer.strategy}")
            print()

        # ============================================================
        # Phase 1: L-BFGS warmup
        # ============================================================
        if verbose >= 1:
            print("Phase 1: L-BFGS warmup...")

        # phase1_failure/phase2_failure record why the pipeline fell back so
        # the final result can be honestly labeled success=False, instead of
        # letting an exception here skip Phase 3 entirely and lose whatever
        # best_params_global was already found.
        phase1_failure: str | None = None
        phase1_start = time.time()
        try:
            phase1_result = self._run_phase1_warmup(
                data_source=(x_data, y_data),
                model=func,
                p0=p0_array,
                bounds=bounds,
            )
        except Exception as exc:
            phase1_failure = f"{type(exc).__name__}: {exc}"
            # Log unconditionally -- see the matching comment on the Phase 2
            # except-block below for why this must not be gated on verbose.
            _logger.warning(
                f"Phase 1 (L-BFGS warmup) raised {phase1_failure}; "
                "falling back to best known parameters.",
            )
            fallback_params = (
                self.best_params_global
                if self.best_params_global is not None
                else self.normalized_params
            )
            phase1_result = {
                "best_params": fallback_params,
                "best_loss": self.best_cost_global,
                "final_loss": self.best_cost_global,
                "switch_reason": f"Phase 1 failed: {phase1_failure}",
                "iterations": 0,
            }
        phase1_duration = time.time() - phase1_start
        phase_timings["phase1_warmup"] = phase1_duration
        phase_iterations["phase1"] = phase1_result["iterations"]

        if verbose >= 1:
            print(
                f"Phase 1 complete: {phase1_result['iterations']} iterations ({phase1_duration:.3f}s)",
            )
            print(f"  Best loss: {phase1_result['best_loss']:.6e}")
            print(f"  Switch reason: {phase1_result['switch_reason']}")
            print()

        # Use best parameters from Phase 1 as starting point for Phase 2
        warmup_params = phase1_result["best_params"]

        # ============================================================
        # Phase 2: Streaming Gauss-Newton
        # ============================================================
        if verbose >= 1:
            print("Phase 2: Streaming Gauss-Newton...")

        phase2_failure: str | None = None
        phase2_start = time.time()
        try:
            phase2_result = self._run_phase2_gauss_newton(
                data_source=(x_data, y_data),
                initial_params=warmup_params,
            )
        except Exception as exc:
            phase2_failure = f"{type(exc).__name__}: {exc}"
            # Log unconditionally (not gated on verbose): this is an anomaly
            # report, not progress output, and verbose=0 is an explicit,
            # commonly-used mode (e.g. batch/production runs) where a caller
            # is most likely to consume result["x"] without also checking
            # result["success"] -- exactly where a silent degraded fit would
            # otherwise go unnoticed.
            _logger.warning(
                f"Phase 2 (Gauss-Newton) raised {phase2_failure}; "
                "falling back to best known parameters for Phase 3.",
            )
            fallback_params = (
                self.best_params_global
                if self.best_params_global is not None
                else warmup_params
            )
            n_params_fb = len(fallback_params)
            # Phase 3 needs JTJ/residual_sum_sq at the fallback point for the
            # covariance transform; recompute it the same way Phase 2 does
            # for its own initial JTJ (this only runs on the failure path).
            # This recompute has its own try/except: without one, a second
            # failure here (e.g. fallback_params itself degenerate) would
            # propagate out of fit() uncaught, defeating the whole point of
            # this fallback -- fit() would crash instead of degrading
            # gracefully to a success=False result.
            try:
                if self._use_scan_for_accumulation():
                    JTJ_fb, JTr_fb, residual_sum_sq_fb = self._accumulate_jtj_jtr_scan(
                        x_data,
                        y_data,
                        fallback_params,
                    )
                else:
                    chunk_size_fb = self.config.chunk_size
                    JTJ_fb = jnp.zeros((n_params_fb, n_params_fb))
                    JTr_fb = jnp.zeros(n_params_fb)
                    residual_sum_sq_fb = 0.0
                    for i in range(0, n_points, chunk_size_fb):
                        x_chunk = x_data[i : i + chunk_size_fb]
                        y_chunk = y_data[i : i + chunk_size_fb]
                        JTJ_fb, JTr_fb, res_sq = self._accumulate_jtj_jtr(
                            x_chunk,
                            y_chunk,
                            fallback_params,
                            JTJ_fb,
                            JTr_fb,
                        )
                        residual_sum_sq_fb += res_sq
                residual_sum_sq_fb = float(residual_sum_sq_fb)
            except Exception as recompute_exc:
                _logger.warning(
                    f"JTJ recompute for the Phase 2 fallback also raised "
                    f"{type(recompute_exc).__name__}: {recompute_exc}; "
                    "returning fallback_params with an uninformative (zero) "
                    "covariance instead of crashing fit().",
                )
                JTJ_fb = jnp.zeros((n_params_fb, n_params_fb))
                residual_sum_sq_fb = float("nan")
                phase2_failure = (
                    f"{phase2_failure}; JTJ recompute also failed: "
                    f"{type(recompute_exc).__name__}: {recompute_exc}"
                )
            phase2_result = {
                "final_params": fallback_params,
                "best_params": fallback_params,
                "best_cost": residual_sum_sq_fb,
                "final_cost": residual_sum_sq_fb,
                "iterations": 0,
                "convergence_reason": f"Phase 2 failed: {phase2_failure}",
                "gradient_norm": float("nan"),
                "JTJ_final": JTJ_fb,
                "residual_sum_sq": residual_sum_sq_fb,
            }
        phase2_duration = time.time() - phase2_start
        phase_timings["phase2_gauss_newton"] = phase2_duration
        phase_iterations["phase2"] = phase2_result["iterations"]

        if verbose >= 1:
            print(
                f"Phase 2 complete: {phase2_result['iterations']} iterations ({phase2_duration:.3f}s)",
            )
            print(f"  Final cost: {phase2_result['final_cost']:.6e}")
            print(f"  Convergence: {phase2_result['convergence_reason']}")
            print(f"  Gradient norm: {phase2_result['gradient_norm']:.6e}")
            print()

        # ============================================================
        # Phase 3: Denormalization and Covariance
        # ============================================================
        if verbose >= 1:
            print("Phase 3: Computing covariance...")

        phase3_start = time.time()
        phase3_result = self._run_phase3_finalize(
            optimized_params_normalized=phase2_result["final_params"],
            JTJ_final=phase2_result["JTJ_final"],
            residual_sum_sq=phase2_result["residual_sum_sq"],
            n_points=n_points,
        )
        phase3_duration = time.time() - phase3_start
        phase_timings["phase3_finalize"] = phase3_duration

        if verbose >= 1:
            print(f"Phase 3 complete ({phase3_duration:.3f}s)")
            print(f"  Residual variance (σ²): {phase3_result['sigma_sq']:.6e}")
            print()

        # ============================================================
        # Assemble Final Result
        # ============================================================
        total_duration = time.time() - total_start_time

        # Compute final residuals for 'fun' field
        final_predictions = func(x_data, *phase3_result["popt"])
        final_residuals = y_data - final_predictions

        # Build streaming diagnostics
        streaming_diagnostics = {
            "phase_timings": phase_timings,
            "phase_iterations": phase_iterations,
            "total_time": total_duration,
            "warmup_diagnostics": {
                "best_loss": phase1_result["best_loss"],
                "final_loss": phase1_result["final_loss"],
                "switch_reason": phase1_result["switch_reason"],
            },
            "gauss_newton_diagnostics": {
                "best_cost": phase2_result["best_cost"],
                "final_cost": phase2_result["final_cost"],
                "gradient_norm": phase2_result["gradient_norm"],
                "convergence_reason": phase2_result["convergence_reason"],
            },
            "phase_history": self.phase_history,
        }

        # A phase falling back to best-known params (see try/except above) is
        # a salvaged, not a genuinely converged, result -- report it as such
        # rather than claiming success=True just because Phase 3 ran.
        pipeline_failure = phase1_failure or phase2_failure
        result_message = (
            f"Fell back to best known parameters ({pipeline_failure})"
            if pipeline_failure
            else phase2_result["convergence_reason"]
        )

        # Format result dictionary (scipy-compatible + NLSQ extensions)
        result = {
            "x": phase3_result["popt"],  # Optimized parameters
            "success": pipeline_failure is None,
            "message": result_message,
            "fun": final_residuals,  # Final residuals
            "pcov": phase3_result["pcov"],  # Covariance matrix
            "perr": phase3_result["perr"],  # Standard errors (NLSQ extension)
            "streaming_diagnostics": streaming_diagnostics,  # NLSQ extension
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
            1: "Phase 1: L-BFGS Warmup",
            2: "Phase 2: Streaming Gauss-Newton",
            3: "Phase 3: Denormalization and Covariance",
        }

        return {
            "current_phase": self.current_phase,
            "phase_name": phase_names.get(self.current_phase, "Unknown"),
            "phase_history": self.phase_history,
            "total_phases": 4,
        }
