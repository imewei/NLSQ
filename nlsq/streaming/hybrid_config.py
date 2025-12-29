"""Configuration for adaptive hybrid streaming optimizer.

This module provides configuration options for the four-phase hybrid optimizer
that combines parameter normalization, L-BFGS warmup, streaming Gauss-Newton, and
exact covariance computation.
"""

from dataclasses import dataclass
from typing import Literal

from nlsq.streaming.validators import (
    validate_cg_config,
    validate_defense_layer_config,
    validate_gauss_newton_config,
    validate_gradient_clip,
    validate_group_variance_config,
    validate_lbfgs_config,
    validate_loop_strategy,
    validate_lr_schedule_config,
    validate_multistart_config,
    validate_normalization_strategy,
    validate_precision,
    validate_streaming_config,
    validate_warmup_config,
)


@dataclass(slots=True)
class HybridStreamingConfig:
    """Configuration for adaptive hybrid streaming optimizer.

    This configuration class controls all aspects of the four-phase hybrid optimizer:
    - Phase 0: Parameter normalization setup
    - Phase 1: L-BFGS warmup with adaptive switching
    - Phase 2: Streaming Gauss-Newton with exact J^T J accumulation
    - Phase 3: Denormalization and covariance transform

    Parameters
    ----------
    normalize : bool, default=True
        Enable parameter normalization. When True, parameters are normalized to
        similar scales to improve gradient signal quality and convergence speed.

    normalization_strategy : str, default='auto'
        Strategy for parameter normalization. Options:

        - **'auto'**: Use bounds-based if bounds provided, else p0-based
        - **'bounds'**: Normalize to [0, 1] using parameter bounds
        - **'p0'**: Scale by initial parameter magnitudes
        - **'none'**: Identity transform (no normalization)

    warmup_iterations : int, default=200
        Number of L-BFGS warmup iterations before checking switch criteria.
        With L-BFGS, typical values are 20-50 (5-10x fewer than Adam).
        More iterations allow better initial convergence before switching
        to Gauss-Newton.

    max_warmup_iterations : int, default=500
        Maximum L-BFGS warmup iterations before forced switch to Phase 2.
        Safety limit to prevent indefinite warmup when loss plateaus slowly.

    warmup_learning_rate : float, default=0.001
        Learning rate for optimizer during warmup phase.
        Typical values: 0.0001-0.01. Higher values converge faster but may overshoot.

    loss_plateau_threshold : float, default=1e-4
        Relative loss improvement threshold for plateau detection.
        Switch to Phase 2 if: abs(loss - prev_loss) / (abs(prev_loss) + eps) < threshold.
        Smaller values = stricter plateau detection = later switching.

    gradient_norm_threshold : float, default=1e-3
        Gradient norm threshold for early Phase 2 switch.
        Switch to Phase 2 if: ||gradient|| < threshold.
        Indicates optimization is close to optimum and Gauss-Newton will be effective.

    active_switching_criteria : list, default=['plateau', 'gradient', 'max_iter']
        List of active switching criteria for Phase 1 -> Phase 2 transition.
        Available criteria:

        - **'plateau'**: Loss plateau detection (loss_plateau_threshold)
        - **'gradient'**: Gradient norm below threshold (gradient_norm_threshold)
        - **'max_iter'**: Maximum iterations reached (max_warmup_iterations)

        Switch occurs when ANY active criterion is met.

    lbfgs_history_size : int, default=10
        Number of previous gradients and updates to store for L-BFGS Hessian
        approximation. Standard default from SciPy, PyTorch, and Nocedal & Wright.
        Larger values give better Hessian approximation but use more memory.

    lbfgs_initial_step_size : float, default=0.1
        Initial step size for L-BFGS during cold start (first m iterations
        while history buffer fills). Small value prevents overshooting when
        Hessian approximation is poor (identity matrix initially).

    lbfgs_line_search : str, default='wolfe'
        Line search method for L-BFGS step acceptance. Options:

        - **'wolfe'**: Standard Wolfe conditions (default)
        - **'strong_wolfe'**: Strong Wolfe conditions (stricter)
        - **'backtracking'**: Simple backtracking line search

    lbfgs_exploration_step_size : float, default=0.1
        L-BFGS initial step size for exploration mode (high relative loss).
        Small value prevents first "Hessian=Identity" step from overshooting.

    lbfgs_refinement_step_size : float, default=1.0
        L-BFGS initial step size for refinement mode (low relative loss).
        Larger value leverages L-BFGS's near-Newton convergence speed when
        close to optimum.

    gauss_newton_max_iterations : int, default=100
        Maximum iterations for Phase 2 Gauss-Newton optimization.
        Typical values: 50-200.

    gauss_newton_tol : float, default=1e-8
        Convergence tolerance for Phase 2 (gradient norm threshold).
        Optimization stops if: ||gradient|| < tol.

    trust_region_initial : float, default=1.0
        Initial trust region radius for Gauss-Newton step control.
        Radius is adapted based on actual vs predicted reduction ratio.

    regularization_factor : float, default=1e-10
        Regularization factor for rank-deficient J^T J matrices.
        Added to diagonal: J^T J + regularization_factor * I.

    cg_max_iterations : int, default=100
        Maximum iterations for Conjugate Gradient solver in Phase 2.
        Used when parameter count exceeds cg_param_threshold.
        Higher values allow better convergence but more computation.

    cg_relative_tolerance : float, default=1e-4
        Relative tolerance for CG solver convergence.
        Convergence check: ||r|| < cg_relative_tolerance * ||J^T r_0||.
        Implements Inexact Newton strategy for efficiency.

    cg_absolute_tolerance : float, default=1e-10
        Absolute tolerance floor for CG solver convergence.
        Safety floor to prevent over-iteration on well-conditioned systems.

    cg_param_threshold : int, default=2000
        Parameter count threshold for auto-selecting CG vs materialized solver.

        - **p < threshold**: Use materialized J^T J with SVD solve (faster for small p)
        - **p >= threshold**: Use CG with implicit matvec (O(p) memory vs O(p^2))

        Threshold balances memory savings vs additional data passes for CG.

    enable_group_variance_regularization : bool, default=False
        Enable variance regularization for parameter groups. When enabled,
        adds a penalty term to the loss function that penalizes variance
        within specified parameter groups. This is essential for preventing
        per-angle parameter absorption in XPCS laminar flow fitting.

        The regularized loss becomes ``L = MSE + group_variance_lambda *
        sum(Var(group_i))`` where each group_i is a slice of parameters
        defined by group_variance_indices.

    group_variance_lambda : float, default=0.01
        Regularization strength for group variance penalty. Larger values
        more strongly penalize variance within parameter groups. Use 0.001-0.01
        for light regularization (allows moderate group variation), 0.1-1.0
        for moderate regularization (constrains groups to be similar), or
        10-1000 for strong regularization (forces groups to be nearly uniform).
        For XPCS with per-angle scaling, use ``lambda ~ 0.1 * n_data /
        (n_phi * sigma^2)`` where sigma is the expected experimental
        variation (~0.05 for 5%).

    group_variance_indices : list of tuple, default=None
        List of (start, end) tuples defining parameter groups for variance
        regularization. Each tuple specifies a slice [start:end] of the
        parameter vector that should have low internal variance.

        Example for XPCS with 23 angles: ``group_variance_indices = [(0, 23),
        (23, 46)]`` constrains contrast params [0:23] and offset params [23:46]
        to each have low variance, preventing them from absorbing
        angle-dependent physical signals.

        If None when enable_group_variance_regularization=True, no groups
        are regularized (effectively disabling the feature).

    chunk_size : int, default=10000
        Size of data chunks for streaming J^T J accumulation.
        Larger chunks = faster but more memory. Typical: 5000-50000.

    enable_checkpoints : bool, default=True
        Enable checkpoint save/resume for fault tolerance.

    checkpoint_frequency : int, default=100
        Save checkpoint every N iterations (across all phases).

    validate_numerics : bool, default=True
        Enable NaN/Inf validation at gradient, parameter, and loss computation points.

    precision : str, default='auto'
        Numerical precision strategy. Options:

        - **'auto'**: float32 for Phase 1 warmup, float64 for Phase 2+ (recommended)
        - **'float32'**: Use float32 throughout (faster, less memory)
        - **'float64'**: Use float64 throughout (more stable)

    enable_multi_device : bool, default=False
        Enable multi-GPU/TPU parallelism for Jacobian computation.
        Uses JAX pmap for data-parallel computation across devices.

    callback_frequency : int, default=10
        Call progress callback every N iterations (if callback provided).

    enable_multistart : bool, default=False
        Enable multi-start optimization with tournament selection during Phase 1.
        When enabled, generates multiple starting points using LHS sampling and
        uses tournament elimination to select the best candidate for Phase 2.

    n_starts : int, default=10
        Number of starting points for multi-start optimization.
        Only used when enable_multistart=True.

    multistart_sampler : str, default='lhs'
        Sampling method for generating starting points.
        Options: 'lhs' (Latin Hypercube), 'sobol', 'halton'.

    elimination_rounds : int, default=3
        Number of tournament elimination rounds.
        Each round eliminates elimination_fraction of candidates.

    elimination_fraction : float, default=0.5
        Fraction of candidates to eliminate per round.
        Must be in (0, 1). Default 0.5 = eliminate half each round.

    batches_per_round : int, default=50
        Number of data batches to use for evaluation in each tournament round.
        More batches = more reliable selection but slower.

    Examples
    --------
    Default configuration:

    >>> from nlsq import HybridStreamingConfig
    >>> config = HybridStreamingConfig()
    >>> config.warmup_iterations
    200

    Aggressive profile (faster convergence with L-BFGS):

    >>> config = HybridStreamingConfig.aggressive()
    >>> config.warmup_iterations
    50

    Conservative profile (higher quality):

    >>> config = HybridStreamingConfig.conservative()
    >>> config.gauss_newton_tol < 1e-8
    True

    Memory-optimized profile:

    >>> config = HybridStreamingConfig.memory_optimized()
    >>> config.chunk_size < 10000
    True

    Custom configuration:

    >>> config = HybridStreamingConfig(
    ...     warmup_iterations=50,
    ...     lbfgs_history_size=15,
    ...     chunk_size=5000,
    ...     precision='float64'
    ... )

    With multi-start tournament selection:

    >>> config = HybridStreamingConfig(
    ...     enable_multistart=True,
    ...     n_starts=20,
    ...     elimination_rounds=3,
    ...     batches_per_round=50,
    ... )

    See Also
    --------
    AdaptiveHybridStreamingOptimizer : Optimizer that uses this configuration
    curve_fit : High-level interface with method='hybrid_streaming'
    TournamentSelector : Tournament selection for multi-start optimization

    Notes
    -----
    Based on Adaptive Hybrid Streaming Optimizer specification:
    ``agent-os/specs/2025-12-18-adaptive-hybrid-streaming-optimizer/spec.md``

    L-BFGS replaces Adam for warmup, providing 5-10x faster convergence to the
    basin of attraction through approximate Hessian information.
    """

    # Phase 0: Parameter normalization
    normalize: bool = True
    normalization_strategy: str = "auto"

    # Phase 1: L-BFGS warmup
    warmup_iterations: int = 200
    max_warmup_iterations: int = 500
    warmup_learning_rate: float = 0.001
    loss_plateau_threshold: float = 1e-4
    gradient_norm_threshold: float = 1e-3
    active_switching_criteria: list[str] | None = None

    # L-BFGS configuration parameters
    lbfgs_history_size: int = 10  # Standard default (SciPy, PyTorch, Nocedal & Wright)
    lbfgs_initial_step_size: float = 0.1  # Cold start scaffolding
    lbfgs_line_search: Literal["wolfe", "strong_wolfe", "backtracking"] = "wolfe"
    lbfgs_exploration_step_size: float = 0.1  # For high relative loss (exploration)
    lbfgs_refinement_step_size: float = 1.0  # For low relative loss (refinement)

    # Optax enhancements
    use_learning_rate_schedule: bool = False
    lr_schedule_warmup_steps: int = 50
    lr_schedule_decay_steps: int = 450
    lr_schedule_end_value: float = 0.0001
    gradient_clip_value: float | None = (
        None  # None = no clipping, e.g., 1.0 for clipping
    )

    # 4-Layer Defense Strategy for Warmup Divergence Prevention
    # Layer 1: Warm Start Detection - skip warmup if already near optimum
    enable_warm_start_detection: bool = True
    warm_start_threshold: float = 0.01  # Skip if relative_loss < this

    # Layer 2: Adaptive Learning Rate - scale LR based on initial loss quality
    enable_adaptive_warmup_lr: bool = True
    warmup_lr_refinement: float = 1e-6  # For relative_loss < 0.1 (excellent)
    warmup_lr_careful: float = 1e-5  # For relative_loss < 1.0 (good)
    # warmup_learning_rate (0.001) used for relative_loss >= 1.0 (poor)

    # Layer 3: Cost-Increase Guard - abort if loss increases during warmup
    enable_cost_guard: bool = True
    cost_increase_tolerance: float = 0.05  # Abort if loss > initial * 1.05

    # Layer 4: Trust Region Constraint - clip update magnitude
    enable_step_clipping: bool = True
    max_warmup_step_size: float = 0.1  # Max L2 norm of parameter update

    # Phase 2: Gauss-Newton
    gauss_newton_max_iterations: int = 100
    gauss_newton_tol: float = 1e-8
    trust_region_initial: float = 1.0
    regularization_factor: float = 1e-10

    # CG-based Gauss-Newton solver configuration
    # These parameters control the Conjugate Gradient solver used when
    # the parameter count exceeds cg_param_threshold
    cg_max_iterations: int = 100  # Cap for high-p problems
    cg_relative_tolerance: float = 1e-4  # Multiplier for ||J^T r|| (Inexact Newton)
    cg_absolute_tolerance: float = 1e-10  # Safety floor
    cg_param_threshold: int = 2000  # Auto-select threshold: p < this -> materialized

    # Group variance regularization (for per-angle parameter absorption prevention)
    enable_group_variance_regularization: bool = False
    group_variance_lambda: float = 0.01
    group_variance_indices: list[tuple[int, int]] | None = None

    # Streaming configuration
    chunk_size: int = 10000

    # Loop strategy for chunk accumulation
    # 'auto': Use scan on GPU/TPU (better fusion), Python loops on CPU (lower overhead)
    # 'scan': Always use JAX lax.scan (best for GPU/TPU)
    # 'loop': Always use Python loops (best for CPU)
    loop_strategy: Literal["auto", "scan", "loop"] = "auto"

    # Fault tolerance
    enable_checkpoints: bool = True
    checkpoint_frequency: int = 100
    checkpoint_dir: str | None = None
    resume_from_checkpoint: str | None = None
    validate_numerics: bool = True
    enable_fault_tolerance: bool = True
    max_retries_per_batch: int = 2
    min_success_rate: float = 0.5

    # Precision control
    precision: str = "auto"

    # Multi-device support
    enable_multi_device: bool = False

    # Progress monitoring
    callback_frequency: int = 10
    verbose: int = 1  # Verbosity level: 0=silent, 1=progress, 2=debug
    log_frequency: int = 1  # Log every N iterations in Phase 2

    # Multi-start optimization with tournament selection
    enable_multistart: bool = False
    n_starts: int = 10
    multistart_sampler: Literal["lhs", "sobol", "halton"] = "lhs"
    elimination_rounds: int = 3
    elimination_fraction: float = 0.5
    batches_per_round: int = 50
    center_on_p0: bool = True
    scale_factor: float = 1.0

    def __post_init__(self):
        """Validate configuration after initialization.

        Delegates to specialized validator functions for each configuration group.
        ConfigValidationError from validators is re-raised as ValueError for
        backwards compatibility.
        """
        from nlsq.streaming.validators import ConfigValidationError

        # Set default for mutable default (list)
        if self.active_switching_criteria is None:
            self.active_switching_criteria = ["plateau", "gradient", "max_iter"]

        try:
            # Validate enum-like parameters
            validate_normalization_strategy(self.normalization_strategy)
            validate_precision(self.precision)
            validate_loop_strategy(self.loop_strategy)

            # Validate Phase 1 warmup configuration
            validate_warmup_config(
                warmup_iterations=self.warmup_iterations,
                max_warmup_iterations=self.max_warmup_iterations,
                warmup_learning_rate=self.warmup_learning_rate,
                loss_plateau_threshold=self.loss_plateau_threshold,
                gradient_norm_threshold=self.gradient_norm_threshold,
            )

            # Validate L-BFGS configuration
            validate_lbfgs_config(
                history_size=self.lbfgs_history_size,
                initial_step_size=self.lbfgs_initial_step_size,
                line_search=self.lbfgs_line_search,
                exploration_step_size=self.lbfgs_exploration_step_size,
                refinement_step_size=self.lbfgs_refinement_step_size,
            )

            # Validate Phase 2 Gauss-Newton configuration
            validate_gauss_newton_config(
                max_iterations=self.gauss_newton_max_iterations,
                tol=self.gauss_newton_tol,
                trust_region_initial=self.trust_region_initial,
                regularization_factor=self.regularization_factor,
            )

            # Validate CG solver configuration
            validate_cg_config(
                max_iterations=self.cg_max_iterations,
                relative_tolerance=self.cg_relative_tolerance,
                absolute_tolerance=self.cg_absolute_tolerance,
                param_threshold=self.cg_param_threshold,
            )

            # Validate group variance regularization
            validate_group_variance_config(
                enabled=self.enable_group_variance_regularization,
                lambda_val=self.group_variance_lambda,
                indices=self.group_variance_indices,
            )

            # Validate streaming configuration
            validate_streaming_config(
                chunk_size=self.chunk_size,
                checkpoint_frequency=self.checkpoint_frequency,
                callback_frequency=self.callback_frequency,
            )

            # Validate learning rate schedule
            validate_lr_schedule_config(
                enabled=self.use_learning_rate_schedule,
                warmup_steps=self.lr_schedule_warmup_steps,
                decay_steps=self.lr_schedule_decay_steps,
                end_value=self.lr_schedule_end_value,
            )

            # Validate gradient clipping
            validate_gradient_clip(self.gradient_clip_value)

            # Validate 4-layer defense strategy
            validate_defense_layer_config(
                enable_warm_start=self.enable_warm_start_detection,
                warm_start_threshold=self.warm_start_threshold,
                enable_adaptive_lr=self.enable_adaptive_warmup_lr,
                lr_refinement=self.warmup_lr_refinement,
                lr_careful=self.warmup_lr_careful,
                lr_default=self.warmup_learning_rate,
                enable_cost_guard=self.enable_cost_guard,
                cost_tolerance=self.cost_increase_tolerance,
                enable_step_clipping=self.enable_step_clipping,
                max_step_size=self.max_warmup_step_size,
            )

            # Validate multi-start configuration
            validate_multistart_config(
                enabled=self.enable_multistart,
                n_starts=self.n_starts,
                sampler=self.multistart_sampler,
                elimination_fraction=self.elimination_fraction,
                elimination_rounds=self.elimination_rounds,
                batches_per_round=self.batches_per_round,
                scale_factor=self.scale_factor,
            )

        except ConfigValidationError as e:
            # Re-raise as ValueError for backwards compatibility
            raise ValueError(str(e)) from e

    @classmethod
    def aggressive(cls):
        """Create aggressive profile: faster convergence with L-BFGS, looser tolerances.

        This preset prioritizes speed over robustness:
        - L-BFGS warmup with reduced iterations (50 vs 300 with Adam)
        - Higher learning rate for faster progress
        - Looser tolerances for earlier Phase 2 switching
        - Larger chunks for better throughput

        Returns
        -------
        HybridStreamingConfig
            Configuration with aggressive settings.

        Examples
        --------
        >>> config = HybridStreamingConfig.aggressive()
        >>> config.warmup_learning_rate
        0.003
        >>> config.warmup_iterations
        50
        """
        return cls(
            # L-BFGS warmup with reduced iterations (5-10x fewer than Adam)
            warmup_iterations=50,
            max_warmup_iterations=100,
            # Higher learning rate for faster progress
            warmup_learning_rate=0.003,
            # Looser tolerances for faster switching
            loss_plateau_threshold=5e-4,
            gradient_norm_threshold=5e-3,
            gauss_newton_tol=1e-7,
            # Larger chunks for throughput
            chunk_size=20000,
            # Keep other defaults
        )

    @classmethod
    def conservative(cls):
        """Create conservative profile: slower but robust, tighter tolerances.

        This preset prioritizes solution quality over speed:
        - L-BFGS warmup with conservative iterations
        - Lower learning rate for stability
        - Tighter tolerances for higher quality
        - More Gauss-Newton iterations

        Returns
        -------
        HybridStreamingConfig
            Configuration with conservative settings.

        Examples
        --------
        >>> config = HybridStreamingConfig.conservative()
        >>> config.gauss_newton_tol
        1e-10
        >>> config.warmup_iterations
        30
        """
        return cls(
            # L-BFGS warmup with reduced iterations, rely on Gauss-Newton
            warmup_iterations=30,
            max_warmup_iterations=80,
            # Lower learning rate for stability
            warmup_learning_rate=0.0003,
            # Tighter tolerances for quality
            loss_plateau_threshold=1e-5,
            gradient_norm_threshold=1e-4,
            gauss_newton_tol=1e-10,
            # More Gauss-Newton iterations
            gauss_newton_max_iterations=200,
            # Smaller trust region for safety
            trust_region_initial=0.5,
            # Keep other defaults
        )

    @classmethod
    def memory_optimized(cls):
        """Create memory-optimized profile: smaller chunks, efficient settings.

        This preset minimizes memory footprint:
        - Smaller chunks to reduce memory usage
        - L-BFGS warmup with reduced iterations
        - Enable checkpoints for recovery (important when memory is tight)
        - float32 precision for 50% memory reduction
        - Lower CG threshold for more aggressive CG usage (avoids O(p^2) J^T J)

        Returns
        -------
        HybridStreamingConfig
            Configuration with memory-optimized settings.

        Examples
        --------
        >>> config = HybridStreamingConfig.memory_optimized()
        >>> config.chunk_size
        5000
        >>> config.warmup_iterations
        40
        >>> config.cg_param_threshold
        1000
        """
        return cls(
            # Smaller chunks for memory efficiency
            chunk_size=5000,
            # L-BFGS warmup with reduced iterations
            warmup_iterations=40,
            max_warmup_iterations=100,
            # Use float32 for 50% memory reduction
            precision="float32",
            # Enable checkpoints (important when memory tight)
            enable_checkpoints=True,
            checkpoint_frequency=50,  # More frequent saves
            # More aggressive CG usage to avoid O(p^2) J^T J storage
            cg_param_threshold=1000,  # Lower threshold for memory savings
            # Keep other defaults
        )

    @classmethod
    def with_multistart(cls, n_starts: int = 10, **kwargs):
        """Create configuration with multi-start tournament selection enabled.

        This preset enables multi-start optimization for finding global optima:
        - Tournament selection during Phase 1 warmup
        - LHS sampling for generating starting points
        - Progressive elimination to select best candidate

        Parameters
        ----------
        n_starts : int, default=10
            Number of starting points to generate.
        **kwargs
            Additional configuration parameters to override.

        Returns
        -------
        HybridStreamingConfig
            Configuration with multi-start enabled.

        Examples
        --------
        >>> config = HybridStreamingConfig.with_multistart(n_starts=20)
        >>> config.enable_multistart
        True
        >>> config.n_starts
        20
        """
        return cls(
            enable_multistart=True,
            n_starts=n_starts,
            **kwargs,
        )

    # =========================================================================
    # Defense Layer Sensitivity Presets
    # =========================================================================

    @classmethod
    def defense_strict(cls):
        """Create strict defense layer profile for near-optimal scenarios.

        This preset maximizes protection against divergence when initial
        parameters are expected to be close to optimal (warm starts, refinement):
        - Very low warm start threshold (triggers at 1% relative loss)
        - Ultra-conservative learning rates for refinement
        - Very tight cost guard tolerance (5% increase aborts)
        - Very small step clipping for stability

        Use this when:
        - Continuing optimization from a previous fit
        - Refining parameters that are already close to optimal
        - Dealing with ill-conditioned problems
        - Prioritizing stability over speed

        Returns
        -------
        HybridStreamingConfig
            Configuration with strict defense layer settings.

        Examples
        --------
        >>> config = HybridStreamingConfig.defense_strict()
        >>> config.warm_start_threshold
        0.01
        >>> config.cost_increase_tolerance
        0.05
        >>> config.warmup_iterations
        25
        """
        return cls(
            # All defense layers enabled
            enable_warm_start_detection=True,
            enable_adaptive_warmup_lr=True,
            enable_cost_guard=True,
            enable_step_clipping=True,
            # Layer 1: Very low threshold (1% relative loss triggers warm start)
            warm_start_threshold=0.01,
            # Layer 2: Ultra-conservative LR progression
            warmup_lr_refinement=1e-7,
            warmup_lr_careful=1e-6,
            warmup_learning_rate=0.0005,
            # Layer 3: Very tight cost guard (5% increase aborts)
            cost_increase_tolerance=0.05,
            # Layer 4: Very small steps
            max_warmup_step_size=0.05,
            # L-BFGS warmup with reduced iterations
            warmup_iterations=25,
            max_warmup_iterations=60,
        )

    @classmethod
    def defense_relaxed(cls):
        """Create relaxed defense layer profile for exploration-heavy scenarios.

        This preset reduces defense layer sensitivity for problems where
        significant parameter exploration is needed:
        - Higher warm start threshold (50% relative loss needed to skip)
        - More aggressive learning rates for exploration
        - Generous cost guard tolerance (50% increase allowed)
        - Larger step clipping for faster exploration

        Use this when:
        - Starting from a rough initial guess
        - Exploring a wide parameter space
        - Problems with multiple local minima
        - Speed is more important than robustness

        Returns
        -------
        HybridStreamingConfig
            Configuration with relaxed defense layer settings.

        Examples
        --------
        >>> config = HybridStreamingConfig.defense_relaxed()
        >>> config.warm_start_threshold
        0.5
        >>> config.cost_increase_tolerance
        0.5
        >>> config.warmup_iterations
        50
        """
        return cls(
            # All defense layers enabled but relaxed
            enable_warm_start_detection=True,
            enable_adaptive_warmup_lr=True,
            enable_cost_guard=True,
            enable_step_clipping=True,
            # Layer 1: High threshold (50% relative loss triggers warm start)
            warm_start_threshold=0.5,
            # Layer 2: Aggressive LR progression
            warmup_lr_refinement=1e-5,
            warmup_lr_careful=1e-4,
            warmup_learning_rate=0.003,
            # Layer 3: Generous cost guard (50% increase allowed)
            cost_increase_tolerance=0.5,
            # Layer 4: Larger steps for exploration
            max_warmup_step_size=0.5,
            # L-BFGS warmup with reduced iterations
            warmup_iterations=50,
            max_warmup_iterations=120,
        )

    @classmethod
    def defense_disabled(cls):
        """Create profile with all defense layers disabled.

        This preset completely disables the 4-layer defense strategy,
        reverting to pre-0.3.6 behavior. Use with caution as this
        removes protection against warmup divergence.

        Use this when:
        - Debugging to isolate defense layer effects
        - Benchmarking without defense overhead
        - Backward compatibility with older code is required

        Returns
        -------
        HybridStreamingConfig
            Configuration with all defense layers disabled.

        Examples
        --------
        >>> config = HybridStreamingConfig.defense_disabled()
        >>> config.enable_warm_start_detection
        False
        """
        return cls(
            enable_warm_start_detection=False,
            enable_adaptive_warmup_lr=False,
            enable_cost_guard=False,
            enable_step_clipping=False,
        )

    @classmethod
    def scientific_default(cls):
        """Create profile optimized for scientific computing workflows.

        This preset is tuned for scientific fitting scenarios like XPCS,
        scattering, spectroscopy, and other physics-based models:
        - Balanced defense layers that protect without being too aggressive
        - Float64 precision for numerical accuracy
        - L-BFGS warmup with moderate iterations
        - Enabled checkpoints for long-running fits

        Use this when:
        - Fitting physics-based models (XPCS, scattering, decay curves)
        - Numerical precision is important
        - Parameters may have multiple scales
        - Reproducibility is required

        Returns
        -------
        HybridStreamingConfig
            Configuration optimized for scientific computing.

        Examples
        --------
        >>> config = HybridStreamingConfig.scientific_default()
        >>> config.precision
        'float64'
        >>> config.warmup_iterations
        35
        """
        return cls(
            # All defense layers enabled with balanced settings
            enable_warm_start_detection=True,
            enable_adaptive_warmup_lr=True,
            enable_cost_guard=True,
            enable_step_clipping=True,
            # Layer 1: Moderate threshold
            warm_start_threshold=0.05,
            # Layer 2: Balanced LR progression
            warmup_lr_refinement=1e-6,
            warmup_lr_careful=1e-5,
            warmup_learning_rate=0.001,
            # Layer 3: Moderate cost guard
            cost_increase_tolerance=0.2,
            # Layer 4: Moderate step clipping
            max_warmup_step_size=0.1,
            # L-BFGS warmup with reduced iterations
            warmup_iterations=35,
            max_warmup_iterations=100,
            # Scientific computing settings
            precision="float64",
            gauss_newton_tol=1e-10,
            gauss_newton_max_iterations=200,
            # Enable checkpoints for long jobs
            enable_checkpoints=True,
            checkpoint_frequency=100,
        )
