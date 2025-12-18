"""Configuration for adaptive hybrid streaming optimizer.

This module provides configuration options for the four-phase hybrid optimizer
that combines parameter normalization, Adam warmup, streaming Gauss-Newton, and
exact covariance computation.
"""

from dataclasses import dataclass


@dataclass
class HybridStreamingConfig:
    """Configuration for adaptive hybrid streaming optimizer.

    This configuration class controls all aspects of the four-phase hybrid optimizer:
    - Phase 0: Parameter normalization setup
    - Phase 1: Adam warmup with adaptive switching
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
        Number of Adam warmup iterations before checking switch criteria.
        Typical values: 100-500. More iterations allow better initial convergence
        before switching to Gauss-Newton.

    max_warmup_iterations : int, default=500
        Maximum Adam warmup iterations before forced switch to Phase 2.
        Safety limit to prevent indefinite warmup when loss plateaus slowly.

    warmup_learning_rate : float, default=0.001
        Learning rate for Adam optimizer during warmup phase.
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

    Examples
    --------
    Default configuration:

    >>> from nlsq import HybridStreamingConfig
    >>> config = HybridStreamingConfig()
    >>> config.warmup_iterations
    200

    Aggressive profile (faster convergence):

    >>> config = HybridStreamingConfig.aggressive()
    >>> config.warmup_iterations > 200
    True

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
    ...     warmup_iterations=300,
    ...     warmup_learning_rate=0.01,
    ...     chunk_size=5000,
    ...     precision='float64'
    ... )

    See Also
    --------
    AdaptiveHybridStreamingOptimizer : Optimizer that uses this configuration
    curve_fit : High-level interface with method='hybrid_streaming'

    Notes
    -----
    Based on Adaptive Hybrid Streaming Optimizer specification:
    ``agent-os/specs/2025-12-18-adaptive-hybrid-streaming-optimizer/spec.md``
    """

    # Phase 0: Parameter normalization
    normalize: bool = True
    normalization_strategy: str = "auto"

    # Phase 1: Adam warmup
    warmup_iterations: int = 200
    max_warmup_iterations: int = 500
    warmup_learning_rate: float = 0.001
    loss_plateau_threshold: float = 1e-4
    gradient_norm_threshold: float = 1e-3
    active_switching_criteria: list = None

    # Optax enhancements
    use_learning_rate_schedule: bool = False
    lr_schedule_warmup_steps: int = 50
    lr_schedule_decay_steps: int = 450
    lr_schedule_end_value: float = 0.0001
    gradient_clip_value: float | None = None  # None = no clipping, e.g., 1.0 for clipping

    # Phase 2: Gauss-Newton
    gauss_newton_max_iterations: int = 100
    gauss_newton_tol: float = 1e-8
    trust_region_initial: float = 1.0
    regularization_factor: float = 1e-10

    # Streaming configuration
    chunk_size: int = 10000

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

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set default for mutable default (list)
        if self.active_switching_criteria is None:
            self.active_switching_criteria = ["plateau", "gradient", "max_iter"]

        # Validate normalization strategy
        valid_strategies = ("auto", "bounds", "p0", "none")
        assert self.normalization_strategy in valid_strategies, (
            f"normalization_strategy must be one of: {valid_strategies}, "
            f"got: {self.normalization_strategy}"
        )

        # Validate precision
        valid_precisions = ("float32", "float64", "auto")
        assert self.precision in valid_precisions, (
            f"precision must be one of: {valid_precisions}, got: {self.precision}"
        )

        # Validate warmup iterations constraint
        assert self.warmup_iterations <= self.max_warmup_iterations, (
            f"warmup_iterations ({self.warmup_iterations}) must be <= "
            f"max_warmup_iterations ({self.max_warmup_iterations})"
        )

        # Validate positive values
        assert self.warmup_iterations >= 0, (
            "warmup_iterations must be non-negative"
        )
        assert self.max_warmup_iterations > 0, (
            "max_warmup_iterations must be positive"
        )
        assert self.warmup_learning_rate > 0, (
            "warmup_learning_rate must be positive"
        )
        assert self.loss_plateau_threshold > 0, (
            "loss_plateau_threshold must be positive"
        )
        assert self.gradient_norm_threshold > 0, (
            "gradient_norm_threshold must be positive"
        )
        assert self.gauss_newton_max_iterations > 0, (
            "gauss_newton_max_iterations must be positive"
        )
        assert self.gauss_newton_tol > 0, (
            "gauss_newton_tol must be positive"
        )
        assert self.trust_region_initial > 0, (
            "trust_region_initial must be positive"
        )
        assert self.regularization_factor >= 0, (
            "regularization_factor must be non-negative"
        )
        assert self.chunk_size > 0, (
            "chunk_size must be positive"
        )
        assert self.checkpoint_frequency > 0, (
            "checkpoint_frequency must be positive"
        )
        assert self.callback_frequency > 0, (
            "callback_frequency must be positive"
        )

        # Validate Optax enhancement parameters
        if self.use_learning_rate_schedule:
            assert self.lr_schedule_warmup_steps >= 0, (
                "lr_schedule_warmup_steps must be non-negative"
            )
            assert self.lr_schedule_decay_steps > 0, (
                "lr_schedule_decay_steps must be positive"
            )
            assert self.lr_schedule_end_value > 0, (
                "lr_schedule_end_value must be positive"
            )

        if self.gradient_clip_value is not None:
            assert self.gradient_clip_value > 0, (
                "gradient_clip_value must be positive"
            )

    @classmethod
    def aggressive(cls):
        """Create aggressive profile: faster convergence, more warmup, looser tolerances.

        This preset prioritizes speed over robustness:
        - More warmup iterations for better initial convergence
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
        """
        return cls(
            # More warmup for better Phase 1 convergence
            warmup_iterations=300,
            max_warmup_iterations=800,
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
        - Less warmup, rely more on Gauss-Newton
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
        """
        return cls(
            # Less warmup, rely on Gauss-Newton
            warmup_iterations=100,
            max_warmup_iterations=300,
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
        - Conservative warmup to limit memory allocation
        - Enable checkpoints for recovery (important when memory is tight)
        - float32 precision for 50% memory reduction

        Returns
        -------
        HybridStreamingConfig
            Configuration with memory-optimized settings.

        Examples
        --------
        >>> config = HybridStreamingConfig.memory_optimized()
        >>> config.chunk_size
        5000
        """
        return cls(
            # Smaller chunks for memory efficiency
            chunk_size=5000,
            # Conservative warmup to reduce memory
            warmup_iterations=150,
            max_warmup_iterations=400,
            # Use float32 for 50% memory reduction
            precision="float32",
            # Enable checkpoints (important when memory tight)
            enable_checkpoints=True,
            checkpoint_frequency=50,  # More frequent saves
            # Keep other defaults
        )
