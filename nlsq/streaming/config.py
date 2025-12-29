"""Configuration for streaming optimizer with fault tolerance support.

This module provides configuration options for the streaming optimizer, including
comprehensive fault tolerance features for production-ready optimizations.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class StreamingConfig:
    """Configuration for streaming optimization with fault tolerance.

    This configuration class controls all aspects of the streaming optimizer,
    including basic optimization parameters, fault tolerance features, checkpoint
    settings, and diagnostic options.

    Parameters
    ----------
    batch_size : int, default=32
        Size of batches to process. Larger batches are more memory-intensive but
        may converge faster. Typical values: 32-256.

    max_epochs : int, default=10
        Maximum number of epochs to run. Each epoch processes the full dataset once.

    learning_rate : float, default=0.001
        Base learning rate for SGD/Adam optimizer. Typical values: 0.0001-0.01.
        Smaller values are more stable but slower to converge.

    momentum : float, default=0.9
        Momentum factor for SGD optimizer (ignored if use_adam=True).
        Higher values (0.9-0.99) provide smoother convergence. Range: [0, 1].

    use_adam : bool, default=True
        Whether to use Adam optimizer instead of SGD with momentum.
        Adam generally converges faster and requires less tuning.

    adam_beta1 : float, default=0.9
        Beta1 parameter for Adam optimizer (exponential decay rate for first moment).
        Range: [0, 1). Typical: 0.9.

    adam_beta2 : float, default=0.999
        Beta2 parameter for Adam optimizer (exponential decay rate for second moment).
        Range: [0, 1). Typical: 0.999.

    adam_eps : float, default=1e-8
        Epsilon parameter for Adam optimizer (numerical stability).
        Typical: 1e-8.

    gradient_clip : float, default=1.0
        Maximum gradient norm for clipping. Prevents exploding gradients.
        Set higher (5-10) for models with large gradients, lower (0.5-1.0) for
        stable models.

    warmup_steps : int, default=100
        Number of warmup steps for learning rate schedule. Learning rate linearly
        increases from 0 to `learning_rate` over first `warmup_steps` iterations.
        Set to 0 to disable warmup.

    convergence_tol : float, default=1e-6
        Convergence tolerance for loss changes. Optimization stops if loss change
        over last 10 batches is less than this value. Typical: 1e-6 to 1e-4.

    checkpoint_dir : str, default="checkpoints"
        Directory to save checkpoints. Created automatically if it doesn't exist.

    checkpoint_frequency : int, default=100
        Save checkpoint every N iterations. More frequent saves (50-100) enable
        finer-grained resume but increase I/O. Less frequent (500-1000) reduces
        overhead.

    checkpoint_interval : int, default=100
        Alias for checkpoint_frequency (backward compatibility).

    enable_checkpoints : bool, default=True
        Whether to enable checkpointing. Disable for maximum performance if
        interruption recovery is not needed.

    resume_from_checkpoint : Union[bool, str, None], default=None
        Resume from checkpoint control:

        - None/False: Start fresh (no resume)
        - True: Auto-detect latest checkpoint in `checkpoint_dir`
        - str: Load from specific checkpoint path

        Examples:
            - `resume_from_checkpoint=True` - Auto-detect latest
            - `resume_from_checkpoint="checkpoints/checkpoint_iter_500.h5"` - Specific

    enable_fault_tolerance : bool, default=True
        Master switch for fault tolerance features. When True, enables:

        - Best parameter tracking
        - NaN/Inf detection (if validate_numerics=True)
        - Adaptive retry strategies
        - Batch statistics collection
        - Detailed diagnostics

        Set to False for fast mode (<1% overhead) when data is trusted.

    validate_numerics : bool, default=True
        Whether to validate for NaN/Inf values at three validation points:

        1. After gradient computation
        2. After parameter update
        3. After loss calculation

        Validation failures cause batch skip with warning. Disable only if you
        are certain your model and data never produce NaN/Inf.

    min_success_rate : float, default=0.5
        Minimum batch success rate required for optimization to succeed.
        Range: [0, 1]. Common values:

        - 0.3-0.4: Permissive (for very noisy data)
        - 0.5-0.6: Standard (default)
        - 0.7-0.9: Strict (for clean data, high reliability requirements)

        Optimization fails if success rate falls below this threshold, but
        best parameters found are still returned.

    max_retries_per_batch : int, default=2
        Maximum retry attempts per batch. Each retry uses adaptive strategy
        based on error type:

        - NaN/Inf: Reduce learning rate by 50% per attempt
        - Singular matrix: Apply 5% parameter perturbation
        - Memory errors: Reduce learning rate + 1% perturbation
        - Generic errors: Apply 1% parameter perturbation

        Common values: 1-3. Set to 0 to disable retries.

    batch_stats_buffer_size : int, default=100
        Size of circular buffer for recent batch statistics. Tracks last N
        batches for diagnostic analysis. Larger values (200-500) provide more
        history but use more memory. Smaller values (50-100) reduce overhead.

    batch_shape_padding : str, default='auto'
        Batch shape padding strategy to eliminate JIT recompilation overhead.
        Options:

        - **'auto'** (recommended): Auto-detect max batch shape during warmup,
          then pad all subsequent batches to that shape. Provides best performance
          (30-50% throughput improvement) with zero recompiles after warmup.

        - **'static'**: User specifies fixed batch shape. All batches padded to
          batch_size. Useful when batch size is known to be uniform.

        - **'dynamic'**: No padding (legacy behavior). Each unique batch shape
          triggers JIT recompilation. Use only if padding causes issues.

        **Why This Matters:**
        JAX JIT compilation is sensitive to array shapes. Variable batch sizes
        (especially the last partial batch) trigger expensive recompilations.
        Padding batches to static shapes eliminates this overhead while preserving
        numerical correctness through masking.

        **Performance Impact:**
        - Auto/static mode: 30-50% throughput improvement, zero post-warmup recompiles
        - Dynamic mode: Current performance with recompiles on shape changes

        **Tradeoffs:**
        - Auto/static: Slight memory overhead from padding (typically <1% for batch_size=100)
        - Dynamic: No memory overhead but suffers from recompilation overhead

    Examples
    --------
    Basic configuration with defaults:

    >>> from nlsq import StreamingConfig
    >>> config = StreamingConfig()
    >>> config.batch_size
    32
    >>> config.enable_fault_tolerance
    True

    Configure for long-running optimization with checkpoint resume:

    >>> config = StreamingConfig(
    ...     batch_size=100,
    ...     max_epochs=50,
    ...     checkpoint_frequency=100,
    ...     resume_from_checkpoint=True,  # Auto-detect latest checkpoint
    ... )

    Configure for noisy data with permissive settings:

    >>> config = StreamingConfig(
    ...     batch_size=100,
    ...     min_success_rate=0.3,         # Allow 70% failures
    ...     max_retries_per_batch=2,      # Standard retry limit
    ...     validate_numerics=True,       # Keep validation
    ... )

    Configure for production (fast mode):

    >>> config = StreamingConfig(
    ...     batch_size=100,
    ...     enable_fault_tolerance=False,  # <1% overhead
    ...     enable_checkpoints=True,       # Still save checkpoints
    ... )

    Configure for high-performance requirements:

    >>> config = StreamingConfig(
    ...     batch_size=200,                # Larger batches
    ...     enable_fault_tolerance=False,  # Fast mode
    ...     validate_numerics=False,       # Skip validation (use with caution)
    ...     enable_checkpoints=False,      # Disable checkpoints for max speed
    ... )

    Notes
    -----
    **Performance Overhead:**
        - Full fault tolerance (enable_fault_tolerance=True): <5% overhead
        - Fast mode (enable_fault_tolerance=False): <1% overhead
        - Checkpoint saves: Negligible (uses async I/O when available)

    **Fault Tolerance Features:**
        When enable_fault_tolerance=True (default):

        - Best parameters always tracked and returned
        - NaN/Inf detection at three critical points
        - Adaptive retry strategies for failed batches
        - Batch statistics collection (circular buffer)
        - Detailed diagnostics (error types, retry counts, etc.)

        When enable_fault_tolerance=False (fast mode):

        - Best parameters still tracked
        - No NaN/Inf validation
        - No retry attempts
        - No batch statistics collection
        - Basic error handling only
        - Checkpoints still saved (if enable_checkpoints=True)

    **Checkpoint Format:**
        Checkpoints use HDF5 format with version metadata:

        - Version 2.0: Full diagnostics (retry counts, error types, etc.)
        - Version 1.0: Basic state only (backward compatible)

        Checkpoint files are named: `checkpoint_iter_{iteration}.h5`

    **Batch Statistics:**
        Uses fixed-size circular buffer for memory efficiency. Older statistics
        are automatically discarded when buffer is full. Buffer size controls
        memory usage vs history depth trade-off.

    **Success Rate Validation:**
        Optimization fails if batch success rate falls below min_success_rate.
        Best parameters found are still returned in this case. Check
        result['success'] and result['message'] to determine if optimization
        succeeded.

    See Also
    --------
    StreamingOptimizer : Optimizer that uses this configuration
    curve_fit_large : High-level interface for large datasets

    References
    ----------
    .. [StreamingFaultTolerance] Specification: Streaming Optimizer Fault Tolerance
           agent-os/specs/2025-10-19-streaming-optimizer-fault-tolerance/spec.md
    """

    # Basic optimization settings
    batch_size: int = 32
    max_epochs: int = 10
    learning_rate: float = 0.001
    momentum: float = 0.9

    # Adam optimizer settings
    use_adam: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # Gradient settings
    gradient_clip: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 100

    # Convergence
    convergence_tol: float = 1e-6

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 100
    checkpoint_interval: int = 100  # Alias for backward compatibility
    enable_checkpoints: bool = True
    resume_from_checkpoint: bool | str | None = None

    # Fault tolerance
    enable_fault_tolerance: bool = True
    validate_numerics: bool = True
    min_success_rate: float = 0.5
    max_retries_per_batch: int = 2

    # Diagnostics
    batch_stats_buffer_size: int = 100

    # Batch shape padding (Task Group 7: Streaming Overhead Reduction)
    batch_shape_padding: str = "auto"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure checkpoint_frequency matches checkpoint_interval (backward compatibility)
        # checkpoint_interval is the legacy parameter, but we use checkpoint_frequency internally
        if (
            hasattr(self, "checkpoint_interval")
            and self.checkpoint_interval != self.checkpoint_frequency
        ):
            # If user explicitly set checkpoint_interval, use it for checkpoint_frequency
            self.checkpoint_frequency = self.checkpoint_interval

        # Validate ranges
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.momentum <= 1, "momentum must be in [0, 1]"
        assert 0 <= self.adam_beta1 < 1, "adam_beta1 must be in [0, 1)"
        assert 0 <= self.adam_beta2 < 1, "adam_beta2 must be in [0, 1)"
        assert self.adam_eps > 0, "adam_eps must be positive"
        assert self.gradient_clip > 0, "gradient_clip must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.convergence_tol > 0, "convergence_tol must be positive"
        assert self.checkpoint_frequency > 0, "checkpoint_frequency must be positive"
        assert 0 <= self.min_success_rate <= 1, "min_success_rate must be in [0, 1]"
        assert self.max_retries_per_batch >= 0, (
            "max_retries_per_batch must be non-negative"
        )
        assert self.batch_stats_buffer_size > 0, (
            "batch_stats_buffer_size must be positive"
        )
        assert self.batch_shape_padding in ("auto", "static", "dynamic"), (
            "batch_shape_padding must be one of: 'auto', 'static', 'dynamic'"
        )
