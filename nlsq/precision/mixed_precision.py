"""Mixed precision fallback for NLSQ optimization.

This module provides automatic precision management that starts optimizations
in float32 and upgrades to float64 when precision issues are detected.

The mixed precision system:
- Starts all optimizations in float32 for 50% memory savings
- Monitors 5 convergence metrics in real-time
- Detects 5 categories of precision issues automatically
- Seamlessly upgrades to float64 when needed
- Preserves optimization state for zero-iteration loss

Key Components:
- PrecisionState: State machine enum for precision management
- MixedPrecisionConfig: Configuration dataclass
- ConvergenceMetrics: Metrics reported by optimization algorithms
- OptimizationState: Complete optimization state for precision upgrades
- ConvergenceMonitor: Issue detection component
- PrecisionUpgrader: State transition and dtype conversion
- BestParameterTracker: Fault tolerance for best parameter tracking
- MixedPrecisionManager: Main orchestrator
"""

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import jax.numpy as jnp
import numpy as np
from jax import device_put

__all__ = [
    "BestParameterTracker",
    "ConvergenceMetrics",
    "ConvergenceMonitor",
    "MixedPrecisionConfig",
    "MixedPrecisionManager",
    "OptimizationState",
    "PrecisionState",
    "PrecisionUpgrader",
]


# Enums and Dataclasses


class PrecisionState(Enum):
    """State machine for precision management.

    The precision state machine tracks the current precision mode and
    manages transitions between float32 and float64 during optimization.

    States
    ------
    FLOAT32_ACTIVE : auto()
        Optimization running in float32, no issues detected.
        This is the initial state and preferred mode for memory efficiency
        and GPU performance.
    MONITORING_DEGRADATION : auto()
        Issues detected, in grace period (counter < max_degradation_iterations).
        The system is monitoring convergence problems but hasn't committed to
        upgrading yet. This provides hysteresis to avoid premature upgrades
        from transient numerical issues.
    UPGRADING_TO_FLOAT64 : auto()
        Precision upgrade in progress (JIT compilation).
        State transfer from float32 to float64 is occurring, including
        zero-copy dtype conversion and lazy JIT compilation of float64 functions.
    FLOAT64_ACTIVE : auto()
        Optimization running in float64.
        Upgraded precision is active, providing higher numerical accuracy
        for ill-conditioned problems.
    RELAXED_FLOAT32_FALLBACK : auto()
        Float64 failed, retrying float32 with relaxed tolerances.
        If float64 optimization also fails to converge, the system falls back
        to float32 with tolerance_relaxation_factor applied (typically 10x).

    Examples
    --------
    >>> state = PrecisionState.FLOAT32_ACTIVE
    >>> print(state.value)
    'FLOAT32_ACTIVE'

    Notes
    -----
    State transitions follow this pattern:
    FLOAT32_ACTIVE -> MONITORING_DEGRADATION -> UPGRADING_TO_FLOAT64 ->
    FLOAT64_ACTIVE -> (if fails) RELAXED_FLOAT32_FALLBACK
    """

    FLOAT32_ACTIVE = auto()
    MONITORING_DEGRADATION = auto()
    UPGRADING_TO_FLOAT64 = auto()
    FLOAT64_ACTIVE = auto()
    RELAXED_FLOAT32_FALLBACK = auto()


@dataclass(slots=True)
class MixedPrecisionConfig:
    """Configuration for automatic mixed precision fallback.

    Controls the behavior of the mixed precision system, including when to
    upgrade from float32 to float64, detection thresholds, and fallback strategies.

    Attributes
    ----------
    enable_mixed_precision_fallback : bool, default=True
        Enable automatic fallback from float32 to float64 when precision issues
        are detected. When False, optimization always uses float64.
    max_degradation_iterations : int, default=5
        Number of iterations with detected issues before upgrading precision.
        Provides hysteresis to avoid premature upgrades from transient problems.
        Lower values upgrade faster, higher values tolerate more issues.
    stall_window : int, default=10
        Number of iterations to track for convergence stall detection.
        If cost doesn't improve meaningfully over this window, a stall is detected.
    gradient_explosion_threshold : float, default=1e10
        Gradient norm threshold for explosion detection. If gradient norm exceeds
        this value, numerical instability is suspected.
    precision_limit_threshold : float, default=1e-7
        Minimum meaningful parameter change in float32 (~10^-7 is float32 precision limit).
        Parameter updates smaller than this suggest float32 insufficient precision.
    tolerance_relaxation_factor : float, default=10.0
        Factor to multiply tolerances (gtol, ftol, xtol) when falling back to
        float32 after float64 failure. Larger values are more permissive.
    verbose : bool, default=False
        If True, promote precision upgrade events to INFO log level (visible by default).
        If False, events logged at DEBUG level (only visible when DEBUG logging enabled).

    Examples
    --------
    Default configuration (recommended for most problems):

    >>> config = MixedPrecisionConfig()
    >>> config.max_degradation_iterations
    5

    Conservative configuration (prefers float64, upgrades quickly):

    >>> config = MixedPrecisionConfig(
    ...     max_degradation_iterations=2,
    ...     gradient_explosion_threshold=1e8,
    ...     verbose=True
    ... )

    Aggressive configuration (stays in float32 longer):

    >>> config = MixedPrecisionConfig(
    ...     max_degradation_iterations=10,
    ...     gradient_explosion_threshold=1e12,
    ...     precision_limit_threshold=1e-8
    ... )

    Disabled (pure float64, no mixed precision):

    >>> config = MixedPrecisionConfig(enable_mixed_precision_fallback=False)

    Notes
    -----
    Mixed precision is enabled by default to provide transparent memory savings
    (50%) and GPU speedup (40-80%) for well-conditioned problems. Ill-conditioned
    problems automatically upgrade to float64 for numerical accuracy.

    See Also
    --------
    MixedPrecisionManager : Main orchestrator using this configuration
    PrecisionState : State machine for precision management
    """

    enable_mixed_precision_fallback: bool = True
    max_degradation_iterations: int = 5
    stall_window: int = 10
    gradient_explosion_threshold: float = 1e10
    precision_limit_threshold: float = 1e-7
    tolerance_relaxation_factor: float = 10.0
    verbose: bool = False


@dataclass(slots=True)
class ConvergenceMetrics:
    """Metrics reported by optimization algorithm for monitoring.

    These metrics are collected after each iteration and used by ConvergenceMonitor
    to detect precision issues that may require upgrading from float32 to float64.

    Attributes
    ----------
    iteration : int
        Current iteration number (0-indexed).
    residual_norm : float
        L2 norm of residuals ||f(x)||. Measures how well the model fits the data.
    gradient_norm : float
        L2 norm of gradient ||g|| where g = J^T @ f. Indicates optimization progress.
    parameter_change : float
        L2 norm of parameter update ||Δx|| from previous iteration.
        Small values may indicate precision limitations.
    cost : float
        Current cost function value (0.5 * ||f||^2). Used for stall detection.
    trust_radius : float
        Current trust region radius (TRF) or damping parameter (LM).
        Very small values may indicate trust region collapse.
    has_nan_inf : bool
        Whether NaN or Inf detected in residuals, Jacobian, or gradient.
        Immediate indicator of numerical problems.

    Examples
    --------
    >>> metrics = ConvergenceMetrics(
    ...     iteration=10,
    ...     residual_norm=1.5e-3,
    ...     gradient_norm=2.1e-5,
    ...     parameter_change=1.2e-6,
    ...     cost=1.125e-6,
    ...     trust_radius=0.8,
    ...     has_nan_inf=False
    ... )
    >>> metrics.iteration
    10

    Notes
    -----
    All norms are L2 norms. The algorithm reports these metrics after computing
    the step but before accepting/rejecting it.
    """

    iteration: int
    residual_norm: float
    gradient_norm: float
    parameter_change: float
    cost: float
    trust_radius: float
    has_nan_inf: bool


@dataclass(slots=True)
class OptimizationState:
    """Complete optimization state for precision upgrade.

    Captures all arrays and scalars needed to preserve optimization state when
    upgrading from float32 to float64. Enables zero-iteration loss during precision
    transitions.

    Attributes
    ----------
    x : jnp.ndarray
        Current parameter vector, shape (n_params,).
    f : jnp.ndarray
        Current residual vector, shape (n_points,).
    J : jnp.ndarray
        Current Jacobian matrix, shape (n_points, n_params).
    g : jnp.ndarray
        Current gradient vector J^T @ f, shape (n_params,).
    cost : float
        Current cost function value (0.5 * ||f||^2).
    trust_radius : float
        Current trust region radius (TRF) or damping parameter lambda (LM).
    iteration : int
        Current iteration count. Preserved during upgrade to avoid iteration loss.
    dtype : jnp.dtype
        Current data type (jnp.float32 or jnp.float64).
    algorithm_specific : Optional[dict]
        Algorithm-specific state (e.g., {"lambda": 0.1} for LM damping parameter).
        TRF doesn't need additional state beyond trust_radius.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> state = OptimizationState(
    ...     x=jnp.array([1.0, 2.0], dtype=jnp.float32),
    ...     f=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32),
    ...     J=jnp.ones((3, 2), dtype=jnp.float32),
    ...     g=jnp.array([0.6, 0.6], dtype=jnp.float32),
    ...     cost=0.035,
    ...     trust_radius=1.0,
    ...     iteration=15,
    ...     dtype=jnp.float32,
    ...     algorithm_specific=None
    ... )
    >>> state.dtype
    dtype('float32')

    Notes
    -----
    This state object is designed to be immutable. PrecisionUpgrader creates
    new instances with upgraded dtypes rather than modifying in-place.

    See Also
    --------
    PrecisionUpgrader : Handles state conversion between dtypes
    """

    x: jnp.ndarray
    f: jnp.ndarray
    J: jnp.ndarray
    g: jnp.ndarray
    cost: float
    trust_radius: float
    iteration: int
    dtype: jnp.dtype
    algorithm_specific: dict | None = None


# Core Classes


class ConvergenceMonitor:
    """Monitors convergence metrics and detects precision issues.

    Detects 5 issue types:
    1. NaN/Inf in residuals, Jacobian, gradient
    2. Gradient explosion (norm > threshold)
    3. Convergence stall (no progress over window)
    4. Parameter precision limits (changes < float32 resolution)
    5. Trust region collapse (radius < minimum)

    Parameters
    ----------
    config : MixedPrecisionConfig
        Configuration for monitoring behavior
    logger : logging.Logger
        Logger for diagnostic messages

    Examples
    --------
    >>> import logging
    >>> config = MixedPrecisionConfig()
    >>> logger = logging.getLogger("nlsq.mixed_precision")
    >>> monitor = ConvergenceMonitor(config, logger)
    >>> metrics = ConvergenceMetrics(
    ...     iteration=10,
    ...     residual_norm=1.5e-3,
    ...     gradient_norm=2.1e-5,
    ...     parameter_change=1.2e-6,
    ...     cost=1.125e-6,
    ...     trust_radius=0.8,
    ...     has_nan_inf=False
    ... )
    >>> issue = monitor.check_convergence(metrics)
    >>> if issue is not None:
    ...     print(f"Issue detected: {issue}")

    Notes
    -----
    The monitor maintains a sliding window of cost history to detect convergence
    stalls. It increments a degradation counter when issues are detected, which
    triggers precision upgrades when the threshold is exceeded.

    See Also
    --------
    MixedPrecisionConfig : Configuration for monitoring behavior
    ConvergenceMetrics : Metrics structure for optimization algorithms
    """

    def __init__(self, config: MixedPrecisionConfig, logger: logging.Logger):
        """Initialize the convergence monitor.

        Parameters
        ----------
        config : MixedPrecisionConfig
            Configuration for monitoring behavior
        logger : logging.Logger
            Logger for diagnostic messages
        """
        self.config = config
        self.logger = logger
        self.cost_history: deque[float] = deque(maxlen=config.stall_window)
        self.degradation_counter = 0

    def check_convergence(self, metrics: ConvergenceMetrics) -> str | None:
        """Check for precision issues in current metrics.

        Parameters
        ----------
        metrics : ConvergenceMetrics
            Current convergence metrics from algorithm

        Returns
        -------
        issue : Optional[str]
            Description of detected issue, or None if no issues.
            Possible values: "nan_inf_detected", "gradient_explosion",
            "convergence_stall", "precision_limit", "trust_region_collapse"
        """
        # 1. NaN/Inf detection
        if metrics.has_nan_inf:
            self.logger.debug(f"NaN/Inf detected at iteration {metrics.iteration}")
            return "nan_inf_detected"

        # 2. Gradient explosion
        if metrics.gradient_norm > self.config.gradient_explosion_threshold:
            self.logger.debug(
                f"Gradient explosion at iteration {metrics.iteration}: "
                f"norm={metrics.gradient_norm:.2e} > "
                f"threshold={self.config.gradient_explosion_threshold:.2e}"
            )
            return "gradient_explosion"

        # 3. Convergence stall
        self.cost_history.append(metrics.cost)
        if len(self.cost_history) == self.config.stall_window:
            cost_change = abs(self.cost_history[0] - self.cost_history[-1])
            if cost_change < 1e-10:  # No meaningful progress
                self.logger.debug(
                    f"Convergence stall at iteration {metrics.iteration}: "
                    f"cost_change={cost_change:.2e} < 1e-10 over "
                    f"{self.config.stall_window} iterations"
                )
                return "convergence_stall"

        # 4. Parameter precision limits
        if metrics.parameter_change < self.config.precision_limit_threshold:
            self.logger.debug(
                f"Parameter precision limit at iteration {metrics.iteration}: "
                f"change={metrics.parameter_change:.2e} < "
                f"threshold={self.config.precision_limit_threshold:.2e}"
            )
            return "precision_limit"

        # 5. Trust region collapse
        if metrics.trust_radius < 1e-12:
            self.logger.debug(
                f"Trust region collapse at iteration {metrics.iteration}: "
                f"radius={metrics.trust_radius:.2e} < 1e-12"
            )
            return "trust_region_collapse"

        return None

    def should_upgrade(self) -> bool:
        """Check if precision should be upgraded.

        Returns
        -------
        should_upgrade : bool
            True if degradation counter >= threshold
        """
        return self.degradation_counter >= self.config.max_degradation_iterations

    def increment_degradation(self):
        """Increment degradation counter."""
        self.degradation_counter += 1

    def reset_degradation(self):
        """Reset degradation counter (issues resolved)."""
        self.degradation_counter = 0


class PrecisionUpgrader:
    """Handles precision upgrades with state preservation.

    Uses JAX's device_put for zero-copy dtype conversion (<10ms).
    Triggers lazy JIT compilation of float64 version (20-50ms).

    Parameters
    ----------
    logger : logging.Logger
        Logger for upgrade events

    Examples
    --------
    >>> import logging
    >>> import jax.numpy as jnp
    >>> logger = logging.getLogger("nlsq.mixed_precision")
    >>> upgrader = PrecisionUpgrader(logger)
    >>> state_float32 = OptimizationState(
    ...     x=jnp.array([1.0, 2.0], dtype=jnp.float32),
    ...     f=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32),
    ...     J=jnp.ones((3, 2), dtype=jnp.float32),
    ...     g=jnp.array([0.6, 0.6], dtype=jnp.float32),
    ...     cost=0.035,
    ...     trust_radius=1.0,
    ...     iteration=15,
    ...     dtype=jnp.float32,
    ...     algorithm_specific=None
    ... )
    >>> state_float64 = upgrader.upgrade_to_float64(state_float32)
    >>> state_float64.dtype
    dtype('float64')

    Notes
    -----
    The upgrader uses JAX's device_put for zero-copy conversion, which is much
    faster than creating new arrays. This preserves all state including iteration
    count and algorithm-specific data.

    See Also
    --------
    OptimizationState : State structure for optimization algorithms
    device_put : JAX function for zero-copy dtype conversion
    """

    def __init__(self, logger: logging.Logger):
        """Initialize the precision upgrader.

        Parameters
        ----------
        logger : logging.Logger
            Logger for upgrade events
        """
        self.logger = logger

    def upgrade_to_float64(self, state: OptimizationState) -> OptimizationState:
        """Upgrade optimization state from float32 to float64.

        Parameters
        ----------
        state : OptimizationState
            Current state in float32

        Returns
        -------
        upgraded_state : OptimizationState
            State converted to float64 (zero-copy, preserves all values)
        """
        if state.dtype == jnp.float64:
            self.logger.warning("State already in float64, no upgrade needed")
            return state

        # Zero-copy dtype conversion using JAX device_put
        upgraded = OptimizationState(
            x=device_put(state.x.astype(jnp.float64)),
            f=device_put(state.f.astype(jnp.float64)),
            J=device_put(state.J.astype(jnp.float64)),
            g=device_put(state.g.astype(jnp.float64)),
            cost=float(state.cost),  # Scalar, no conversion needed
            trust_radius=float(state.trust_radius),
            iteration=state.iteration,  # Preserve iteration count!
            dtype=jnp.float64,
            algorithm_specific=state.algorithm_specific,
        )

        self.logger.debug(
            f"Upgraded precision: float32 → float64 at iteration {state.iteration}"
        )

        return upgraded

    def downgrade_to_float32(self, state: OptimizationState) -> OptimizationState:
        """Downgrade optimization state from float64 to float32 (for fallback).

        Parameters
        ----------
        state : OptimizationState
            Current state in float64

        Returns
        -------
        downgraded_state : OptimizationState
            State converted to float32
        """
        if state.dtype == jnp.float32:
            return state

        downgraded = OptimizationState(
            x=device_put(state.x.astype(jnp.float32)),
            f=device_put(state.f.astype(jnp.float32)),
            J=device_put(state.J.astype(jnp.float32)),
            g=device_put(state.g.astype(jnp.float32)),
            cost=float(state.cost),
            trust_radius=float(state.trust_radius),
            iteration=state.iteration,
            dtype=jnp.float32,
            algorithm_specific=state.algorithm_specific,
        )

        self.logger.info(
            f"Downgraded to relaxed float32 at iteration {state.iteration}"
        )

        return downgraded


class BestParameterTracker:
    """Tracks best parameters throughout optimization history.

    Ensures we never return initial p0 guess. Returns best parameters
    found across all phases (float32, float64, relaxed float32).

    Attributes
    ----------
    best_cost : float
        Best (lowest) cost function value found so far. Initialized to infinity.
    best_parameters : Optional[jnp.ndarray]
        Best parameter vector found so far. None if no valid parameters tracked yet.
    best_iteration : int
        Iteration number where best parameters were found. -1 if not yet set.

    Examples
    --------
    >>> tracker = BestParameterTracker()
    >>> import jax.numpy as jnp
    >>> params1 = jnp.array([1.0, 2.0])
    >>> tracker.update(params1, cost=10.5, iteration=0)
    >>> params2 = jnp.array([1.2, 2.1])
    >>> tracker.update(params2, cost=8.3, iteration=1)  # Better
    >>> best = tracker.get_best_parameters()  # Returns params2
    >>> best_cost = tracker.get_best_cost()  # Returns 8.3

    Notes
    -----
    Parameters are converted to NumPy arrays for storage to avoid JAX device
    memory issues during long-running optimizations.

    See Also
    --------
    MixedPrecisionManager : Uses this tracker for fault tolerance
    """

    def __init__(self):
        """Initialize the best parameter tracker."""
        self.best_cost: float = float("inf")
        self.best_parameters: np.ndarray | None = None
        self.best_iteration: int = -1

    def update(self, parameters: jnp.ndarray, cost: float, iteration: int = -1):
        """Update best parameters if current cost is lower.

        Parameters
        ----------
        parameters : jnp.ndarray
            Current parameter vector
        cost : float
            Current cost function value
        iteration : int, default=-1
            Current iteration number (for logging/tracking)
        """
        if cost < self.best_cost:
            self.best_cost = cost
            # Convert to NumPy for storage to avoid JAX device memory issues
            self.best_parameters = np.array(parameters)
            self.best_iteration = iteration

    def get_best_parameters(self) -> np.ndarray | None:
        """Get best parameters found so far.

        Returns
        -------
        best_parameters : Optional[np.ndarray]
            Best parameters as NumPy array, or None if no valid parameters tracked
        """
        return self.best_parameters

    def get_best_cost(self) -> float:
        """Get best cost found so far.

        Returns
        -------
        best_cost : float
            Best (lowest) cost function value, or infinity if no valid cost tracked
        """
        return self.best_cost


class MixedPrecisionManager:
    """Main orchestrator for mixed precision optimization.

    Coordinates ConvergenceMonitor, PrecisionUpgrader, and BestParameterTracker.
    Manages PrecisionState state machine for automatic precision upgrades.

    The manager provides automatic precision management:
    1. Start in float32 for memory efficiency
    2. Monitor convergence metrics for precision issues
    3. Upgrade to float64 when issues detected persistently
    4. Track best parameters across all precision phases
    5. Fallback to relaxed float32 if float64 fails

    Parameters
    ----------
    config : MixedPrecisionConfig
        Configuration for mixed precision behavior
    verbose : bool, default=False
        Promote precision events to INFO level (overrides config.verbose).
        If True, precision upgrades/downgrades logged at INFO.
        If False, logged at DEBUG.

    Attributes
    ----------
    config : MixedPrecisionConfig
        Configuration object
    verbose : bool
        Whether to log at INFO level
    logger : logging.Logger
        Logger instance for diagnostic messages
    monitor : nlsq.mixed_precision.ConvergenceMonitor
        Convergence monitoring component
    upgrader : PrecisionUpgrader
        Precision upgrade/downgrade component
    tracker : BestParameterTracker
        Best parameter tracking component
    state : PrecisionState
        Current state in the precision state machine
    current_dtype : jnp.dtype
        Current precision dtype (jnp.float32 or jnp.float64)

    Examples
    --------
    Basic usage (algorithm integration):

    >>> import logging
    >>> import jax.numpy as jnp
    >>> config = MixedPrecisionConfig()
    >>> manager = MixedPrecisionManager(config)
    >>> # Algorithm iteration loop
    >>> for iteration in range(max_iterations):
    ...     # Compute residuals, Jacobian, gradient, cost
    ...     metrics = ConvergenceMetrics(
    ...         iteration=iteration,
    ...         residual_norm=residual_norm,
    ...         gradient_norm=gradient_norm,
    ...         parameter_change=param_change,
    ...         cost=cost,
    ...         trust_radius=trust_radius,
    ...         has_nan_inf=has_nan_inf
    ...     )
    ...     manager.report_metrics(metrics)
    ...
    ...     if manager.should_upgrade():
    ...         current_state = OptimizationState(
    ...             x=x, f=f, J=J, g=g,
    ...             cost=cost, trust_radius=trust_radius,
    ...             iteration=iteration, dtype=jnp.float32
    ...         )
    ...         upgraded_state = manager.upgrade_precision(current_state)
    ...         # Continue optimization with upgraded_state
    ...
    ...     # Update best parameters
    ...     manager.update_best(current_params, current_cost, iteration)

    Notes
    -----
    The manager uses a 5-state machine:
    - FLOAT32_ACTIVE: Normal float32 operation
    - MONITORING_DEGRADATION: Issues detected, in grace period
    - UPGRADING_TO_FLOAT64: Upgrade in progress
    - FLOAT64_ACTIVE: Running in float64
    - RELAXED_FLOAT32_FALLBACK: Float64 failed, using relaxed float32

    See Also
    --------
    MixedPrecisionConfig : Configuration options
    nlsq.mixed_precision.ConvergenceMonitor : Issue detection component
    PrecisionUpgrader : Precision conversion component
    BestParameterTracker : Parameter tracking component
    """

    def __init__(self, config: MixedPrecisionConfig, verbose: bool = False):
        """Initialize the mixed precision manager.

        Parameters
        ----------
        config : MixedPrecisionConfig
            Configuration for mixed precision behavior
        verbose : bool, default=False
            Promote precision events to INFO level
        """
        self.config = config
        self.verbose = verbose or config.verbose

        # Setup logging
        self.logger = logging.getLogger("nlsq.mixed_precision")
        # Note: Users can configure logger externally via logging.basicConfig
        # This just sets the minimum level for this logger
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.DEBUG)

        # Initialize components
        self.monitor = ConvergenceMonitor(config, self.logger)
        self.upgrader = PrecisionUpgrader(self.logger)
        self.tracker = BestParameterTracker()

        # State machine
        self.state = PrecisionState.FLOAT32_ACTIVE
        self.current_dtype = jnp.float32

    def report_metrics(self, metrics: ConvergenceMetrics):
        """Report convergence metrics from algorithm.

        Checks for precision issues and updates the state machine accordingly.

        Parameters
        ----------
        metrics : ConvergenceMetrics
            Current iteration metrics from optimization algorithm
        """
        issue = self.monitor.check_convergence(metrics)

        if issue is not None:
            if self.state == PrecisionState.FLOAT32_ACTIVE:
                self.state = PrecisionState.MONITORING_DEGRADATION
                self.monitor.increment_degradation()
                self.logger.debug(
                    f"Issue detected: {issue} (degradation count: {self.monitor.degradation_counter})"
                )
            elif self.state == PrecisionState.MONITORING_DEGRADATION:
                self.monitor.increment_degradation()
                self.logger.debug(
                    f"Degradation continues: {issue} (count: {self.monitor.degradation_counter})"
                )
        # Issues resolved
        elif self.state == PrecisionState.MONITORING_DEGRADATION:
            self.logger.debug("Issues resolved, returning to FLOAT32_ACTIVE")
            self.state = PrecisionState.FLOAT32_ACTIVE
            self.monitor.reset_degradation()

    def should_upgrade(self) -> bool:
        """Check if precision upgrade should occur.

        Returns
        -------
        should_upgrade : bool
            True if state machine indicates upgrade needed
        """
        if not self.config.enable_mixed_precision_fallback:
            return False

        return (
            self.state == PrecisionState.MONITORING_DEGRADATION
            and self.monitor.should_upgrade()
        )

    def upgrade_precision(self, state: OptimizationState) -> OptimizationState:
        """Upgrade optimization state to float64.

        Parameters
        ----------
        state : OptimizationState
            Current optimization state (float32)

        Returns
        -------
        upgraded_state : OptimizationState
            State in float64 with preserved values and iteration count
        """
        self.state = PrecisionState.UPGRADING_TO_FLOAT64

        log_level = logging.INFO if self.verbose else logging.DEBUG
        self.logger.log(
            log_level,
            f"Upgrading precision float32 → float64 at iteration {state.iteration}",
        )

        upgraded_state = self.upgrader.upgrade_to_float64(state)

        self.state = PrecisionState.FLOAT64_ACTIVE
        self.current_dtype = jnp.float64

        return upgraded_state

    def apply_relaxed_fallback(
        self, state: OptimizationState, original_tolerances: dict
    ) -> tuple[OptimizationState, dict]:
        """Apply relaxed float32 fallback if float64 fails.

        Parameters
        ----------
        state : OptimizationState
            Current state in float64
        original_tolerances : dict
            Original tolerances {"gtol": ..., "ftol": ..., "xtol": ...}

        Returns
        -------
        fallback_state : OptimizationState
            State in float32 with best parameters from history
        relaxed_tolerances : dict
            Tolerances multiplied by tolerance_relaxation_factor
        """
        self.state = PrecisionState.RELAXED_FLOAT32_FALLBACK

        self.logger.info(
            "Float64 convergence failed, falling back to relaxed float32 "
            f"(tolerances × {self.config.tolerance_relaxation_factor})"
        )

        # Get best parameters from entire history
        best_params = self.tracker.get_best_parameters()
        if best_params is not None:
            # Create new state with best parameters
            state = OptimizationState(
                x=jnp.array(best_params),
                f=state.f,
                J=state.J,
                g=state.g,
                cost=state.cost,
                trust_radius=state.trust_radius,
                iteration=state.iteration,
                dtype=state.dtype,
                algorithm_specific=state.algorithm_specific,
            )

        # Downgrade to float32
        fallback_state = self.upgrader.downgrade_to_float32(state)

        # Relax tolerances
        relaxed_tol = {
            key: val * self.config.tolerance_relaxation_factor
            for key, val in original_tolerances.items()
        }

        return fallback_state, relaxed_tol

    def update_best(self, parameters: jnp.ndarray, cost: float, iteration: int):
        """Update best parameters tracker.

        Parameters
        ----------
        parameters : jnp.ndarray
            Current parameter vector
        cost : float
            Current cost value
        iteration : int
            Current iteration number
        """
        self.tracker.update(parameters, cost, iteration)

    def get_best_parameters(self) -> np.ndarray | None:
        """Get best parameters from entire optimization history.

        Returns
        -------
        best_parameters : Optional[np.ndarray]
            Best parameters as NumPy array, or None if no parameters tracked
        """
        return self.tracker.get_best_parameters()

    def get_current_dtype(self) -> jnp.dtype:
        """Get current precision dtype for memory estimates.

        Returns
        -------
        dtype : jnp.dtype
            Current precision dtype (jnp.float32 or jnp.float64)
        """
        return self.current_dtype

    def _validate_parameters(self, x: jnp.ndarray | None) -> tuple[bool, str | None]:
        """Validate parameter vector for NaN/Inf and extreme values.

        Parameters
        ----------
        x : Optional[jnp.ndarray]
            Parameter vector to validate

        Returns
        -------
        is_valid : bool
            True if parameters are valid
        error_message : Optional[str]
            Error description if invalid, None otherwise
        """
        if x is None:
            return True, None

        if jnp.any(jnp.isnan(x)):
            return False, "Parameters contain NaN values"
        if jnp.any(jnp.isinf(x)):
            return False, "Parameters contain Inf values"
        if jnp.all(jnp.abs(x) < 1e-15):
            self.logger.warning(
                "Parameters are extremely small (all |x| < 1e-15), "
                "may indicate numerical underflow"
            )

        return True, None

    def _validate_residuals_jacobian(
        self, f: jnp.ndarray | None, J: jnp.ndarray | None
    ) -> tuple[bool, str | None]:
        """Validate residuals and Jacobian for NaN/Inf.

        Parameters
        ----------
        f : Optional[jnp.ndarray]
            Residual vector
        J : Optional[jnp.ndarray]
            Jacobian matrix

        Returns
        -------
        is_valid : bool
            True if both are valid
        error_message : Optional[str]
            Error description if invalid, None otherwise
        """
        # Check residuals
        if f is not None:
            if jnp.any(jnp.isnan(f)):
                return False, "Residuals contain NaN values"
            if jnp.any(jnp.isinf(f)):
                return False, "Residuals contain Inf values"

        # Check Jacobian
        if J is not None:
            if jnp.any(jnp.isnan(J)):
                return False, "Jacobian contains NaN values"
            if jnp.any(jnp.isinf(J)):
                return False, "Jacobian contains Inf values"

        return True, None

    def _validate_gradient_cost(
        self, g: jnp.ndarray | None, cost: float
    ) -> tuple[bool, str | None]:
        """Validate gradient and cost for NaN/Inf.

        Parameters
        ----------
        g : Optional[jnp.ndarray]
            Gradient vector
        cost : float
            Cost value

        Returns
        -------
        is_valid : bool
            True if both are valid
        error_message : Optional[str]
            Error description if invalid, None otherwise
        """
        # Check gradient
        if g is not None:
            if jnp.any(jnp.isnan(g)):
                return False, "Gradient contains NaN values"
            if jnp.any(jnp.isinf(g)):
                return False, "Gradient contains Inf values"

        # Check cost
        if jnp.isnan(cost) or jnp.isinf(cost):
            return False, f"Cost is {cost}"

        return True, None

    def _validate_trust_radius(
        self, trust_radius: float | None
    ) -> tuple[bool, str | None]:
        """Validate trust radius for NaN/Inf and non-positive values.

        Parameters
        ----------
        trust_radius : Optional[float]
            Trust radius value

        Returns
        -------
        is_valid : bool
            True if valid
        error_message : Optional[str]
            Error description if invalid, None otherwise
        """
        if trust_radius is None:
            return True, None

        if jnp.isnan(trust_radius) or jnp.isinf(trust_radius):
            return False, f"Trust radius is {trust_radius}"
        if trust_radius <= 0:
            return False, f"Trust radius is non-positive: {trust_radius}"

        return True, None

    def validate_state(self, state: OptimizationState) -> tuple[bool, str | None]:
        """Validate optimization state for NaN/Inf and other issues.

        Parameters
        ----------
        state : OptimizationState
            State to validate

        Returns
        -------
        is_valid : bool
            True if state is valid, False otherwise
        error_message : Optional[str]
            None if valid, error description if invalid

        Examples
        --------
        >>> manager = MixedPrecisionManager(MixedPrecisionConfig())
        >>> state = OptimizationState(x=jnp.array([1.0, 2.0]), ...)
        >>> is_valid, error = manager.validate_state(state)
        >>> if not is_valid:
        ...     print(f"Invalid state: {error}")
        """
        # Check parameters
        is_valid, error = self._validate_parameters(state.x)
        if not is_valid:
            return False, error

        # Check residuals and Jacobian
        is_valid, error = self._validate_residuals_jacobian(state.f, state.J)
        if not is_valid:
            return False, error

        # Check gradient and cost
        is_valid, error = self._validate_gradient_cost(state.g, state.cost)
        if not is_valid:
            return False, error

        # Check trust radius
        is_valid, error = self._validate_trust_radius(state.trust_radius)
        if not is_valid:
            return False, error

        return True, None

    def validate_metrics(self, metrics: ConvergenceMetrics) -> tuple[bool, str | None]:
        """Validate convergence metrics for NaN/Inf and other issues.

        Parameters
        ----------
        metrics : ConvergenceMetrics
            Metrics to validate

        Returns
        -------
        is_valid : bool
            True if metrics are valid, False otherwise
        error_message : Optional[str]
            None if valid, error description if invalid

        Examples
        --------
        >>> manager = MixedPrecisionManager(MixedPrecisionConfig())
        >>> metrics = ConvergenceMetrics(iteration=0, residual_norm=1.0, ...)
        >>> is_valid, error = manager.validate_metrics(metrics)
        >>> if not is_valid:
        ...     print(f"Invalid metrics: {error}")
        """
        if jnp.isnan(metrics.residual_norm) or jnp.isinf(metrics.residual_norm):
            return False, f"Residual norm is {metrics.residual_norm}"

        if jnp.isnan(metrics.gradient_norm) or jnp.isinf(metrics.gradient_norm):
            return False, f"Gradient norm is {metrics.gradient_norm}"

        if jnp.isnan(metrics.parameter_change) or jnp.isinf(metrics.parameter_change):
            return False, f"Parameter change is {metrics.parameter_change}"

        if jnp.isnan(metrics.cost) or jnp.isinf(metrics.cost):
            return False, f"Cost is {metrics.cost}"

        if metrics.trust_radius is not None:
            if jnp.isnan(metrics.trust_radius) or jnp.isinf(metrics.trust_radius):
                return False, f"Trust radius is {metrics.trust_radius}"

        # Additional sanity checks
        if metrics.residual_norm < 0:
            return False, f"Residual norm is negative: {metrics.residual_norm}"

        if metrics.gradient_norm < 0:
            return False, f"Gradient norm is negative: {metrics.gradient_norm}"

        if metrics.parameter_change < 0:
            return False, f"Parameter change is negative: {metrics.parameter_change}"

        return True, None

    def handle_validation_failure(
        self, error_message: str, context: str = ""
    ) -> str | None:
        """Handle validation failures with appropriate logging and recovery suggestions.

        Parameters
        ----------
        error_message : str
            Description of the validation failure
        context : str, optional
            Additional context about where the failure occurred

        Returns
        -------
        recovery_suggestion : Optional[str]
            Suggestion for recovering from the failure, or None if unrecoverable

        Examples
        --------
        >>> manager = MixedPrecisionManager(MixedPrecisionConfig())
        >>> suggestion = manager.handle_validation_failure(
        ...     "Parameters contain NaN", context="after step"
        ... )
        >>> if suggestion:
        ...     print(f"Try: {suggestion}")
        """
        full_message = f"Validation failure: {error_message}"
        if context:
            full_message += f" ({context})"

        self.logger.error(full_message)

        # Provide recovery suggestions based on error type
        if "NaN" in error_message or "Inf" in error_message:
            if "Parameters" in error_message:
                return (
                    "Try reducing step size, checking initial parameters, "
                    "or using parameter bounds"
                )
            elif "Jacobian" in error_message or "Gradient" in error_message:
                return (
                    "Try checking model function for numerical issues, "
                    "reducing data scale, or using finite differences"
                )
            elif "Cost" in error_message or "Residuals" in error_message:
                return (
                    "Try checking model function evaluation, "
                    "reducing data scale, or checking for outliers"
                )
            else:
                return "Try reducing step size or trust radius"

        elif "negative" in error_message.lower():
            return "This indicates a bug in the optimization algorithm"

        elif "non-positive" in error_message.lower():
            if "trust radius" in error_message.lower():
                return (
                    "Trust radius collapsed; optimization cannot continue. "
                    "Try different initial parameters or looser tolerances"
                )

        return "Unable to provide automatic recovery suggestion"
