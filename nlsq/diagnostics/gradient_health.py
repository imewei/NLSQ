"""Gradient health monitoring for nonlinear least squares optimization.

This module provides the GradientMonitor class for tracking gradient behavior
during optimization iterations. It detects:

- Vanishing gradients (GRAD-001): Gradient magnitude becomes very small
  while cost remains significant
- Gradient imbalance (GRAD-002): Large disparity in gradient magnitudes
  across parameters
- Gradient stagnation (GRAD-003): Gradient norm remains nearly constant
  for multiple iterations

Memory usage is bounded at <1KB regardless of iteration count using:
- Sliding window for gradient norm history (configurable, default 100)
- Welford's online algorithm for running mean/variance per parameter

Issue codes follow the pattern GRAD-NNN for consistency with the
Model Health Diagnostics System.

Integration with TRF Optimizer
------------------------------
The GradientMonitor can be integrated with the TRF optimizer via callbacks:

>>> from nlsq import curve_fit
>>> from nlsq.diagnostics import DiagnosticsConfig, GradientMonitor
>>>
>>> config = DiagnosticsConfig()
>>> monitor = GradientMonitor(config)
>>> callback = monitor.create_callback()
>>>
>>> # Fit with gradient monitoring
>>> result = curve_fit(model, x, y, p0=p0, callback=callback)
>>>
>>> # Get gradient health report
>>> report = monitor.get_report()
>>> print(report)
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from nlsq.diagnostics.recommendations import get_recommendation
from nlsq.diagnostics.types import (
    DiagnosticsConfig,
    GradientHealthReport,
    HealthStatus,
    IssueCategory,
    IssueSeverity,
    ModelHealthIssue,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class GradientMonitor:
    """Monitor gradient health during optimization iterations.

    This class tracks gradient behavior to detect potential optimization
    issues such as vanishing gradients, gradient imbalance between parameters,
    and gradient stagnation. It uses memory-efficient algorithms to ensure
    memory usage stays below 1KB regardless of iteration count.

    Parameters
    ----------
    config : DiagnosticsConfig
        Configuration containing thresholds and settings for gradient monitoring.

    Attributes
    ----------
    config : DiagnosticsConfig
        Configuration for the monitor.
    iteration_count : int
        Total number of iterations recorded.

    Examples
    --------
    >>> from nlsq.diagnostics import DiagnosticsConfig
    >>> from nlsq.diagnostics.gradient_health import GradientMonitor
    >>> import numpy as np
    >>>
    >>> config = DiagnosticsConfig()
    >>> monitor = GradientMonitor(config)
    >>>
    >>> # Record gradients during optimization
    >>> for i in range(100):
    ...     gradient = np.array([0.1, 0.08, 0.12]) / (i + 1)
    ...     monitor.record_gradient(gradient, cost=1.0 / (i + 1))
    >>>
    >>> report = monitor.get_report()
    >>> print(report.health_status)
    HealthStatus.HEALTHY

    Integration with curve_fit callback:

    >>> from nlsq import curve_fit
    >>> from nlsq.diagnostics import DiagnosticsConfig, GradientMonitor
    >>>
    >>> config = DiagnosticsConfig()
    >>> monitor = GradientMonitor(config)
    >>> callback = monitor.create_callback()
    >>>
    >>> # Use in curve_fit (gradient is estimated from Jacobian)
    >>> # result = curve_fit(model, x, y, p0=p0, callback=callback)
    >>> # report = monitor.get_report()

    Notes
    -----
    Memory efficiency is achieved through:

    1. **Sliding window**: Stores only the last N gradient norms (default 100),
       using a deque with maxlen for O(1) append/pop.

    2. **Welford's algorithm**: Computes running mean and variance in O(1) space
       per parameter, without storing individual values.

    The total memory footprint is approximately:
    - Sliding window: window_size * 8 bytes (floats)
    - Per-parameter stats: 3 * n_params * 8 bytes (mean, M2, count)
    - Overhead: ~100 bytes for scalars and bookkeeping

    For 100 window size and 10 parameters: ~900 bytes < 1KB
    """

    __slots__ = (
        "_cost_history",
        "_gradient_norm_history",
        "_has_numerical_issues",
        "_initial_cost",
        "_initial_gradient_norm",
        "_last_gradient_norm",
        "_last_params",
        "_max_imbalance_ratio",
        "_n_params",
        "_param_count",
        "_param_m2",
        "_param_means",
        "config",
        "iteration_count",
    )

    def __init__(self, config: DiagnosticsConfig) -> None:
        """Initialize the gradient monitor.

        Parameters
        ----------
        config : DiagnosticsConfig
            Configuration containing monitoring thresholds.
        """
        self.config = config
        self.iteration_count: int = 0

        # Sliding window for gradient norms (bounded memory)
        self._gradient_norm_history: deque[float] = deque(
            maxlen=config.gradient_window_size
        )
        self._cost_history: deque[float] = deque(maxlen=config.gradient_window_size)

        # Welford's algorithm state for per-parameter running statistics
        self._param_means: np.ndarray = np.array([])
        self._param_m2: np.ndarray = np.array([])  # Sum of squared differences
        self._param_count: int = 0

        # Tracking variables
        self._last_gradient_norm: float = 0.0
        self._has_numerical_issues: bool = False
        self._n_params: int = 0
        self._max_imbalance_ratio: float = 1.0
        self._initial_gradient_norm: float = 0.0
        self._initial_cost: float = 0.0
        self._last_params: np.ndarray | None = None

    def record_gradient(
        self,
        gradient: np.ndarray | Sequence[float],
        cost: float,
    ) -> None:
        """Record a gradient observation from an optimization iteration.

        Parameters
        ----------
        gradient : np.ndarray or Sequence[float]
            The gradient vector (partial derivatives w.r.t. each parameter).
        cost : float
            The current cost/loss value at this iteration.

        Raises
        ------
        ValueError
            If gradient is empty.

        Notes
        -----
        This method uses Welford's online algorithm to update running statistics
        for per-parameter gradient magnitudes. This allows computing mean and
        variance without storing individual values, achieving O(1) memory per
        parameter.

        The algorithm maintains:
        - mean: Running mean of absolute gradient values
        - M2: Sum of squared differences from the mean

        Variance is computed as M2 / (n - 1) when needed.
        """
        gradient = np.asarray(gradient)

        if gradient.size == 0:
            raise ValueError("Gradient array cannot be empty")

        # Check for numerical issues
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            self._has_numerical_issues = True
            # Replace non-finite values with safe large values (avoid overflow in Welford's)
            gradient = np.nan_to_num(gradient, nan=0.0, posinf=1e100, neginf=-1e100)

        self.iteration_count += 1
        n_params = len(gradient)

        # Initialize per-parameter stats on first call
        if self._n_params == 0:
            self._n_params = n_params
            self._param_means = np.zeros(n_params)
            self._param_m2 = np.zeros(n_params)
            self._initial_cost = cost

        # Compute gradient norm
        gradient_norm = float(np.linalg.norm(gradient))
        self._gradient_norm_history.append(gradient_norm)
        self._cost_history.append(cost)
        self._last_gradient_norm = gradient_norm

        # Store initial gradient norm
        if self.iteration_count == 1:
            self._initial_gradient_norm = gradient_norm

        # Update per-parameter running statistics (Welford's algorithm)
        abs_gradient = np.abs(gradient)
        self._param_count += 1
        delta = abs_gradient - self._param_means
        self._param_means += delta / self._param_count
        delta2 = abs_gradient - self._param_means
        self._param_m2 += delta * delta2

        # Track max imbalance ratio (with overflow guard)
        min_grad = (
            np.min(abs_gradient[abs_gradient > 0]) if np.any(abs_gradient > 0) else 1.0
        )
        max_grad = np.max(abs_gradient)
        if min_grad > 0 and np.isfinite(max_grad) and np.isfinite(min_grad):
            imbalance = max_grad / min_grad
            if np.isfinite(imbalance):
                self._max_imbalance_ratio = max(self._max_imbalance_ratio, imbalance)

    def create_callback(
        self,
        user_callback: Callable[..., None] | None = None,
    ) -> Callable[..., None]:
        """Create a callback function for integration with curve_fit/TRF.

        This method creates a callback compatible with NLSQ's optimization
        callbacks. The callback extracts gradient information from the
        optimization state and records it in the monitor.

        Parameters
        ----------
        user_callback : callable, optional
            An optional user callback to chain with the gradient monitor.
            Will be called after gradient recording with the same arguments.

        Returns
        -------
        callable
            A callback function compatible with curve_fit's callback parameter.

        Examples
        --------
        >>> from nlsq import curve_fit
        >>> from nlsq.diagnostics import DiagnosticsConfig, GradientMonitor
        >>>
        >>> monitor = GradientMonitor(DiagnosticsConfig())
        >>> callback = monitor.create_callback()
        >>>
        >>> # result = curve_fit(model, x, y, p0=p0, callback=callback)
        >>> # report = monitor.get_report()

        Notes
        -----
        The callback receives iteration information including:
        - iteration: Current iteration number
        - cost: Current cost value
        - params: Current parameter values
        - info: Dictionary with gradient_norm, nfev, step_norm, etc.

        When the gradient is not directly available, we estimate it from
        changes in parameters and cost, or use gradient_norm from info.
        """

        def gradient_monitor_callback(
            iteration: int,
            cost: float,
            params: np.ndarray,
            info: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            """Callback function for gradient monitoring.

            Parameters
            ----------
            iteration : int
                Current iteration number.
            cost : float
                Current cost value.
            params : np.ndarray
                Current parameter values.
            info : dict, optional
                Additional information including gradient_norm.
            **kwargs : Any
                Additional keyword arguments (ignored).
            """
            # Extract gradient information
            if info is not None and "gradient" in info:
                # Direct gradient available
                gradient = np.asarray(info["gradient"])
            elif info is not None and "gradient_norm" in info:
                # Only gradient norm available - estimate gradient direction
                # from parameter changes
                gradient_norm = info["gradient_norm"]
                if self._last_params is not None and gradient_norm > 0:
                    # Estimate gradient direction from parameter change
                    delta_params = params - self._last_params
                    delta_norm = np.linalg.norm(delta_params)
                    if delta_norm > 0:
                        # Scale to match gradient norm (rough estimate)
                        gradient = -delta_params * (gradient_norm / delta_norm)
                    else:
                        # No parameter change - use uniform gradient
                        n_params = len(params)
                        gradient = np.ones(n_params) * (
                            gradient_norm / np.sqrt(n_params)
                        )
                else:
                    # First iteration or no norm - use uniform gradient
                    n_params = len(params)
                    gradient = (
                        np.ones(n_params) * (gradient_norm / np.sqrt(n_params))
                        if gradient_norm > 0
                        else np.ones(n_params)
                    )
            # No gradient info - estimate from parameters
            elif self._last_params is not None:
                gradient = -(params - self._last_params)
                if np.linalg.norm(gradient) == 0:
                    gradient = np.ones_like(params) * 1e-10
            else:
                gradient = np.ones_like(params)

            # Store current params for next iteration
            self._last_params = params.copy()

            # Record in monitor
            self.record_gradient(gradient, cost)

            # Call user callback if provided
            if user_callback is not None:
                user_callback(
                    iteration=iteration, cost=cost, params=params, info=info, **kwargs
                )

        return gradient_monitor_callback

    def get_report(self) -> GradientHealthReport:
        """Generate a gradient health report from recorded observations.

        Returns
        -------
        GradientHealthReport
            Report containing gradient health metrics and any detected issues.

        Notes
        -----
        The report includes:

        - Overall health score (0-1, higher is healthier)
        - Mean and final gradient norms
        - Per-parameter mean and variance of gradient magnitudes
        - Detection of vanishing gradients, imbalance, and stagnation
        - List of ModelHealthIssue objects for any detected problems
        """
        start_time = time.perf_counter()

        if self.iteration_count == 0:
            return GradientHealthReport(
                available=True,
                n_iterations=0,
                health_score=1.0,
                issues=[],
                health_status=HealthStatus.HEALTHY,
                computation_time_ms=0.0,
            )

        # Compute statistics from sliding window
        norm_history = list(self._gradient_norm_history)
        cost_history = list(self._cost_history)

        mean_gradient_norm = float(np.mean(norm_history)) if norm_history else 0.0
        final_gradient_norm = self._last_gradient_norm

        # Compute per-parameter variance from Welford's M2
        if self._param_count > 1:
            variance = self._param_m2 / (self._param_count - 1)
        else:
            variance = np.zeros_like(self._param_means)

        # Detect issues
        issues: list[ModelHealthIssue] = []

        # Check for vanishing gradients (GRAD-001)
        vanishing_detected = self._detect_vanishing_gradients(
            norm_history, cost_history
        )
        if vanishing_detected:
            issues.append(self._create_grad_001_issue(norm_history, cost_history))

        # Check for gradient imbalance (GRAD-002)
        imbalance_detected = self._detect_gradient_imbalance()
        if imbalance_detected:
            issues.append(self._create_grad_002_issue())

        # Check for gradient stagnation (GRAD-003)
        stagnation_detected = self._detect_gradient_stagnation(norm_history)
        if stagnation_detected:
            issues.append(self._create_grad_003_issue(norm_history))

        # Compute health score
        health_score = self._compute_health_score(
            vanishing_detected, imbalance_detected, stagnation_detected
        )

        # Determine overall health status
        health_status = self._determine_health_status(issues)

        computation_time = (time.perf_counter() - start_time) * 1000

        return GradientHealthReport(
            available=True,
            n_iterations=self.iteration_count,
            health_score=health_score,
            mean_gradient_norm=mean_gradient_norm,
            final_gradient_norm=final_gradient_norm,
            mean_gradient_magnitudes=self._param_means.copy(),
            variance_gradient_magnitudes=variance,
            max_imbalance_ratio=self._max_imbalance_ratio,
            has_numerical_issues=self._has_numerical_issues,
            vanishing_detected=vanishing_detected,
            imbalance_detected=imbalance_detected,
            stagnation_detected=stagnation_detected,
            issues=issues,
            health_status=health_status,
            computation_time_ms=computation_time,
        )

    def reset(self) -> None:
        """Reset the monitor to its initial state.

        Clears all recorded gradients and statistics. Useful when starting
        a new optimization run.
        """
        self.iteration_count = 0
        self._gradient_norm_history.clear()
        self._cost_history.clear()
        self._param_means = np.array([])
        self._param_m2 = np.array([])
        self._param_count = 0
        self._last_gradient_norm = 0.0
        self._has_numerical_issues = False
        self._n_params = 0
        self._max_imbalance_ratio = 1.0
        self._initial_gradient_norm = 0.0
        self._initial_cost = 0.0
        self._last_params = None

    def _detect_vanishing_gradients(
        self,
        norm_history: list[float],
        cost_history: list[float],
    ) -> bool:
        """Detect if gradients are vanishing while cost remains significant.

        Vanishing gradients occur when the gradient magnitude becomes very
        small relative to the vanishing_threshold, but the cost function
        is still significant (not converged).

        Parameters
        ----------
        norm_history : list[float]
            Recent gradient norm history.
        cost_history : list[float]
            Recent cost history.

        Returns
        -------
        bool
            True if vanishing gradients detected.
        """
        if len(norm_history) < 5:
            return False

        # Check recent gradient norms
        recent_norms = norm_history[-10:] if len(norm_history) >= 10 else norm_history
        recent_costs = cost_history[-10:] if len(cost_history) >= 10 else cost_history

        avg_recent_norm = np.mean(recent_norms)
        avg_recent_cost = np.mean(recent_costs)

        # Gradient is vanishing if:
        # 1. Average norm is below threshold
        # 2. Cost is still significant (not effectively zero)
        # 3. Initial gradient was not already vanishing (we had signal to start)
        threshold = self.config.vanishing_threshold

        if self._initial_gradient_norm > 0:
            # Use relative threshold based on initial gradient
            relative_threshold = self._initial_gradient_norm * threshold
        else:
            relative_threshold = threshold

        # Cost is significant if it's above a small epsilon
        cost_significant = avg_recent_cost > 1e-10

        # Also check if gradient has dropped significantly from initial
        gradient_dropped = (
            self._initial_gradient_norm > 0 and avg_recent_norm < relative_threshold
        )

        return gradient_dropped and cost_significant

    def _detect_gradient_imbalance(self) -> bool:
        """Detect if gradient magnitudes are severely imbalanced across parameters.

        Imbalance occurs when the ratio between the largest and smallest
        gradient components exceeds the imbalance_threshold.

        Returns
        -------
        bool
            True if gradient imbalance detected.
        """
        if self._n_params < 2:
            return False

        return self._max_imbalance_ratio > self.config.imbalance_threshold

    def _detect_gradient_stagnation(self, norm_history: list[float]) -> bool:
        """Detect if gradient norm has stagnated (no significant change).

        Stagnation occurs when the gradient norm remains nearly constant
        for multiple consecutive iterations, which may indicate the optimizer
        is stuck or has reached a saddle point.

        Parameters
        ----------
        norm_history : list[float]
            Recent gradient norm history.

        Returns
        -------
        bool
            True if gradient stagnation detected.
        """
        window = self.config.stagnation_window
        tolerance = self.config.stagnation_tolerance

        if len(norm_history) < window:
            return False

        recent = norm_history[-window:]
        mean_norm = np.mean(recent)

        if mean_norm < 1e-15:
            # Effectively zero gradient - this is convergence, not stagnation
            return False

        # Check if relative variation is below tolerance
        std_norm = np.std(recent)
        relative_variation = std_norm / mean_norm

        return relative_variation < tolerance

    def _compute_health_score(
        self,
        vanishing: bool,
        imbalance: bool,
        stagnation: bool,
    ) -> float:
        """Compute overall gradient health score.

        The score ranges from 0 (poor) to 1 (healthy).

        Parameters
        ----------
        vanishing : bool
            Whether vanishing gradients were detected.
        imbalance : bool
            Whether gradient imbalance was detected.
        stagnation : bool
            Whether gradient stagnation was detected.

        Returns
        -------
        float
            Health score in [0, 1].
        """
        score = 1.0

        # Deduct for each detected issue
        if vanishing:
            score -= 0.3
        if imbalance:
            score -= 0.3
        if stagnation:
            score -= 0.2
        if self._has_numerical_issues:
            score -= 0.2

        return max(0.0, score)

    def _determine_health_status(self, issues: list[ModelHealthIssue]) -> HealthStatus:
        """Determine overall health status from detected issues.

        Parameters
        ----------
        issues : list[ModelHealthIssue]
            List of detected issues.

        Returns
        -------
        HealthStatus
            Overall health status.
        """
        if not issues:
            return HealthStatus.HEALTHY

        has_critical = any(issue.severity == IssueSeverity.CRITICAL for issue in issues)
        if has_critical:
            return HealthStatus.CRITICAL

        has_warning = any(issue.severity == IssueSeverity.WARNING for issue in issues)
        if has_warning:
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def _create_grad_001_issue(
        self,
        norm_history: list[float],
        cost_history: list[float],
    ) -> ModelHealthIssue:
        """Create GRAD-001 issue for vanishing gradients.

        Parameters
        ----------
        norm_history : list[float]
            Recent gradient norm history.
        cost_history : list[float]
            Recent cost history.

        Returns
        -------
        ModelHealthIssue
            Issue describing vanishing gradients.
        """
        recent_norm = (
            np.mean(norm_history[-10:])
            if len(norm_history) >= 10
            else np.mean(norm_history)
        )
        recent_cost = (
            np.mean(cost_history[-10:])
            if len(cost_history) >= 10
            else np.mean(cost_history)
        )

        return ModelHealthIssue(
            category=IssueCategory.GRADIENT,
            severity=IssueSeverity.WARNING,
            code="GRAD-001",
            message=(
                f"Vanishing gradients detected: gradient norm ({recent_norm:.2e}) "
                f"is very small while cost ({recent_cost:.2e}) remains significant."
            ),
            affected_parameters=None,
            details={
                "recent_gradient_norm": recent_norm,
                "recent_cost": recent_cost,
                "threshold": self.config.vanishing_threshold,
                "initial_gradient_norm": self._initial_gradient_norm,
            },
            recommendation=get_recommendation("GRAD-001"),
        )

    def _create_grad_002_issue(self) -> ModelHealthIssue:
        """Create GRAD-002 issue for gradient imbalance.

        Returns
        -------
        ModelHealthIssue
            Issue describing gradient imbalance.
        """
        # Identify which parameters have extreme gradients
        if len(self._param_means) > 0:
            min_idx = int(np.argmin(self._param_means))
            max_idx = int(np.argmax(self._param_means))
            affected = (min_idx, max_idx) if min_idx != max_idx else (min_idx,)
        else:
            affected = None

        return ModelHealthIssue(
            category=IssueCategory.GRADIENT,
            severity=IssueSeverity.WARNING,
            code="GRAD-002",
            message=(
                f"Gradient imbalance detected: ratio between largest and smallest "
                f"gradient components is {self._max_imbalance_ratio:.2e}, "
                f"exceeding threshold {self.config.imbalance_threshold:.2e}."
            ),
            affected_parameters=affected,
            details={
                "imbalance_ratio": self._max_imbalance_ratio,
                "threshold": self.config.imbalance_threshold,
                "mean_gradient_magnitudes": self._param_means.tolist(),
            },
            recommendation=get_recommendation("GRAD-002"),
        )

    def _create_grad_003_issue(self, norm_history: list[float]) -> ModelHealthIssue:
        """Create GRAD-003 issue for gradient stagnation.

        Parameters
        ----------
        norm_history : list[float]
            Recent gradient norm history.

        Returns
        -------
        ModelHealthIssue
            Issue describing gradient stagnation.
        """
        window = self.config.stagnation_window
        recent = norm_history[-window:] if len(norm_history) >= window else norm_history
        mean_norm = np.mean(recent)
        std_norm = np.std(recent)

        return ModelHealthIssue(
            category=IssueCategory.GRADIENT,
            severity=IssueSeverity.WARNING,
            code="GRAD-003",
            message=(
                f"Gradient stagnation detected: gradient norm has remained "
                f"nearly constant ({mean_norm:.2e} +/- {std_norm:.2e}) "
                f"over the last {len(recent)} iterations."
            ),
            affected_parameters=None,
            details={
                "mean_gradient_norm": mean_norm,
                "std_gradient_norm": std_norm,
                "relative_variation": std_norm / mean_norm if mean_norm > 0 else 0,
                "stagnation_window": window,
                "tolerance": self.config.stagnation_tolerance,
            },
            recommendation=get_recommendation("GRAD-003"),
        )
