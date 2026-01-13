"""Core types and data structures for the Model Health Diagnostics System.

This module provides the foundational types used throughout the diagnostics
package, including enumerations for health status and issue severity, and
dataclasses for issues, analysis results, and configuration.

All dataclasses use __slots__ for memory efficiency following NLSQ v0.4.2
patterns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class HealthStatus(Enum):
    """Overall model health status.

    Attributes
    ----------
    HEALTHY : auto
        No issues detected, high confidence in results.
    WARNING : auto
        Minor issues detected, results may be reliable.
    CRITICAL : auto
        Serious issues detected, results may be unreliable.
    """

    HEALTHY = auto()
    WARNING = auto()
    CRITICAL = auto()


class IssueSeverity(Enum):
    """Severity level of a detected issue.

    Attributes
    ----------
    INFO : auto
        Informational, no action required.
    WARNING : auto
        Potential problem, review recommended.
    CRITICAL : auto
        Serious problem, action required.
    """

    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


class IssueCategory(Enum):
    """Category of detected issue.

    Attributes
    ----------
    IDENTIFIABILITY : auto
        Parameter identifiability issues.
    GRADIENT : auto
        Gradient health issues.
    CORRELATION : auto
        Parameter correlation issues.
    CONDITIONING : auto
        Numerical conditioning issues.
    CONVERGENCE : auto
        Convergence-related issues.
    SENSITIVITY : auto
        Parameter sensitivity spectrum issues (wide eigenvalue spread).
    """

    IDENTIFIABILITY = auto()
    GRADIENT = auto()
    CORRELATION = auto()
    CONDITIONING = auto()
    CONVERGENCE = auto()
    SENSITIVITY = auto()


class DiagnosticLevel(Enum):
    """Diagnostic analysis depth level.

    Attributes
    ----------
    BASIC : auto
        Fast analysis: identifiability + gradient health.
    FULL : auto
        Comprehensive analysis: includes parameter sensitivity analysis.
    """

    BASIC = auto()
    FULL = auto()


@dataclass(slots=True, frozen=True)
class ModelHealthIssue:
    """A single detected model health issue.

    This dataclass represents an actionable issue detected during
    diagnostic analysis, including its category, severity, and
    a recommendation for addressing it.

    Attributes
    ----------
    category : IssueCategory
        Category of the issue.
    severity : IssueSeverity
        Severity level.
    code : str
        Unique issue code (e.g., "IDENT-001", "GRAD-002").
    message : str
        Human-readable description of the issue.
    affected_parameters : tuple[int, ...] | None
        Indices of affected parameters, if applicable.
    details : dict[str, Any]
        Additional issue-specific details.
    recommendation : str
        Actionable recommendation for addressing the issue.

    Examples
    --------
    >>> issue = ModelHealthIssue(
    ...     category=IssueCategory.IDENTIFIABILITY,
    ...     severity=IssueSeverity.CRITICAL,
    ...     code="IDENT-001",
    ...     message="Parameters 0 and 1 are structurally unidentifiable",
    ...     affected_parameters=(0, 1),
    ...     details={"numerical_rank": 2, "n_params": 3},
    ...     recommendation="Consider reparameterizing the model",
    ... )
    >>> issue.code
    'IDENT-001'
    >>> issue.severity
    <IssueSeverity.CRITICAL: 3>
    """

    category: IssueCategory
    severity: IssueSeverity
    code: str
    message: str
    affected_parameters: tuple[int, ...] | None
    details: dict[str, Any]
    recommendation: str

    def __post_init__(self) -> None:
        """Validate issue attributes after initialization."""
        if not self.code:
            raise ValueError("Issue code cannot be empty")
        if not self.message:
            raise ValueError("Issue message cannot be empty")


@dataclass(slots=True)
class AnalysisResult:
    """Base class for analysis results.

    Provides common attributes for tracking whether an analysis
    completed successfully, any error messages, and timing information.

    Attributes
    ----------
    available : bool
        Whether the analysis completed successfully.
    error_message : str | None
        Error message if analysis failed.
    computation_time_ms : float
        Time taken to compute this analysis in milliseconds.

    Examples
    --------
    >>> result = AnalysisResult()
    >>> result.available
    True
    >>> result = AnalysisResult(available=False, error_message="SVD failed")
    >>> result.available
    False
    >>> result.error_message
    'SVD failed'
    """

    available: bool = True
    error_message: str | None = None
    computation_time_ms: float = 0.0


@dataclass(slots=True)
class IdentifiabilityReport(AnalysisResult):
    """Report from identifiability analysis.

    Contains results from analyzing the Fisher Information Matrix (FIM)
    including condition number, numerical rank, correlation structure,
    and any detected identifiability issues.

    This dataclass extends AnalysisResult to include identifiability-specific
    information such as condition number, rank, and correlation analysis.

    Attributes
    ----------
    condition_number : float
        Condition number of the FIM. High values (> 1e8) indicate
        practical unidentifiability.
    numerical_rank : int
        Numerical rank of the FIM. If less than n_params, indicates
        structural unidentifiability.
    n_params : int
        Total number of parameters in the model.
    correlation_matrix : np.ndarray | None
        Parameter correlation matrix derived from FIM. None if
        computation failed.
    highly_correlated_pairs : list[tuple[int, int, float]]
        List of highly correlated parameter pairs as (i, j, correlation).
        Only includes pairs with absolute correlation greater than
        correlation_threshold.
    issues : list[ModelHealthIssue]
        List of detected identifiability issues (IDENT-001, IDENT-002, CORR-001).
    health_status : HealthStatus
        Overall health status based on detected issues.

    Examples
    --------
    >>> report = IdentifiabilityReport(
    ...     condition_number=1e5,
    ...     numerical_rank=3,
    ...     n_params=3,
    ...     correlation_matrix=np.eye(3),
    ...     highly_correlated_pairs=[],
    ...     issues=[],
    ...     health_status=HealthStatus.HEALTHY,
    ... )
    >>> report.available
    True
    >>> report.condition_number
    100000.0

    >>> # Report with issues
    >>> from nlsq.diagnostics.types import ModelHealthIssue, IssueCategory, IssueSeverity
    >>> issue = ModelHealthIssue(
    ...     category=IssueCategory.IDENTIFIABILITY,
    ...     severity=IssueSeverity.CRITICAL,
    ...     code="IDENT-001",
    ...     message="Structural unidentifiability detected",
    ...     affected_parameters=(0, 1),
    ...     details={"numerical_rank": 2, "n_params": 3},
    ...     recommendation="Reparameterize model",
    ... )
    >>> report = IdentifiabilityReport(
    ...     condition_number=float('inf'),
    ...     numerical_rank=2,
    ...     n_params=3,
    ...     correlation_matrix=None,
    ...     highly_correlated_pairs=[],
    ...     issues=[issue],
    ...     health_status=HealthStatus.CRITICAL,
    ... )
    >>> len(report.issues)
    1
    """

    condition_number: float = float("inf")
    numerical_rank: int = 0
    n_params: int = 0
    correlation_matrix: np.ndarray | None = None
    highly_correlated_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    issues: list[ModelHealthIssue] = field(default_factory=list)
    health_status: HealthStatus = HealthStatus.HEALTHY

    def __str__(self) -> str:
        """Return a human-readable summary of the identifiability report."""
        if not self.available:
            return f"IdentifiabilityReport: UNAVAILABLE - {self.error_message}"

        lines = [
            "Identifiability Analysis Report",
            "=" * 40,
            f"Health Status: {self.health_status.name}",
            f"Condition Number: {self.condition_number:.2e}",
            f"Numerical Rank: {self.numerical_rank}/{self.n_params}",
            f"Computation Time: {self.computation_time_ms:.2f} ms",
        ]

        if self.highly_correlated_pairs:
            lines.append(
                f"\nHighly Correlated Pairs ({len(self.highly_correlated_pairs)}):"
            )
            for i, j, corr in self.highly_correlated_pairs:
                lines.append(f"  Parameters {i} and {j}: {corr:.4f}")

        if self.issues:
            lines.append(f"\nIssues Detected ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"  [{issue.severity.name}] {issue.code}: {issue.message}")  # noqa: PERF401

        return "\n".join(lines)

    def summary(self) -> str:
        """Return a summary string of the report.

        Returns
        -------
        str
            Human-readable summary of the identifiability analysis.
        """
        return str(self)


@dataclass(slots=True)
class GradientHealthReport(AnalysisResult):
    """Report from gradient health monitoring during optimization.

    Contains results from monitoring gradient behavior across iterations,
    including detection of vanishing gradients, gradient imbalance,
    and gradient stagnation.

    This dataclass extends AnalysisResult to include gradient-specific
    metrics tracked during optimization using memory-efficient algorithms
    (sliding window for norms, Welford's algorithm for running statistics).

    Memory usage is bounded at <1KB regardless of iteration count.

    Attributes
    ----------
    n_iterations : int
        Total number of iterations monitored.
    health_score : float
        Overall gradient health score in [0, 1]. Higher is healthier.
    mean_gradient_norm : float
        Mean gradient norm across all iterations.
    final_gradient_norm : float
        Gradient norm at the final iteration.
    mean_gradient_magnitudes : np.ndarray
        Mean gradient magnitude per parameter (from Welford's algorithm).
    variance_gradient_magnitudes : np.ndarray
        Variance of gradient magnitude per parameter (from Welford's algorithm).
    max_imbalance_ratio : float
        Maximum ratio between largest and smallest gradient components.
    has_numerical_issues : bool
        Whether NaN or Inf values were detected in gradients.
    vanishing_detected : bool
        Whether vanishing gradients were detected.
    imbalance_detected : bool
        Whether gradient imbalance was detected.
    stagnation_detected : bool
        Whether gradient stagnation was detected.
    issues : list[ModelHealthIssue]
        List of detected gradient issues (GRAD-001, GRAD-002, GRAD-003).
    health_status : HealthStatus
        Overall health status based on detected issues.

    Examples
    --------
    >>> report = GradientHealthReport(
    ...     n_iterations=100,
    ...     health_score=0.95,
    ...     mean_gradient_norm=0.1,
    ...     final_gradient_norm=0.001,
    ...     mean_gradient_magnitudes=np.array([0.1, 0.08, 0.12]),
    ...     variance_gradient_magnitudes=np.array([0.01, 0.01, 0.01]),
    ...     max_imbalance_ratio=1.5,
    ...     has_numerical_issues=False,
    ...     vanishing_detected=False,
    ...     imbalance_detected=False,
    ...     stagnation_detected=False,
    ...     issues=[],
    ...     health_status=HealthStatus.HEALTHY,
    ... )
    >>> report.available
    True
    >>> report.health_score
    0.95
    """

    n_iterations: int = 0
    health_score: float = 1.0
    mean_gradient_norm: float = 0.0
    final_gradient_norm: float = 0.0
    mean_gradient_magnitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_gradient_magnitudes: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    max_imbalance_ratio: float = 1.0
    has_numerical_issues: bool = False
    vanishing_detected: bool = False
    imbalance_detected: bool = False
    stagnation_detected: bool = False
    issues: list[ModelHealthIssue] = field(default_factory=list)
    health_status: HealthStatus = HealthStatus.HEALTHY

    def __str__(self) -> str:
        """Return a human-readable summary of the gradient health report."""
        if not self.available:
            return f"GradientHealthReport: UNAVAILABLE - {self.error_message}"

        lines = [
            "Gradient Health Report",
            "=" * 40,
            f"Health Status: {self.health_status.name}",
            f"Health Score: {self.health_score:.2f}",
            f"Iterations Monitored: {self.n_iterations}",
            f"Mean Gradient Norm: {self.mean_gradient_norm:.2e}",
            f"Final Gradient Norm: {self.final_gradient_norm:.2e}",
            f"Max Imbalance Ratio: {self.max_imbalance_ratio:.2e}",
            f"Computation Time: {self.computation_time_ms:.2f} ms",
        ]

        if self.has_numerical_issues:
            lines.append("\n[!] Numerical issues (NaN/Inf) detected in gradients")

        if self.issues:
            lines.append(f"\nIssues Detected ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"  [{issue.severity.name}] {issue.code}: {issue.message}")  # noqa: PERF401

        return "\n".join(lines)

    def summary(self) -> str:
        """Return a summary string of the report.

        Returns
        -------
        str
            Human-readable summary of the gradient health analysis.
        """
        return str(self)


@dataclass(slots=True)
class ParameterSensitivityReport(AnalysisResult):
    """Report from parameter sensitivity spectrum analysis.

    Contains results from eigenvalue spectrum analysis to identify
    well-determined vs poorly-determined parameter directions based
    on the spread of eigenvalues in the Fisher Information Matrix.

    Attributes
    ----------
    is_sloppy : bool
        Whether the model exhibits wide eigenvalue spread (sensitivity spectrum).
    eigenvalues : np.ndarray
        Eigenvalue spectrum of the Fisher Information Matrix.
    eigenvectors : np.ndarray | None
        Eigenvectors of the FIM (columns are eigenvectors).
    eigenvalue_range : float
        Log10 range of eigenvalues (orders of magnitude).
    effective_dimensionality : float
        Effective number of well-determined parameter combinations.
    stiff_indices : list[int]
        Indices of stiff (well-determined) directions.
    sloppy_indices : list[int]
        Indices of poorly-determined directions.
    issues : list[ModelHealthIssue]
        List of detected sensitivity issues (SENS-001, SENS-002).
    health_status : HealthStatus
        Overall health status based on detected issues.
    """

    is_sloppy: bool = False
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray | None = None
    eigenvalue_range: float = 0.0
    effective_dimensionality: float = 0.0
    stiff_indices: list[int] = field(default_factory=list)
    sloppy_indices: list[int] = field(default_factory=list)
    issues: list[ModelHealthIssue] = field(default_factory=list)
    health_status: HealthStatus = HealthStatus.HEALTHY

    def get_sloppy_combinations(self) -> list[tuple[np.ndarray, float]]:
        """Get poorly determined parameter combinations.

        Returns
        -------
        list[tuple[np.ndarray, float]]
            List of (eigenvector, eigenvalue) tuples for poorly-determined directions.
        """
        if self.eigenvectors is None or len(self.sloppy_indices) == 0:
            return []
        return [
            (self.eigenvectors[:, idx], self.eigenvalues[idx])
            for idx in self.sloppy_indices
        ]


@dataclass(slots=True)
class PluginResult:
    """Result from a diagnostic plugin execution.

    Attributes
    ----------
    plugin_name : str
        Name of the plugin that produced this result.
    available : bool
        Whether the plugin executed successfully.
    error_message : str | None
        Error message if plugin execution failed.
    data : dict[str, Any]
        Plugin-specific result data.
    issues : list[ModelHealthIssue]
        Issues detected by the plugin.
    computation_time_ms : float
        Time taken for plugin execution.
    """

    plugin_name: str = ""
    available: bool = True
    error_message: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    issues: list[ModelHealthIssue] = field(default_factory=list)
    computation_time_ms: float = 0.0


@dataclass(slots=True)
class DiagnosticsReport:
    """Aggregated diagnostics report containing all analysis results.

    This class aggregates results from all diagnostic analyses into
    a single report. It provides access to individual analysis results
    and an overall health assessment.

    Attributes
    ----------
    identifiability : IdentifiabilityReport | None
        Results from identifiability analysis.
    gradient_health : GradientHealthReport | None
        Results from gradient health monitoring.
    overall_status : HealthStatus
        Overall health status based on all analyses.
    computation_time_ms : float
        Total time for all diagnostic computations.

    Examples
    --------
    >>> from nlsq.diagnostics.types import IdentifiabilityReport
    >>> ident = IdentifiabilityReport(
    ...     condition_number=1e5,
    ...     numerical_rank=3,
    ...     n_params=3,
    ...     correlation_matrix=np.eye(3),
    ...     highly_correlated_pairs=[],
    ...     issues=[],
    ...     health_status=HealthStatus.HEALTHY,
    ... )
    >>> report = DiagnosticsReport(identifiability=ident)
    >>> report.overall_status
    <HealthStatus.HEALTHY: 1>
    """

    identifiability: IdentifiabilityReport | None = None
    gradient_health: GradientHealthReport | None = None
    overall_status: HealthStatus = HealthStatus.HEALTHY
    computation_time_ms: float = 0.0

    def __post_init__(self) -> None:
        """Compute overall status from individual analyses."""
        self._compute_overall_status()

    def _compute_overall_status(self) -> None:
        """Compute overall health status from individual analyses."""
        # Check identifiability
        if self.identifiability is not None:
            if self.identifiability.health_status == HealthStatus.CRITICAL:
                self.overall_status = HealthStatus.CRITICAL
            elif self.identifiability.health_status == HealthStatus.WARNING:
                if self.overall_status != HealthStatus.CRITICAL:
                    self.overall_status = HealthStatus.WARNING

        # Check gradient health
        if self.gradient_health is not None:
            if self.gradient_health.health_status == HealthStatus.CRITICAL:
                self.overall_status = HealthStatus.CRITICAL
            elif self.gradient_health.health_status == HealthStatus.WARNING:
                if self.overall_status != HealthStatus.CRITICAL:
                    self.overall_status = HealthStatus.WARNING

    def __str__(self) -> str:
        """Return a human-readable summary of all diagnostics."""
        lines = [
            "Model Health Diagnostics Report",
            "=" * 50,
            f"Overall Status: {self.overall_status.name}",
            f"Total Computation Time: {self.computation_time_ms:.2f} ms",
            "",
        ]

        if self.identifiability is not None:
            lines.append(str(self.identifiability))
            lines.append("")

        if self.gradient_health is not None:
            lines.append(str(self.gradient_health))

        return "\n".join(lines)

    def summary(self) -> str:
        """Return a summary string of all diagnostics.

        Returns
        -------
        str
            Human-readable summary of all diagnostic analyses.
        """
        return str(self)


@dataclass(slots=True, frozen=True)
class DiagnosticsConfig:
    """Configuration for diagnostic computation.

    This frozen dataclass contains all thresholds and settings used
    by the diagnostic analyzers. Being frozen ensures configuration
    immutability during analysis.

    Attributes
    ----------
    level : DiagnosticLevel
        Diagnostic analysis depth.
    condition_threshold : float
        FIM condition number threshold for practical identifiability.
        Default: 1e8.
    correlation_threshold : float
        Correlation coefficient threshold for high correlation warning.
        Default: 0.95.
    imbalance_threshold : float
        Gradient imbalance ratio threshold.
        Default: 1e6.
    vanishing_threshold : float
        Relative gradient magnitude threshold for vanishing detection.
        Default: 1e-6.
    sloppy_threshold : float
        Eigenvalue ratio threshold for sensitivity classification.
        Default: 1e-6.
    gradient_window_size : int
        Window size for gradient norm history.
        Default: 100.
    stagnation_window : int
        Number of iterations to check for gradient stagnation.
        Default: 10.
    stagnation_tolerance : float
        Relative tolerance for detecting gradient stagnation.
        Default: 0.01 (1% change).
    verbose : bool
        Print diagnostic summary to console.
        Default: True.
    emit_warnings : bool
        Emit Python warnings for critical issues.
        Default: True.

    Examples
    --------
    >>> config = DiagnosticsConfig()
    >>> config.level
    <DiagnosticLevel.BASIC: 1>
    >>> config.condition_threshold
    100000000.0

    >>> config = DiagnosticsConfig(
    ...     level=DiagnosticLevel.FULL,
    ...     condition_threshold=1e10,
    ...     verbose=False,
    ... )
    >>> config.level
    <DiagnosticLevel.FULL: 2>
    """

    level: DiagnosticLevel = DiagnosticLevel.BASIC
    condition_threshold: float = 1e8
    correlation_threshold: float = 0.95
    imbalance_threshold: float = 1e6
    vanishing_threshold: float = 1e-6
    sloppy_threshold: float = 1e-6
    gradient_window_size: int = 100
    stagnation_window: int = 10
    stagnation_tolerance: float = 0.01
    verbose: bool = True
    emit_warnings: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        # Validate thresholds are positive
        if self.condition_threshold <= 0:
            raise ValueError("condition_threshold must be positive")
        if self.correlation_threshold <= 0 or self.correlation_threshold > 1:
            raise ValueError("correlation_threshold must be in (0, 1]")
        if self.imbalance_threshold <= 0:
            raise ValueError("imbalance_threshold must be positive")
        if self.vanishing_threshold <= 0:
            raise ValueError("vanishing_threshold must be positive")
        if self.sloppy_threshold <= 0:
            raise ValueError("sloppy_threshold must be positive")
        if self.gradient_window_size <= 0:
            raise ValueError("gradient_window_size must be positive")
        if self.stagnation_window <= 0:
            raise ValueError("stagnation_window must be positive")
        if self.stagnation_tolerance <= 0:
            raise ValueError("stagnation_tolerance must be positive")


@dataclass(slots=True)
class ModelHealthReport:
    """Aggregated model health report with overall assessment.

    This dataclass aggregates results from all diagnostic components
    (identifiability, gradient health, parameter sensitivity, and plugins) into
    a unified health report with overall status, health score, and
    actionable recommendations.

    Attributes
    ----------
    identifiability : IdentifiabilityReport | None
        Results from identifiability analysis.
    gradient_health : GradientHealthReport | None
        Results from gradient health monitoring.
    sloppy_model : ParameterSensitivityReport | None
        Results from parameter sensitivity analysis (level=FULL only).
    plugin_results : dict[str, PluginResult]
        Results from diagnostic plugins, keyed by plugin name.
    status : HealthStatus
        Overall health status (HEALTHY, WARNING, or CRITICAL).
    health_score : float
        Overall health score in [0.0, 1.0]. Higher is healthier.
    all_issues : list[ModelHealthIssue]
        Aggregated issues from all components, sorted by severity.
    config : DiagnosticsConfig | None
        Configuration used for diagnostics.
    computation_time_ms : float
        Total computation time for all diagnostics in milliseconds.

    Examples
    --------
    >>> from nlsq.diagnostics.health_report import create_health_report
    >>> report = create_health_report(
    ...     identifiability=healthy_ident_report,
    ...     gradient_health=healthy_grad_report,
    ... )
    >>> report.status
    <HealthStatus.HEALTHY: 1>
    >>> report.health_score
    1.0
    """

    identifiability: IdentifiabilityReport | None = None
    gradient_health: GradientHealthReport | None = None
    sloppy_model: ParameterSensitivityReport | None = None
    plugin_results: dict[str, PluginResult] = field(default_factory=dict)
    status: HealthStatus = HealthStatus.HEALTHY
    health_score: float = 1.0
    all_issues: list[ModelHealthIssue] = field(default_factory=list)
    config: DiagnosticsConfig | None = None
    computation_time_ms: float = 0.0

    def summary(self, verbose: bool = True) -> str:
        """Generate human-readable summary.

        Parameters
        ----------
        verbose : bool, default=True
            Include detailed issue descriptions and recommendations.

        Returns
        -------
        str
            Formatted summary string suitable for console output.
        """
        return _format_model_health_summary(self, verbose=verbose)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the report.
        """
        return _model_health_to_dict(self)

    def __str__(self) -> str:
        """Return summary as string representation."""
        return self.summary()


def _format_model_health_summary(
    report: ModelHealthReport, verbose: bool = True
) -> str:
    """Format the health report as a human-readable summary.

    This function implements contract B4 (healthy format) and B5 (issues format).

    Parameters
    ----------
    report : ModelHealthReport
        The report to format.
    verbose : bool, default=True
        Include detailed descriptions and recommendations.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines: list[str] = []
    separator = "=" * 70

    # Header
    lines.append(separator)
    lines.append("Model Health Report")
    lines.append(separator)
    lines.append("")

    # Status and score
    lines.append(f"Status: {report.status.name}")
    lines.append(f"Health Score: {report.health_score:.2f}")
    lines.append("")

    # Issues section (only if there are issues)
    if report.all_issues:
        lines.append(f"--- Issues ({len(report.all_issues)}) ---")
        for issue in report.all_issues:
            lines.append(f"[{issue.severity.name}] {issue.code}: {issue.message}")
            if verbose:
                # Add recommendation with arrow prefix
                lines.append(f"  -> {issue.recommendation}")
        lines.append("")

    # Identifiability section
    if report.identifiability is not None:
        lines.append("--- Identifiability ---")
        if report.identifiability.available:
            is_structural = (
                report.identifiability.numerical_rank >= report.identifiability.n_params
            )
            is_practical = (
                is_structural and report.identifiability.condition_number < 1e8
            )
            lines.append(
                f"Structurally identifiable: {'Yes' if is_structural else 'No'}"
            )
            lines.append(f"Practically identifiable: {'Yes' if is_practical else 'No'}")

            # Format condition number appropriately
            cond = report.identifiability.condition_number
            if math.isinf(cond):
                lines.append("FIM condition number: Inf")
            else:
                lines.append(f"FIM condition number: {cond:.2e}")

            # Highly correlated pairs
            if report.identifiability.highly_correlated_pairs:
                pairs_str = ", ".join(
                    f"({i}, {j}): {r:.2f}"
                    for i, j, r in report.identifiability.highly_correlated_pairs
                )
                lines.append(f"Highly correlated pairs: {pairs_str}")
            else:
                lines.append("Highly correlated pairs: None")
        else:
            lines.append(f"UNAVAILABLE: {report.identifiability.error_message}")
        lines.append("")

    # Gradient Health section
    if report.gradient_health is not None:
        lines.append("--- Gradient Health ---")
        if report.gradient_health.available:
            lines.append(f"Health score: {report.gradient_health.health_score:.2f}")
            lines.append(
                f"Vanishing gradients: {'Yes' if report.gradient_health.vanishing_detected else 'No'}"
            )
            lines.append(
                f"Gradient imbalance: {'Yes' if report.gradient_health.imbalance_detected else 'No'}"
            )
        else:
            lines.append(f"UNAVAILABLE: {report.gradient_health.error_message}")
        lines.append("")

    # Sloppy Model section (only for FULL level)
    config = report.config or DiagnosticsConfig()
    if config.level == DiagnosticLevel.FULL and report.sloppy_model is not None:
        lines.append("--- Sloppy Model ---")
        if report.sloppy_model.available:
            lines.append(
                f"Is sloppy: {'Yes' if report.sloppy_model.is_sloppy else 'No'}"
            )
            lines.append(
                f"Eigenvalue range: {report.sloppy_model.eigenvalue_range:.1f} orders of magnitude"
            )
            lines.append(
                f"Effective dimensionality: {report.sloppy_model.effective_dimensionality:.1f}"
            )
        else:
            lines.append(f"UNAVAILABLE: {report.sloppy_model.error_message}")
        lines.append("")

    # Plugin results section (if any)
    if report.plugin_results:
        lines.append("--- Plugin Results ---")
        for name, result in report.plugin_results.items():
            if result.available:
                n_issues = len(result.issues)
                lines.append(f"{name}: {n_issues} issue(s)")
            else:
                lines.append(f"{name}: UNAVAILABLE - {result.error_message}")
        lines.append("")

    # Recommendations section (only if there are issues and verbose)
    if verbose and report.all_issues:
        lines.append("--- Recommendations ---")
        # Deduplicate recommendations while preserving order
        seen_recommendations: set[str] = set()
        recommendation_list: list[str] = []
        for issue in report.all_issues:
            if issue.recommendation not in seen_recommendations:
                seen_recommendations.add(issue.recommendation)
                recommendation_list.append(issue.recommendation)

        for i, rec in enumerate(recommendation_list, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    # Footer
    lines.append(separator)

    return "\n".join(lines)


def _model_health_to_dict(report: ModelHealthReport) -> dict[str, Any]:
    """Convert a ModelHealthReport to a dictionary for serialization.

    Parameters
    ----------
    report : ModelHealthReport
        The report to convert.

    Returns
    -------
    dict[str, Any]
        Dictionary representation with JSON-serializable values.
    """
    result: dict[str, Any] = {
        "status": report.status.name,
        "health_score": report.health_score,
        "computation_time_ms": report.computation_time_ms,
        "issues": [
            {
                "category": issue.category.name,
                "severity": issue.severity.name,
                "code": issue.code,
                "message": issue.message,
                "affected_parameters": issue.affected_parameters,
                "details": issue.details,
                "recommendation": issue.recommendation,
            }
            for issue in report.all_issues
        ],
    }

    # Add identifiability if present
    if report.identifiability is not None:
        result["identifiability"] = {
            "available": report.identifiability.available,
            "error_message": report.identifiability.error_message,
            "condition_number": (
                None
                if math.isinf(report.identifiability.condition_number)
                else report.identifiability.condition_number
            ),
            "numerical_rank": report.identifiability.numerical_rank,
            "n_params": report.identifiability.n_params,
            "highly_correlated_pairs": report.identifiability.highly_correlated_pairs,
            "health_status": report.identifiability.health_status.name,
            "computation_time_ms": report.identifiability.computation_time_ms,
        }

    # Add gradient health if present
    if report.gradient_health is not None:
        result["gradient_health"] = {
            "available": report.gradient_health.available,
            "error_message": report.gradient_health.error_message,
            "n_iterations": report.gradient_health.n_iterations,
            "health_score": report.gradient_health.health_score,
            "mean_gradient_norm": report.gradient_health.mean_gradient_norm,
            "final_gradient_norm": report.gradient_health.final_gradient_norm,
            "max_imbalance_ratio": report.gradient_health.max_imbalance_ratio,
            "has_numerical_issues": report.gradient_health.has_numerical_issues,
            "vanishing_detected": report.gradient_health.vanishing_detected,
            "imbalance_detected": report.gradient_health.imbalance_detected,
            "stagnation_detected": report.gradient_health.stagnation_detected,
            "health_status": report.gradient_health.health_status.name,
            "computation_time_ms": report.gradient_health.computation_time_ms,
        }

    # Add sloppy model if present
    if report.sloppy_model is not None:
        result["sloppy_model"] = {
            "available": report.sloppy_model.available,
            "error_message": report.sloppy_model.error_message,
            "is_sloppy": report.sloppy_model.is_sloppy,
            "eigenvalue_range": report.sloppy_model.eigenvalue_range,
            "effective_dimensionality": report.sloppy_model.effective_dimensionality,
            "stiff_indices": report.sloppy_model.stiff_indices,
            "sloppy_indices": report.sloppy_model.sloppy_indices,
            "health_status": report.sloppy_model.health_status.name,
            "computation_time_ms": report.sloppy_model.computation_time_ms,
        }

    # Add plugin results if present
    if report.plugin_results:
        result["plugin_results"] = {
            name: {
                "available": pr.available,
                "error_message": pr.error_message,
                "data": pr.data,
                "issues": [
                    {
                        "code": issue.code,
                        "severity": issue.severity.name,
                        "message": issue.message,
                    }
                    for issue in pr.issues
                ],
                "computation_time_ms": pr.computation_time_ms,
            }
            for name, pr in report.plugin_results.items()
        }

    # Add config if present
    if report.config is not None:
        result["config"] = {
            "level": report.config.level.name,
            "condition_threshold": report.config.condition_threshold,
            "correlation_threshold": report.config.correlation_threshold,
            "imbalance_threshold": report.config.imbalance_threshold,
            "vanishing_threshold": report.config.vanishing_threshold,
            "sloppy_threshold": report.config.sloppy_threshold,
            "verbose": report.config.verbose,
            "emit_warnings": report.config.emit_warnings,
        }

    return result
