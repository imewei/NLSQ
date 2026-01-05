"""Model Health Report aggregation and formatting.

This module provides the create_health_report() factory function for
creating aggregated health reports from component analyses, as well as
summary formatting and serialization utilities.

The health report aggregates results from:
- Identifiability analysis
- Gradient health monitoring
- Parameter sensitivity analysis (level=FULL)
- Custom diagnostic plugins

It computes an overall health score and status, collects all issues,
and provides human-readable summaries with recommendations.
"""

import math
import time
import warnings
from typing import Any

from nlsq.diagnostics.types import (
    DiagnosticLevel,
    DiagnosticsConfig,
    GradientHealthReport,
    HealthStatus,
    IdentifiabilityReport,
    IssueSeverity,
    ModelHealthIssue,
    ModelHealthReport,
    ParameterSensitivityReport,
    PluginResult,
)


def create_health_report(
    identifiability: IdentifiabilityReport | None = None,
    gradient_health: GradientHealthReport | None = None,
    sloppy_model: ParameterSensitivityReport | None = None,
    plugin_results: dict[str, PluginResult] | None = None,
    config: DiagnosticsConfig | None = None,
) -> ModelHealthReport:
    """Create aggregated health report from component reports.

    This factory function aggregates results from all diagnostic components
    into a unified ModelHealthReport with:
    - Overall status determination (HEALTHY/WARNING/CRITICAL)
    - Health score computation with weighted contributions
    - Issue aggregation sorted by severity and code
    - Optional warnings emission for critical issues
    - Optional verbose console output

    Parameters
    ----------
    identifiability : IdentifiabilityReport, optional
        Identifiability analysis results.
    gradient_health : GradientHealthReport, optional
        Gradient health monitoring results.
    sloppy_model : ParameterSensitivityReport, optional
        Sloppy model analysis results.
    plugin_results : dict[str, PluginResult], optional
        Results from diagnostic plugins, keyed by plugin name.
    config : DiagnosticsConfig, optional
        Configuration for formatting and warnings.

    Returns
    -------
    ModelHealthReport
        Complete aggregated report with status, score, and all issues.

    Examples
    --------
    >>> from nlsq.diagnostics.health_report import create_health_report
    >>> report = create_health_report(
    ...     identifiability=ident_report,
    ...     gradient_health=grad_report,
    ... )
    >>> print(report.status)
    HealthStatus.HEALTHY
    >>> print(report.health_score)
    0.95
    """
    start_time = time.perf_counter()

    # Use default config if not provided
    if config is None:
        config = DiagnosticsConfig()

    # Initialize plugin results dict if None
    if plugin_results is None:
        plugin_results = {}

    # Collect all issues from all components
    all_issues = _collect_issues(
        identifiability=identifiability,
        gradient_health=gradient_health,
        sloppy_model=sloppy_model,
        plugin_results=plugin_results,
    )

    # Sort issues by severity (CRITICAL > WARNING > INFO) then by code
    all_issues = _sort_issues(all_issues)

    # Determine overall status from issues
    status = _determine_status(all_issues, identifiability, gradient_health)

    # Compute health score
    health_score = _compute_health_score(
        identifiability=identifiability,
        gradient_health=gradient_health,
        sloppy_model=sloppy_model,
        config=config,
    )

    # Compute total computation time from components
    computation_time_ms = _compute_total_time(
        identifiability=identifiability,
        gradient_health=gradient_health,
        sloppy_model=sloppy_model,
        plugin_results=plugin_results,
    )

    # Add time for this function
    computation_time_ms += (time.perf_counter() - start_time) * 1000

    # Create the report
    report = ModelHealthReport(
        identifiability=identifiability,
        gradient_health=gradient_health,
        sloppy_model=sloppy_model,
        plugin_results=plugin_results,
        status=status,
        health_score=health_score,
        all_issues=all_issues,
        config=config,
        computation_time_ms=computation_time_ms,
    )

    # Emit warnings for critical issues if configured (FR-019)
    if config.emit_warnings:
        _emit_warnings(all_issues)

    # Print verbose output if configured (FR-020)
    if config.verbose:
        print(report.summary(verbose=True))

    return report


def _collect_issues(
    identifiability: IdentifiabilityReport | None,
    gradient_health: GradientHealthReport | None,
    sloppy_model: ParameterSensitivityReport | None,
    plugin_results: dict[str, PluginResult],
) -> list[ModelHealthIssue]:
    """Collect all issues from all component reports.

    Parameters
    ----------
    identifiability : IdentifiabilityReport | None
        Identifiability analysis results.
    gradient_health : GradientHealthReport | None
        Gradient health monitoring results.
    sloppy_model : ParameterSensitivityReport | None
        Sloppy model analysis results.
    plugin_results : dict[str, PluginResult]
        Results from diagnostic plugins.

    Returns
    -------
    list[ModelHealthIssue]
        All collected issues (unsorted).
    """
    issues: list[ModelHealthIssue] = []

    # Collect from identifiability
    if identifiability is not None and identifiability.available:
        issues.extend(identifiability.issues)

    # Collect from gradient health
    if gradient_health is not None and gradient_health.available:
        issues.extend(gradient_health.issues)

    # Collect from sloppy model
    if sloppy_model is not None and sloppy_model.available:
        issues.extend(sloppy_model.issues)

    # Collect from plugins
    for plugin_result in plugin_results.values():
        if plugin_result.available:
            issues.extend(plugin_result.issues)

    return issues


def _sort_issues(issues: list[ModelHealthIssue]) -> list[ModelHealthIssue]:
    """Sort issues by severity (CRITICAL > WARNING > INFO) then by code.

    Parameters
    ----------
    issues : list[ModelHealthIssue]
        Unsorted list of issues.

    Returns
    -------
    list[ModelHealthIssue]
        Sorted list with CRITICAL first, then WARNING, then INFO.
    """
    # Define severity order by name (lower value = higher priority)
    # Using names as keys avoids enum identity issues across import paths
    severity_order = {
        "CRITICAL": 0,
        "WARNING": 1,
        "INFO": 2,
    }

    return sorted(
        issues, key=lambda i: (severity_order.get(i.severity.name, 3), i.code)
    )


def _determine_status(
    all_issues: list[ModelHealthIssue],
    identifiability: IdentifiabilityReport | None,
    gradient_health: GradientHealthReport | None,
) -> HealthStatus:
    """Determine overall health status from issues.

    Logic (Contract B1):
    - If any CRITICAL severity issue exists: CRITICAL
    - Else if any WARNING severity issue exists: WARNING
    - Else if all components unavailable or None: WARNING
    - Else: HEALTHY

    Parameters
    ----------
    all_issues : list[ModelHealthIssue]
        All aggregated issues.
    identifiability : IdentifiabilityReport | None
        Identifiability report for checking availability.
    gradient_health : GradientHealthReport | None
        Gradient health report for checking availability.

    Returns
    -------
    HealthStatus
        Overall status.
    """
    # Check for critical issues (use .name to avoid enum identity issues)
    if any(issue.severity.name == "CRITICAL" for issue in all_issues):
        return HealthStatus.CRITICAL

    # Check for warning issues (use .name to avoid enum identity issues)
    if any(issue.severity.name == "WARNING" for issue in all_issues):
        return HealthStatus.WARNING

    # Check if all components are unavailable or None (per contract error handling)
    ident_unavailable = identifiability is None or not identifiability.available
    grad_unavailable = gradient_health is None or not gradient_health.available

    if ident_unavailable and grad_unavailable:
        return HealthStatus.WARNING

    return HealthStatus.HEALTHY


def _compute_health_score(
    identifiability: IdentifiabilityReport | None,
    gradient_health: GradientHealthReport | None,
    sloppy_model: ParameterSensitivityReport | None,
    config: DiagnosticsConfig,
) -> float:
    """Compute overall health score from component scores.

    Health score computation (Contract B2):
    - Start with base score = 1.0
    - Identifiability contribution:
      - -0.4 if structurally unidentifiable
      - -0.2 if practically unidentifiable
      - -0.05 per highly correlated pair (max -0.2)
    - Gradient health contribution:
      - Use gradient_health.health_score * 0.3 weight
    - Sloppy model contribution (if level=FULL):
      - -0.1 if is_sloppy
    - Clamp to [0.0, 1.0]

    Parameters
    ----------
    identifiability : IdentifiabilityReport | None
        Identifiability analysis results.
    gradient_health : GradientHealthReport | None
        Gradient health monitoring results.
    sloppy_model : ParameterSensitivityReport | None
        Sloppy model analysis results.
    config : DiagnosticsConfig
        Configuration for thresholds.

    Returns
    -------
    float
        Health score in [0.0, 1.0].
    """
    score = 1.0

    # Identifiability contribution
    if identifiability is not None and identifiability.available:
        # Check for structural unidentifiability (rank deficient)
        is_structurally_unidentifiable = (
            identifiability.numerical_rank < identifiability.n_params
        )
        # Check for practical unidentifiability (high condition number)
        is_practically_unidentifiable = (
            not is_structurally_unidentifiable
            and identifiability.condition_number > config.condition_threshold
        )

        if is_structurally_unidentifiable:
            score -= 0.4
        elif is_practically_unidentifiable:
            score -= 0.2

        # Deduction for correlated pairs: -0.05 per pair, max -0.2
        n_correlated_pairs = len(identifiability.highly_correlated_pairs)
        correlation_deduction = 0.05 * min(n_correlated_pairs, 4)
        score -= correlation_deduction

    # Gradient health contribution (0.3 weight)
    if gradient_health is not None and gradient_health.available:
        # The gradient health contributes 0.3 to the total score
        # If gradient is perfect (1.0), no deduction
        # If gradient is poor (0.0), deduct 0.3
        gradient_contribution = gradient_health.health_score * 0.3
        # We have a 0.7 weight for base + identifiability, 0.3 for gradient
        # Recalculate: score = score * 0.7 + gradient_health.health_score * 0.3
        score = score * 0.7 + gradient_contribution

    # Sloppy model contribution (level=FULL only)
    if (
        sloppy_model is not None
        and sloppy_model.available
        and config.level == DiagnosticLevel.FULL
    ):
        if sloppy_model.is_sloppy:
            score -= 0.1

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, score))


def _compute_total_time(
    identifiability: IdentifiabilityReport | None,
    gradient_health: GradientHealthReport | None,
    sloppy_model: ParameterSensitivityReport | None,
    plugin_results: dict[str, PluginResult],
) -> float:
    """Compute total computation time from all components.

    Parameters
    ----------
    identifiability : IdentifiabilityReport | None
        Identifiability analysis results.
    gradient_health : GradientHealthReport | None
        Gradient health monitoring results.
    sloppy_model : ParameterSensitivityReport | None
        Sloppy model analysis results.
    plugin_results : dict[str, PluginResult]
        Results from diagnostic plugins.

    Returns
    -------
    float
        Total computation time in milliseconds.
    """
    total = 0.0

    if identifiability is not None:
        total += identifiability.computation_time_ms

    if gradient_health is not None:
        total += gradient_health.computation_time_ms

    if sloppy_model is not None:
        total += sloppy_model.computation_time_ms

    for plugin_result in plugin_results.values():
        total += plugin_result.computation_time_ms

    return total


def _emit_warnings(issues: list[ModelHealthIssue]) -> None:
    """Emit Python warnings for critical issues (FR-019).

    Parameters
    ----------
    issues : list[ModelHealthIssue]
        All aggregated issues.
    """
    for issue in issues:
        if issue.severity == IssueSeverity.CRITICAL:
            warnings.warn(
                f"[{issue.code}] {issue.message}",
                UserWarning,
                stacklevel=4,  # Point to user's code calling create_health_report
            )


def _format_summary(report: ModelHealthReport, verbose: bool = True) -> str:
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


def _to_dict(report: ModelHealthReport) -> dict[str, Any]:
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
