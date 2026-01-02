"""Model Health Diagnostics System for NLSQ.

This package provides comprehensive diagnostic capabilities for nonlinear
least squares curve fitting, including:

- Identifiability analysis (structural and practical)
- Gradient health monitoring during optimization
- Parameter sensitivity spectrum analysis
- Aggregated health reports with actionable recommendations
- Plugin system for domain-specific diagnostics

Basic Usage
-----------
>>> from nlsq import curve_fit
>>> result = curve_fit(model, x, y, compute_diagnostics=True)
>>> print(result.diagnostics.summary())

Plugin Usage
------------
>>> from nlsq.diagnostics import DiagnosticPlugin, PluginRegistry, PluginResult
>>> class MyPlugin:
...     @property
...     def name(self) -> str:
...         return "my-plugin"
...     def analyze(self, jacobian, parameters, residuals, **context):
...         return PluginResult(plugin_name=self.name, data={}, issues=[])
>>> PluginRegistry.register(MyPlugin())

Exports
-------
Types and Enumerations:
    HealthStatus : Overall health status (HEALTHY, WARNING, CRITICAL)
    IssueSeverity : Issue severity level (INFO, WARNING, CRITICAL)
    IssueCategory : Issue category (IDENTIFIABILITY, GRADIENT, etc.)
    DiagnosticLevel : Diagnostic depth (BASIC, FULL)
    ModelHealthIssue : Single detected issue with recommendation
    AnalysisResult : Base class for analysis results
    IdentifiabilityReport : Report from identifiability analysis
    GradientHealthReport : Report from gradient health monitoring
    ParameterSensitivityReport : Report from parameter sensitivity analysis
    PluginResult : Result from a diagnostic plugin
    ModelHealthReport : Aggregated health report with overall assessment
    DiagnosticsReport : Aggregated diagnostics report (legacy)
    DiagnosticsConfig : Configuration for diagnostic computation

Analyzers:
    IdentifiabilityAnalyzer : Analyzer for parameter identifiability
    GradientMonitor : Monitor for gradient health during optimization
    ParameterSensitivityAnalyzer : Analyzer for parameter sensitivity spectrum

Plugin System:
    DiagnosticPlugin : Protocol for custom diagnostic plugins
    PluginRegistry : Global registry for plugin registration
    run_plugins : Execute all registered plugins

Factory Functions:
    create_health_report : Create aggregated health report from components

Recommendations:
    RECOMMENDATIONS : Mapping of issue codes to recommendation text
    get_recommendation : Get recommendation text for an issue code

Deprecated Names (will be removed in v0.6.0):
    SloppyModelAnalyzer : Use ParameterSensitivityAnalyzer instead
    SloppyModelReport : Use ParameterSensitivityReport instead
"""

import warnings

from nlsq.diagnostics.gradient_health import GradientMonitor
from nlsq.diagnostics.health_report import create_health_report
from nlsq.diagnostics.identifiability import IdentifiabilityAnalyzer
from nlsq.diagnostics.parameter_sensitivity import ParameterSensitivityAnalyzer
from nlsq.diagnostics.plugin import DiagnosticPlugin, PluginRegistry, run_plugins
from nlsq.diagnostics.recommendations import RECOMMENDATIONS, get_recommendation
from nlsq.diagnostics.types import (
    AnalysisResult,
    DiagnosticLevel,
    DiagnosticsConfig,
    DiagnosticsReport,
    GradientHealthReport,
    HealthStatus,
    IdentifiabilityReport,
    IssueCategory,
    IssueSeverity,
    ModelHealthIssue,
    ModelHealthReport,
    ParameterSensitivityReport,
    PluginResult,
)

# Mapping of deprecated names to their new names and the actual objects
_DEPRECATED_NAME_ALIASES = {
    "SloppyModelAnalyzer": (
        "ParameterSensitivityAnalyzer",
        ParameterSensitivityAnalyzer,
    ),
    "SloppyModelReport": (
        "ParameterSensitivityReport",
        ParameterSensitivityReport,
    ),
}


def __getattr__(name: str):
    """Provide deprecation warnings for renamed classes.

    This function is called when an attribute is not found in the module.
    It provides backwards compatibility for deprecated names while emitting
    deprecation warnings.

    Parameters
    ----------
    name : str
        The name of the attribute being accessed.

    Returns
    -------
    object
        The object corresponding to the deprecated name.

    Raises
    ------
    AttributeError
        If the name is not a known deprecated alias.
    """
    if name in _DEPRECATED_NAME_ALIASES:
        new_name, obj = _DEPRECATED_NAME_ALIASES[name]
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead. "
            "Will be removed in v0.6.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RECOMMENDATIONS",
    "AnalysisResult",
    "DiagnosticLevel",
    "DiagnosticPlugin",
    "DiagnosticsConfig",
    "DiagnosticsReport",
    "GradientHealthReport",
    "GradientMonitor",
    "HealthStatus",
    "IdentifiabilityAnalyzer",
    "IdentifiabilityReport",
    "IssueCategory",
    "IssueSeverity",
    "ModelHealthIssue",
    "ModelHealthReport",
    "ParameterSensitivityAnalyzer",
    "ParameterSensitivityReport",
    "PluginRegistry",
    "PluginResult",
    # Deprecated names - kept for backwards compatibility
    "SloppyModelAnalyzer",
    "SloppyModelReport",
    "create_health_report",
    "get_recommendation",
    "run_plugins",
]
