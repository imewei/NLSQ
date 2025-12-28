# nlsq/utils/__init__.py
"""Utility modules for diagnostics, profiling, logging, and validation.

This subpackage contains utility modules:
- diagnostics: Convergence monitoring and optimization diagnostics
- profiler: Performance profiling
- profiler_visualization: Profiling visualizations
- profiling: Profiling utilities
- logging: Logging configuration
- async_logger: Asynchronous logging
- validators: Input validation
- error_messages: Error message formatting
"""

from nlsq.utils.diagnostics import ConvergenceMonitor, OptimizationDiagnostics
from nlsq.utils.profiler import (
    PerformanceProfiler,
    ProfileMetrics,
    clear_profiling_data,
    get_global_profiler,
)
from nlsq.utils.validators import InputValidator

__all__ = [
    "ConvergenceMonitor",
    "InputValidator",
    "OptimizationDiagnostics",
    "PerformanceProfiler",
    "ProfileMetrics",
    "clear_profiling_data",
    "get_global_profiler",
]
