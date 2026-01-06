"""Result types for NLSQ optimization.

This package provides the canonical location for optimization result types:

- OptimizeResult: Base container for optimization results with attribute access
- OptimizeResultV2: Memory-efficient frozen dataclass (v0.4.3+)
- OptimizeWarning: Warning class for non-critical optimization issues
- CurveFitResult: Enhanced result with statistical properties and visualization

Example:
    >>> from nlsq.result import OptimizeResult, OptimizeResultV2
    >>> result = OptimizeResult(x=[1.0, 2.0], success=True)
    >>> result.x
    [1.0, 2.0]
"""

from nlsq.result.curve_fit_result import CurveFitResult
from nlsq.result.optimize_result import (
    OptimizeResult,
    OptimizeResultLegacy,
    OptimizeResultV2,
)
from nlsq.result.optimize_warning import OptimizeWarning, _check_unknown_options

__all__ = [
    "CurveFitResult",
    "OptimizeResult",
    "OptimizeResultLegacy",
    "OptimizeResultV2",
    "OptimizeWarning",
    "_check_unknown_options",
]
