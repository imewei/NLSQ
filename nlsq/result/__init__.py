"""Result types for NLSQ optimization.

This package provides the canonical location for optimization result types:

- OptimizeResult: Base container for optimization results with attribute access
- OptimizeWarning: Warning class for non-critical optimization issues
- CurveFitResult: Enhanced result with statistical properties and visualization

For backward compatibility, OptimizeResult and OptimizeWarning are also
re-exported from nlsq.core._optimize (deprecated, will be removed in v0.6.0).

Example:
    >>> from nlsq.result import OptimizeResult, OptimizeWarning, CurveFitResult
    >>> result = OptimizeResult(x=[1.0, 2.0], success=True)
    >>> result.x
    [1.0, 2.0]
"""

from nlsq.result.curve_fit_result import CurveFitResult
from nlsq.result.optimize_result import OptimizeResult
from nlsq.result.optimize_warning import OptimizeWarning, _check_unknown_options

__all__ = [
    "CurveFitResult",
    "OptimizeResult",
    "OptimizeWarning",
    "_check_unknown_options",
]
