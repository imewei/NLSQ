# nlsq/stability/__init__.py
"""Numerical stability and fallback modules.

This subpackage contains numerical stability utilities:
- guard: NumericalStabilityGuard for detecting numerical issues
- svd_fallback: SVD fallback with GPU/CPU switching
- recovery: OptimizationRecovery for recovering from failures
- fallback: FallbackOrchestrator for fallback strategies
"""

from nlsq.stability.fallback import (
    FallbackOrchestrator,
    FallbackResult,
    FallbackStrategy,
)
from nlsq.stability.guard import (
    NumericalStabilityGuard,
    apply_automatic_fixes,
    check_problem_stability,
    detect_collinearity,
    detect_parameter_scale_mismatch,
    estimate_condition_number,
)
from nlsq.stability.recovery import OptimizationRecovery

__all__ = [
    "FallbackOrchestrator",
    "FallbackResult",
    "FallbackStrategy",
    "NumericalStabilityGuard",
    "OptimizationRecovery",
    "apply_automatic_fixes",
    "check_problem_stability",
    "detect_collinearity",
    "detect_parameter_scale_mismatch",
    "estimate_condition_number",
]
