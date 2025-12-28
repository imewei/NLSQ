# nlsq/stability/__init__.py
"""Numerical stability and fallback modules.

This subpackage contains numerical stability utilities:
- guard: NumericalStabilityGuard for detecting numerical issues
- svd_fallback: SVD fallback with GPU/CPU switching
- robust_decomposition: Robust matrix decomposition
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
from nlsq.stability.robust_decomposition import RobustDecomposition, robust_decomp

__all__ = [
    "FallbackOrchestrator",
    "FallbackResult",
    "FallbackStrategy",
    "NumericalStabilityGuard",
    "OptimizationRecovery",
    "RobustDecomposition",
    "apply_automatic_fixes",
    "check_problem_stability",
    "detect_collinearity",
    "detect_parameter_scale_mismatch",
    "estimate_condition_number",
    "robust_decomp",
]
