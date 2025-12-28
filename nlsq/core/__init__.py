# nlsq/core/__init__.py
"""Core optimization pipeline modules.

This subpackage contains the core optimization algorithms:
- minpack: SciPy-compatible curve_fit API
- least_squares: LeastSquares class for optimization orchestration
- trf: Trust Region Reflective algorithm
- _optimize: OptimizeResult and OptimizeWarning
- optimizer_base: Base classes for optimizers
- sparse_jacobian: Sparse Jacobian computation
- functions: Model function utilities
- loss_functions: Loss function implementations
- workflow: Workflow system for automatic optimization selection
"""

from nlsq.core import functions
from nlsq.core._optimize import OptimizeResult, OptimizeWarning
from nlsq.core.least_squares import LeastSquares
from nlsq.core.minpack import CurveFit, curve_fit
from nlsq.core.sparse_jacobian import SparseJacobianComputer, SparseOptimizer
from nlsq.core.trf import TrustRegionReflective

__all__ = [
    "CurveFit",
    "LeastSquares",
    "OptimizeResult",
    "OptimizeWarning",
    "SparseJacobianComputer",
    "SparseOptimizer",
    "TrustRegionReflective",
    "curve_fit",
    "functions",
]
