"""
NLSQ: Nonlinear Least Squares Curve Fitting for GPU/TPU

A JAX-based implementation of curve fitting algorithms with automatic
differentiation and GPU/TPU acceleration.
"""

# Version information
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

# Main API imports
from ._optimize import OptimizeResult, OptimizeWarning
from .least_squares import LeastSquares
from .minpack import CurveFit, curve_fit

# Public API - only expose main user-facing functions
__all__ = [
    "CurveFit",
    # Advanced API
    "LeastSquares",
    # Result types
    "OptimizeResult",
    "OptimizeWarning",
    # Version
    "__version__",
    # Main curve fitting API
    "curve_fit",
]

# Optional: Provide convenience access to submodules for advanced users
# Users can still access internal functions via:
# from nlsq.loss_functions import LossFunctionsJIT
# from nlsq.trf import TrustRegionReflective
# etc.
