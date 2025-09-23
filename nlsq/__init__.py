# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:26:59 2022

@author: hofer
"""

# Explicit imports from minpack module
from .minpack import CurveFit, curve_fit

# Explicit imports from least_squares module
from .least_squares import LeastSquares, prepare_bounds

# Explicit imports from loss_functions module
from .loss_functions import LossFunctionsJIT

# Explicit imports from trf module
from .trf import TrustRegionReflective

# Explicit imports from common_jax module
from .common_jax import CommonJIT

# Explicit imports from common_scipy module
from .common_scipy import (
    EPS,
    in_bounds,
    make_strictly_feasible,
    update_tr_radius,
    solve_lsq_trust_region,
    check_termination,
    CL_scaling_vector,
    find_active_constraints,
    step_size_to_bound,
    intersect_trust_region,
    minimize_quadratic_1d,
    print_header_nonlinear,
    print_iteration_nonlinear
)

# Explicit imports from _optimize module
from ._optimize import OptimizeResult, OptimizeWarning

# Explicitly define exported names
__all__ = [
    # Main API
    'CurveFit',
    'curve_fit',
    # Least squares
    'LeastSquares',
    'prepare_bounds',
    # Loss functions
    'LossFunctionsJIT',
    # Trust region
    'TrustRegionReflective',
    # Common JAX
    'CommonJIT',
    # Common SciPy utilities
    'EPS',
    'in_bounds',
    'make_strictly_feasible',
    'update_tr_radius',
    'solve_lsq_trust_region',
    'check_termination',
    'CL_scaling_vector',
    'find_active_constraints',
    'step_size_to_bound',
    'intersect_trust_region',
    'minimize_quadratic_1d',
    'print_header_nonlinear',
    'print_iteration_nonlinear',
    # Optimize results
    'OptimizeResult',
    'OptimizeWarning'
]