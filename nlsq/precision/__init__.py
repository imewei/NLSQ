# nlsq/precision/__init__.py
"""Precision and parameter handling modules.

This subpackage contains numerical precision utilities:
- parameter_normalizer: Parameter normalization
- parameter_estimation: Parameter estimation utilities
- bound_inference: Automatic bounds inference
- algorithm_selector: Automatic algorithm selection
"""

from nlsq.precision.algorithm_selector import AlgorithmSelector, auto_select_algorithm
from nlsq.precision.bound_inference import (
    BoundsInference,
    infer_bounds,
    infer_bounds_for_multistart,
    merge_bounds,
)
from nlsq.precision.parameter_estimation import estimate_initial_parameters
from nlsq.precision.parameter_normalizer import (
    NormalizedModelWrapper,
    ParameterNormalizer,
)

__all__ = [
    "AlgorithmSelector",
    "BoundsInference",
    "NormalizedModelWrapper",
    "ParameterNormalizer",
    "auto_select_algorithm",
    "estimate_initial_parameters",
    "infer_bounds",
    "infer_bounds_for_multistart",
    "merge_bounds",
]
