# nlsq/precision/__init__.py
"""Precision and parameter handling modules.

This subpackage contains numerical precision utilities:
- mixed_precision: Mixed precision optimization
- parameter_normalizer: Parameter normalization
- parameter_estimation: Parameter estimation utilities
- bound_inference: Automatic bounds inference
- algorithm_selector: Automatic algorithm selection
"""

from nlsq.precision.algorithm_selector import AlgorithmSelector, auto_select_algorithm
from nlsq.precision.bound_inference import BoundsInference, infer_bounds, merge_bounds
from nlsq.precision.parameter_normalizer import ParameterNormalizer

__all__ = [
    "AlgorithmSelector",
    "BoundsInference",
    "ParameterNormalizer",
    "auto_select_algorithm",
    "infer_bounds",
    "merge_bounds",
]
