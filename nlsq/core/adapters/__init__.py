"""Protocol adapters for core optimization algorithms.

This package provides adapter classes that wrap core optimization
functionality behind protocol interfaces, enabling loose coupling
and dependency injection.

The adapters implement protocols defined in nlsq.interfaces and
provide a clean abstraction layer between high-level curve fitting
APIs and low-level optimization algorithms.
"""

from nlsq.core.adapters.curve_fit_adapter import CurveFitAdapter

__all__ = ["CurveFitAdapter"]
