"""Orchestration components for CurveFit decomposition.

This package contains extracted components from the CurveFit God class,
each handling a single responsibility:

- DataPreprocessor: Input validation, array conversion, data padding
- OptimizationSelector: Method selection, bounds preparation, initial guess
- CovarianceComputer: Covariance via SVD, sigma transformation
- StreamingCoordinator: Memory analysis, streaming strategy selection

These components are used internally by CurveFit and controlled via feature flags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlsq.core.orchestration.covariance_computer import CovarianceComputer
    from nlsq.core.orchestration.data_preprocessor import DataPreprocessor
    from nlsq.core.orchestration.entities import (
        CovarianceResult,
        OptimizationConfig,
        PreprocessedData,
        StreamingDecision,
    )
    from nlsq.core.orchestration.optimization_selector import OptimizationSelector
    from nlsq.core.orchestration.streaming_coordinator import StreamingCoordinator

__all__ = [
    "CovarianceComputer",
    "CovarianceResult",
    "DataPreprocessor",
    "OptimizationConfig",
    "OptimizationSelector",
    "PreprocessedData",
    "StreamingCoordinator",
    "StreamingDecision",
]


def __getattr__(name: str):
    """Lazy import for orchestration components."""
    if name == "DataPreprocessor":
        from nlsq.core.orchestration.data_preprocessor import DataPreprocessor

        return DataPreprocessor
    if name == "OptimizationSelector":
        from nlsq.core.orchestration.optimization_selector import OptimizationSelector

        return OptimizationSelector
    if name == "CovarianceComputer":
        from nlsq.core.orchestration.covariance_computer import CovarianceComputer

        return CovarianceComputer
    if name == "StreamingCoordinator":
        from nlsq.core.orchestration.streaming_coordinator import StreamingCoordinator

        return StreamingCoordinator
    if name in (
        "PreprocessedData",
        "OptimizationConfig",
        "CovarianceResult",
        "StreamingDecision",
    ):
        from nlsq.core.orchestration import entities

        return getattr(entities, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
