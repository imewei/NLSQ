"""Data entities for CurveFit orchestration components.

This module re-exports the dataclasses used by the orchestration components.
The actual definitions are in nlsq/interfaces/orchestration_protocol.py
to keep protocols and their associated data types together.
"""

from __future__ import annotations

from nlsq.interfaces.orchestration_protocol import (
    CovarianceResult,
    OptimizationConfig,
    PreprocessedData,
    StreamingDecision,
)

__all__ = [
    "CovarianceResult",
    "OptimizationConfig",
    "PreprocessedData",
    "StreamingDecision",
]
