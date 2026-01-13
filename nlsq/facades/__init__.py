"""Facades for breaking circular dependencies in NLSQ.

This package provides lazy-loading facades that break circular import cycles:

- OptimizationFacade: Breaks global_optimization <-> minpack cycle
- StabilityFacade: Breaks stability.fallback <-> minpack cycle
- DiagnosticsFacade: Breaks diagnostics.types <-> health_report cycle

All facades use function-level lazy imports to defer module loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlsq.facades.diagnostics_facade import DiagnosticsFacade
    from nlsq.facades.optimization_facade import OptimizationFacade
    from nlsq.facades.stability_facade import StabilityFacade

__all__ = [
    "DiagnosticsFacade",
    "OptimizationFacade",
    "StabilityFacade",
]


def __getattr__(name: str):
    """Lazy import for facade components."""
    if name == "OptimizationFacade":
        from nlsq.facades.optimization_facade import OptimizationFacade

        return OptimizationFacade
    if name == "StabilityFacade":
        from nlsq.facades.stability_facade import StabilityFacade

        return StabilityFacade
    if name == "DiagnosticsFacade":
        from nlsq.facades.diagnostics_facade import DiagnosticsFacade

        return DiagnosticsFacade
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
