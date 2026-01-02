"""Phase-based streaming optimization modules.

This subpackage contains the phase implementations for the
AdaptiveHybridStreamingOptimizer, organized by optimization phase:

- Phase 0: Setup and normalization
- Phase 1: L-BFGS warmup (WarmupPhase)
- Phase 2: Gauss-Newton streaming optimization (GaussNewtonPhase)
- Phase 3: Finalization and denormalization

The orchestrator coordinates phase transitions based on convergence criteria.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlsq.streaming.phases.checkpoint import CheckpointManager, CheckpointState
    from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase, GNResult
    from nlsq.streaming.phases.orchestrator import (
        PhaseOrchestrator,
        PhaseOrchestratorResult,
    )
    from nlsq.streaming.phases.warmup import WarmupPhase, WarmupResult

# Lazy import mapping for deferred module loading
# This preserves import time by only loading modules when accessed
_LAZY_IMPORTS: dict[str, str] = {
    "WarmupPhase": "nlsq.streaming.phases.warmup",
    "WarmupResult": "nlsq.streaming.phases.warmup",
    "GaussNewtonPhase": "nlsq.streaming.phases.gauss_newton",
    "GNResult": "nlsq.streaming.phases.gauss_newton",
    "CheckpointManager": "nlsq.streaming.phases.checkpoint",
    "CheckpointState": "nlsq.streaming.phases.checkpoint",
    "PhaseOrchestrator": "nlsq.streaming.phases.orchestrator",
    "PhaseOrchestratorResult": "nlsq.streaming.phases.orchestrator",
}


def __getattr__(name: str):
    """Lazy import handler for phase modules.

    This defers importing phase modules until they are actually accessed,
    preserving the ~620ms import time target.
    """
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available attributes including lazy imports."""
    return list(_LAZY_IMPORTS.keys()) + list(globals().keys())


__all__ = [
    "CheckpointManager",
    "CheckpointState",
    "GNResult",
    "GaussNewtonPhase",
    "PhaseOrchestrator",
    "PhaseOrchestratorResult",
    "WarmupPhase",
    "WarmupResult",
]
