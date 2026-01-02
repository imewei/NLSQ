"""Phase-based streaming optimization modules.

This subpackage contains the phase implementations for the
AdaptiveHybridStreamingOptimizer, organized by optimization phase:

- Phase 0: Setup and normalization
- Phase 1: L-BFGS warmup
- Phase 2: Gauss-Newton streaming optimization
- Phase 3: Finalization and denormalization

The orchestrator coordinates phase transitions based on convergence criteria.
"""

from __future__ import annotations

# Phase modules will be added as they are extracted
# from adaptive_hybrid.py

__all__: list[str] = []
