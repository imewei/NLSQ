"""Phase orchestrator for streaming optimization workflow.

This module provides the PhaseOrchestrator class that coordinates
the multi-phase optimization workflow (setup, warmup, GN, finalize).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from nlsq.utils.logging import get_logger

if TYPE_CHECKING:
    from jax import Array

    from nlsq.precision.parameter_normalizer import NormalizedModelWrapper
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase, GNResult
    from nlsq.streaming.phases.warmup import WarmupPhase, WarmupResult

_logger = get_logger("phase_orchestrator")


@dataclass(frozen=True, slots=True)
class PhaseOrchestratorResult:
    """Complete result from phase orchestration.

    Attributes:
        params: Final optimized parameters in original space.
        normalized_params: Final parameters in normalized space.
        cost: Final cost value.
        warmup_result: Result from Phase 1 warmup.
        gn_result: Result from Phase 2 Gauss-Newton.
        phase_history: List of phase transition records.
        total_time: Total optimization time.
    """

    params: Array
    normalized_params: Array
    cost: float
    warmup_result: WarmupResult | None
    gn_result: GNResult | None
    phase_history: list[dict[str, Any]]
    total_time: float


class PhaseOrchestrator:
    """Orchestrates the multi-phase streaming optimization workflow.

    The orchestrator coordinates:
    - Phase 0: Setup (normalization, validation)
    - Phase 1: L-BFGS warmup (WarmupPhase)
    - Phase 2: Streaming Gauss-Newton (GaussNewtonPhase)
    - Phase 3: Finalization (denormalization, covariance)

    Parameters
    ----------
    config : HybridStreamingConfig
        Configuration for streaming optimization.

    Attributes
    ----------
    config : HybridStreamingConfig
        Configuration object.
    warmup_phase : WarmupPhase or None
        The warmup phase handler (lazy initialized).
    gn_phase : GaussNewtonPhase or None
        The Gauss-Newton phase handler (lazy initialized).
    phase_history : list
        Records of phase transitions.
    """

    def __init__(self, config: HybridStreamingConfig) -> None:
        """Initialize PhaseOrchestrator.

        Parameters
        ----------
        config : HybridStreamingConfig
            Configuration for streaming optimization.
        """
        self.config = config
        self.warmup_phase: WarmupPhase | None = None
        self.gn_phase: GaussNewtonPhase | None = None
        self.phase_history: list[dict[str, Any]] = []

        # Best parameter tracking (shared across phases)
        self._best_tracker: dict[str, Any] = {
            "best_params_global": None,
            "best_cost_global": float("inf"),
        }

        # State
        self.current_phase: int = 0
        self._start_time: float | None = None

    def initialize_phases(
        self,
        normalized_model: NormalizedModelWrapper,
        normalized_bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> None:
        """Initialize phase handlers.

        Parameters
        ----------
        normalized_model : NormalizedModelWrapper
            Model wrapper operating in normalized parameter space.
        normalized_bounds : tuple of array_like or None
            Parameter bounds in normalized space.
        """
        from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase
        from nlsq.streaming.phases.warmup import WarmupPhase

        self.warmup_phase = WarmupPhase(self.config, normalized_model)
        self.gn_phase = GaussNewtonPhase(
            self.config, normalized_model, normalized_bounds
        )

    def run(
        self,
        data_source: tuple[jnp.ndarray, jnp.ndarray],
        initial_params: jnp.ndarray,
        normalizer: Any | None = None,
    ) -> dict[str, Any]:
        """Run the full multi-phase optimization workflow.

        Parameters
        ----------
        data_source : tuple of array_like
            Data as (x_data, y_data).
        initial_params : array_like
            Initial parameters in normalized space.
        normalizer : ParameterNormalizer or None
            For denormalization in Phase 3.

        Returns
        -------
        result : dict
            Complete optimization result with keys:
            - 'final_params': Final parameters in original space
            - 'normalized_params': Final parameters in normalized space
            - 'best_cost': Best cost achieved
            - 'warmup_result': WarmupResult from Phase 1
            - 'gn_result': GNResult from Phase 2
            - 'phase_history': List of phase records
            - 'JTJ_final': Final J^T J matrix
            - 'residual_sum_sq': Final residual sum of squares
        """
        self._start_time = time.time()
        self.phase_history = []
        self._best_tracker = {
            "best_params_global": initial_params,
            "best_cost_global": float("inf"),
        }

        verbose = getattr(self.config, "verbose", 1)

        # =====================================================
        # PHASE 0: Setup
        # =====================================================
        self.current_phase = 0
        if verbose >= 1:
            _logger.info("Phase 0: Setup and validation")

        x_data, y_data = data_source
        x_data = jnp.asarray(x_data, dtype=jnp.float64)
        y_data = jnp.asarray(y_data, dtype=jnp.float64)

        current_params = initial_params

        # =====================================================
        # PHASE 1: L-BFGS Warmup
        # =====================================================
        self.current_phase = 1
        if verbose >= 1:
            _logger.info("Phase 1: L-BFGS warmup")

        warmup_result = None
        if self.warmup_phase is not None:
            phase1_result = self.warmup_phase.run(
                data_source=(x_data, y_data),
                initial_params=current_params,
                phase_history=self.phase_history,
                best_tracker=self._best_tracker,
            )
            current_params = phase1_result["best_params"]
            warmup_result = phase1_result.get("warmup_result")

            if verbose >= 1:
                _logger.info(
                    f"Phase 1 complete: {phase1_result['iterations']} iterations, "
                    f"loss={phase1_result['best_loss']:.6e}, "
                    f"reason: {phase1_result['switch_reason']}"
                )
        else:
            # Skip warmup if no phase handler
            self.phase_history.append(
                {
                    "phase": 1,
                    "name": "lbfgs_warmup",
                    "iterations": 0,
                    "final_loss": float("inf"),
                    "best_loss": float("inf"),
                    "switch_reason": "Warmup phase not initialized",
                    "timestamp": time.time(),
                    "skipped": True,
                }
            )

        # =====================================================
        # PHASE 2: Streaming Gauss-Newton
        # =====================================================
        self.current_phase = 2
        if verbose >= 1:
            _logger.info("Phase 2: Streaming Gauss-Newton")

        gn_result = None
        JTJ_final = None
        residual_sum_sq = 0.0

        if self.gn_phase is not None:
            phase2_result = self.gn_phase.run(
                data_source=(x_data, y_data),
                initial_params=current_params,
                phase_history=self.phase_history,
                best_tracker=self._best_tracker,
            )
            current_params = phase2_result["best_params"]
            gn_result = phase2_result.get("gn_result")
            JTJ_final = phase2_result.get("JTJ_final")
            residual_sum_sq = phase2_result.get("residual_sum_sq", 0.0)

            if verbose >= 1:
                _logger.info(
                    f"Phase 2 complete: {phase2_result['iterations']} iterations, "
                    f"cost={phase2_result['best_cost']:.6e}, "
                    f"reason: {phase2_result['convergence_reason']}"
                )
        else:
            # Skip GN if no phase handler
            self.phase_history.append(
                {
                    "phase": 2,
                    "name": "gauss_newton",
                    "iterations": 0,
                    "final_cost": float("inf"),
                    "best_cost": float("inf"),
                    "convergence_reason": "GN phase not initialized",
                    "timestamp": time.time(),
                    "skipped": True,
                }
            )

        # =====================================================
        # PHASE 3: Finalization
        # =====================================================
        self.current_phase = 3
        if verbose >= 1:
            _logger.info("Phase 3: Finalization")

        # Use best parameters found globally
        final_normalized_params = self._best_tracker["best_params_global"]
        if final_normalized_params is None:
            final_normalized_params = current_params

        # Denormalize parameters if normalizer available
        if normalizer is not None and hasattr(normalizer, "denormalize"):
            final_params = normalizer.denormalize(final_normalized_params)
        else:
            final_params = final_normalized_params

        total_time = time.time() - self._start_time

        # Record Phase 3
        self.phase_history.append(
            {
                "phase": 3,
                "name": "finalization",
                "final_cost": self._best_tracker["best_cost_global"],
                "timestamp": time.time(),
                "total_time": total_time,
            }
        )

        if verbose >= 1:
            best_cost = self._best_tracker["best_cost_global"]
            _logger.info(
                f"Optimization complete: cost={best_cost:.6e}, "
                f"total_time={total_time:.1f}s"
            )

        return {
            "final_params": final_params,
            "normalized_params": final_normalized_params,
            "best_cost": self._best_tracker["best_cost_global"],
            "warmup_result": warmup_result,
            "gn_result": gn_result,
            "phase_history": self.phase_history,
            "JTJ_final": JTJ_final,
            "residual_sum_sq": residual_sum_sq,
            "total_time": total_time,
        }

    def get_phase_history(self) -> list[dict[str, Any]]:
        """Get the phase transition history.

        Returns
        -------
        phase_history : list
            List of phase transition records.
        """
        return self.phase_history

    def get_best_params(self) -> jnp.ndarray | None:
        """Get the best parameters found across all phases.

        Returns
        -------
        best_params : array_like or None
            Best parameters in normalized space.
        """
        return self._best_tracker.get("best_params_global")

    def get_best_cost(self) -> float:
        """Get the best cost found across all phases.

        Returns
        -------
        best_cost : float
            Best cost value.
        """
        return self._best_tracker.get("best_cost_global", float("inf"))


__all__ = ["PhaseOrchestrator", "PhaseOrchestratorResult"]
