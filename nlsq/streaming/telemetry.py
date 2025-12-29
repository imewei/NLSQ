"""Telemetry for monitoring defense layer activations.

This module provides telemetry tracking for the 4-layer defense strategy
used during L-BFGS warmup in the adaptive hybrid streaming optimizer.
"""

from __future__ import annotations

import time

__all__ = [
    "DefenseLayerTelemetry",
    "get_defense_telemetry",
    "reset_defense_telemetry",
]


class DefenseLayerTelemetry:
    """Telemetry for monitoring 4-layer defense strategy activations.

    Tracks when each defense layer is triggered during warmup to help
    with production monitoring and tuning. This class maintains thread-safe
    statistics that can be queried or exported for monitoring dashboards.

    The 4 layers tracked are:
        - Layer 1: Warm start detection (skips warmup)
        - Layer 2: Adaptive learning rate selection (refinement/careful/exploration)
        - Layer 3: Cost-increase guard (aborts warmup if loss increases)
        - Layer 4: Step clipping (limits update magnitude)

    Attributes
    ----------
    layer1_warm_start_triggers : int
        Count of warm start detection activations (warmup skipped)
    layer2_lr_mode_counts : dict[str, int]
        Counts per LR mode: {"refinement": n, "careful": m, "exploration": k}
    layer3_cost_guard_triggers : int
        Count of cost-increase guard aborts
    layer4_clip_triggers : int
        Count of step clipping activations
    total_warmup_calls : int
        Total number of warmup phase executions
    """

    def __init__(self) -> None:
        """Initialize telemetry with zeroed counters."""
        self.reset()

    def reset(self) -> None:
        """Reset all telemetry counters to zero."""
        self.layer1_warm_start_triggers: int = 0
        self.layer2_lr_mode_counts: dict[str, int] = {
            "refinement": 0,
            "careful": 0,
            "exploration": 0,
            "fixed": 0,
        }
        self.layer3_cost_guard_triggers: int = 0
        self.layer4_clip_triggers: int = 0
        self.total_warmup_calls: int = 0

        # L-BFGS-specific telemetry counters
        self.lbfgs_history_buffer_fill_events: int = 0
        self.lbfgs_line_search_failures: int = 0

        # Detailed event log (last N events)
        self._event_log: list[dict] = []
        self._max_events: int = 1000

    def record_warmup_start(self) -> None:
        """Record start of a warmup phase."""
        self.total_warmup_calls += 1

    def record_layer1_trigger(self, relative_loss: float, threshold: float) -> None:
        """Record Layer 1 warm start detection trigger.

        Parameters
        ----------
        relative_loss : float
            Relative loss that triggered warm start
        threshold : float
            Threshold value that was exceeded
        """
        self.layer1_warm_start_triggers += 1
        self._log_event(
            "layer1_warm_start",
            {"relative_loss": relative_loss, "threshold": threshold},
        )

    def record_layer2_lr_mode(self, mode: str, relative_loss: float) -> None:
        """Record Layer 2 adaptive LR mode selection.

        Parameters
        ----------
        mode : str
            Selected LR mode: "refinement", "careful", "exploration", or "fixed"
        relative_loss : float
            Relative loss that determined the mode
        """
        if mode in self.layer2_lr_mode_counts:
            self.layer2_lr_mode_counts[mode] += 1
        self._log_event(
            "layer2_lr_mode", {"mode": mode, "relative_loss": relative_loss}
        )

    def record_layer3_trigger(
        self, cost_ratio: float, tolerance: float, iteration: int
    ) -> None:
        """Record Layer 3 cost-increase guard trigger.

        Parameters
        ----------
        cost_ratio : float
            Cost increase ratio that triggered the guard
        tolerance : float
            Tolerance threshold that was exceeded
        iteration : int
            Iteration number when triggered
        """
        self.layer3_cost_guard_triggers += 1
        self._log_event(
            "layer3_cost_guard",
            {"cost_ratio": cost_ratio, "tolerance": tolerance, "iteration": iteration},
        )

    def record_layer4_clip(self, original_norm: float, max_norm: float) -> None:
        """Record Layer 4 step clipping activation.

        Parameters
        ----------
        original_norm : float
            Original update norm before clipping
        max_norm : float
            Maximum allowed norm (clipping threshold)
        """
        self.layer4_clip_triggers += 1
        self._log_event(
            "layer4_clip", {"original_norm": original_norm, "max_norm": max_norm}
        )

    def record_lbfgs_history_fill(self, iteration: int) -> None:
        """Record L-BFGS history buffer fill event.

        Called when the L-BFGS history buffer becomes fully populated,
        signaling transition from cold start to full L-BFGS mode.

        Parameters
        ----------
        iteration : int
            Iteration number when history buffer filled
        """
        self.lbfgs_history_buffer_fill_events += 1
        self._log_event(
            "lbfgs_history_fill",
            {"iteration": iteration},
        )

    def record_lbfgs_line_search_failure(
        self, iteration: int, reason: str = ""
    ) -> None:
        """Record L-BFGS line search failure event.

        Called when the L-BFGS line search fails to find an acceptable step.

        Parameters
        ----------
        iteration : int
            Iteration number when line search failed
        reason : str, optional
            Reason for line search failure
        """
        self.lbfgs_line_search_failures += 1
        self._log_event(
            "lbfgs_line_search_failure",
            {"iteration": iteration, "reason": reason},
        )

    def _log_event(self, event_type: str, data: dict) -> None:
        """Log an event with timestamp.

        Parameters
        ----------
        event_type : str
            Type of event
        data : dict
            Event data
        """
        event = {"type": event_type, "timestamp": time.time(), "data": data}
        self._event_log.append(event)

        # Trim if over limit
        if len(self._event_log) > self._max_events:
            self._event_log = self._event_log[-self._max_events :]

    def get_trigger_rates(self) -> dict[str, float]:
        """Get trigger rates as percentage of total warmup calls.

        Returns
        -------
        dict[str, float]
            Trigger rates for each layer as percentages (0-100)
        """
        if self.total_warmup_calls == 0:
            return {
                "layer1_warm_start_rate": 0.0,
                "layer2_refinement_rate": 0.0,
                "layer2_careful_rate": 0.0,
                "layer2_exploration_rate": 0.0,
                "layer3_cost_guard_rate": 0.0,
                "layer4_clip_rate": 0.0,
                "lbfgs_history_buffer_fill_rate": 0.0,
                "lbfgs_line_search_failure_rate": 0.0,
            }

        total = self.total_warmup_calls
        return {
            "layer1_warm_start_rate": 100.0 * self.layer1_warm_start_triggers / total,
            "layer2_refinement_rate": 100.0
            * self.layer2_lr_mode_counts["refinement"]
            / total,
            "layer2_careful_rate": 100.0
            * self.layer2_lr_mode_counts["careful"]
            / total,
            "layer2_exploration_rate": 100.0
            * self.layer2_lr_mode_counts["exploration"]
            / total,
            "layer3_cost_guard_rate": 100.0 * self.layer3_cost_guard_triggers / total,
            "layer4_clip_rate": 100.0 * self.layer4_clip_triggers / total,
            "lbfgs_history_buffer_fill_rate": 100.0
            * self.lbfgs_history_buffer_fill_events
            / total,
            "lbfgs_line_search_failure_rate": 100.0
            * self.lbfgs_line_search_failures
            / total,
        }

    def get_summary(self) -> dict:
        """Get summary statistics for all defense layers.

        Returns
        -------
        dict
            Summary with counts and rates for each layer
        """
        rates = self.get_trigger_rates()
        return {
            "total_warmup_calls": self.total_warmup_calls,
            "layer1": {
                "name": "warm_start_detection",
                "triggers": self.layer1_warm_start_triggers,
                "rate_pct": rates["layer1_warm_start_rate"],
            },
            "layer2": {
                "name": "adaptive_lr_selection",
                "mode_counts": self.layer2_lr_mode_counts.copy(),
                "rates_pct": {
                    "refinement": rates["layer2_refinement_rate"],
                    "careful": rates["layer2_careful_rate"],
                    "exploration": rates["layer2_exploration_rate"],
                },
            },
            "layer3": {
                "name": "cost_increase_guard",
                "triggers": self.layer3_cost_guard_triggers,
                "rate_pct": rates["layer3_cost_guard_rate"],
            },
            "layer4": {
                "name": "step_clipping",
                "triggers": self.layer4_clip_triggers,
                "rate_pct": rates["layer4_clip_rate"],
            },
        }

    def get_recent_events(self, n: int = 10) -> list[dict]:
        """Get most recent N events.

        Parameters
        ----------
        n : int
            Number of recent events to return

        Returns
        -------
        list[dict]
            Most recent events
        """
        return self._event_log[-n:]

    def export_metrics(self) -> dict:
        """Export metrics in a format suitable for monitoring systems.

        Returns
        -------
        dict
            Metrics with consistent naming for Prometheus/Grafana/etc.
        """
        return {
            "nlsq_defense_warmup_calls_total": self.total_warmup_calls,
            "nlsq_defense_layer1_triggers_total": self.layer1_warm_start_triggers,
            "nlsq_defense_layer2_refinement_total": self.layer2_lr_mode_counts[
                "refinement"
            ],
            "nlsq_defense_layer2_careful_total": self.layer2_lr_mode_counts["careful"],
            "nlsq_defense_layer2_exploration_total": self.layer2_lr_mode_counts[
                "exploration"
            ],
            "nlsq_defense_layer3_triggers_total": self.layer3_cost_guard_triggers,
            "nlsq_defense_layer4_triggers_total": self.layer4_clip_triggers,
            "nlsq_defense_lbfgs_history_fill_total": self.lbfgs_history_buffer_fill_events,
            "nlsq_defense_lbfgs_line_search_failures_total": self.lbfgs_line_search_failures,
        }


# Global telemetry instance for monitoring
_defense_telemetry: DefenseLayerTelemetry | None = None


def get_defense_telemetry() -> DefenseLayerTelemetry:
    """Get global defense layer telemetry instance.

    Returns
    -------
    DefenseLayerTelemetry
        Global telemetry instance (created on first call)
    """
    global _defense_telemetry  # noqa: PLW0603
    if _defense_telemetry is None:
        _defense_telemetry = DefenseLayerTelemetry()
    return _defense_telemetry


def reset_defense_telemetry() -> None:
    """Reset global defense layer telemetry."""
    global _defense_telemetry  # noqa: PLW0602
    if _defense_telemetry is not None:
        _defense_telemetry.reset()
