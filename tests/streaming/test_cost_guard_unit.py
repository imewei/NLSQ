"""Direct unit tests for the Layer-3 cost-increase guard.

The end-to-end warmup tests never trip the guard with their configured
learning rates, so the guard's positive path is exercised here directly.
"""

import jax.numpy as jnp
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.telemetry import DefenseLayerTelemetry


def _guard(loss_value, tol=0.05, iteration=1):
    opt = AdaptiveHybridStreamingOptimizer(
        HybridStreamingConfig(
            enable_cost_guard=True, cost_increase_tolerance=tol, verbose=0
        )
    )
    opt._warmup_initial_loss = 1.0
    best = jnp.array([1.0, 2.0])
    out = opt._check_cost_increase_guard(
        iteration=iteration,
        loss_value=loss_value,
        best_loss=0.9,
        best_params=best,
        lr_mode="exploration",
        relative_loss=1.0,
        telemetry=DefenseLayerTelemetry(),
    )
    return out, best


def test_cost_guard_fires_and_returns_best_params():
    (should_exit, res), best = _guard(loss_value=2.0)
    assert should_exit is True
    assert res["cost_guard_triggered"] is True
    assert res["final_params"] is best
    assert res["best_loss"] == 0.9
    assert res["cost_increase_ratio"] == pytest.approx(2.0)


def test_cost_guard_quiet_within_tolerance_and_on_first_iteration():
    assert _guard(loss_value=1.04)[0] == (False, None)
    assert _guard(loss_value=2.0, iteration=0)[0] == (False, None)
