"""Phase boundaries must preserve the objective and the best point."""

import jax.numpy as jnp
import numpy as np

from nlsq.precision.parameter_normalizer import (
    NormalizedModelWrapper,
    ParameterNormalizer,
)
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase
from nlsq.streaming.phases.orchestrator import PhaseOrchestrator
from nlsq.streaming.phases.warmup import WarmupPhase


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def test_phase_warmup_and_gn_share_regularized_objective():
    cfg = HybridStreamingConfig(
        enable_group_variance_regularization=True,
        group_variance_lambda=0.1,
        group_variance_indices=[(0, 2)],
        verbose=0,
    )
    normalizer = ParameterNormalizer(jnp.array([10.0, 0.1]), strategy="p0")
    model = NormalizedModelWrapper(_model, normalizer)
    warmup = WarmupPhase(cfg, model)
    gn = GaussNewtonPhase(cfg, model)
    x = jnp.linspace(0, 5, 16)
    params = jnp.array([1.0, 1.0])
    y = model(x, *params)
    warmup_ssr = warmup._create_loss_fn()(params, x, y) * len(y)
    np.testing.assert_allclose(warmup_ssr, gn._compute_cost(params, x, y))


def test_orchestrator_preserves_initial_point_when_only_trial_is_worse():
    cfg = HybridStreamingConfig(
        chunk_size=256,
        gauss_newton_max_iterations=1,
        trust_region_initial=50.0,
        verbose=0,
    )
    x = jnp.linspace(0, 5, 200)
    y = _model(x, 2.0, 0.3)
    initial = jnp.array([2.0, 5.0])
    orchestrator = PhaseOrchestrator(cfg)
    orchestrator.gn_phase = GaussNewtonPhase(cfg, _model)
    result = orchestrator.run((x, y), initial)
    np.testing.assert_allclose(result["final_params"], initial)
    np.testing.assert_allclose(
        result["best_cost"], orchestrator.gn_phase._compute_cost(initial, x, y)
    )


def test_warmup_cost_guard_accepts_exact_start_when_detection_disabled():
    cfg = HybridStreamingConfig(
        warmup_iterations=2,
        max_warmup_iterations=3,
        active_switching_criteria=[],
        enable_warm_start_detection=False,
        enable_cost_guard=True,
        verbose=0,
    )
    x = jnp.linspace(0, 2, 16)
    warmup = WarmupPhase(cfg, lambda x, a: a * x)
    result = warmup.run(
        (x, 2 * x), jnp.array([2.0]), [], {"best_cost_global": float("inf")}
    )
    np.testing.assert_allclose(result["best_params"], [2.0])
    assert result["best_loss"] == 0.0
