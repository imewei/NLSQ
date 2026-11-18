"""Regression tests for the 2026-11-18 three-brain review of the streaming hybrid
engine (adaptive_hybrid.py, hybrid_config.py, telemetry.py, phases/warmup.py)."""

import warnings
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.warmup import WarmupPhase
from nlsq.streaming.telemetry import DefenseLayerTelemetry


def _cfg(**kw):
    base = {"enable_checkpoints": False, "verbose": 0, "normalize": False}
    return HybridStreamingConfig(**{**base, **kw})


def _fit(f, x, y, p0, cfg_kw=None, **fit_kw):
    opt = AdaptiveHybridStreamingOptimizer(_cfg(**(cfg_kw or {})))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return opt.fit((x, y), f, p0, verbose=0, **fit_kw)


def test_active_bound_does_not_freeze_small_valued_free_parameters():
    # b sits on its lower bound; a's curvature (2e-22) is far below the old absolute 1.0
    # placed on the fixed row's diagonal, which zeroed a's step.
    def f(x, a, b):
        return 1e-11 * jnp.where(x == 0, a, b)

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = 1e-11 * np.array([2.0, -1.0, 2.0, -1.0])
    r = _fit(
        f,
        x,
        y,
        [1.0, 0.0],
        {
            "warmup_iterations": 0,
            "max_warmup_iterations": 1,
            "gauss_newton_max_iterations": 3,
        },
        bounds=([-np.inf, 0.0], [np.inf, np.inf]),
    )
    np.testing.assert_allclose(np.asarray(r["x"]), [2.0, 0.0], atol=1e-6)


def test_zero_data_residual_does_not_end_fit_before_group_penalty_is_minimized():
    def f(x, a, b):
        return a + b + jnp.zeros_like(x)

    r = _fit(
        f,
        np.arange(4.0),
        np.full(4, 4.0),
        [1.0, 3.0],
        {
            "warmup_iterations": 0,
            "max_warmup_iterations": 1,
            "enable_group_variance_regularization": True,
            "group_variance_indices": [(0, 2)],
            "group_variance_lambda": 1.0,
            "trust_region_initial": 0.01,
        },
    )
    np.testing.assert_allclose(np.asarray(r["x"]), [2.0, 2.0], atol=1e-4)


def test_ill_conditioned_polynomial_reaches_the_least_squares_optimum():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 5000)
    true = rng.normal(size=9) * np.array([10.0**-k for k in range(9)])

    def poly(x, *c):
        return sum(ci * x**i for i, ci in enumerate(c))

    y = poly(x, *true) + 0.01 * rng.normal(size=x.size)
    opt = AdaptiveHybridStreamingOptimizer(_cfg(normalize=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = opt.fit((x, y), poly, list(1.5 * true), verbose=0)
    A = np.vander(x, 9, increasing=True)
    best = np.sum((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) ** 2)
    ssr = np.sum((y - poly(x, *np.asarray(r["x"]))) ** 2)
    assert ssr <= best * (1 + 2e-7)


@pytest.mark.parametrize("normalize", [False, True])
def test_collinear_covariance_is_the_moore_penrose_pinv(normalize):
    # the pseudo-inverse must be taken in original coordinates, also when the
    # optimizer works in normalized ones
    r = _fit(
        lambda x, a, b: (a + 2 * b) * x,
        np.array([1.0, 2.0, 3.0]),
        np.array([5.0, 10.0, 15.0]),
        [1.0, 2.0],
        {"normalize": normalize},
        absolute_sigma=True,
    )
    JTJ = np.array([[1.0, 2.0], [2.0, 4.0]]) * 14.0
    np.testing.assert_allclose(np.asarray(r["pcov"]), np.linalg.pinv(JTJ), rtol=1e-8)


@pytest.mark.parametrize(
    ("x", "y"),
    [(np.array([]), np.array([])), (np.array([1.0, 2.0, 3.0]), np.array([2.0]))],
)
def test_empty_or_misaligned_observations_rejected(x, y):
    with pytest.raises(ValueError, match="same number of observations"):
        _fit(lambda x, a: a * x, x, y, [1.0])


def test_warmup_phase_final_loss_pairs_with_final_params():
    class Model:
        normalizer = None

        def __call__(self, x, a):
            return a * x

    cfg = _cfg(
        warmup_iterations=0,
        max_warmup_iterations=1,
        enable_warm_start_detection=False,
        active_switching_criteria=[],
    )
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    phase = WarmupPhase(cfg, Model())
    res = phase.run((x, 2 * x), jnp.array([1.0]), [], {})
    loss_fn = phase._create_loss_fn()
    final = float(loss_fn(res["final_params"], x, 2 * x))
    assert res["final_loss"] == pytest.approx(final, rel=1e-12)
    assert res["best_loss"] <= final * (1 + 1e-12)
    assert float(res["best_params"][0]) == pytest.approx(
        float(res["final_params"][0]), abs=1e-12
    )


def test_phase1_callback_is_called_for_the_iteration_that_switches():
    calls = []

    def cb(iteration, cost, params, info):
        calls.append((iteration, info.get("phase")))
        raise StopIteration

    r = _fit(
        lambda x, a: a * a + jnp.zeros_like(x),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.ones(4),
        [0.1],
        {
            "warmup_iterations": 0,
            "max_warmup_iterations": 1,
            "callback_frequency": 1,
        },
        callback=cb,
    )
    assert calls and calls[0][0] == 1
    assert calls[0][1] == 1  # phase 1, not a Phase 2 iteration
    assert r["x"] is not None


def test_plateau_cannot_fire_at_warmup_iteration_zero():
    opt = AdaptiveHybridStreamingOptimizer(_cfg())
    stop, _ = opt._check_phase1_switch_criteria(
        iteration=0, current_loss=5.0, prev_loss=5.0, grad_norm=1.0
    )
    assert not stop
    stop, why = opt._check_phase1_switch_criteria(
        iteration=1, current_loss=5.0, prev_loss=5.0, grad_norm=1.0
    )
    assert stop and "plateau" in why


def test_regularization_factor_reaches_the_gauss_newton_solve():
    JTJ = jnp.array([[4.0, 0.0], [0.0, 1.0]])
    JTr = jnp.array([1.0, 1.0])
    small = AdaptiveHybridStreamingOptimizer(_cfg(regularization_factor=1e-10))
    large = AdaptiveHybridStreamingOptimizer(_cfg(regularization_factor=1.0))
    s_small, _ = small._solve_gauss_newton_step(JTJ, JTr, 1e6)
    s_large, _ = large._solve_gauss_newton_step(JTJ, JTr, 1e6)
    np.testing.assert_allclose(np.asarray(s_small), [0.25, 1.0], rtol=1e-8)
    np.testing.assert_allclose(np.asarray(s_large), [0.125, 0.5], rtol=1e-8)


def test_set_residual_weights_survives_fit_setup():
    opt = AdaptiveHybridStreamingOptimizer(
        _cfg(enable_residual_weighting=True, residual_weights=[1.0, 1.0, 1.0])
    )
    new = np.array([0.3, 0.7, 1.0])
    opt.set_residual_weights(new)
    opt._setup_residual_weights()  # what fit() runs
    np.testing.assert_allclose(np.asarray(opt._residual_weights_jax), new)


def test_telemetry_tokens_stay_unique_across_reset():
    t = DefenseLayerTelemetry()
    a = t.record_warmup_start()
    t.reset()
    b = t.record_warmup_start()
    assert a != b
    t.record_lbfgs_iteration()
    t.record_lbfgs_iteration()
    t.record_lbfgs_line_search_failure(0, "a", warmup=a)
    t.record_lbfgs_line_search_failure(0, "b", warmup=b)
    # a began before reset(): its failure belongs to the wiped counters (round 2:
    # counting it exceeded 100% against the new denominator)
    assert t.lbfgs_line_search_failed_steps == 1


# --- wave 2 (Agy P04) -----------------------------------------------------------


class _LinearModel:
    normalizer = None

    def __call__(self, x, a):
        return a * x


def test_gauss_newton_phase_jacobian_handles_2d_x():
    from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase

    def model_2d(x, a, b):
        return a * x[:, 0] + b * x[:, 1]

    gn = GaussNewtonPhase(_cfg(chunk_size=10), model_2d)
    J = gn._compute_jacobian_chunk(
        jnp.stack([jnp.arange(10.0), jnp.ones(10)], axis=1), jnp.array([1.0, 2.0])
    )
    np.testing.assert_allclose(np.asarray(J[:, 0]), np.arange(10.0))
    np.testing.assert_allclose(np.asarray(J[:, 1]), 1.0)


def test_gauss_newton_phase_rejected_step_is_not_convergence_far_from_optimum():
    from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase

    x = jnp.linspace(1.0, 4.0, 40)
    y = 2.0 * x
    gn = GaussNewtonPhase(
        _cfg(chunk_size=40, gauss_newton_max_iterations=3), _LinearModel()
    )
    p0 = jnp.array([10.0])
    cost0 = gn._compute_cost(p0, x, y)
    orig = gn._gauss_newton_iteration

    def rejected(data, params, radius):
        r = orig(data, params, radius)
        r.update(
            new_params=params,
            new_cost=cost0 * (1 + 1e-12),
            actual_reduction=-cost0 * 1e-12,
            gradient_norm=1.0,
        )
        return r

    gn._gauss_newton_iteration = rejected
    res = gn.run((x, y), p0, [], {})
    assert res["convergence_reason"] == "Maximum iterations reached"


def test_warmup_phase_counts_lbfgs_iterations_for_telemetry_rates():
    from nlsq.streaming.telemetry import get_defense_telemetry, reset_defense_telemetry

    reset_defense_telemetry()
    cfg = _cfg(
        warmup_iterations=5,
        max_warmup_iterations=5,
        max_warmup_step_size=0.001,
        enable_warm_start_detection=False,
        active_switching_criteria=[],
    )
    x = jnp.linspace(1, 10, 50)
    WarmupPhase(cfg, _LinearModel()).run((x, 2 * x), jnp.array([10.0]), [], {})
    tel = get_defense_telemetry()
    assert tel.total_lbfgs_iterations == 5
    assert tel.get_trigger_rates()["layer4_clip_rate"] > 0.0


def test_warmup_phase_careful_step_reads_config():
    phase = WarmupPhase(_cfg(lbfgs_careful_step_size=0.3), _LinearModel())
    step, mode = phase._select_initial_step_size(0.5, MagicMock())
    assert (step, mode) == (0.3, "careful")


def test_warmup_phase_clips_updates_to_normalized_bounds():
    cfg = _cfg(
        warmup_iterations=0,
        max_warmup_iterations=8,
        max_warmup_step_size=100.0,
        enable_warm_start_detection=False,
        active_switching_criteria=[],
    )
    x = jnp.linspace(1, 4, 20)
    bounds = (jnp.array([5.0]), jnp.array([12.0]))
    phase = WarmupPhase(cfg, _LinearModel(), bounds)
    res = phase.run((x, 2 * x), jnp.array([10.0]), [], {})
    assert float(res["final_params"][0]) >= 5.0
    assert float(res["best_params"][0]) >= 5.0


def test_orchestrator_forwards_residual_weights_to_warmup():
    from nlsq.streaming.phases.orchestrator import PhaseOrchestrator

    cfg = _cfg(enable_residual_weighting=True, residual_weights=[1.0, 2.0])
    orch = PhaseOrchestrator(cfg)
    orch.initialize_phases(_LinearModel())
    np.testing.assert_allclose(
        np.asarray(orch.warmup_phase._residual_weights), [1.0, 2.0]
    )


def test_warm_start_updates_shared_best_tracker():
    cfg = _cfg(enable_warm_start_detection=True, warm_start_threshold=0.5)
    x = jnp.linspace(1, 4, 20)
    tracker = {"best_cost_global": float("inf"), "best_params_global": None}
    res = WarmupPhase(cfg, _LinearModel()).run(
        (x, 2 * x + 1e-3 * jnp.sin(x)), jnp.array([2.0]), [], tracker
    )
    assert res.get("warm_start") is True
    assert np.isfinite(tracker["best_cost_global"])
    assert tracker["best_params_global"] is not None


# --- wave 3 ---------------------------------------------------------------------


def test_gauss_newton_phase_stationarity_guard_uses_penalized_objective():
    from nlsq.precision.parameter_normalizer import (
        NormalizedModelWrapper,
        ParameterNormalizer,
    )
    from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase

    p0 = jnp.array([1.0, 3.0])
    wrapped = NormalizedModelWrapper(
        lambda x, a, b: a + b + jnp.zeros_like(x),
        ParameterNormalizer(p0=p0, strategy="none"),
    )
    cfg = _cfg(
        chunk_size=4,
        gauss_newton_tol=1e-8,
        gauss_newton_max_iterations=2,
        trust_region_initial=1e-10,
        enable_group_variance_regularization=True,
        group_variance_indices=[(0, 2)],
        group_variance_lambda=1.0,
    )
    res = GaussNewtonPhase(cfg, wrapped).run(
        (jnp.arange(4.0), jnp.full(4, 4.0)), p0, [], {}
    )
    # the penalized gradient is non-zero at [1, 3]: a one-step cost-change stop is wrong
    assert res["convergence_reason"] != "Cost change below tolerance"
