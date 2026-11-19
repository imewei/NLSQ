"""Regression tests for round 2 of the 2026-11-18 three-brain review of the streaming
hybrid engine (adaptive_hybrid.py, hybrid_config.py, telemetry.py, phases/)."""

import os
import stat
import sys
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.checkpoint import CheckpointManager, CheckpointState
from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase
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


_GN_ONLY = {"warmup_iterations": 0, "max_warmup_iterations": 1}


def _tracker():
    return {"best_params_global": None, "best_cost_global": float("inf")}


def _phase_cfg(**kw):
    return HybridStreamingConfig(
        chunk_size=256, verbose=0, enable_checkpoints=False, **kw
    )


def _lin(x, a):
    return a * x


# --- adaptive engine ---------------------------------------------------------


def test_batch_dependent_model_is_differentiated_as_evaluated():
    def f(x, a):
        return a * (x - jnp.mean(x))

    x = np.arange(1.0, 9.0)
    r = _fit(f, x, np.asarray(f(x, 2.0)), [1.0], {**_GN_ONLY, "chunk_size": 8})
    np.testing.assert_allclose(np.asarray(r["x"]), [2.0], rtol=1e-8)


def test_gradient_convergence_does_not_underflow_for_tiny_valued_data():
    def f(x, a):
        return 1e-100 * a * x

    x = np.array([1.0, 2.0, 3.0])
    r = _fit(f, x, 100e-100 * x, [1.0], {**_GN_ONLY, "gauss_newton_max_iterations": 20})
    np.testing.assert_allclose(np.asarray(r["x"]), [100.0], rtol=1e-6)


def test_weak_but_identifiable_column_is_not_dropped():
    def f(x, a, b):
        return a * x[:, 0] + 1e-11 * b * x[:, 1]

    X = np.array([[1.0, 0.0], [0.0, 1.0]] * 2)
    r = _fit(f, X, np.asarray(f(X, 2.0, 3.0)), [1.0, 1.0], _GN_ONLY)
    np.testing.assert_allclose(np.asarray(r["x"]), [2.0, 3.0], rtol=1e-6)


def test_constrained_trust_region_step_uses_configured_damping():
    H, g = jnp.diag(jnp.array([1.0, 100.0])), jnp.array([1.0, 10.0])
    steps = []
    for reg in (1e-12, 1.0):
        opt = AdaptiveHybridStreamingOptimizer(_cfg(regularization_factor=reg))
        steps.append(np.asarray(opt._solve_gauss_newton_step(H, g, 0.05)[0]))
    np.testing.assert_allclose(steps[0], [0.00954669, 0.04908015], rtol=1e-6)
    np.testing.assert_allclose(steps[1], [0.02647676, 0.0424144], rtol=1e-6)


def test_column_response_is_flattened_not_broadcast():
    x = np.array([1.0, 2.0, 3.0])
    r = _fit(_lin, x, (2 * x)[:, None], [1.0], _GN_ONLY)
    assert np.asarray(r["fun"]).shape == (3,)
    np.testing.assert_allclose(np.asarray(r["x"]), [2.0], rtol=1e-8)


def test_matrix_response_is_rejected():
    with pytest.raises(ValueError, match="same number of observations"):
        _fit(_lin, np.ones(3), np.ones((3, 2)), [1.0], _GN_ONLY)


def test_result_residuals_are_model_minus_data_like_trf():
    x, y = np.arange(4.0), np.array([1.0, 2.0, 4.0, 8.0])
    r = _fit(_lin, x, y, [1.0], _GN_ONLY)
    np.testing.assert_allclose(
        np.asarray(r["fun"]), np.asarray(r["x"])[0] * x - y, atol=1e-12
    )


def test_phase_status_advances_without_checkpoints():
    opt = AdaptiveHybridStreamingOptimizer(_cfg())
    x = np.linspace(0.0, 1.0, 50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt.fit((x, 3.0 * x), _lin, [1.0], verbose=0)
    assert opt.phase_status["current_phase"] == 3


def test_set_residual_weights_validates_and_rejects_complex():
    opt = AdaptiveHybridStreamingOptimizer(
        _cfg(enable_residual_weighting=True, residual_weights=[1.0, 1.0])
    )
    with pytest.raises(ValueError, match="positive"):
        opt.set_residual_weights([1.0, -1.0])
    with pytest.raises(ValueError, match="complex"):
        opt.set_residual_weights(np.array([1 + 4j, 2 + 5j]))


def test_residual_weight_update_during_warmup_changes_the_objective():
    def f(x, c):
        return c + 0.0 * x[:, 1]

    x = np.column_stack([[0.0, 1.0, 0.0, 1.0], np.zeros(4)])
    y = np.array([0.0, 10.0, 0.0, 10.0])
    opt = AdaptiveHybridStreamingOptimizer(
        _cfg(
            enable_residual_weighting=True,
            residual_weights=[1.0, 1.0],
            warmup_iterations=2,
            max_warmup_iterations=3,
            enable_warm_start_detection=False,
            active_switching_criteria=[],
            callback_frequency=1,
        )
    )
    seen = []

    def cb(it, cost, params, info):
        seen.append((it, info["gradient_norm"]))
        if it == 1:
            opt.set_residual_weights([1.0, 100.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt.fit((x, y), f, [5.0], callback=cb, verbose=0)
    # equal weights leave c=5 stationary (gradient 0); the update must be seen
    assert seen[0][1] == pytest.approx(0.0, abs=1e-9)
    assert seen[1][1] > 1e-3


def test_warmup_callback_cost_is_unweighted_half_ssr():
    def f(x, c):
        return c + 0.0 * x[:, 1]

    x = np.column_stack([[0.0, 1.0], np.zeros(2)])
    y = np.array([0.0, 10.0])
    opt = AdaptiveHybridStreamingOptimizer(
        _cfg(
            enable_residual_weighting=True,
            residual_weights=[1.0, 100.0],
            warmup_iterations=1,
            max_warmup_iterations=1,
            enable_warm_start_detection=False,
            active_switching_criteria=[],
            callback_frequency=1,
        )
    )
    costs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt.fit(
            (x, y),
            f,
            [0.0],
            callback=lambda it, cost, p, info: costs.append((info["phase"], cost)),
            verbose=0,
        )
    assert costs[0] == (1, pytest.approx(50.0))


# --- config ------------------------------------------------------------------


@pytest.mark.parametrize("bad", [["gradent"], "plateau", ["plateau", "grad"]])
def test_unknown_switching_criteria_are_rejected(bad):
    with pytest.raises(ValueError, match="active_switching_criteria"):
        HybridStreamingConfig(active_switching_criteria=bad)


def test_memory_optimized_preset_fits_1200_parameters_past_the_cg_guard():
    cfg = HybridStreamingConfig.memory_optimized()
    assert cfg.cg_param_threshold > 1200


# --- telemetry ---------------------------------------------------------------


def test_failure_from_a_warmup_started_before_reset_is_not_counted():
    t = DefenseLayerTelemetry()
    a = t.record_warmup_start()
    t.record_lbfgs_iteration()
    t.reset()
    b = t.record_warmup_start()
    t.record_lbfgs_iteration()
    t.record_lbfgs_line_search_failure(0, "a", warmup=a)
    t.record_lbfgs_line_search_failure(0, "b", warmup=b)
    assert t.get_trigger_rates()["lbfgs_line_search_failure_rate"] == pytest.approx(
        100.0
    )


def test_recent_events_are_detached_copies():
    t = DefenseLayerTelemetry()
    t.record_lbfgs_line_search_failure(0, "x")
    t.get_recent_events()[-1]["data"]["iteration"] = 99
    assert t.get_recent_events()[-1]["data"]["iteration"] == 0


# --- standalone phases ---------------------------------------------------------


def test_standalone_gauss_newton_ignores_absolute_gradient_for_small_units():
    x = jnp.arange(1.0, 5.0) * 1e-6
    r = GaussNewtonPhase(_phase_cfg(), _lin).run(
        (x, 2 * x), jnp.array([0.0]), [], _tracker()
    )
    np.testing.assert_allclose(np.asarray(r["final_params"]), [2.0], rtol=1e-5)


def test_standalone_gauss_newton_reaches_bounded_optimum_with_active_set():
    def f(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    x = jnp.linspace(0.0, 5.0, 200)
    y = f(x, 3.0, 0.7, 0.5)
    cfg = _phase_cfg(gauss_newton_max_iterations=200)
    bounds = (jnp.array([-10.0, -10.0, -10.0]), jnp.array([10.0, 10.0, 0.4]))
    r = GaussNewtonPhase(cfg, f, bounds).run(
        (x, y), jnp.array([2.5, 0.6, 0.3]), [], _tracker()
    )
    p = np.asarray(r["final_params"])
    assert p[2] == pytest.approx(0.4)
    # constrained optimum (SciPy bounded TRF): SSR ~2.12, not the 20 of a clipped step
    assert float(jnp.sum((f(x, *p) - y) ** 2)) < 2.5


@pytest.mark.parametrize("phase", ["warmup", "gauss_newton"])
def test_standalone_phase_projects_infeasible_start_onto_bounds(phase):
    x = jnp.arange(1.0, 5.0)
    bounds = (jnp.array([0.0]), jnp.array([1.0]))
    cfg = _phase_cfg(
        warmup_iterations=0,
        max_warmup_iterations=1,
        enable_warm_start_detection=False,
    )
    p = (WarmupPhase if phase == "warmup" else GaussNewtonPhase)(cfg, _lin, bounds)
    tracker = _tracker()
    r = p.run((x, 2 * x), jnp.array([2.0]), [], tracker)
    assert float(np.asarray(r["best_params"])[0]) <= 1.0
    assert float(np.asarray(tracker["best_params_global"])[0]) <= 1.0


@pytest.mark.parametrize("phase", [WarmupPhase, GaussNewtonPhase])
def test_standalone_phase_rejects_unmatched_response_length(phase):
    x = jnp.arange(1.0, 9.0)
    y = jnp.concatenate([2 * x, jnp.array([500.0])])
    with pytest.raises(ValueError, match="xdata and ydata"):
        phase(_phase_cfg(), _lin).run((x, y), jnp.array([2.0]), [], _tracker())


def test_standalone_gauss_newton_ratio_uses_the_bounded_step():
    # slope 0 -> 0.1 is clipped by the bound; the executed step reduces SSR by 0.76
    # exactly as the quadratic model predicts, so the radius must not shrink.
    x = jnp.ones(4)
    gn = GaussNewtonPhase(_phase_cfg(), _lin, (jnp.array([0.0]), jnp.array([0.1])))
    out = gn._gauss_newton_iteration((x, x), jnp.array([0.0]), 1.0)
    assert out["trust_radius"] >= 1.0


def test_standalone_gauss_newton_solver_honors_configured_damping():
    H, g = jnp.diag(jnp.array([1.0, 100.0])), jnp.array([1.0, 10.0])
    steps = [
        np.asarray(
            GaussNewtonPhase(
                _phase_cfg(regularization_factor=reg), _lin
            )._solve_gauss_newton_step(H, g, 0.05)[0]
        )
        for reg in (1e-12, 1.0)
    ]
    assert not np.allclose(steps[0], steps[1])


def test_standalone_gauss_newton_keeps_scaled_independent_direction():
    X = jnp.array([[1000.0, 0.0], [0.0, 0.001]] * 2)

    def f(x, a, b):
        return a * x[:, 0] + b * x[:, 1]

    r = GaussNewtonPhase(_phase_cfg(gauss_newton_max_iterations=20), f).run(
        (X, f(X, 2.0, 2.0)), jnp.array([1.0, 1.0]), [], _tracker()
    )
    np.testing.assert_allclose(np.asarray(r["final_params"]), [2.0, 2.0], rtol=1e-6)


def test_all_rejected_run_reports_cost_at_retained_start():
    def f(x, a):
        return a**2 * x

    x = jnp.arange(1.0, 5.0)
    cfg = _phase_cfg(gauss_newton_max_iterations=1, trust_region_initial=10.0)
    r = GaussNewtonPhase(cfg, f).run((x, 2 * x), jnp.array([0.1]), [], _tracker())
    if float(np.asarray(r["final_params"])[0]) == pytest.approx(0.1):
        assert r["final_cost"] == pytest.approx(r["best_cost"])


def test_standalone_warmup_does_not_declare_plateau_at_iteration_zero():
    x = jnp.arange(1.0, 5.0)
    cfg = _phase_cfg(
        warmup_iterations=0,
        max_warmup_iterations=5,
        enable_warm_start_detection=False,
        active_switching_criteria=["plateau"],
    )
    r = WarmupPhase(cfg, _lin).run((x, 2 * x), jnp.array([1.0]), [], _tracker())
    assert r["iterations"] > 1


def test_standalone_warmup_weight_setter_accepts_lists_and_validates():
    w = WarmupPhase(
        _phase_cfg(enable_residual_weighting=True, residual_weights=[1.0, 1.0]), _lin
    )
    w.set_residual_weights([1.0, 2.0])
    assert isinstance(w._residual_weights, jnp.ndarray)
    with pytest.raises(ValueError):
        w.set_residual_weights([1.0, 0.0])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_checkpoint_replace_keeps_restricted_mode(tmp_path):
    state = CheckpointState(
        current_phase=1,
        normalized_params=jnp.array([1.0]),
        phase1_optimizer_state=None,
        phase2_JTJ_accumulator=None,
        phase2_JTr_accumulator=None,
        best_params_global=None,
        best_cost_global=float("inf"),
        phase_history=[],
        normalizer=None,
        tournament_selector=None,
        multistart_candidates=None,
    )
    mgr = CheckpointManager(_phase_cfg())
    path = tmp_path / "ck.h5"
    mgr.save(path, state)
    os.chmod(path, 0o600)
    mgr.save(path, state)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_fixed_learning_rate_mode_is_reported():
    t = DefenseLayerTelemetry()
    t.record_warmup_start()
    t.record_layer2_lr_mode("fixed", 0.1)
    assert t.get_trigger_rates()["layer2_fixed_rate"] == pytest.approx(100.0)
    assert t.get_summary()["layer2"]["rates_pct"]["fixed"] == pytest.approx(100.0)
    assert t.export_metrics()["nlsq_defense_layer2_fixed_total"] == 1
