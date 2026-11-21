"""Regression tests for the 2026-11-16 round-1 review of the streaming
hybrid/chunked path (adaptive_hybrid.py, large_dataset.py, phases/)."""

import warnings

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from nlsq.callbacks import StopOptimization
from nlsq.result import OptimizeWarning
from nlsq.streaming.adaptive_hybrid import (
    CALLBACK_STOP_MESSAGE,
    AdaptiveHybridStreamingOptimizer,
)
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig
from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase
from nlsq.streaming.phases.warmup import _create_lbfgs_optimizer


def _exp_model(x, a, b):
    return a * jnp.exp(-b * x)


def _cfg(**kw):
    base = {
        "enable_checkpoints": False,
        "verbose": 0,
        "max_warmup_iterations": 20,
        "warmup_iterations": 5,
        "gauss_newton_max_iterations": 10,
    }
    return HybridStreamingConfig(**{**base, **kw})


def _lin(x, a, b):
    return a * x + b


def _fitter(**kw):
    kw.setdefault("min_chunk_size", 1000)
    kw.setdefault("max_chunk_size", 1000)
    return LargeDatasetFitter(config=LDMemoryConfig(**kw))


# --- callback StopIteration -------------------------------------------------


def test_stop_iteration_callback_stops_hybrid_fit():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 3000)
    y = 3 * np.exp(-0.7 * x) + 0.01 * rng.normal(size=x.size)
    calls = []

    def cb(iteration, cost, params, info):
        calls.append(iteration)
        if iteration >= 3:
            raise StopIteration

    opt = AdaptiveHybridStreamingOptimizer(_cfg(chunk_size=1000, callback_frequency=1))
    r = opt.fit((x, y), _exp_model, [1.0, 1.0], verbose=0, callback=cb)
    assert r["status"] == -2
    assert r["message"] == CALLBACK_STOP_MESSAGE
    assert calls[-1] == 3


# --- covariance never fabricated -------------------------------------------


def test_ill_conditioned_information_gives_inf_pcov_and_warns():
    fitter = LargeDatasetFitter()
    fitter._accum_information = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-14]])
    with pytest.warns(OptimizeWarning, match="not estimated"):
        pcov = fitter._compute_covariance_from_history(
            [np.ones(2), np.ones(2) * 1.001], np.ones(2)
        )
    assert pcov.shape == (2, 2)
    assert np.all(np.isinf(pcov))


def test_no_accumulated_information_gives_inf_pcov_and_warns():
    fitter = LargeDatasetFitter()
    fitter._accum_information = None
    with pytest.warns(OptimizeWarning, match="not estimated"):
        pcov = fitter._compute_covariance_from_history([np.ones(2)], np.ones(2))
    assert np.all(np.isinf(pcov))


def test_chunk_acceptance_shares_the_1e12_gate():
    # correlation matrix with cond ~1e13: above the 1e12 gate, below the old 1.5e15
    eps = 1e-13
    pcov = np.array([[1.0, 1.0 - eps], [1.0 - eps, 1.0]])
    assert LargeDatasetFitter._chunk_precision(pcov, 2) is None
    ok = np.array([[1.0, 0.9], [0.9, 1.0]])
    assert LargeDatasetFitter._chunk_precision(ok, 2) is not None


# --- non-finite data / masks ----------------------------------------------------


def test_missing_check_finite_does_not_silently_drop_chunk():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 8000)
    y = 2 * x + 1 + 0.01 * rng.normal(size=x.size)
    y[123] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _fitter(min_success_rate=0.5).fit(_lin, x, y, p0=[1.0, 0.0])


def test_explicit_check_finite_false_is_still_an_opt_out():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 4000)
    y = 2 * x + 1 + 0.01 * rng.normal(size=x.size)
    y[5] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _fitter(min_success_rate=0.5).fit(
            _lin, x, y, p0=[1.0, 0.0], check_finite=False
        )
    assert res is not None  # no ValueError: explicit opt-out preserved


def test_validation_ignores_masked_out_rows():
    x = np.arange(10.0) + 1
    y = 2 * x
    sigma = np.ones(10)
    sigma[0] = 0.0  # excluded by the mask
    x_nan = x.copy()
    x_nan[0] = np.nan
    mask = np.ones(10, dtype=bool)
    mask[0] = False
    fitter = _fitter(min_chunk_size=2, max_chunk_size=4)
    res = fitter.fit(
        lambda x, a: a * x, x_nan, y, p0=[1.0], sigma=sigma, data_mask=mask
    )
    assert res.success
    np.testing.assert_allclose(res.popt, [2.0], rtol=1e-6)


# --- retry respects the inner success flag ------------------------------------


def test_retry_non_converged_is_not_a_success():
    from nlsq.result import CurveFitResult

    class _Res(tuple):
        success = False
        message = "Maximum number of function evaluations reached"

    fitter = LargeDatasetFitter()
    fitter.curve_fit.curve_fit = lambda *a, **k: _Res((np.array([5.0]), np.eye(1)))
    x = np.linspace(0, 1, 50)
    chunk_result, popt = fitter._retry_failed_chunk(
        f=lambda x, a: a * x,
        x_chunk=x,
        y_chunk=x,
        chunk_idx=0,
        chunk_start_time=0.0,
        chunk_times=[],
        current_params=np.array([1.0]),
        initial_error=RuntimeError("first attempt"),
        bounds=(-np.inf, np.inf),
        method="trf",
        solver="auto",
    )
    assert popt is None
    assert chunk_result["success"] is False


# --- accumulator not double counted ---------------------------------------------


def test_chunk_not_folded_twice_when_result_building_raises():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 1, 4000)
    y = 2 * x + 1 + 0.01 * rng.normal(size=x.size)
    fitter = _fitter(min_success_rate=0.5)
    real = fitter._create_chunk_result
    state = {"raised": False}

    def flaky(*a, **k):
        if not state["raised"] and k.get("success"):
            state["raised"] = True
            raise RuntimeError("diagnostics failed")
        return real(*a, **k)

    fitter._create_chunk_result = flaky
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter.fit(_lin, x, y, p0=[1.0, 0.0])
    assert state["raised"]
    assert fitter._accum_n == len(x)


# --- validation cache keyed on p0 length ----------------------------------------


def test_validation_cache_revalidates_for_different_p0_length():
    x = np.linspace(0, 1, 100)
    y = x.copy()
    fitter = LargeDatasetFitter()

    def model(x, *p):
        return p[0] * x + (p[1] if len(p) > 1 else 0.0)

    fitter._validate_model_function(model, x, y, [1.0, 0.0])
    key = next(iter(fitter._validated_functions))
    assert fitter._validated_functions[key] == 2
    fitter._validate_model_function(model, x, y, [1.0])
    assert fitter._validated_functions[key] == 1


# --- Gauss-Newton fault tolerance does not hide deterministic bugs --------------


def test_deterministic_gn_error_is_raised_not_swallowed():
    opt = AdaptiveHybridStreamingOptimizer(_cfg())
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise TypeError("a real bug")

    opt._gauss_newton_iteration = boom
    with pytest.raises(TypeError, match="a real bug"):
        opt._gn_iteration_with_retry((None, None), jnp.zeros(2), 1.0, jnp.zeros(2), 1.0)
    assert len(calls) == 1  # not retried


def test_numeric_gn_error_is_retried_then_reported():
    opt = AdaptiveHybridStreamingOptimizer(_cfg())
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise FloatingPointError("overflow")

    opt._gauss_newton_iteration = boom
    res, _ = opt._gn_iteration_with_retry(
        (None, None), jnp.zeros(2), 1.0, jnp.zeros(2), 1.0
    )
    assert len(calls) == opt.config.max_retries_per_batch + 1
    assert res["retry_exhausted"] and "FloatingPointError" in res["error"]


def test_deterministic_gn_error_surfaces_in_fit_result():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 500)
    y = 3 * np.exp(-0.7 * x) + 0.01 * rng.normal(size=x.size)
    opt = AdaptiveHybridStreamingOptimizer(_cfg(gauss_newton_max_iterations=6))

    def boom(*a, **k):
        raise TypeError("a real bug")

    opt._gauss_newton_iteration = boom
    r = opt.fit((x, y), _exp_model, [1.0, 1.0], verbose=0)
    assert r["success"] is False
    assert "TypeError" in r["message"] and "a real bug" in r["message"]


# --- model never sees more than chunk_size rows ---------------------------------


@pytest.mark.parametrize("p0", [[3.0, 0.7], [1.0, 0.2]], ids=["exact", "inexact"])
def test_hybrid_evaluates_model_in_chunks(p0, monkeypatch):
    # a batch that fits in memory is evaluated whole; force the over-budget route
    monkeypatch.setattr(
        "nlsq.streaming.phases.warmup._full_batch_fits", lambda n, k: False
    )
    n, chunk = 2048, 256
    x = np.linspace(0, 5, n)
    y = 3.0 * np.exp(-0.7 * x)

    def model(x, a, b):
        if x.shape[0] > 1024:  # static shape: raises at trace time
            raise ValueError(f"model called on {x.shape[0]} rows")
        return a * jnp.exp(-b * x)

    opt = AdaptiveHybridStreamingOptimizer(_cfg(chunk_size=chunk))
    r = opt.fit((x, y), model, p0, verbose=0)
    assert "Fell back" not in r["message"], r["message"]
    assert np.asarray(r["fun"]).shape == (n,)
    np.testing.assert_allclose(r["x"], [3.0, 0.7], rtol=1e-4)


def test_chunked_loss_fn_matches_whole_value_and_grad():
    import jax

    from nlsq.streaming.phases.warmup import build_warmup_loss, chunked_loss_fn

    loss = build_warmup_loss(
        lambda x, a, b: a * jnp.exp(-b * x),
        denormalize=lambda p: p,
        var_reg_enabled=False,
        var_lambda=0.0,
        var_indices=None,
    )
    x = jnp.linspace(0, 5, 1003)  # ragged tail
    y = 3 * jnp.exp(-0.7 * x) + 0.1
    p = jnp.array([2.0, 0.5])
    v0, g0 = jax.value_and_grad(loss)(p, x, y)
    v1, g1 = jax.value_and_grad(
        chunked_loss_fn(loss, 100, full_batch_fits=lambda n, k: False)
    )(p, x, y)
    np.testing.assert_allclose(v1, v0, rtol=1e-12)
    np.testing.assert_allclose(g1, g0, rtol=1e-10)


# --- L-BFGS backtracking step is not scaled by s**2 ---------------------------------


def test_backtracking_lbfgs_step_not_squared_by_initial_step_size():
    cfg = HybridStreamingConfig(lbfgs_line_search="backtracking")
    target = jnp.array([3.0, -2.0])

    def f(p):
        return 0.5 * jnp.sum((p - target) ** 2)

    p = jnp.zeros(2)
    opt, state = _create_lbfgs_optimizer(cfg, p, 0.1)
    for _ in range(5):
        v, g = jax_value_and_grad(f, p)
        upd, state = opt.update(g, state, p, value=v, grad=g, value_fn=f)
        p = optax.apply_updates(p, upd)
    # s**2 = 1% steps left f near 5.96 (from 6.5); the fix reaches well below.
    assert float(f(p)) < 3.0


def jax_value_and_grad(f, p):
    import jax

    return jax.value_and_grad(f)(p)


# --- phases/gauss_newton: truncated pseudo-inverse -------------------------------------


def test_phases_gauss_newton_zeroes_null_space():
    from unittest.mock import MagicMock

    phase = GaussNewtonPhase(HybridStreamingConfig(), MagicMock())
    jtj = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    jtr = jnp.array([1.0, -1.0])  # entirely along the null space
    step, _ = phase._solve_gauss_newton_step(
        jtj, jtr, trust_radius=1e6, regularization=0.0
    )
    assert float(jnp.linalg.norm(step)) < 1.0


# --- warmup telemetry tokens ------------------------------------------------------


def test_each_optimizer_warmup_gets_its_own_telemetry_token():
    from nlsq.streaming.telemetry import get_defense_telemetry, reset_defense_telemetry

    reset_defense_telemetry()
    x = np.linspace(0, 5, 400)
    y = 3 * np.exp(-0.7 * x)
    opts = [AdaptiveHybridStreamingOptimizer(_cfg(chunk_size=200)) for _ in range(2)]
    for o in opts:
        o.fit((x, y), _exp_model, [1.0, 0.2], verbose=0)
    t0, t1 = (o._warmup_token for o in opts)
    assert t0 is not None and t1 is not None and t0 != t1
    # Both warmups report a failure at iteration 0: two failed steps, not one.
    tel = get_defense_telemetry()
    before = tel.lbfgs_line_search_failed_steps
    tel.record_lbfgs_line_search_failure(0, "a", warmup=t0)
    tel.record_lbfgs_line_search_failure(0, "b", warmup=t1)
    assert tel.lbfgs_line_search_failed_steps == before + 2


# --- round 2: callback returning True stops the fit ---------------------------------


def test_callback_returning_true_stops_hybrid_fit():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 3000)
    y = 3 * np.exp(-0.7 * x) + 0.01 * rng.normal(size=x.size)
    calls = []

    def cb(iteration, cost, params, info):
        calls.append(iteration)
        return iteration >= 3

    opt = AdaptiveHybridStreamingOptimizer(_cfg(chunk_size=1000, callback_frequency=1))
    r = opt.fit((x, y), _exp_model, [1.0, 1.0], verbose=0, callback=cb)
    assert r["status"] == -2
    assert r["message"] == CALLBACK_STOP_MESSAGE
    assert calls[-1] == 3


# --- round 2: flat-valley trap must not be reported as success -------------------


def test_flat_valley_trap_is_not_reported_as_success():
    # p0 = [8, 0.2, 3] sends the L-BFGS warmup into the b ~ 0 valley (a, c -> +-inf,
    # a*(1 - b*x) + c ~ a straight line): SSR ~32x the optimum, relative gradient
    # ~1e-5 against ~1e-9 at the optimum. A cost-change stop there is not convergence.
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 3000)
    y = 2 * np.exp(-1.3 * x) + 0.5 + 0.05 * rng.normal(size=x.size)

    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    opt = AdaptiveHybridStreamingOptimizer(
        _cfg(
            chunk_size=1000,
            max_warmup_iterations=500,
            warmup_iterations=200,
            gauss_newton_max_iterations=100,
        )
    )
    r = opt.fit((x, y), model, [8.0, 0.2, 3.0], verbose=0)
    ssr = float(np.sum((y - np.asarray(model(x, *r["x"]))) ** 2))
    ssr_opt = float(np.sum((y - np.asarray(model(x, 2.0022, 1.3005, 0.4981))) ** 2))
    assert r["success"] is False or ssr < 1.01 * ssr_opt
    if not r["success"]:
        assert "not first-order stationary" in r["message"]


def test_converged_fit_still_reports_success():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 3000)
    y = 2 * np.exp(-1.3 * x) + 0.5 + 0.05 * rng.normal(size=x.size)

    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    r = AdaptiveHybridStreamingOptimizer(_cfg(chunk_size=1000)).fit(
        (x, y), model, [1.8, 1.2, 0.4], verbose=0
    )
    assert r["success"]
    np.testing.assert_allclose(r["x"], [2.0022, 1.3005, 0.4981], rtol=1e-3)


def test_trust_region_step_is_damped_not_shrunk_newton():
    opt = AdaptiveHybridStreamingOptimizer(_cfg())
    jtj = jnp.array([[1e6, 0.0], [0.0, 1e-4]])
    jtr = jnp.array([1e3, 1.0])
    radius = 0.5
    step, pred = opt._solve_gauss_newton_step(jtj, jtr, trust_radius=radius)
    assert abs(float(jnp.linalg.norm(step)) - radius) < 1e-6
    newton = jnp.array([1e3 / 1e6, 1.0 / 1e-4])
    shrunk = newton * (radius / float(jnp.linalg.norm(newton)))
    pred_shrunk = float(jtr @ shrunk - 0.5 * shrunk @ jtj @ shrunk)
    assert float(pred) >= pred_shrunk - 1e-12


# --- round 2 (Codex) --------------------------------------------------------------


def test_damped_step_is_euclidean_trust_region_optimum():
    # H = diag(1, 100), g = [1, 100], radius 1: the Euclidean TR optimum solves
    # (H + lam I) p = g with ||p|| = 1, p ~ [0.2436, 0.9699], gain ~ 50.17
    # (damping in Jacobi-scaled coordinates gave [0.7071, 0.7071], gain 46.17).
    opt = AdaptiveHybridStreamingOptimizer(_cfg())
    h = jnp.diag(jnp.array([1.0, 100.0]))
    g = jnp.array([1.0, 100.0])
    step, pred = opt._solve_gauss_newton_step(h, g, trust_radius=1.0)
    assert abs(float(jnp.linalg.norm(step)) - 1.0) < 1e-8
    np.testing.assert_allclose(np.asarray(step), [0.2436, 0.9699], atol=2e-4)
    assert float(pred) > 50.1


def _penalized_cfg():
    return HybridStreamingConfig(
        chunk_size=32,
        warmup_iterations=1,
        max_warmup_iterations=2,
        gauss_newton_max_iterations=30,
        gauss_newton_tol=1e-14,
        enable_group_variance_regularization=True,
        group_variance_lambda=0.1,
        group_variance_indices=[(0, 2)],
        enable_checkpoints=False,
        verbose=0,
    )


def test_stationarity_check_includes_group_variance_penalty():
    x = np.linspace(0, 1, 80)
    y = 2 + 3 * x + 0.01 * np.sin(40 * x)
    opt = AdaptiveHybridStreamingOptimizer(_penalized_cfg())
    r = opt.fit((x, y), lambda x, a, b: a + b * x, [1.0, 1.0], verbose=0)
    assert r["success"], r["message"]


@pytest.mark.parametrize("n", [128, 20])
def test_scalar_model_output_is_broadcast_in_final_residuals(n):
    x = np.arange(float(n))
    y = np.full(n, 2.0)
    opt = AdaptiveHybridStreamingOptimizer(_cfg(chunk_size=32))
    r = opt.fit((x, y), lambda x, a: a, [2.0], verbose=0)
    assert np.asarray(r["fun"]).shape == (n,)


def test_telemetry_does_not_evict_live_warmup_dedup_state():
    from nlsq.streaming.telemetry import DefenseLayerTelemetry

    tel = DefenseLayerTelemetry()
    t_long = tel.record_warmup_start()
    tel.record_lbfgs_line_search_failure(0, "a", warmup=t_long)
    for _ in range(70):  # many later warmups start and finish
        t = tel.record_warmup_start()
        tel.record_lbfgs_line_search_failure(0, "x", warmup=t)  # triggers eviction
        tel.record_warmup_end(t)
    before = tel.lbfgs_line_search_failed_steps
    tel.record_lbfgs_line_search_failure(0, "again", warmup=t_long)  # same step
    assert tel.lbfgs_line_search_failed_steps == before


# --- round 3 ----------------------------------------------------------------------


def _exp_c(x, a, b, c):
    return a * jnp.exp(-b * x) + c


@pytest.mark.serial
@pytest.mark.parametrize(
    "p0, cfg",
    [
        # plain iteration-budget stop (message is exactly "Maximum iterations reached")
        (
            [2.0, 1.3, 0.5],
            {"gauss_newton_max_iterations": 1},
        ),
        # flat-valley start: message gets the "not first-order stationary" suffix
        (
            [8.0, 0.2, 3.0],
            {"gauss_newton_max_iterations": 3},
        ),
    ],
    ids=["max_iter", "not_stationary"],
)
def test_curve_fit_large_hybrid_max_iterations_warns_not_raises(p0, cfg):
    from nlsq import curve_fit_large
    from nlsq.streaming.specs import Hybrid

    rng = np.random.default_rng(0)
    x = np.linspace(0, 4, 3000)
    y = 2 * np.exp(-1.3 * x) + 0.5 + 0.05 * rng.standard_normal(x.size)
    config = HybridStreamingConfig(
        chunk_size=1000, warmup_iterations=1, max_warmup_iterations=1, **cfg
    )
    with pytest.warns(OptimizeWarning):
        popt, _pcov = curve_fit_large(
            _exp_c, x, y, p0=p0, strategy=Hybrid(config=config, verbose=0)
        )
    assert np.all(np.isfinite(popt))


@pytest.mark.parametrize("fail", [False, True])
def test_warmup_phase_ends_its_telemetry_token(fail):
    from nlsq.streaming.phases.warmup import WarmupPhase
    from nlsq.streaming.telemetry import get_defense_telemetry

    class Model:
        normalizer = None

        def __call__(self, x, a, b):
            if fail:
                raise RuntimeError("boom")
            return a * x + b

    x = jnp.linspace(0.0, 1.0, 200)
    cfg = HybridStreamingConfig(
        normalize=False, chunk_size=64, warmup_iterations=2, max_warmup_iterations=3
    )
    tel = get_defense_telemetry()
    tel._live_warmups.clear()
    try:
        WarmupPhase(cfg, Model()).run((x, 2 * x + 1), jnp.array([1.0, 0.0]), [], {})
    except RuntimeError:
        assert fail
    assert tel._live_warmups == set()


def test_damped_step_survives_huge_finite_hessian():
    # H = diag(1e308, 1e306): 0.5*(H + H.T) overflowed and the old shrunk-Newton
    # direction [.7071, .7071] was returned silently (gain 4.62e307 < 5.02e307).
    opt = AdaptiveHybridStreamingOptimizer(_cfg())
    h = jnp.diag(jnp.array([1e308, 1e306]))
    g = jnp.array([1e308, 1e306])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        step, pred = opt._solve_gauss_newton_step(h, g, trust_radius=1.0)
    assert abs(float(jnp.linalg.norm(step)) - 1.0) < 1e-8
    np.testing.assert_allclose(np.asarray(step), [0.9699, 0.2436], atol=2e-4)
    assert float(pred) > 5.0e307


# --- tournament: sigma / multi-dimensional x ------------------------------------


def test_hybrid_tournament_batches_multi_x_points_first():
    # The engine contract is points-first (M, k); callers transpose scipy's (k, M).
    rng = np.random.default_rng(0)
    m = 600
    x = rng.uniform(0.1, 2.0, size=(m, 2))
    y = 2.0 * x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.01 * rng.standard_normal(m)

    seen = []

    def model(xc, a, b):
        seen.append(tuple(xc.shape))
        return a * xc[:, 0] + b * xc[:, 1] ** 2

    cfg = HybridStreamingConfig(
        chunk_size=100,
        enable_multistart=True,
        n_starts=4,
        elimination_rounds=1,
        batches_per_round=1,
        warmup_iterations=1,
        max_warmup_iterations=1,
        gauss_newton_max_iterations=5,
        verbose=0,
    )
    opt = AdaptiveHybridStreamingOptimizer(cfg)
    opt._setup_normalization(model, jnp.array([1.0, 1.0]), None)
    opt.multistart_candidates = jnp.array(
        [[1.0, 1.0], [2.0, 0.5], [3.0, 1.0], [0.5, 2.0]]
    )
    seen.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a failed tournament only warns
        best = opt._run_tournament_selection((x, y), model)
    assert np.all(np.isfinite(np.asarray(best)))
    assert opt.multistart_diagnostics.get("fallback") is not True
    assert seen and all(s[1] == 2 and s[0] <= 100 for s in seen), seen


@pytest.mark.parametrize("h, g", [(1e-200, 1e-20), (1e-300, 1e-20)])
def test_finite_oversized_newton_step_is_clipped_not_zeroed(h, g):
    # Newton step g/h ~ 1e180 (1e280): norm() overflows when squaring, and the old
    # code returned step 0 / gain 0 instead of the radius-1 trust-region step.
    opt = AdaptiveHybridStreamingOptimizer(_cfg())
    step, pred = opt._solve_gauss_newton_step(
        jnp.array([[h]]), jnp.array([g]), trust_radius=1.0
    )
    np.testing.assert_allclose(np.asarray(step), [1.0], rtol=1e-12)
    assert float(pred) == pytest.approx(g, rel=1e-6)


# --- L-BFGS warmup: norms of finite vectors above ~1e154 --------------------------


def test_clip_update_norm_clips_huge_finite_update_to_radius():
    from nlsq.streaming.phases.warmup import _clip_update_norm

    out = np.asarray(_clip_update_norm(jnp.array([3e160, 4e160]), 0.1))
    np.testing.assert_allclose(out, [0.06, 0.08], rtol=1e-9)
    # non-finite input still propagates; zero stays zero; small stays unchanged
    assert not np.all(
        np.isfinite(np.asarray(_clip_update_norm(jnp.array([np.inf, 1.0]), 0.1)))
    )
    np.testing.assert_array_equal(np.asarray(_clip_update_norm(jnp.zeros(2), 0.1)), 0.0)
    np.testing.assert_array_equal(
        np.asarray(_clip_update_norm(jnp.array([0.01, 0.02]), 0.1)), [0.01, 0.02]
    )


def test_lbfgs_step_with_gradient_above_1e154_is_clipped_not_zeroed():
    # optax's own line search returns a zero update at this scale, so drive the
    # step with a plain steepest-descent transform: the update is ~1e160 and the
    # Layer-4 clip must bring it to the radius (it used to give inf norm -> 0).
    def model(x, a, b):
        return a * x + b

    x = jnp.full(8, 1e80)
    y = jnp.zeros(8)
    opt = AdaptiveHybridStreamingOptimizer(
        _cfg(normalize=False, max_warmup_step_size=0.1)
    )
    opt._setup_normalization(model, jnp.array([1.0, 1.0]), None)
    loss_fn = opt._create_warmup_loss_fn()
    sgd = optax.GradientTransformationExtraArgs(
        lambda p: optax.EmptyState(), lambda g, st, p=None, **kw: (-g, st)
    )
    p0 = opt.normalized_params
    new_p, _loss, grad_norm, _st, _failed = opt._lbfgs_step(
        p0, optax.EmptyState(), sgd, loss_fn, x, y, 0
    )
    assert np.isfinite(grad_norm) and grad_norm > 1e154  # was inf
    moved = float(np.linalg.norm(np.asarray(new_p - p0)))
    assert moved == pytest.approx(0.1, rel=1e-6)  # was 0.0


# --- warmup perf: jitted norm, full-batch loss when the batch fits in memory ------


def test_safe_norm_is_jitted_and_matches_plain_norm():
    import jax

    from nlsq.streaming.phases.warmup import _safe_norm

    v = jnp.array([3.0, 4.0, 12.0])
    assert float(_safe_norm(v)) == float(jnp.linalg.norm(v))  # bit-identical
    assert float(jax.jit(lambda u: 2.0 * _safe_norm(u))(v)) == 26.0  # outer jit
    assert float(_safe_norm(jnp.array([3e160, 4e160]))) == pytest.approx(5e160)
    assert float(_safe_norm(jnp.zeros(3))) == 0.0
    assert not np.isfinite(float(_safe_norm(jnp.array([np.inf, 1.0]))))
    assert hasattr(_safe_norm, "lower")  # a jax.jit-wrapped function


def _chunk_loss_setup(n=5000, chunk=1000):
    from nlsq.streaming.phases.warmup import build_warmup_loss

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    cfg = _cfg(normalize=False, chunk_size=chunk)
    opt = AdaptiveHybridStreamingOptimizer(cfg)
    opt._setup_normalization(model, jnp.array([1.0, 1.0]), None)
    loss_fn = opt._create_warmup_loss_fn()
    rng = np.random.default_rng(0)
    x = jnp.asarray(np.linspace(0, 3, n))
    y = jnp.asarray(2 * np.exp(-0.7 * np.asarray(x)) + 0.01 * rng.standard_normal(n))
    return loss_fn, x, y, chunk


def test_chunked_and_full_batch_loss_routes_agree_and_limit_selects_chunked():
    import jax

    from nlsq.config import MemoryConfig, memory_context
    from nlsq.streaming.phases.warmup import _full_batch_fits, chunked_loss_fn

    loss_fn, x, y, chunk = _chunk_loss_setup()
    p = jnp.array([1.3, 0.4])

    full = chunked_loss_fn(loss_fn, chunk)  # default budget: the batch fits
    assert "scan" not in str(jax.make_jaxpr(full)(p, x, y))

    # 5000 points x ~100 B each is far above half of a 1 KB budget -> chunked
    with memory_context(MemoryConfig(memory_limit_gb=0.1, safety_factor=0.8)):
        assert not _full_batch_fits(1_000_000, 2)
    forced = chunked_loss_fn(loss_fn, chunk, full_batch_fits=lambda n, k: False)
    assert "scan" in str(jax.make_jaxpr(forced)(p, x, y))

    v_full, g_full = jax.value_and_grad(full)(p, x, y)
    v_ch, g_ch = jax.value_and_grad(forced)(p, x, y)
    assert float(v_full) == pytest.approx(float(v_ch), rel=1e-12)
    np.testing.assert_allclose(np.asarray(g_full), np.asarray(g_ch), rtol=1e-12)


def test_full_batch_fits_honours_memory_limit():
    from nlsq.config import MemoryConfig, memory_context
    from nlsq.streaming.phases.warmup import _full_batch_fits

    assert _full_batch_fits(1_000_000, 3)
    with memory_context(MemoryConfig(memory_limit_gb=0.1, safety_factor=0.8)):
        assert not _full_batch_fits(1_000_000, 3)  # ~146 MB > half of 0.08 GB
        assert _full_batch_fits(10_000, 3)


# --- standalone WarmupPhase shares the memory-aware chunked loss ---------------------


class _RowCapModel:
    normalizer = None

    def __init__(self, cap):
        self.cap = cap

    def __call__(self, x, a, b):
        if x.shape[0] > self.cap:  # static shape: raises at trace time
            raise ValueError(f"model called on {x.shape[0]} rows")
        return a * x + b


def _warmup_cfg(**kw):
    return HybridStreamingConfig(
        normalize=False,
        chunk_size=10_000,
        warmup_iterations=1,
        max_warmup_iterations=1,
        enable_warm_start_detection=False,
        enable_checkpoints=False,
        verbose=0,
        **kw,
    )


@pytest.mark.serial
def test_warmup_phase_never_evaluates_model_beyond_chunk_when_over_budget():
    from nlsq.config import MemoryConfig, memory_context
    from nlsq.streaming.phases.warmup import WarmupPhase

    n = 1_000_000
    x = jnp.arange(n, dtype=jnp.float64) / n
    y = 2 * x
    cfg = _warmup_cfg(enable_fault_tolerance=False)
    with memory_context(MemoryConfig(memory_limit_gb=0.1, safety_factor=0.8)):
        res = WarmupPhase(cfg, _RowCapModel(10_000)).run(
            (x, y), jnp.array([1.0, 0.0]), [], {}
        )
    assert np.all(np.isfinite(np.asarray(res["final_params"])))


def test_warmup_phase_chunked_and_full_routes_agree(monkeypatch):
    from nlsq.streaming.phases.warmup import WarmupPhase

    n = 30_000
    x = jnp.linspace(0.0, 1.0, n)
    y = 2 * x + 1 + 0.01 * jnp.sin(50 * x)
    out = {}
    for name, fits in (("full", lambda n, k: True), ("chunked", lambda n, k: False)):
        monkeypatch.setattr("nlsq.streaming.phases.warmup._full_batch_fits", fits)
        res = WarmupPhase(_warmup_cfg(), _RowCapModel(10**9)).run(
            (x, y), jnp.array([1.0, 0.0]), [], {}
        )
        out[name] = (np.asarray(res["final_params"]), res["final_loss"])
    np.testing.assert_allclose(out["full"][0], out["chunked"][0], rtol=1e-10)
    assert out["full"][1] == pytest.approx(out["chunked"][1], rel=1e-12)


# --- the full/chunked choice follows the CURRENT memory context ----------------------


def _recording_loss(seen):
    from nlsq.streaming.phases.warmup import build_warmup_loss

    def model(x, a, b):
        seen.append(x.shape[0])  # runs at trace time only
        return a * x + b

    return build_warmup_loss(
        model,
        denormalize=lambda p: p,
        var_reg_enabled=False,
        var_lambda=0.0,
        var_indices=None,
    )


@pytest.mark.parametrize("via", ["chunked_loss_fn", "lbfgs_step_fns"])
def test_memory_decision_is_made_on_every_call_not_once_per_shape(via):
    import jax

    from nlsq.config import MemoryConfig, memory_context
    from nlsq.streaming.phases.warmup import chunked_loss_fn, lbfgs_step_fns

    n, chunk = 1_000_000, 10_000
    x = jnp.linspace(0.0, 1.0, n)
    y = 2 * x
    p = jnp.array([1.0, 0.0])
    seen = []
    loss_fn = _recording_loss(seen)
    if via == "chunked_loss_fn":
        fn = chunked_loss_fn(loss_fn, chunk)
    else:
        vg, _upd = lbfgs_step_fns(loss_fn, optax.lbfgs(), chunk)
        fn = lambda p, x, y: vg(p, x, y)[0]

    with memory_context(MemoryConfig(memory_limit_gb=8.0, safety_factor=0.8)):
        v_full = fn(p, x, y)
    assert max(seen) == n  # the whole batch fits in 8 GB
    seen.clear()
    with memory_context(MemoryConfig(memory_limit_gb=0.1, safety_factor=0.8)):
        v_chunk = fn(p, x, y)  # same shapes, tighter budget
    assert seen and max(seen) <= chunk, seen
    assert float(v_full) == pytest.approx(float(v_chunk), rel=1e-12)

    # no recompilation under an unchanged context, in either mode
    seen.clear()
    with memory_context(MemoryConfig(memory_limit_gb=8.0, safety_factor=0.8)):
        fn(p, x, y)
        fn(p, x, y)
    with memory_context(MemoryConfig(memory_limit_gb=0.1, safety_factor=0.8)):
        fn(p, x, y)
        fn(p, x, y)
    assert seen == []
    del jax


# --- one memory decision per L-BFGS step; no accumulation across fits ----------------


def test_lbfgs_step_fns_use_the_decision_passed_in():
    from nlsq.streaming.phases.warmup import lbfgs_step_fns

    seen = []
    loss_fn = _recording_loss(seen)
    vg, upd = lbfgs_step_fns(loss_fn, optax.lbfgs(), 1000)
    x = jnp.linspace(0.0, 1.0, 5000)
    y = 2 * x
    p = jnp.array([1.0, 0.0])
    for full in (True, False):
        seen.clear()
        _v, g = vg(p, x, y, full=full)
        upd(g, optax.lbfgs().init(p), p, _v, x, y, full=full)
        assert (max(seen) == 5000) is full, (full, seen)


@pytest.mark.parametrize("which", ["adaptive", "warmup_phase"])
def test_lbfgs_step_makes_one_memory_decision(which, monkeypatch):
    from nlsq.streaming.phases import warmup as wmod
    from nlsq.streaming.phases.warmup import WarmupPhase

    calls = []
    monkeypatch.setattr(
        wmod, "_full_batch_fits", lambda n, k: calls.append(n) or len(calls) % 2 == 1
    )

    def model(x, a, b):
        return a * x + b

    x = jnp.linspace(0.0, 1.0, 5000)
    y = 2 * x + 1
    if which == "adaptive":
        opt = AdaptiveHybridStreamingOptimizer(_cfg(normalize=False, chunk_size=1000))
        opt._setup_normalization(model, jnp.array([0.1, 0.1]), None)
        loss_fn = opt._create_warmup_loss_fn()
        p = opt.normalized_params
        lbfgs, state = opt._create_lbfgs_optimizer(p)
        opt._lbfgs_step(p, state, lbfgs, loss_fn, x, y, 0)
    else:

        class Model:
            normalizer = None

            def __call__(self, x, a, b):
                return model(x, a, b)

        phase = WarmupPhase(_cfg(normalize=False, chunk_size=1000), Model())
        loss_fn = phase._create_loss_fn()
        p = jnp.array([0.1, 0.1])
        lbfgs, state = wmod._create_lbfgs_optimizer(phase.config, p, 0.1)
        phase._lbfgs_step(p, state, lbfgs, loss_fn, x, y, 0, {})
    assert len(calls) == 1, calls


def test_repeated_fits_on_one_optimizer_do_not_retain_old_loss_closures():
    import gc
    import weakref

    refs = []
    orig = AdaptiveHybridStreamingOptimizer._create_warmup_loss_fn

    def spy(self):
        fn = orig(self)
        refs.append(weakref.ref(fn))
        return fn

    x = np.linspace(0, 3, 400)
    y = 2 * np.exp(-0.7 * x)
    opt = AdaptiveHybridStreamingOptimizer(_cfg(chunk_size=100))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(AdaptiveHybridStreamingOptimizer, "_create_warmup_loss_fn", spy)
        for _ in range(4):
            opt.fit((x, y), lambda x, a, b: a * jnp.exp(-b * x), [1.0, 1.0], verbose=0)
    gc.collect()
    assert len(refs) >= 4
    assert sum(r() is not None for r in refs) <= 1
