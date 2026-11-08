"""Regression tests for the 2026-11-07 streaming review of adaptive_hybrid.py.

S1: Phase 2 Gauss-Newton with an active bound returned the warmup point.
S2: Phase 1 L-BFGS retraced optax's line search on every step.
S3: Phase 2 evaluated J^T J / J^T r / SSR twice at identical parameters.
S4: ``fit(verbose=0)`` still printed Phase 2 progress.
"""

import warnings

import jax
import jax.monitoring
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy.optimize import curve_fit

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig

pytestmark = pytest.mark.slow


def _model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


def _np_model(x, a, b, c):
    return a * np.exp(-b * x) + c


def _data(n=3003, seed=1):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 5, n))
    sig = rng.uniform(0.05, 0.3, n)
    y = 2.5 * np.exp(-1.3 * x) + 0.5 + rng.normal(0, 1, n) * sig
    return x, y


def _cfg(**kw):
    base = {
        "chunk_size": 1000,
        "warmup_iterations": 20,
        "max_warmup_iterations": 20,
        "verbose": 0,
        "enable_checkpoints": False,
    }
    return HybridStreamingConfig(**{**base, **kw})


def _ssr(x, y, p):
    return float(np.sum((y - _np_model(x, *p)) ** 2))


class TestActiveBoundsGaussNewton:
    def test_active_lower_bound_matches_curve_fit(self):
        x, y = _data()
        p0 = np.array([1.0, 1.0, 0.7])
        bounds = (np.array([0.0, 0.0, 0.6]), np.array([10.0, 10.0, 10.0]))
        r = AdaptiveHybridStreamingOptimizer(_cfg()).fit(
            (x, y), _model, p0=p0, bounds=bounds, verbose=0
        )
        p = np.asarray(r["x"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref, _ = curve_fit(
                _np_model, x, y, p0=p0, bounds=bounds, method="trf", ftol=1e-12
            )
        assert p[2] == pytest.approx(0.6)
        assert _ssr(x, y, p) <= _ssr(x, y, ref) * 1.01
        np.testing.assert_allclose(p, ref, rtol=2e-3)

    def test_active_bound_terminates_early_at_kkt_point(self):
        """Projected gradient must vanish at a bounded optimum (no max-iter spin)."""
        x, y = _data()
        bounds = (np.array([0.0, 0.0, 0.6]), np.array([10.0, 10.0, 10.0]))
        opt = AdaptiveHybridStreamingOptimizer(_cfg(gauss_newton_max_iterations=100))
        opt.fit((x, y), _model, p0=np.array([1.0, 1.0, 0.7]), bounds=bounds, verbose=0)
        rec = [r for r in opt.phase_history if r.get("name") == "gauss_newton"][-1]
        assert rec["convergence_reason"] != "Maximum iterations reached", rec

    @pytest.mark.parametrize(
        "bounds",
        [
            None,
            (np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0])),
        ],
        ids=["unbounded", "inactive-bounds"],
    )
    def test_unbounded_and_inactive_bounds_match_curve_fit(self, bounds):
        x, y = _data()
        p0 = np.array([1.0, 1.0, 0.0])
        r = AdaptiveHybridStreamingOptimizer(_cfg()).fit(
            (x, y), _model, p0=p0, bounds=bounds, verbose=0
        )
        ref, _ = curve_fit(_np_model, x, y, p0=p0)
        np.testing.assert_allclose(np.asarray(r["x"]), ref, rtol=1e-5)


def _compile_counter():
    counts = [0]
    active = [False]

    def listener(event, duration, **kw):
        if active[0] and event.endswith("backend_compile_duration"):
            counts[0] += 1

    jax.monitoring.register_event_duration_secs_listener(listener)
    return counts, active


class TestLbfgsStepJitted:
    def test_step_matches_eager_optax_trajectory(self):
        x, y = _data(500)
        opt = AdaptiveHybridStreamingOptimizer(_cfg(enable_step_clipping=False))
        opt._setup_normalization(_model, np.array([1.0, 1.0, 0.0]), None)
        loss_fn = opt._create_warmup_loss_fn()
        xj, yj = jnp.asarray(x), jnp.asarray(y)
        p = opt.normalized_params
        optimizer, st = opt._create_lbfgs_optimizer(p, initial_step_size=0.1)
        p_ref, st_ref = p, st
        for it in range(6):
            p, loss, _, st, _ = opt._lbfgs_step(p, st, optimizer, loss_fn, xj, yj, it)
            # eager reference (the pre-fix implementation)
            v, g = jax.value_and_grad(loss_fn)(p_ref, xj, yj)
            upd, st_ref = optimizer.update(
                g, st_ref, p_ref, value=v, grad=g, value_fn=lambda q: loss_fn(q, xj, yj)
            )
            p_ref = optax.apply_updates(p_ref, upd)
            if opt.normalized_bounds is not None:
                p_ref = jnp.clip(p_ref, *opt.normalized_bounds)
            assert float(loss) == pytest.approx(float(v), rel=1e-12)
            np.testing.assert_allclose(np.asarray(p), np.asarray(p_ref), rtol=1e-10)

    def test_no_per_step_recompilation(self):
        x, y = _data(500)
        counts, active = _compile_counter()
        # warm every fit-independent compile (model, loss, SVD, ...)
        AdaptiveHybridStreamingOptimizer(
            _cfg(warmup_iterations=3, max_warmup_iterations=3)
        ).fit((x, y), _model, p0=np.array([1.0, 1.0, 0.0]), verbose=0)
        opt = AdaptiveHybridStreamingOptimizer(
            _cfg(warmup_iterations=40, max_warmup_iterations=40)
        )
        steps = [0]
        orig = opt._lbfgs_step

        def counting_step(*a, **k):
            steps[0] += 1
            return orig(*a, **k)

        opt._lbfgs_step = counting_step
        active[0] = True
        try:
            opt.fit((x, y), _model, p0=np.array([1.0, 1.0, 0.0]), verbose=0)
        finally:
            active[0] = False
        assert steps[0] >= 10
        # Before the fix: 3 backend compiles per step (>= 3 * steps).
        assert counts[0] < steps[0], (counts[0], steps[0])


class TestPhase2PassReuse:
    def test_one_pass_per_accepted_iteration(self):
        x, y = _data(2000)
        opt = AdaptiveHybridStreamingOptimizer(
            _cfg(chunk_size=2000, gauss_newton_tol=1e-14, gauss_newton_max_iterations=8)
        )
        calls = []
        orig = opt._accumulate_pass

        def counting(params, *a, **k):
            calls.append(np.asarray(params).copy())
            return orig(params, *a, **k)

        opt._accumulate_pass = counting
        r = opt.fit((x, y), _model, p0=np.array([1.0, 1.0, 0.0]), verbose=0)
        n_iter = [h for h in opt.phase_history if h.get("name") == "gauss_newton"][-1][
            "iterations"
        ]
        # no two passes at bitwise-identical parameters
        for i in range(len(calls)):
            for j in range(i + 1, len(calls)):
                assert not np.array_equal(calls[i], calls[j]), (i, j)
        assert len(calls) <= n_iter + 2
        assert np.all(np.isfinite(np.asarray(r["x"])))


class TestVerboseZeroSilent:
    def test_fit_verbose_zero_prints_nothing(self, capsys):
        x, y = _data(1500)
        # config.verbose left at its default (1): the fit() argument must win
        cfg = HybridStreamingConfig(
            chunk_size=500,
            warmup_iterations=5,
            max_warmup_iterations=5,
            enable_checkpoints=False,
        )
        AdaptiveHybridStreamingOptimizer(cfg).fit(
            (x, y), _model, p0=np.array([1.0, 1.0, 0.0]), verbose=0
        )
        out = capsys.readouterr().out
        assert out == "", out
