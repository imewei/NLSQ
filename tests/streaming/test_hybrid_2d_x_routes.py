"""R5-S1: 2-D (k, M) scipy-layout x through the shared hybrid Strategy.fit."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit, fit
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.strategy import (
    build_large_strategy,
    scipy_layout_to_points_first,
)


def _model(x, a, b, c):
    return a * x[0] + b * jnp.exp(-c * x[1])


def _data(m=3000):
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (2, m))
    y = np.asarray(_model(X, 2.0, 1.5, 3.0)) + 0.05 * rng.standard_normal(m)
    return X, y


def _cfg():
    return HybridStreamingConfig(
        chunk_size=1000,
        warmup_iterations=20,
        gauss_newton_max_iterations=30,
        enable_checkpoints=False,
        verbose=0,
    )


def test_fit_workflow_hybrid_config_2d_x_matches_curve_fit():
    X, y = _data()
    ref_p, ref_c = curve_fit(_model, X, y, p0=[1, 1, 1])
    r = fit(_model, X, y, p0=[1, 1, 1], workflow=_cfg())
    assert r["success"]
    np.testing.assert_allclose(np.asarray(r["x"]), ref_p, rtol=1e-2)
    pcov = np.asarray(r["pcov"])
    assert np.all(np.isfinite(pcov))
    np.testing.assert_allclose(
        np.sqrt(np.diag(pcov)), np.sqrt(np.diag(ref_c)), rtol=0.2
    )


def test_strategy_fit_via_scipy_layout_helper_with_bounds():
    X, y = _data()
    strategy = build_large_strategy("hybrid", _cfg())
    f_engine, x_engine = scipy_layout_to_points_first(_model, X)
    native = strategy.fit(
        f_engine,
        x_engine,
        y,
        np.array([1.0, 1.0, 1.0]),
        bounds=(np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0])),
        callback=None,
        verbose=0,
    )
    x, pcov = strategy.unpack(native)
    np.testing.assert_allclose(x, [2.0, 1.5, 3.0], rtol=0.05)
    assert np.all(np.isfinite(pcov))


def test_strategy_fit_1d_x_unchanged():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 2000)
    y = 3.0 * np.exp(-0.7 * x) + 0.01 * rng.standard_normal(x.size)
    seen = []

    def f(x, a, b):
        seen.append(jnp.shape(x))
        return a * jnp.exp(-b * x)

    strategy = build_large_strategy("hybrid", _cfg())
    native = strategy.fit(f, x, y, np.array([1.0, 1.0]), verbose=0)
    assert native["success"]
    assert all(len(s) == 1 for s in seen)


@pytest.mark.filterwarnings("ignore:The .preset. parameter:FutureWarning")
def test_fit_streaming_preset_2d_x():
    X, y = _data(2000)
    r = fit(_model, X, y, p0=[1, 1, 1], preset="streaming", verbose=0)
    assert r["success"]
    np.testing.assert_allclose(np.asarray(r["x"]), [2.0, 1.5, 3.0], rtol=0.1)
    assert np.all(np.isfinite(np.asarray(r["pcov"])))


def test_layout_contract_both_routes_agree():
    """curve_fit(method='hybrid_streaming') takes (n, k); fit(workflow=...) takes (k, M)."""
    X, y = _data(2000)
    cfg = _cfg()

    def model_pf(x, a, b, c):  # points-first chunks (n, k)
        return a * x[:, 0] + b * jnp.exp(-c * x[:, 1])

    p_cf, _ = curve_fit(
        model_pf,
        X.T,
        y,
        p0=[1, 1, 1],
        method="hybrid_streaming",
        config=cfg,
        verbose=0,
    )
    r = fit(_model, X, y, p0=[1, 1, 1], workflow=cfg)
    np.testing.assert_allclose(np.asarray(r["x"]), np.asarray(p_cf), rtol=1e-6)


def test_scipy_layout_helper_noop_for_1d_and_transposes_2d():
    from nlsq.streaming.strategy import scipy_layout_to_points_first

    x1 = np.arange(5.0)
    f1, xo = scipy_layout_to_points_first(_model, x1)
    assert f1 is _model and xo is x1
    X = np.arange(12.0).reshape(2, 6)
    f2, xt = scipy_layout_to_points_first(_model, X)
    assert xt.shape == (6, 2) and f2.__wrapped__ is _model
    np.testing.assert_allclose(f2(xt, 1.0, 0.0, 0.0), X[0])
