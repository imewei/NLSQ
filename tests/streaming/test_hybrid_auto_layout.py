"""R6H-1: method='auto' forced onto the streaming strategy takes scipy (k, M) x."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit


def _model(x, a, b, c):
    return a * x[0] + b * jnp.exp(-c * x[1])


def _data(m=3000):
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (2, m))
    y = np.asarray(_model(X, 2.0, 1.5, 3.0)) + 0.05 * rng.standard_normal(m)
    return X, y


@pytest.mark.filterwarnings("ignore")
def test_auto_streaming_2d_x_matches_trf():
    X, y = _data()
    ref = curve_fit(_model, X, y, p0=[1, 1, 1])
    r = curve_fit(_model, X, y, p0=[1, 1, 1], method="auto", memory_limit_gb=1e-7)
    np.testing.assert_allclose(np.asarray(r.x), np.asarray(ref[0]), rtol=1e-2)
    assert np.all(np.isfinite(np.asarray(r["pcov"])))
    # result keeps the caller's (scipy) layout and model
    assert r["xdata"].shape == X.shape
    np.testing.assert_allclose(
        np.asarray(r["model"](r["xdata"], *r.x)), np.asarray(_model(X, *r.x))
    )


@pytest.mark.filterwarnings("ignore")
def test_auto_streaming_2d_x_return_eval_scipy_layout():
    X, y = _data()
    popt, _pcov, yfit = curve_fit(
        _model,
        X,
        y,
        p0=[1, 1, 1],
        method="auto",
        memory_limit_gb=1e-7,
        return_eval=True,
    )
    np.testing.assert_allclose(yfit, np.asarray(_model(X, *popt)), rtol=1e-10)


@pytest.mark.filterwarnings("ignore")
def test_auto_streaming_1d_x_unchanged():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 2000)
    y = 3.0 * np.exp(-0.7 * x) + 0.01 * rng.standard_normal(x.size)

    def f(x, a, b):
        return a * jnp.exp(-b * x)

    r = curve_fit(f, x, y, p0=[1, 1], method="auto", memory_limit_gb=1e-7)
    np.testing.assert_allclose(np.asarray(r.x), [3.0, 0.7], rtol=2e-2)
