"""Residual/Jacobian wrapper cache: reuse only when a re-trace would be identical.

curve_fit() builds a new LeastSquares per call; the module-level cache in
nlsq.core.least_squares lets repeated fits of the same model skip re-tracing
and re-compiling. These tests pin that results never depend on the cache.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.core import least_squares as ls_mod
from nlsq.core.least_squares import _wrapper_cache_key

X = np.linspace(0.0, 4.0, 200)
P0 = [1.0, 1.0]


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


def _fit(model, y):
    popt, pcov = curve_fit(model, X, y, p0=P0)
    return np.asarray(popt), np.asarray(pcov)


def test_cached_fit_matches_uncached_fit_bitwise():
    y = 2.5 * np.exp(-1.3 * X) + 0.01 * np.random.default_rng(0).standard_normal(X.size)
    ls_mod._wrapper_cache.clear()
    popt1, pcov1 = _fit(_exp, y)  # builds and stores the wrappers
    assert any(k[0] is _exp for k in ls_mod._wrapper_cache)
    popt2, pcov2 = _fit(_exp, y)  # reuses them
    np.testing.assert_array_equal(popt1, popt2)
    np.testing.assert_array_equal(pcov1, pcov2)


def test_changed_defaults_are_not_reused():
    def model(x, a, b, k=1.0):
        return k * a * jnp.exp(-b * x)

    y = 2.0 * np.exp(-0.5 * X)
    popt1, _ = _fit(model, y)
    model.__defaults__ = (2.0,)
    popt2, _ = _fit(model, y)
    np.testing.assert_allclose(popt1[0], 2.0, rtol=1e-6)
    np.testing.assert_allclose(popt2[0], 1.0, rtol=1e-6)


def test_same_code_different_captured_constants():
    def make(k):
        return lambda x, a, b: k * a * jnp.exp(-b * x)

    y = 2.0 * np.exp(-0.5 * X)
    for k in (1.0, 2.0, 4.0):
        popt, _ = _fit(make(k), y)
        np.testing.assert_allclose(popt[0], 2.0 / k, rtol=1e-6)


_SCALE = 1.0


def _global_model(x, a, b):
    return _SCALE * a * jnp.exp(-b * x)


def test_changed_global_is_not_reused(monkeypatch):
    y = 2.0 * np.exp(-0.5 * X)
    np.testing.assert_allclose(_fit(_global_model, y)[0][0], 2.0, rtol=1e-6)
    monkeypatch.setitem(_global_model.__globals__, "_SCALE", 4.0)
    np.testing.assert_allclose(_fit(_global_model, y)[0][0], 0.5, rtol=1e-6)


def test_mutable_captures_are_not_cached():
    arr = np.ones(1)
    lst = [1.0]
    assert _wrapper_cache_key(lambda x, a: arr[0] * a * x, 1, "fwd") is None
    assert _wrapper_cache_key(lambda x, a: lst[0] * a * x, 1, "fwd") is None

    # and the uncached path still follows in-place mutation
    def model(x, a, b):
        return arr[0] * a * jnp.exp(-b * x)

    y = 2.0 * np.exp(-0.5 * X)
    np.testing.assert_allclose(_fit(model, y)[0][0], 2.0, rtol=1e-6)
    arr[0] = 4.0
    np.testing.assert_allclose(_fit(model, y)[0][0], 0.5, rtol=1e-6)


@pytest.mark.parametrize(
    ("u", "v"), [(0.0, -0.0), (1, 1.0), (True, 1), (1.0, np.float64(1.0))]
)
def test_equal_but_distinct_constants_get_distinct_keys(u, v):
    def make(k):
        return lambda x, a: k * a * x

    f = make(u)
    g = make(v)
    g.__code__ = f.__code__
    key_f = _wrapper_cache_key(f, 1, "fwd")
    key_g = _wrapper_cache_key(g, 1, "fwd")
    assert key_f is not None and key_g is not None
    assert key_f[1:] != key_g[1:]


def test_non_function_callables_are_not_cached():
    class Model:
        def __call__(self, x, a):
            return a * x

    assert _wrapper_cache_key(Model(), 1, "fwd") is None
    assert _wrapper_cache_key(Model().__call__, 1, "fwd") is None
