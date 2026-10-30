"""Residual/Jacobian wrapper cache: reuse only when a re-trace would be identical.

curve_fit() builds a new LeastSquares per call; the module-level cache in
nlsq.core.least_squares lets repeated fits of the same model skip re-tracing
and re-compiling. These tests pin that results never depend on the cache.
"""

import dataclasses
import functools
import os
import types
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq import curve_fit
from nlsq.core import least_squares as ls_mod
from nlsq.core.least_squares import _wrapper_cache_key, clear_wrapper_cache

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
    assert any(k[0][0] is _exp for k in ls_mod._wrapper_cache)
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
    assert key_f[0][1:] != key_g[0][1:]  # same snapshot apart from the fn object


def test_non_function_callables_are_not_cached():
    class Model:
        def __call__(self, x, a):
            return a * x

    assert _wrapper_cache_key(Model(), 1, "fwd") is None
    assert _wrapper_cache_key(Model().__call__, 1, "fwd") is None


_TEMP = 1.0


def _boltz(x):
    return jnp.exp(-x / _TEMP)


def _helper_model(x, a, b):
    return a * _boltz(b * x)


def test_global_read_by_helper_is_not_reused(monkeypatch):
    y = 2.0 * np.exp(-0.5 * X)
    np.testing.assert_allclose(_fit(_helper_model, y)[0], [2.0, 0.5], rtol=1e-6)
    monkeypatch.setitem(_boltz.__globals__, "_TEMP", 2.0)
    np.testing.assert_allclose(_fit(_helper_model, y)[0], [2.0, 1.0], rtol=1e-6)


def _inner_lambda_model(x, a, b):
    scale = lambda v: _TEMP * v
    return scale(a) * jnp.exp(-b * x)


def test_global_read_by_nested_code_is_not_reused(monkeypatch):
    y = 2.0 * np.exp(-0.5 * X)
    np.testing.assert_allclose(_fit(_inner_lambda_model, y)[0][0], 2.0, rtol=1e-6)
    monkeypatch.setitem(_inner_lambda_model.__globals__, "_TEMP", 4.0)
    np.testing.assert_allclose(_fit(_inner_lambda_model, y)[0][0], 0.5, rtol=1e-6)


def test_class_captures_are_not_cached():
    class Params:
        k = 1.0

    assert _wrapper_cache_key(lambda x, a: Params.k * a * x, 1, "fwd") is None


# --- Fix round 1: default-deny probes. Each mutates state the model reads
# between two fits; the second fit must equal a cold fit (cache cleared).

Y2 = 2.0 * np.exp(-0.5 * X)


def _assert_second_fit_is_cold(model, mutate, expected_a):
    _fit(model, Y2)
    mutate()
    warm = _fit(model, Y2)[0]
    clear_wrapper_cache()
    cold = _fit(model, Y2)[0]
    np.testing.assert_array_equal(warm, cold)
    np.testing.assert_allclose(warm[0], expected_a, rtol=1e-6)


def test_probe_a_function_attribute():
    def model(x, a, b):
        return model.scale * a * jnp.exp(-b * x)

    model.scale = 1.0
    assert _wrapper_cache_key(model, 2, "fwd") is None
    _assert_second_fit_is_cold(model, lambda: setattr(model, "scale", 2.0), 1.0)


def test_probe_a_helper_function_attribute():
    def helper(v):
        return helper.scale * v

    helper.scale = 1.0

    def model(x, a, b):
        return helper(a) * jnp.exp(-b * x)

    assert _wrapper_cache_key(model, 2, "fwd") is None
    _assert_second_fit_is_cold(model, lambda: setattr(helper, "scale", 4.0), 0.5)


def test_probe_b_bound_builtin_methods():
    state = {"k": 1.0}
    getter = state.get
    arr = np.array([1.0])
    item, mean_fn = arr.item, arr.mean

    def m_get(x, a, b):
        return getter("k") * a * jnp.exp(-b * x)

    def m_item(x, a, b):
        return item() * a * jnp.exp(-b * x)

    def m_mean(x, a, b):
        return mean_fn() * a * jnp.exp(-b * x)

    for m in (m_get, m_item, m_mean):
        assert _wrapper_cache_key(m, 2, "fwd") is None
    _assert_second_fit_is_cold(m_get, lambda: state.update(k=2.0), 1.0)
    arr[0] = 1.0
    _assert_second_fit_is_cold(m_item, lambda: arr.__setitem__(0, 4.0), 0.5)
    arr[0] = 1.0
    _assert_second_fit_is_cold(m_mean, lambda: arr.__setitem__(0, 2.0), 1.0)


def test_probe_c_user_module_state():
    settings = types.ModuleType("settings")
    settings.scale = 1.0
    settings.PARAMS = {"K": 1.0}

    def m_attr(x, a, b):
        return settings.scale * a * jnp.exp(-b * x)

    def m_dict(x, a, b):
        return settings.PARAMS["K"] * a * jnp.exp(-b * x)

    def m_env(x, a, b):
        return float(os.environ.get("NLSQ_TEST_K", "1")) * a * jnp.exp(-b * x)

    for m in (m_attr, m_dict, m_env):
        assert _wrapper_cache_key(m, 2, "fwd") is None
    _assert_second_fit_is_cold(m_attr, lambda: setattr(settings, "scale", 2.0), 1.0)
    _assert_second_fit_is_cold(m_dict, lambda: settings.PARAMS.update(K=4.0), 0.5)


def test_probe_c_os_environ(monkeypatch):
    def m_env(x, a, b):
        return float(os.environ.get("NLSQ_TEST_K", "1")) * a * jnp.exp(-b * x)

    monkeypatch.delenv("NLSQ_TEST_K", raising=False)
    _assert_second_fit_is_cold(
        m_env, lambda: monkeypatch.setenv("NLSQ_TEST_K", "2"), 1.0
    )


@dataclasses.dataclass(frozen=True)
class _Cfg:
    k: float = 1.0
    factor: ClassVar[float] = 1.0


def test_probe_d_frozen_dataclass_classvar(monkeypatch):
    cfg = _Cfg()

    def model(x, a, b):
        return cfg.factor * cfg.k * a * jnp.exp(-b * x)

    assert _wrapper_cache_key(model, 2, "fwd") is None
    _assert_second_fit_is_cold(
        model, lambda: monkeypatch.setattr(_Cfg, "factor", 2.0), 1.0
    )


@pytest.mark.parametrize(
    "value",
    [
        np.timedelta64(1, "s"),
        np.datetime64(1, "s"),
        np.void(b"\x00" * 8),
        np.str_("a"),
    ],
)
def test_probe_e_numpy_scalars_outside_allowlist(value):
    assert _wrapper_cache_key(lambda x, a: (value, a * x)[1], 1, "fwd") is None


def test_probe_e_numpy_scalar_dtype_is_part_of_key():
    def make(k):
        return lambda x, a: k * a * x

    f, g = make(np.float64(1.0)), make(np.float32(1.0))
    g.__code__ = f.__code__
    assert (
        _wrapper_cache_key(f, 1, "fwd")[0][1:] != _wrapper_cache_key(g, 1, "fwd")[0][1:]
    )


def test_probe_f_custom_builtins():
    ns = {"__builtins__": {"K": 1.0, "float": float}, "jnp": jnp}
    exec("def model(x, a, b):\n    return K * a * jnp.exp(-b * x)\n", ns)
    model = ns["model"]
    assert _wrapper_cache_key(model, 2, "fwd") is None
    _assert_second_fit_is_cold(model, lambda: ns["__builtins__"].update(K=2.0), 1.0)


def _plain_jnp_model(x, a, b):
    return a * jnp.exp(-b * x) + 0.0 * jnp.sin(x)


def test_allowlisted_model_is_cached_and_reused():
    clear_wrapper_cache()
    _fit(_plain_jnp_model, Y2)
    assert len(ls_mod._wrapper_cache) == 1
    (bundle,) = ls_mod._wrapper_cache.values()
    _fit(_plain_jnp_model, Y2)
    assert len(ls_mod._wrapper_cache) == 1
    (again,) = ls_mod._wrapper_cache.values()
    assert again is bundle  # reused, not rebuilt


def test_global_named_like_attribute_does_not_block_caching():
    # A module that did `from jax.numpy import exp` has a global `exp` and the
    # attribute name `exp` in `jnp.exp`; only the bytecode's global loads
    # matter. (Kept in its own namespace so the other tests' modules are clean.)
    ns = {"jnp": jnp, "exp": jnp.exp, "np": np}
    exec(
        "def attr_model(x, a, b):\n    return a * jnp.exp(-b * x)\n"
        "def global_model(x, a, b):\n    return a * exp(-b * x)\n"
        "def ufunc_model(x, a, b):\n    return a * jnp.exp(-b * x) + 0.0 * np.pi\n",
        ns,
    )
    for name in ("attr_model", "global_model", "ufunc_model"):
        assert _wrapper_cache_key(ns[name], 2, "fwd") is not None, name
    np.testing.assert_allclose(_fit(ns["global_model"], Y2)[0], [2.0, 0.5], rtol=1e-6)


def test_nlsq_clear_cache_empties_wrapper_cache():
    _fit(_exp, Y2)
    assert len(ls_mod._wrapper_cache) >= 1
    nlsq.clear_cache()
    assert len(ls_mod._wrapper_cache) == 0


def test_env_var_disables_wrapper_cache(monkeypatch):
    clear_wrapper_cache()
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    assert _wrapper_cache_key(_exp, 2, "fwd") is None
    np.testing.assert_allclose(_fit(_exp, Y2)[0], [2.0, 0.5], rtol=1e-6)
    assert len(ls_mod._wrapper_cache) == 0


def test_loop_lambdas_and_partials_are_correct():
    for k in (1.0, 2.0):
        popt = _fit(lambda x, a, b, k=k: k * a * jnp.exp(-b * x), Y2)[0]
        np.testing.assert_allclose(popt[0], 2.0 / k, rtol=1e-6)

    def model(x, a, b, k):
        return k * a * jnp.exp(-b * x)

    for k in (1.0, 4.0):
        part = functools.partial(model, k=k)
        assert _wrapper_cache_key(part, 2, "fwd") is None
        popt = _fit(part, Y2)[0]
        np.testing.assert_allclose(popt[0], 2.0 / k, rtol=1e-6)
