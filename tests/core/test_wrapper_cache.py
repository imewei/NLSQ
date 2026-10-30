"""Residual/Jacobian wrapper cache: a hit requires an identical traced computation.

curve_fit() builds a new LeastSquares per call; nlsq.core.least_squares keeps
the compiled wrappers of a model keyed by its traced jaxpr, so any model state
read at trace time is part of the key. Each probe below changes such state
between fits and checks that the fit misses the cache and equals a cold fit.
"""

import contextvars
import dataclasses
import functools
import os
import sys
import types
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq import curve_fit
from nlsq.core import least_squares as ls_mod
from nlsq.core.least_squares import clear_wrapper_cache


@pytest.fixture(autouse=True)
def _clean_cache_state(monkeypatch):
    """Start every test with the cache enabled and empty, whatever the shell sets."""
    monkeypatch.delenv("NLSQ_DISABLE_WRAPPER_CACHE", raising=False)
    clear_wrapper_cache()


X = np.linspace(0.0, 4.0, 200)
Y2 = 2.0 * np.exp(-0.5 * X)
P0 = [1.0, 1.0]


def _fit(model, y=Y2, **kw):
    popt, pcov = curve_fit(model, X, y, p0=P0, **kw)
    return np.asarray(popt), np.asarray(pcov)


def _counted_fit(model, **kw):
    """(popt, number of new cache entries): 1 = miss, 0 = hit or uncacheable."""
    before = len(ls_mod._wrapper_cache)
    popt = _fit(model, **kw)[0]
    return popt, len(ls_mod._wrapper_cache) - before


def _cacheable(model, n=2):
    return (
        ls_mod._trace_model(model, jnp.asarray(X), jnp.asarray(Y2), np.ones(n), "fwd")
        is not None
    )


def _assert_state_change_misses(model, mutate, expected_a):
    """Miss, hit, mutate -> miss; the post-mutation fit equals a cold fit."""
    clear_wrapper_cache()
    assert _cacheable(model)
    assert _counted_fit(model)[1] == 1
    assert _counted_fit(model)[1] == 0
    mutate()
    warm, new = _counted_fit(model)
    assert new == 1
    clear_wrapper_cache()
    cold = _fit(model)[0]
    np.testing.assert_array_equal(warm, cold)
    np.testing.assert_allclose(warm[0], expected_a, rtol=1e-6)


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


@pytest.mark.slow
def test_cached_fit_matches_uncached_fit_bitwise(monkeypatch):
    y = 2.5 * np.exp(-1.3 * X) + 0.01 * np.random.default_rng(0).standard_normal(X.size)
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    popt0, pcov0 = _fit(_exp, y)
    monkeypatch.delenv("NLSQ_DISABLE_WRAPPER_CACHE")
    clear_wrapper_cache()
    popt1, pcov1 = _fit(_exp, y)  # miss: builds from the jaxpr
    popt2, pcov2 = _fit(_exp, y)  # hit
    for a, b in ((popt0, popt1), (pcov0, pcov1), (popt1, popt2), (pcov1, pcov2)):
        np.testing.assert_array_equal(a, b)


# --- earlier rounds' probes ------------------------------------------------


def test_changed_defaults():
    def model(x, a, b, k=1.0):
        return k * a * jnp.exp(-b * x)

    _assert_state_change_misses(
        model, lambda: setattr(model, "__defaults__", (2.0,)), 1.0
    )


def test_same_code_different_captured_constants():
    def make(k):
        return lambda x, a, b: k * a * jnp.exp(-b * x)

    for k in (1.0, 2.0, 4.0):
        np.testing.assert_allclose(_fit(make(k))[0][0], 2.0 / k, rtol=1e-6)


_SCALE = 1.0


def _global_model(x, a, b):
    return _SCALE * a * jnp.exp(-b * x)


def test_changed_global(monkeypatch):
    _assert_state_change_misses(
        _global_model,
        lambda: monkeypatch.setitem(_global_model.__globals__, "_SCALE", 4.0),
        0.5,
    )


_TEMP = 1.0


def _boltz(x):
    return jnp.exp(-x / _TEMP)


def _helper_model(x, a, b):
    return a * _boltz(b * x)


def test_global_read_by_helper(monkeypatch):
    _assert_state_change_misses(
        _helper_model,
        lambda: monkeypatch.setitem(_boltz.__globals__, "_TEMP", 2.0),
        2.0,
    )


def _inner_lambda_model(x, a, b):
    scale = lambda v: _TEMP * v
    return scale(a) * jnp.exp(-b * x)


def test_global_read_by_nested_code(monkeypatch):
    _assert_state_change_misses(
        _inner_lambda_model,
        lambda: monkeypatch.setitem(_inner_lambda_model.__globals__, "_TEMP", 4.0),
        0.5,
    )


def test_mutable_array_and_list_captures():
    arr = np.ones(1)
    lst = [1.0]

    def m_arr(x, a, b):
        return arr[0] * a * jnp.exp(-b * x)

    def m_lst(x, a, b):
        return lst[0] * a * jnp.exp(-b * x)

    _assert_state_change_misses(m_arr, lambda: arr.__setitem__(0, 4.0), 0.5)
    _assert_state_change_misses(m_lst, lambda: lst.__setitem__(0, 2.0), 1.0)


def test_captured_array_constant_is_part_of_key():
    w = jnp.ones(X.size)

    def model(x, a, b):
        return w * a * jnp.exp(-b * x)

    def mutate():
        nonlocal w
        w = 2.0 * jnp.ones(X.size)

    _assert_state_change_misses(model, mutate, 1.0)


def test_callable_object_and_class_capture():
    class Params:
        k = 1.0

    class Model:
        def __init__(self):
            self.k = 1.0

        def __call__(self, x, a, b):
            return self.k * Params.k * a * jnp.exp(-b * x)

    m = Model()
    _assert_state_change_misses(m, lambda: setattr(m, "k", 2.0), 1.0)
    _assert_state_change_misses(m, lambda: setattr(Params, "k", 2.0), 0.5)


def test_probe_a_function_attribute():
    def model(x, a, b):
        return model.scale * a * jnp.exp(-b * x)

    model.scale = 1.0
    _assert_state_change_misses(model, lambda: setattr(model, "scale", 2.0), 1.0)


def test_probe_a_helper_function_attribute():
    def helper(v):
        return helper.scale * v

    helper.scale = 1.0

    def model(x, a, b):
        return helper(a) * jnp.exp(-b * x)

    _assert_state_change_misses(model, lambda: setattr(helper, "scale", 4.0), 0.5)


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

    _assert_state_change_misses(m_get, lambda: state.update(k=2.0), 1.0)
    _assert_state_change_misses(m_item, lambda: arr.__setitem__(0, 4.0), 0.5)
    arr[0] = 1.0
    _assert_state_change_misses(m_mean, lambda: arr.__setitem__(0, 2.0), 1.0)


def test_probe_c_user_module_state(monkeypatch):
    settings = types.ModuleType("settings")
    settings.scale = 1.0
    settings.PARAMS = {"K": 1.0}

    def m_attr(x, a, b):
        return settings.scale * a * jnp.exp(-b * x)

    def m_dict(x, a, b):
        return settings.PARAMS["K"] * a * jnp.exp(-b * x)

    def m_env(x, a, b):
        return float(os.environ.get("NLSQ_TEST_K", "1")) * a * jnp.exp(-b * x)

    _assert_state_change_misses(m_attr, lambda: setattr(settings, "scale", 2.0), 1.0)
    _assert_state_change_misses(m_dict, lambda: settings.PARAMS.update(K=4.0), 0.5)
    monkeypatch.delenv("NLSQ_TEST_K", raising=False)
    _assert_state_change_misses(
        m_env, lambda: monkeypatch.setenv("NLSQ_TEST_K", "2"), 1.0
    )


@dataclasses.dataclass(frozen=True)
class _Cfg:
    k: float = 1.0
    factor: ClassVar[float] = 1.0


@pytest.mark.slow
def test_probe_d_frozen_dataclass_classvar(monkeypatch):
    cfg = _Cfg()

    def model(x, a, b):
        return cfg.factor * cfg.k * a * jnp.exp(-b * x)

    _assert_state_change_misses(
        model, lambda: monkeypatch.setattr(_Cfg, "factor", 2.0), 1.0
    )


def test_probe_e_numpy_scalar_dtype():
    k = np.float64(1.0)

    def model(x, a, b):
        return k * a * jnp.exp(-b * x)

    def mutate():
        nonlocal k
        k = np.float32(2.0)

    _assert_state_change_misses(model, mutate, 1.0)


def test_probe_f_custom_builtins():
    ns = {"__builtins__": {"K": 1.0}, "jnp": jnp}
    exec("def model(x, a, b):\n    return K * a * jnp.exp(-b * x)\n", ns)
    _assert_state_change_misses(
        ns["model"], lambda: ns["__builtins__"].update(K=2.0), 1.0
    )


# --- round 2 probes (N1-N7) ------------------------------------------------


@pytest.fixture
def mymod(monkeypatch):
    mod = types.ModuleType("nlsq_test_mymod")
    mod.K = 1.0
    monkeypatch.setitem(sys.modules, "nlsq_test_mymod", mod)
    return mod


def test_n1_import_inside_model_body(mymod):
    def m_import(x, a, b):
        import nlsq_test_mymod

        return nlsq_test_mymod.K * a * jnp.exp(-b * x)

    def m_from(x, a, b):
        from nlsq_test_mymod import K

        return K * a * jnp.exp(-b * x)

    _assert_state_change_misses(m_import, lambda: setattr(mymod, "K", 2.0), 1.0)
    mymod.K = 1.0
    _assert_state_change_misses(m_from, lambda: setattr(mymod, "K", 4.0), 0.5)


_DYN = 1.0


@pytest.mark.slow
def test_n2_dynamic_reads(monkeypatch, mymod, tmp_path):
    def m_globals(x, a, b):
        return globals()["_DYN"] * a * jnp.exp(-b * x)

    def m_eval(x, a, b):
        return eval("_DYN") * a * jnp.exp(-b * x)

    def m_dunder_import(x, a, b):
        return __import__("nlsq_test_mymod").K * a * jnp.exp(-b * x)

    path = tmp_path / "k.txt"
    path.write_text("1.0")

    def m_file(x, a, b):
        return float(path.read_text()) * a * jnp.exp(-b * x)

    cv = contextvars.ContextVar("k", default=1.0)
    get = cv.get

    def m_ctx(x, a, b):
        return get() * a * jnp.exp(-b * x)

    g = m_globals.__globals__
    _assert_state_change_misses(
        m_globals, lambda: monkeypatch.setitem(g, "_DYN", 2.0), 1.0
    )
    monkeypatch.setitem(g, "_DYN", 1.0)
    _assert_state_change_misses(
        m_eval, lambda: monkeypatch.setitem(g, "_DYN", 4.0), 0.5
    )
    _assert_state_change_misses(m_dunder_import, lambda: setattr(mymod, "K", 2.0), 1.0)
    _assert_state_change_misses(m_file, lambda: path.write_text("4.0"), 0.5)
    token = cv.set(1.0)
    try:
        _assert_state_change_misses(m_ctx, lambda: cv.set(2.0), 1.0)
    finally:
        cv.reset(token)


def test_n3_numpy_scalar_subclass_with_mutable_attribute():
    class Gain(np.float64):
        pass

    gain = Gain(1.0)
    gain.mult = 1.0

    def model(x, a, b):
        return gain * gain.mult * a * jnp.exp(-b * x)

    _assert_state_change_misses(model, lambda: setattr(gain, "mult", 2.0), 1.0)


def test_n4_library_state_through_allowlisted_module():
    saved = np.get_printoptions()

    def model(x, a, b):
        return np.get_printoptions()["precision"] / 8.0 * a * jnp.exp(-b * x)

    try:
        np.set_printoptions(precision=8)
        _assert_state_change_misses(
            model, lambda: np.set_printoptions(precision=16), 1.0
        )
    finally:
        np.set_printoptions(**saved)


_GAIN = 1.0


def _own_globals_model(x, a, b):
    return _own_globals_model.__globals__["_GAIN"] * a * jnp.exp(-b * x)


def test_n5_function_metadata_introspection(monkeypatch):
    def helper(v, gain: float = 1.0):
        return v

    helper.__annotations__["gain"] = 1.0

    def m_ann(x, a, b):
        return helper.__annotations__["gain"] * a * jnp.exp(-b * x)

    _assert_state_change_misses(
        _own_globals_model,
        lambda: monkeypatch.setitem(_own_globals_model.__globals__, "_GAIN", 2.0),
        1.0,
    )
    _assert_state_change_misses(
        m_ann, lambda: helper.__annotations__.update(gain=4.0), 0.5
    )


def test_n6_descriptors_only_run_during_the_traces():
    class Cfg:
        reads = 0

        @property
        def scale(self):
            Cfg.reads += 1
            return float(2 ** (Cfg.reads - 1))  # 1.0, 2.0, 4.0, ... per read

        def __getattr__(self, name):
            Cfg.reads += 100
            raise AttributeError(name)

    cfg = Cfg()

    def model(x, a, b):
        return cfg.scale * a * jnp.exp(-b * x)

    clear_wrapper_cache()
    popt, new = _counted_fit(model)
    # One trace per fit (no custom rule, so no Jacobian trace); the cache never
    # inspects the object, so __getattr__ is never reached.
    assert Cfg.reads == 1
    assert new == 1
    np.testing.assert_allclose(popt[0], 2.0, rtol=1e-6)  # traced at 1.0
    popt, new = _counted_fit(model)
    assert Cfg.reads == 2
    assert new == 1  # the property returned a new value: miss
    np.testing.assert_allclose(popt[0], 1.0, rtol=1e-6)


def test_n7_function_builtins_differ_from_globals_builtins():
    ns = {"__builtins__": {"K": 1.0}, "jnp": jnp}
    exec("def model(x, a, b):\n    return K * a * jnp.exp(-b * x)\n", ns)
    model = ns["model"]
    real = model.__builtins__
    ns["__builtins__"] = {"K": 99.0}  # does not affect the function
    _assert_state_change_misses(model, lambda: real.update(K=2.0), 1.0)


# --- models that must stay cached (hit on the second fit) --------------------

_MODULE_CONST = 1.0


def _module_const_model(x, a, b):
    return _MODULE_CONST * a * jnp.exp(-b * x)


def _norm_model(x, a, b):
    from jax.numpy.linalg import norm

    return a * jnp.exp(-b * x) + 0.0 * norm(x)


def _float64_model(x, a, b):
    from jax.numpy import float64

    return float64(a) * jnp.exp(-b * x)


def _wraps(f):
    @functools.wraps(f)
    def inner(*args):
        return f(*args)

    return inner


_wrapped_model = _wraps(_exp)


def _make_closure_const(k):
    return lambda x, a, b: k * a * jnp.exp(-b * x)


@pytest.mark.parametrize(
    "model",
    [
        _exp,
        _module_const_model,
        _norm_model,
        _float64_model,
        _wrapped_model,
        _make_closure_const(1.0),
        lambda x, a, b: a * jax.nn.relu(jnp.exp(-b * x)),
    ],
    ids=["jnp", "module_const", "linalg_norm", "float64", "wraps", "closure", "relu"],
)
def test_unchanged_models_hit(model):
    clear_wrapper_cache()
    assert _cacheable(model)
    popt1, new1 = _counted_fit(model)
    (bundle,) = ls_mod._wrapper_cache.values()
    popt2, new2 = _counted_fit(model)
    assert (new1, new2) == (1, 0)
    assert next(iter(ls_mod._wrapper_cache.values())) is bundle
    np.testing.assert_array_equal(popt1, popt2)
    np.testing.assert_allclose(popt1, [2.0, 0.5], rtol=1e-6)


def test_lambda_recreated_every_call_hits():
    clear_wrapper_cache()
    news = [_counted_fit(lambda x, a, b: a * jnp.exp(-b * x))[1] for _ in range(3)]
    assert news == [1, 0, 0]


def test_partials_hit_and_stay_correct():
    def model(x, a, b, k):
        return k * a * jnp.exp(-b * x)

    clear_wrapper_cache()
    for k in (1.0, 4.0):
        popt = _fit(functools.partial(model, k=k))[0]
        np.testing.assert_allclose(popt[0], 2.0 / k, rtol=1e-6)
    assert _counted_fit(functools.partial(model, k=4.0))[1] == 0


def test_user_jac_is_not_cached_and_follows_state():
    k = [1.0]

    def jac(x, a, b):
        e = jnp.exp(-b * x)
        return jnp.stack([k[0] * e, -k[0] * a * x * e])

    def model(x, a, b):
        return k[0] * a * jnp.exp(-b * x)

    clear_wrapper_cache()
    popt, new = _counted_fit(model, jac=jac)
    assert new == 0
    np.testing.assert_allclose(popt, [2.0, 0.5], rtol=1e-6)
    k[0] = 2.0
    np.testing.assert_allclose(_fit(model, jac=jac)[0], [1.0, 0.5], rtol=1e-6)
    assert len(ls_mod._wrapper_cache) == 0


def test_effects_and_callbacks_are_not_cached(capfd):
    def m_print(x, a, b):
        jax.debug.print("a={a}", a=a)
        return a * jnp.exp(-b * x)

    def m_callback(x, a, b):
        e = jax.pure_callback(np.exp, jax.ShapeDtypeStruct(x.shape, x.dtype), -b * x)
        return a * e

    assert not _cacheable(m_callback)  # (not differentiable, so not fitted)
    assert not _cacheable(m_print)
    clear_wrapper_cache()
    popt, new = _counted_fit(m_print)
    assert new == 0
    np.testing.assert_allclose(popt, [2.0, 0.5], rtol=1e-6)


def test_nlsq_clear_cache_empties_wrapper_cache():
    _fit(_exp)
    assert len(ls_mod._wrapper_cache) >= 1
    nlsq.clear_cache()
    assert len(ls_mod._wrapper_cache) == 0


def test_env_var_disables_wrapper_cache(monkeypatch):
    clear_wrapper_cache()
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    assert not _cacheable(_exp)
    np.testing.assert_allclose(_fit(_exp)[0], [2.0, 0.5], rtol=1e-6)
    assert len(ls_mod._wrapper_cache) == 0


def test_cached_wrappers_never_call_the_user_function():
    calls = [0]

    def model(x, a, b):
        calls[0] += 1
        return a * jnp.exp(-b * x)

    clear_wrapper_cache()
    _fit(model)
    _fit(model)
    assert calls[0] == 2  # one cache-key trace per fit (no custom rule)


@pytest.mark.parametrize("disable", ["1", "0"])
def test_reused_instance_with_changed_parameter_count(monkeypatch, disable):
    from nlsq.core.least_squares import LeastSquares

    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", disable)

    def poly(x, *p):
        return sum(c * x**i for i, c in enumerate(p))

    y = 1.0 + 2.0 * X + 0.5 * X**2
    ls = LeastSquares()
    xd, yd = jnp.asarray(X), jnp.asarray(y)
    r2 = ls.least_squares(poly, np.ones(2), xdata=xd, ydata=yd)
    r3 = ls.least_squares(poly, np.ones(3), xdata=xd, ydata=yd)
    assert r2.x.shape == (2,)
    np.testing.assert_allclose(r3.x, [1.0, 2.0, 0.5], rtol=1e-6)


# --- round 3: structural key (Codex F1-F7, Opus D1) --------------------------

XS = np.linspace(0.1, 2.0, 40)


def _fit_xs(model, y, **kw):
    popt, _ = curve_fit(model, XS, y, p0=kw.pop("p0", [1.0]), **kw)
    return np.asarray(popt)


def _cached_vs_uncached(model, y, monkeypatch, **kw):
    """(cached popt, new cache entries, popt with the cache disabled)."""
    before = len(ls_mod._wrapper_cache)
    cached = _fit_xs(model, y, **dict(kw))
    new = len(ls_mod._wrapper_cache) - before
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    uncached = _fit_xs(model, y, **dict(kw))
    monkeypatch.delenv("NLSQ_DISABLE_WRAPPER_CACHE")
    return cached, new, uncached


def test_f1_nested_jit_constants_are_keyed(monkeypatch):
    def make(k):
        c = np.full_like(XS, k)

        @jax.jit
        def transform(z):
            return c * z

        return lambda z, a: a * transform(z)

    clear_wrapper_cache()
    _fit_xs(make(1.0), 6 * XS)
    cached, new, uncached = _cached_vs_uncached(make(2.0), 6 * XS, monkeypatch)
    assert new == 1
    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_allclose(cached, [3.0], rtol=1e-9)


def test_f2_constants_are_private_snapshots(monkeypatch):
    c = np.ones_like(XS)

    def first(z, a):
        return a * z + c

    clear_wrapper_cache()
    _fit_xs(first, 3 * XS + 1)
    c[:] = 2.0  # mutate the array the first entry was traced with
    fresh = np.ones_like(XS)

    def second(z, a):
        return a * z + fresh

    sigma = np.ones_like(XS)
    cached, new, uncached = _cached_vs_uncached(
        second, 3 * XS + 1, monkeypatch, sigma=sigma
    )
    assert new == 0  # same computation as the first entry's snapshot: hit
    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_allclose(cached, [3.0], rtol=1e-9)


def test_f3_custom_jvp_rule_state_is_keyed(monkeypatch):
    slope = {"zero": 1.0}
    relu = jax.custom_jvp(jax.nn.relu)

    @relu.defjvp
    def rule(primals, tangents):
        (z,), (dz,) = primals, tangents
        s = jnp.where(z > 0, 1.0, jnp.where(z < 0, 0.0, slope["zero"]))
        return jax.nn.relu(z), s * dz

    def model(z, a):
        return relu(a * z)

    clear_wrapper_cache()
    _fit_xs(model, 3 * XS, p0=[0.0])
    slope["zero"] = 0.0
    cached, new, uncached = _cached_vs_uncached(model, 3 * XS, monkeypatch, p0=[0.0])
    assert new == 1
    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_allclose(cached, [0.0])


def test_f3_custom_vjp_rule_state_is_keyed_in_rev_mode(monkeypatch):
    monkeypatch.setenv("NLSQ_JACOBIAN_MODE", "rev")
    slope = {"zero": 1.0}

    @jax.custom_vjp
    def relu(z):
        return jax.nn.relu(z)

    def fwd(z):
        return jax.nn.relu(z), z

    def bwd(z, g):
        s = jnp.where(z > 0, 1.0, jnp.where(z < 0, 0.0, slope["zero"]))
        return (s * g,)

    relu.defvjp(fwd, bwd)

    def model(z, a):
        return relu(a * z)

    clear_wrapper_cache()
    _fit_xs(model, 3 * XS, p0=[0.0])
    slope["zero"] = 0.0
    cached, new, uncached = _cached_vs_uncached(model, 3 * XS, monkeypatch, p0=[0.0])
    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_allclose(cached, [0.0])
    assert new == 1  # the bwd rule state is in the (jacrev) key


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["defjvp", "defjvps"])
def test_d1_user_jvp_rule_reading_a_global(monkeypatch, variant):
    state = {"sc": 1.0}
    g = jax.custom_jvp(jnp.exp)
    if variant == "defjvp":
        g.defjvp(lambda p, t: (jnp.exp(p[0]), state["sc"] * jnp.exp(p[0]) * t[0]))
    else:
        g.defjvps(lambda t, primal_out, z: state["sc"] * primal_out * t)
    x = np.linspace(0, 4, 200)
    y = 2 * np.exp(-0.7 * x) + 0.01 * np.sin(9 * x)

    def model(z, a, b):
        return a * g(-b * z)

    def fit():
        popt, pcov = curve_fit(model, x, y, p0=(1.0, 1.0), max_nfev=3)
        return np.asarray(popt), np.asarray(pcov)

    clear_wrapper_cache()
    fit()
    state["sc"] = 5.0
    before = len(ls_mod._wrapper_cache)
    cached = fit()
    assert len(ls_mod._wrapper_cache) - before == 1  # rule state is in the key
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    uncached = fit()
    np.testing.assert_array_equal(cached[0], uncached[0])
    np.testing.assert_array_equal(cached[1], uncached[1])


def test_f4_literal_bits_are_keyed(monkeypatch):
    def make(gain):
        bits = np.float64(np.nan).view(np.uint64) | np.uint64(int(gain))
        lit = bits.view(np.float64)  # NaN whose payload encodes the gain

        def model(z, a):
            g = jax.lax.bitcast_convert_type(lit, jnp.uint64) & jnp.uint64(3)
            return g.astype(jnp.float64) * a * z

        return model

    clear_wrapper_cache()
    _fit_xs(make(1), 6 * XS)
    cached, new, uncached = _cached_vs_uncached(make(2), 6 * XS, monkeypatch)
    assert new == 1
    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_allclose(cached, [3.0], rtol=1e-9)


_ARRAY_LITERAL_SCRIPT = """
import os
import numpy as np
import jax.numpy as jnp
from nlsq import curve_fit
from nlsq.core import least_squares as ls
x = np.linspace(.1, 2., 40)
def make(k):
    c = jnp.full_like(jnp.asarray(x), k)
    return lambda z, a: a * c * z
def fit(f):
    p, _ = curve_fit(f, x, 6 * x, p0=[1.])
    return p
ls.clear_wrapper_cache()
fit(make(1.))
g = make(2.)
n0 = len(ls._wrapper_cache)
cached = fit(g)
assert len(ls._wrapper_cache) == n0 + 1, "array literal change must miss"
os.environ["NLSQ_DISABLE_WRAPPER_CACHE"] = "1"
uncached = fit(g)
np.testing.assert_array_equal(cached, uncached)
np.testing.assert_allclose(cached, [3.0], rtol=1e-9)
print("OK")
"""


@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.parametrize("simplified", ["1", "0"])
def test_f4_array_literals_are_keyed(simplified, tmp_path):
    import subprocess

    env = dict(os.environ, JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=simplified)
    env.update(NLSQ_SKIP_GPU_CHECK="1")
    env.pop("NLSQ_DISABLE_WRAPPER_CACHE", None)
    proc = subprocess.run(
        [sys.executable, "-c", _ARRAY_LITERAL_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout


def test_f5_user_primitives_are_not_cached(monkeypatch):
    from jax.extend import core as jex_core
    from jax.interpreters import ad, mlir

    def make(k):
        prim = jex_core.Primitive("scale_probe")
        prim.def_impl(lambda z: k * z)
        prim.def_abstract_eval(lambda aval: aval)
        mlir.register_lowering(
            prim, mlir.lower_fun(lambda z: k * z, multiple_results=False)
        )
        ad.primitive_jvps[prim] = lambda ps, ts: (prim.bind(ps[0]), k * ts[0])
        return lambda z, a: prim.bind(a * z)

    clear_wrapper_cache()
    _fit_xs(make(1.0), 6 * XS)
    cached, new, uncached = _cached_vs_uncached(make(2.0), 6 * XS, monkeypatch)
    assert new == 0 and len(ls_mod._wrapper_cache) == 0  # refused
    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_allclose(cached, [3.0], rtol=1e-9)


def test_f6_weak_type_input_is_traced_as_delivered(monkeypatch):
    from nlsq.core.least_squares import LeastSquares

    x = jnp.broadcast_to(jnp.asarray(2.0), (40,))
    assert x.weak_type
    y = 6 * x

    def model(z, a):
        return (2.0 if z.weak_type else 1.0) * a * z

    def fit():
        return LeastSquares().least_squares(model, np.ones(1), xdata=x, ydata=y).x

    clear_wrapper_cache()
    cached = fit()
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    uncached = fit()
    np.testing.assert_array_equal(cached, uncached)


def test_f7_linear_solve_is_cacheable(monkeypatch):
    def model(z, a):
        return a * jnp.linalg.solve(2 * jnp.eye(2), jnp.stack((z, z)))[0]

    clear_wrapper_cache()
    cached, new, uncached = _cached_vs_uncached(model, 6 * XS, monkeypatch)
    assert new == 1 and len(ls_mod._wrapper_cache) > 0
    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_allclose(cached, [12.0], rtol=1e-9)


# --- round 4 (Codex phase 3d) -------------------------------------------------


def _assert_cached_and_hits(model, y, p0=(1.0,)):
    clear_wrapper_cache()
    popt1, _ = curve_fit(model, XS, y, p0=list(p0))
    assert len(ls_mod._wrapper_cache) == 1
    (bundle,) = ls_mod._wrapper_cache.values()
    popt2, _ = curve_fit(model, XS, y, p0=list(p0))
    assert len(ls_mod._wrapper_cache) == 1
    assert next(iter(ls_mod._wrapper_cache.values())) is bundle  # hit
    np.testing.assert_array_equal(popt1, popt2)
    return np.asarray(popt1)


def test_b1_namedtuple_params_keyed_by_type_and_fields(monkeypatch):
    from collections import namedtuple

    fields_a = "offset_dims collapsed_slice_dims start_index_map"
    fields_b = "offset_dims start_index_map collapsed_slice_dims"
    tail = " operand_batching_dims start_indices_batching_dims"
    A = namedtuple("Dims", fields_a + tail)
    B = namedtuple("Dims", fields_b + tail)
    da = A((1,), (0,), (1,), (), ())
    db = B((1,), (0,), (1,), (), ())
    x = np.array([[1.0, 2.0], [4.0, 8.0]])
    y = np.array([6.0, 12.0])
    idx = jnp.array([[0], [1]], dtype=jnp.int32)

    def make(d):
        return lambda z, a: a * jax.lax.gather(z, idx, d, (1, 1)).ravel()

    def fit(f):
        popt, _ = curve_fit(f, x, y, p0=[1.0])
        return np.asarray(popt)

    clear_wrapper_cache()
    fit(make(da))
    g = make(db)
    before = len(ls_mod._wrapper_cache)
    cached = fit(g)
    assert len(ls_mod._wrapper_cache) == before + 1  # different class: miss
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    np.testing.assert_array_equal(cached, fit(g))


@jax.tree_util.register_pytree_node_class
class _Scaled:
    def __init__(self, data, scale=2.0):
        self.data = data
        self.scale = scale

    def tree_flatten(self):
        return (self.data,), self.scale

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], aux)

    def __sub__(self, other):
        return self.scale * self.data - other

    def __jax_array__(self):
        return self.scale * self.data

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.scale * self.data, dtype=dtype)


def test_b2_pytree_output_is_not_cached(monkeypatch):
    def model(z, a):
        return _Scaled(a * z)

    clear_wrapper_cache()
    cached = _fit_xs(model, 6 * XS)
    assert len(ls_mod._wrapper_cache) == 0  # refused: non-array output
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    np.testing.assert_array_equal(cached, _fit_xs(model, 6 * XS))


def test_p1_jax_operation_objects_and_policies_are_cached():
    knots = jnp.array([0.0, 1.0, 2.0, 3.0])

    def m_interp(z, a):
        return a * jnp.interp(z, knots, knots)

    def m_policy(z, a):
        f = jax.checkpoint(
            lambda t: a * t, policy=jax.checkpoint_policies.everything_saveable
        )
        return f(z)

    np.testing.assert_allclose(_assert_cached_and_hits(m_interp, 6 * XS), [6.0])
    np.testing.assert_allclose(_assert_cached_and_hits(m_policy, 6 * XS), [6.0])


def test_p1_user_checkpoint_policy_is_refused(monkeypatch):
    calls = {"save": True}

    def policy(prim, *args, **params):
        return calls["save"]

    def model(z, a):
        return jax.checkpoint(lambda t: a * t, policy=policy)(z)

    clear_wrapper_cache()
    cached = _fit_xs(model, 6 * XS)
    assert len(ls_mod._wrapper_cache) == 0
    np.testing.assert_allclose(cached, [6.0])


def test_p2_frozenset_params_are_cached():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.array(jax.devices()), ("d",))
    sharding = NamedSharding(mesh, PartitionSpec())

    def model(z, a):
        return jax.lax.with_sharding_constraint(a * z, sharding)

    np.testing.assert_allclose(_assert_cached_and_hits(model, 6 * XS), [6.0])


def test_p3_typed_prng_key_constants_are_cached(monkeypatch):
    def make(seed):
        key = jax.random.key(seed)
        return lambda z, a: a * z + 0.01 * jax.random.normal(key, z.shape)

    y = 6 * XS + 0.01 * np.asarray(jax.random.normal(jax.random.key(0), XS.shape))
    popt = _assert_cached_and_hits(make(0), y)
    np.testing.assert_allclose(popt, [6.0], rtol=1e-9)
    before = len(ls_mod._wrapper_cache)
    cached = _fit_xs(make(1), y)  # different key bits: miss
    assert len(ls_mod._wrapper_cache) == before + 1
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    np.testing.assert_array_equal(cached, _fit_xs(make(1), y))


# --- round 5 (Codex phase 3e, Opus E1) ----------------------------------------


def _assert_refused_and_matches_uncached(model, y, monkeypatch):
    clear_wrapper_cache()
    cached = _fit_xs(model, y)
    assert len(ls_mod._wrapper_cache) == 0
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    np.testing.assert_array_equal(cached, _fit_xs(model, y))
    return cached


def test_e1_partial_params_are_refused(monkeypatch):
    from jax.tree_util import Partial

    def make(save):
        pol = Partial(lambda *a, **k: save)
        return lambda z, a: jax.checkpoint(lambda t: a * t, policy=pol)(z)

    with pytest.raises(ls_mod._Uncacheable):
        ls_mod._canon_param(Partial(lambda u: u))
    with pytest.raises(ls_mod._Uncacheable):
        ls_mod._canon_param(functools.partial(lambda u: u))
    _assert_refused_and_matches_uncached(make(True), 6 * XS, monkeypatch)
    _assert_refused_and_matches_uncached(make(False), 6 * XS, monkeypatch)


def test_c3_jax_function_params_need_module_identity(monkeypatch):
    import jax._src.ad_checkpoint as mod

    def policy(prim, *args, gain=1.0, **params):
        return True

    policy.__module__ = "jax.numerical_probe"  # module string never imported
    with pytest.raises(ls_mod._Uncacheable):
        ls_mod._canon_param(policy)
    policy.__module__ = mod.__name__  # real module, but not its attribute
    with pytest.raises(ls_mod._Uncacheable):
        ls_mod._canon_param(policy)

    def model(z, a):
        return jax.checkpoint(lambda t: a * t, policy=policy)(z)

    _assert_refused_and_matches_uncached(model, 6 * XS, monkeypatch)

    monkeypatch.setattr(mod, policy.__name__, policy, raising=False)
    key = ls_mod._canon_param(policy)[1]
    policy.__kwdefaults__ = {"gain": 2.0}
    assert ls_mod._canon_param(policy)[1] != key


def test_c3_spoofed_module_callable_object_is_refused(monkeypatch):
    class Op:
        __module__ = "jax._src.numpy.hijax"

        def __call__(self, z):
            return z

    with pytest.raises(ls_mod._Uncacheable):
        ls_mod._canon_param(Op())


def test_c4_custom_prng_implementation_is_refused(monkeypatch):
    from jax.extend import random as ext_random

    def make_impl(name):
        return ext_random.define_prng_impl(
            key_shape=(1,),
            seed=lambda n: jnp.asarray([n], dtype=jnp.uint32),
            split=lambda k, shape: jax.random.bits(
                jax.random.key(k[0]), shape, dtype=jnp.uint32
            )[..., None],
            random_bits=lambda k, width, shape: jax.random.bits(
                jax.random.key(k[0]), shape, dtype=jnp.uint32
            ),
            fold_in=lambda k, n: k + jnp.asarray(n, dtype=jnp.uint32),
            name=name,
            tag=name[:3],
        )

    def make(key):
        def model(z, a):
            gain = jax.random.bits(key, z.shape, dtype=jnp.uint32) & jnp.uint32(3)
            return (gain.astype(z.dtype) + 0.5) * a * z

        return model

    for name in ("xor_random", "threefry2x32"):  # user impl, incl. a spoofed name
        key = jax.random.key(0, impl=make_impl(name))
        with pytest.raises(ls_mod._Uncacheable):
            ls_mod._array_key(key)
        model = make(key)
        y = np.asarray(model(jnp.asarray(XS), 6.0))
        monkeypatch.delenv("NLSQ_DISABLE_WRAPPER_CACHE", raising=False)
        _assert_refused_and_matches_uncached(model, y, monkeypatch)
    for name in ("threefry2x32", "rbg", "unsafe_rbg"):
        assert ls_mod._array_key(jax.random.key(0, impl=name))[1] == name


@jax.tree_util.register_pytree_node_class
class _AbstractAwareScaled(_Scaled):
    @classmethod
    def tree_unflatten(cls, aux, children):
        if type(children[0]) is jax.ShapeDtypeStruct:
            return children[0]  # lies to return_shape=True
        return cls(children[0], aux)


def test_c5_output_structure_comes_from_the_trace(monkeypatch):
    def model(z, a):
        return _AbstractAwareScaled(a * z)

    assert type(jax.jit(model)(XS, 1.0)) is _AbstractAwareScaled
    cached = _assert_refused_and_matches_uncached(model, 6 * XS, monkeypatch)
    np.testing.assert_allclose(cached, [3.0])


@pytest.mark.slow
def test_c6_jax_config_is_part_of_the_key(monkeypatch):
    key = jax.random.key(0)

    def model(z, a):
        return a * z + jax.random.uniform(key, z.shape, dtype=z.dtype)

    clear_wrapper_cache()
    with jax.threefry_partitionable(True):
        _fit_xs(model, 6 * XS)
        _fit_xs(model, 6 * XS)
        assert len(ls_mod._wrapper_cache) == 1  # same config: hit
    with jax.threefry_partitionable(False):
        cached = _fit_xs(model, 6 * XS)
        assert len(ls_mod._wrapper_cache) == 2  # config change: miss
        monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
        np.testing.assert_array_equal(cached, _fit_xs(model, 6 * XS))


def test_c6_unrelated_flag_toggle_restores_a_hit():
    def model(z, a):
        return a * z

    popt = _assert_cached_and_hits(model, 6 * XS)
    with jax.default_matmul_precision("highest"):
        _fit_xs(model, 6 * XS)
        assert len(ls_mod._wrapper_cache) == 2  # other config: miss
    keys = set(ls_mod._wrapper_cache)
    np.testing.assert_array_equal(_fit_xs(model, 6 * XS), popt)
    assert set(ls_mod._wrapper_cache) == keys  # back to the first config: hit


def test_cache_is_lru_bounded(monkeypatch):
    """The size bound evicts the least recently used entry, not the newest."""
    monkeypatch.setattr(ls_mod, "_WRAPPER_CACHE_SIZE", 2)
    m1, m2, m3, m4 = (_make_closure_const(k) for k in (1.0, 2.0, 3.0, 4.0))

    def keys_after(model):
        _fit(model)
        assert len(ls_mod._wrapper_cache) <= 2
        return set(ls_mod._wrapper_cache)

    k1 = keys_after(m1)
    k12 = keys_after(m2)
    (key2,) = k12 - k1
    k23 = keys_after(m3)
    assert key2 in k23 and not (k1 & k23)  # m1 was evicted
    assert keys_after(m2) == k23  # hit: m2 is now the most recent
    k24 = keys_after(m4)
    assert key2 in k24 and len(k24) == 2  # m3, the LRU entry, was evicted
