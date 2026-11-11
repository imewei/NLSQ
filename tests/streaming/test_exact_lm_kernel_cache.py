"""The exact solver reuses its jitted chunk kernel across calls for identically tracing models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nlsq.streaming.exact_lm as lm
from nlsq.core import least_squares as ls

x = np.linspace(0, 4, 2000)
y = 2.5 * np.exp(-1.3 * x) + 0.5
P0 = [1.0, 1.0, 0.0]
BLOCK = jax.ShapeDtypeStruct((x.size,), jnp.float64)


def _model():
    # a fresh closure per call: same computation, different Python object
    return lambda x, a, b, c: a * jnp.exp(-b * x) + c


@pytest.fixture(autouse=True)
def _clean_cache():
    ls.clear_wrapper_cache()
    yield
    ls.clear_wrapper_cache()


def test_identical_models_share_one_kernel():
    m1, s1 = lm._cached_stats(_model(), BLOCK, 3, False)
    m2, s2 = lm._cached_stats(_model(), BLOCK, 3, False)
    assert s1 is s2 and m1 is m2


def test_weighted_and_shape_are_part_of_the_key():
    _, s = lm._cached_stats(_model(), BLOCK, 3, False)
    _, sw = lm._cached_stats(_model(), BLOCK, 3, True)
    _, s2 = lm._cached_stats(
        _model(), jax.ShapeDtypeStruct((10,), jnp.float64), 3, False
    )
    assert len({id(s), id(sw), id(s2)}) == 3


def test_mutated_closure_data_misses_and_fits_the_new_data():
    """A hit must never evaluate stale captured data."""
    scale = np.array([1.0])

    def model(x, a, b, c):
        return scale[0] * (a * jnp.exp(-b * x) + c)

    r1 = lm.exact_lm_fit(model, x, y, P0)
    scale[0] = 2.0  # same function object, different captured value
    r2 = lm.exact_lm_fit(model, x, y, P0)
    np.testing.assert_allclose(r1.x, [2.5, 1.3, 0.5], rtol=1e-6)
    np.testing.assert_allclose(r2.x, [1.25, 1.3, 0.25], rtol=1e-6)


def test_cache_disabled_returns_the_callers_model(monkeypatch):
    monkeypatch.setenv("NLSQ_DISABLE_WRAPPER_CACHE", "1")
    m = _model()
    got, _ = lm._cached_stats(m, BLOCK, 3, False)
    assert got is m
    r = lm.exact_lm_fit(m, x, y, P0)
    np.testing.assert_allclose(r.x, [2.5, 1.3, 0.5], rtol=1e-6)


def test_uncacheable_model_still_fits():
    def model(x, a, b, c):
        jax.debug.callback(lambda: None)  # effects cannot be keyed exactly
        return a * jnp.exp(-b * x) + c

    got, _ = lm._cached_stats(model, BLOCK, 3, False)
    assert got is model
    r = lm.exact_lm_fit(model, x, y, P0)
    np.testing.assert_allclose(r.x, [2.5, 1.3, 0.5], rtol=1e-6)


def test_clear_cache_drops_the_kernel():
    _, s1 = lm._cached_stats(_model(), BLOCK, 3, False)
    ls.clear_wrapper_cache()
    _, s2 = lm._cached_stats(_model(), BLOCK, 3, False)
    assert s1 is not s2


def test_second_fit_does_not_recompile():
    import logging

    seen = []

    class H(logging.Handler):
        def emit(self, record):
            if "compilation of jit(stats)" in record.getMessage():
                seen.append(record.getMessage())

    h = H()
    # the JAX config is part of the cache key: log for both fits
    with jax.log_compiles(True):
        logging.getLogger("jax").addHandler(h)
        try:
            lm.exact_lm_fit(_model(), x, y, P0)
            assert seen, "the first fit compiles the kernel"
            seen.clear()
            lm.exact_lm_fit(_model(), x, y, P0)
        finally:
            logging.getLogger("jax").removeHandler(h)
    assert seen == []


@pytest.mark.parametrize("cap,cached", [(10_000_000, True), (1_000, False)])
def test_large_captured_constants_skip_the_cache(monkeypatch, cap, cached):
    """A captured table above the byte cap is neither copied into the cache nor kept."""
    monkeypatch.setattr(lm, "_CACHE_CONST_BYTES", cap)
    table = np.linspace(0.0, 1.0, x.size)  # 16 kB float64, captured by the model

    def model(xx, a, b):
        # pointwise lookup: x is linspace(0, 4, n), so this index is the row number
        i = jnp.round(xx * (table.size - 1) / 4.0).astype(jnp.int32)
        return a * jnp.asarray(table)[i] + b

    got, _ = lm._cached_stats(model, BLOCK, 2, False)
    assert (got is not model) is cached
    assert (len(ls._wrapper_cache) == 1) is cached
    r = lm.exact_lm_fit(model, x, 2.0 * table + 1.0, [1.0, 0.0])
    np.testing.assert_allclose(r.x, [2.0, 1.0], rtol=1e-8)


def _lookup(table):
    def inner(xx, a, b):
        i = jnp.round(xx * (table.size - 1) / 4.0).astype(jnp.int32)
        return a * jnp.asarray(table)[i] + b

    return inner


@pytest.mark.parametrize("cap,cached", [(10_000_000, True), (1_000, False)])
def test_capture_inside_nested_jit_counts_toward_the_cap(monkeypatch, cap, cached):
    monkeypatch.setattr(lm, "_CACHE_CONST_BYTES", cap)
    inner = jax.jit(_lookup(np.linspace(0.0, 1.0, x.size)))  # 16 kB in a nested jaxpr

    def model(xx, a, b):
        return inner(xx, a, b)

    scalar = jax.ShapeDtypeStruct((), jnp.float64)
    closed = jax.make_jaxpr(model)(BLOCK, scalar, scalar)
    assert sum(c.nbytes for c in closed.consts) < 1_000  # invisible at the top level
    got, _ = lm._cached_stats(model, BLOCK, 2, False)
    assert (got is not model) is cached
    assert (len(ls._wrapper_cache) == 1) is cached


def test_growing_capture_is_measured_on_a_fresh_trace(monkeypatch):
    monkeypatch.setattr(lm, "_CACHE_CONST_BYTES", 1_000)
    state = {"t": np.linspace(0.0, 1.0, 10)}  # 80 B: under the cap

    def model(xx, a, b):
        return _lookup(state["t"])(xx, a, b)

    got, _ = lm._cached_stats(model, BLOCK, 2, False)
    assert got is not model  # small capture: cached
    ls.clear_wrapper_cache()
    state["t"] = np.linspace(
        0.0, 1.0, x.size
    )  # 16 kB: over the cap, same function object
    got, _ = lm._cached_stats(model, BLOCK, 2, False)
    assert got is model and len(ls._wrapper_cache) == 0


def test_budget_is_scoped_to_the_exact_cache():
    """The byte budget is set only around the exact kernel digest, never left behind."""
    lm._cached_stats(_model(), BLOCK, 3, False)
    assert ls._snapshot_budget.get() is None


@pytest.mark.parametrize("cap,cached", [(10 * 16_000, True), (3 * 16_000, False)])
def test_one_table_in_several_nested_jaxprs_counts_each_copy(monkeypatch, cap, cached):
    """The digest snapshots the table once per nested jaxpr: five copies, not one."""
    table = np.linspace(0.0, 1.0, x.size)  # 16 kB
    fns = [jax.jit(_lookup(table)) for _ in range(5)]

    def model(xx, a, b):
        return sum(f(xx, a, b) for f in fns) / 5.0

    monkeypatch.setattr(lm, "_CACHE_CONST_BYTES", cap)
    got, _ = lm._cached_stats(model, BLOCK, 2, False)
    assert (got is not model) is cached


def _rule_model(state):
    """`a*x` whose custom JVP reads live Python state (a 1-element array)."""

    @jax.custom_jvp
    def f(xx, a):
        return a * xx

    @f.defjvp
    def _jvp(primals, tangents):
        xx, a = primals
        _, da = tangents
        return a * xx, state[0] * xx * da

    return f


def test_custom_jvp_models_are_not_cached():
    """The rule is a live callable JAX re-invokes at compile time: never shared."""
    owner_state, victim_state = np.array([2.0]), np.array([2.0])
    owner, victim = _rule_model(owner_state), _rule_model(victim_state)
    got, s_owner = lm._cached_stats(owner, BLOCK, 1, False)
    assert got is owner and len(ls._wrapper_cache) == 0
    _, s_victim = lm._cached_stats(victim, BLOCK, 1, False)
    owner_state[0] = 5.0  # before either kernel compiles
    ones, mask = jnp.ones(x.size), jnp.ones(x.size, bool)
    *_, h_victim = s_victim(jnp.ones(1), ones, ones, mask)
    *_, h_owner = s_owner(jnp.ones(1), ones, ones, mask)
    np.testing.assert_allclose(h_victim[0, 0], 4.0 * x.size)  # (2*1)^2 per row
    np.testing.assert_allclose(h_owner[0, 0], 25.0 * x.size)


def test_rule_closure_is_not_retained_after_the_fit():
    import gc
    import weakref

    table = np.ones(1000)
    ref = weakref.ref(table)
    state = np.array([1.0])

    @jax.custom_jvp
    def f(xx, a):
        return a * xx

    @f.defjvp
    def _jvp(primals, tangents):
        xx, a = primals
        _, da = tangents
        return a * xx, (1.0 if len(table) else 2.0) * state[0] * xx * da

    def model(xx, a):
        return f(xx, a)

    r = lm.exact_lm_fit(model, x, 3.0 * x, [1.0])
    np.testing.assert_allclose(r.x, [3.0], rtol=1e-8)
    del model, f, _jvp, table, r
    jax.clear_caches()
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize("cap,cached", [(10_000_000, True), (1_000, False)])
@pytest.mark.parametrize("nested", [False, True])
def test_captured_typed_prng_keys_count_toward_the_cap(
    monkeypatch, cap, cached, nested
):
    monkeypatch.setattr(lm, "_CACHE_CONST_BYTES", cap)
    keys = jax.random.split(jax.random.key(7), 20_000)  # 160 kB of key data

    def noise():
        return jax.random.uniform(keys[0])

    inner = jax.jit(noise) if nested else noise

    def model(xx, a):
        return a * xx + inner()

    got, _ = lm._cached_stats(model, BLOCK, 1, False)
    assert (got is not model) is cached
    assert (len(ls._wrapper_cache) == 1) is cached
