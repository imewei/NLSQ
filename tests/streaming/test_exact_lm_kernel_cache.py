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
