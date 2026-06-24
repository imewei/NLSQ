"""Regression tests for JITCompilationCache function identity.

Adversarial-review finding (high): the cache key hashed only
``func.__code__.co_code``. Two functions sharing ``__module__`` + ``__name__``
and identical bytecode -- e.g. closures from a common factory, or functions
differing only in literal constants -- collided to one key, so the second call
silently reused the first's compiled function and returned wrong results.
"""

import jax.numpy as jnp
import numpy as np

from nlsq.caching.smart_cache import JITCompilationCache


def _make_model(scale):
    """Factory producing closures that share bytecode but differ in behavior."""

    def model(x):
        return scale * x

    return model


def test_factory_closures_do_not_collide():
    """Two closures from the same factory share co_code, __module__, __name__.
    The cache must compile each separately, not serve the first for the second."""
    m1 = _make_model(1.0)
    m2 = _make_model(2.0)

    # Sanity: they genuinely share the collision-prone attributes.
    assert m1.__code__.co_code == m2.__code__.co_code
    assert m1.__name__ == m2.__name__
    assert m1.__module__ == m2.__module__

    cache = JITCompilationCache()
    c1 = cache.get_or_compile(m1)
    c2 = cache.get_or_compile(m2)

    x = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(c1(x)), 1.0 * np.asarray(x))
    np.testing.assert_allclose(np.asarray(c2(x)), 2.0 * np.asarray(x))
    assert len(cache.compiled_functions) == 2


def test_same_function_reuses_cache_entry():
    """A genuinely identical call must hit the cache (no needless recompile)."""
    cache = JITCompilationCache()
    m = _make_model(3.0)
    c1 = cache.get_or_compile(m)
    c2 = cache.get_or_compile(m)
    assert c1 is c2
    assert len(cache.compiled_functions) == 1


def test_same_name_different_constants_do_not_collide():
    """Functions with identical bytecode but different literal constants and the
    same name/module must not collide (co_code omits co_consts)."""
    ns1: dict = {}
    ns2: dict = {}
    exec("def model(x):\n    return x + 1.0", {}, ns1)
    exec("def model(x):\n    return x + 2.0", {}, ns2)
    f1, f2 = ns1["model"], ns2["model"]
    f1.__module__ = f2.__module__ = "collision_test_mod"

    # Sanity: identical bytecode, not closures, differing only in constants.
    assert f1.__code__.co_code == f2.__code__.co_code
    assert f1.__closure__ is None and f2.__closure__ is None
    assert f1.__code__.co_consts != f2.__code__.co_consts

    cache = JITCompilationCache()
    c1 = cache.get_or_compile(f1)
    c2 = cache.get_or_compile(f2)

    x = jnp.array([5.0])
    np.testing.assert_allclose(np.asarray(c1(x)), np.asarray(x) + 1.0)
    np.testing.assert_allclose(np.asarray(c2(x)), np.asarray(x) + 2.0)
    assert len(cache.compiled_functions) == 2


def test_distinct_named_functions_still_cache_independently():
    """Baseline: ordinary distinct functions remain independently cached."""

    def linear(x):
        return x + 1.0

    def quadratic(x):
        return x * x

    cache = JITCompilationCache()
    cl = cache.get_or_compile(linear)
    cq = cache.get_or_compile(quadratic)
    x = jnp.array([3.0])
    np.testing.assert_allclose(np.asarray(cl(x)), np.asarray(x) + 1.0)
    np.testing.assert_allclose(np.asarray(cq(x)), np.asarray(x) * np.asarray(x))
    assert len(cache.compiled_functions) == 2
