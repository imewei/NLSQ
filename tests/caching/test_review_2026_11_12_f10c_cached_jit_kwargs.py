"""cached_jit must key traced scalar kwargs by type, not value."""

import jax.numpy as jnp

from nlsq.caching.unified_cache import UnifiedCache, cached_jit


def test_scalar_kwarg_values_share_one_entry():
    cache = UnifiedCache(maxsize=8)
    calls = []

    def g(x, s=1.0):
        calls.append(1)
        return x * s

    x = jnp.ones(3)
    for i in range(50):
        f = cache.get_or_compile(g, (x,), {"s": float(i)})
        assert float(f(x, s=float(i))[0]) == float(i)
    assert len(cache._cache) == 1
    assert len(calls) == 1  # traced once


def test_unrelated_entry_not_evicted_by_scalar_sweep():
    cache = UnifiedCache(maxsize=4)

    def k(x):
        return x + 1

    def g(x, s=1.0):
        return x * s

    x = jnp.ones(3)
    ck = cache.get_or_compile(k, (x,), {})
    for i in range(50):
        cache.get_or_compile(g, (x,), {"s": float(i)})
    assert cache.get_or_compile(k, (x,), {}) is ck


def test_kwarg_type_and_array_rank_still_distinguish():
    cache = UnifiedCache(maxsize=8)

    def g(x, s=1.0):
        return x * s

    x = jnp.ones(3)
    a = cache.get_or_compile(g, (x,), {"s": 1.0})
    b = cache.get_or_compile(g, (x,), {"s": 1})
    c = cache.get_or_compile(g, (x,), {"s": jnp.ones(3)})
    assert a is not b and a is not c


def test_static_positional_still_value_keyed():
    cache = UnifiedCache(maxsize=8)

    def g(x, n):
        return x * n

    x = jnp.ones(3)
    a = cache.get_or_compile(g, (x, 2), {}, static_argnums=(1,))
    b = cache.get_or_compile(g, (x, 3), {}, static_argnums=(1,))
    assert a is not b
    assert float(b(x, 3)[0]) == 3.0


def test_decorator_end_to_end_kwarg_sweep():
    @cached_jit
    def h(x, s=1.0):
        return x * s

    x = jnp.ones(2)
    for i in range(5):
        assert float(h(x, s=float(i))[0]) == float(i)
