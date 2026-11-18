"""Regression tests for the 2026-11-18 three-brain review (fixer F2: caching/)."""

import jax.numpy as jnp

from nlsq.caching.unified_cache import UnifiedCache, cached_jit


def test_c06_cached_jit_hashes_function_source_once(monkeypatch):
    calls = []
    real = UnifiedCache._get_function_hash

    def spy(self, func):
        calls.append(func)
        return real(self, func)

    monkeypatch.setattr(UnifiedCache, "_get_function_hash", spy)

    @cached_jit
    def double(x):
        return x * 2

    x = jnp.ones(3)
    for _ in range(5):
        assert float(double(x)[0]) == 2.0
    assert len(calls) == 1


def test_agy_p07_clear_cache_drops_jax_traces():
    from nlsq.caching.unified_cache import clear_cache

    offset = [1.0]

    @cached_jit
    def shifted(x):
        return x + offset[0]

    assert float(shifted(jnp.array(1.0))) == 2.0
    offset[0] = 2.0
    clear_cache()
    assert float(shifted(jnp.array(1.0))) == 3.0
