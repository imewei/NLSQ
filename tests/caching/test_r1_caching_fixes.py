"""Regression tests for round-1 review findings in nlsq/caching."""

import numpy as np
import pytest

from nlsq.caching.memory_manager import MemoryManager
from nlsq.caching.unified_cache import UnifiedCache


class _Holder:
    def f(self, x):
        return x * 2.0

    def g(self, x):
        return x * 3.0


def test_identity_tag_distinguishes_methods_of_same_instance():
    h = _Holder()
    assert UnifiedCache._identity_tag(h.f) != UnifiedCache._identity_tag(h.g)
    # Re-created bound method objects of the same method still share a tag.
    assert UnifiedCache._identity_tag(h.f) == UnifiedCache._identity_tag(h.f)


@pytest.mark.parametrize("maxsize", [0, -1])
def test_unified_cache_rejects_nonpositive_maxsize(maxsize):
    with pytest.raises(ValueError, match="maxsize"):
        UnifiedCache(maxsize=maxsize)


def test_memory_guard_measures_real_delta_within_ttl():
    mm = MemoryManager(
        enable_adaptive_safety=True, memory_cache_ttl=1000.0, adaptive_ttl=False
    )
    mm.get_memory_usage_bytes()  # prime the TTL cache
    nbytes = 64 * 1024 * 1024
    with mm.memory_guard(nbytes):
        keep = np.ones(nbytes // 8)
    assert mm.allocation_history[-1]["bytes_used"] > nbytes // 2
    del keep
