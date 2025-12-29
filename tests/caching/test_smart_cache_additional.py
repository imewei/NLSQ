"""Additional fast tests for SmartCache edge paths."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from nlsq.caching import smart_cache
from nlsq.caching.smart_cache import SmartCache, cached_function


@pytest.mark.cache
def test_memory_eviction_increments_stats() -> None:
    """LRU eviction should bump eviction stats when max_memory_items exceeded."""
    cache = SmartCache(max_memory_items=1, disk_cache_enabled=False)

    key1 = cache.cache_key(1)
    key2 = cache.cache_key(2)

    cache.set(key1, np.array([1.0]))
    cache.set(key2, np.array([2.0]))

    stats = cache.get_stats()
    assert stats["evictions"] == 1
    assert key2 in cache.memory_cache
    assert key1 not in cache.memory_cache


@pytest.mark.cache
def test_disk_save_warning_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disk cache failures should emit a warning and continue."""
    cache = SmartCache(max_memory_items=2, disk_cache_enabled=True)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cache, "_save_to_disk", _raise)

    with pytest.warns(UserWarning, match="Could not save to disk cache"):
        cache.set(cache.cache_key(3), np.array([3.0]))


@pytest.mark.cache
def test_load_from_disk_unknown_structure(tmp_path: Path) -> None:
    """Unknown cache file structures should raise a ValueError."""
    cache = SmartCache(cache_dir=str(tmp_path), disk_cache_enabled=True)
    cache_file = tmp_path / "bad_cache.npz"
    np.savez_compressed(cache_file, unexpected=np.array([1.0]))

    with pytest.raises(ValueError, match="Unknown cache file structure"):
        cache._load_from_disk(str(cache_file))


@pytest.mark.cache
def test_cached_function_ttl_expiration() -> None:
    """cached_function should recompute after TTL expiration."""
    cache = SmartCache(disk_cache_enabled=False)
    calls = {"count": 0}

    @cached_function(cache=cache, ttl=0.01)
    def add_one(x: int) -> int:
        calls["count"] += 1
        return x + 1

    assert add_one(1) == 2
    time.sleep(0.02)
    assert add_one(1) == 2
    assert calls["count"] == 2


@pytest.mark.cache
def test_cache_key_large_and_small_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    """cache_key should handle both small and large arrays on fallback path."""
    cache = SmartCache(disk_cache_enabled=False)
    monkeypatch.setattr(smart_cache, "HAS_XXHASH", False)

    small = np.ones(10, dtype=np.float64)
    large = np.ones(smart_cache.LARGE_ARRAY_THRESHOLD + 1, dtype=np.float64)

    key_small = cache.cache_key(small)
    key_large = cache.cache_key(large)

    assert key_small.startswith("v2_")
    assert key_large.startswith("v2_")
    assert key_small != key_large


@pytest.mark.cache
def test_disk_cache_corrupt_file_removed(tmp_path: Path) -> None:
    """Corrupted disk cache files should be removed after load failure."""
    cache = SmartCache(cache_dir=str(tmp_path), disk_cache_enabled=True)
    key = cache.cache_key("x")
    cache_file = tmp_path / f"{key}.npz"
    cache_file.write_bytes(b"not a zip")

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ValueError("bad cache")

    cache._load_from_disk = _raise  # type: ignore[assignment]

    with pytest.warns(UserWarning, match="Could not load from disk cache"):
        assert cache.get(key) is None

    assert not cache_file.exists()


@pytest.mark.cache
def test_save_to_disk_unsupported_type_warns(tmp_path: Path) -> None:
    """Unsupported types should warn and skip disk caching."""
    cache = SmartCache(cache_dir=str(tmp_path), disk_cache_enabled=True)

    class BadArray:
        def __array__(self):
            raise ValueError("no array")

    with pytest.warns(UserWarning, match="Cannot safely cache type"):
        cache._save_to_disk(str(tmp_path / "x.npz"), BadArray())


@pytest.mark.cache
def test_invalidate_warns_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """invalidate should warn if disk cache cannot be cleared."""
    cache = SmartCache(disk_cache_enabled=True)

    monkeypatch.setattr(
        smart_cache.os,
        "listdir",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("nope")),
    )

    with pytest.warns(UserWarning, match="Could not clear disk cache"):
        cache.invalidate()
