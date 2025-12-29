"""Protocol definition for caching mechanisms.

This module defines the CacheProtocol that caching implementations
should follow, enabling different caching strategies.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for caching mechanisms.

    This protocol defines the interface for cache implementations,
    allowing different strategies (LRU, TTL, memory-bounded) to be used.

    Methods
    -------
    get(key, default)
        Retrieve value from cache.
    set(key, value)
        Store value in cache.
    clear()
        Clear all cached values.
    """

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve value from cache.

        Parameters
        ----------
        key : str
            Cache key.
        default : Any
            Value to return if key not found.

        Returns
        -------
        Any
            Cached value or default.
        """
        ...

    def set(self, key: str, value: Any) -> None:
        """Store value in cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to store.
        """
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...


@runtime_checkable
class BoundedCacheProtocol(Protocol):
    """Protocol for memory-bounded caches.

    Extended protocol for caches that track memory usage and
    can evict items based on size limits.
    """

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve value from cache."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...

    @property
    def size_bytes(self) -> int:
        """Current cache size in bytes."""
        ...

    @property
    def max_size_bytes(self) -> int:
        """Maximum cache size in bytes."""
        ...

    def evict(self, n_bytes: int) -> int:
        """Evict items to free at least n_bytes.

        Parameters
        ----------
        n_bytes : int
            Minimum bytes to free.

        Returns
        -------
        int
            Actual bytes freed.
        """
        ...


class DictCache:
    """Simple dictionary-based cache implementation.

    A minimal cache implementation for testing and simple use cases.
    """

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve value from cache."""
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        self._cache[key] = value

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self._cache)
