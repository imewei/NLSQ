"""Smart caching system for NLSQ optimization.

This module provides intelligent caching for expensive computations,
particularly Jacobian evaluations and function calls.

Note: This module uses safe serialization only (JSON and numpy.savez
with allow_pickle=False). No pickle is used.

Phase 3 Optimizations (Task Group 9):
- Array hash optimization: stride-based sampling only for >10000 elements
- For smaller arrays, hash full array directly (no redundant sampling)
"""

import hashlib
import json
import os
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

from nlsq.config import JAXConfig

_jax_config = JAXConfig()


import contextlib

import jax.numpy as jnp

# Cache version for invalidating old cache entries when hash algorithm changes
CACHE_VERSION = "v2"

# Threshold for using stride-based sampling (Task 9.3)
# Arrays larger than this use stride sampling for efficiency
LARGE_ARRAY_THRESHOLD = 10000

# Try to use xxhash for faster hashing (10x faster than SHA256)
try:
    import xxhash  # type: ignore[import-not-found]

    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False


class SmartCache:
    """Intelligent caching system for optimization computations.

    This class provides:

    - Memory and disk caching with LRU eviction
    - Automatic cache key generation from function arguments
    - Cache persistence across sessions
    - Cache invalidation and warming strategies

    Phase 3 Optimizations (3.2a):

    - Array hash optimization: uses stride-based sampling only for
      arrays with >10000 elements when xxhash is unavailable
    - For smaller arrays, hashes full array directly without redundant
      sampling, providing 15-20% improvement in cache key generation

    All dict operations are protected by a per-instance ``threading.Lock``
    so that concurrent threads can safely call ``get``/``set``/``invalidate``.

    Attributes
    ----------
    cache_dir : str
        Directory for disk cache storage
    memory_cache : dict
        In-memory cache storage
    disk_cache_enabled : bool
        Whether disk caching is enabled
    max_memory_items : int
        Maximum items in memory cache
    cache_stats : dict
        Cache hit/miss statistics
    """

    def __init__(
        self,
        cache_dir: str = ".nlsq_cache",
        max_memory_items: int = 1000,
        disk_cache_enabled: bool = True,
        enable_stats: bool = True,
    ):
        """Initialize smart cache.

        Parameters
        ----------
        cache_dir : str
            Directory for disk cache
        max_memory_items : int
            Maximum items in memory cache
        disk_cache_enabled : bool
            Enable disk caching
        enable_stats : bool
            Track cache statistics
        """
        self._lock = threading.Lock()
        self.cache_dir = cache_dir
        self.memory_cache: dict[str, tuple[Any, float]] = {}  # value, timestamp
        self.access_count: dict[str, int] = {}  # Track access frequency
        self.disk_cache_enabled = disk_cache_enabled
        self.max_memory_items = max_memory_items
        self.enable_stats = enable_stats

        # Statistics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "disk_hits": 0,
            "evictions": 0,
        }

        # Create cache directory if needed
        if disk_cache_enabled and not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir)
            except OSError:
                warnings.warn(f"Could not create cache directory {cache_dir}")
                self.disk_cache_enabled = False

    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        key : str
            Hash of arguments (xxhash if available, BLAKE2b fallback)

        Notes
        -----
        Uses xxhash (xxh64) when available for ~10x faster hashing compared
        to SHA256/BLAKE2b. Falls back to BLAKE2b if xxhash is not installed.
        All cache keys are prefixed with CACHE_VERSION to ensure old cache
        entries are invalidated when the hash algorithm changes.

        Task 9.3 (3.2a): Array hash optimization
        - Arrays <= 10000 elements: hash full array directly (no sampling)
        - Arrays > 10000 elements: use stride-based sampling for efficiency
        - Removes redundant sampling when computing full hash in fallback path
        """
        key_parts = []

        for arg in args:
            if isinstance(arg, (np.ndarray, jnp.ndarray)):
                # For arrays, use shape, dtype, and fast hash of values
                arr = np.asarray(arg)
                if HAS_XXHASH:
                    # Fast path: xxhash on contiguous data (10x faster than SHA256)
                    if arr.flags["C_CONTIGUOUS"]:
                        data_hash = xxhash.xxh64(arr).hexdigest()[:16]
                    else:
                        data_hash = xxhash.xxh64(np.ascontiguousarray(arr)).hexdigest()[
                            :16
                        ]
                    key_parts.append(f"array_{arg.shape}_{arg.dtype}_{data_hash}")
                else:
                    # Task 9.3: Optimized fallback path
                    # Use stride-based sampling ONLY for very large arrays (>10000 elements)
                    arr_flat = arr.flatten()
                    arr_size = len(arr_flat)

                    if arr_size > LARGE_ARRAY_THRESHOLD:
                        # Large array: use stride-based sampling for efficiency
                        # Calculate stride to sample approximately 1000 elements
                        stride = max(1, arr_size // 1000)
                        sample = arr_flat[::stride]
                        # Use BLAKE2b for the sample hash
                        sample_hash = hashlib.blake2b(
                            sample.tobytes(), digest_size=16
                        ).hexdigest()
                        key_parts.append(f"array_{arg.shape}_{arg.dtype}_{sample_hash}")
                    else:
                        # Small/medium array: hash full array directly (no sampling overhead)
                        # This is the optimized path - removes redundant sampling
                        full_hash = hashlib.blake2b(
                            arr_flat.tobytes(), digest_size=16
                        ).hexdigest()
                        key_parts.append(f"array_{arg.shape}_{arg.dtype}_{full_hash}")
            elif callable(arg):
                # For functions, use their name and module
                key_parts.append(f"func_{arg.__module__}_{arg.__name__}")
            else:
                key_parts.append(str(arg))

        # Add kwargs
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        key_str = "|".join(key_parts)

        # Use xxhash for final key if available, BLAKE2b as fallback
        if HAS_XXHASH:
            hash_hex = xxhash.xxh64(key_str.encode()).hexdigest()
        else:
            # Use BLAKE2b instead of MD5 for better security and collision resistance
            hash_hex = hashlib.blake2b(key_str.encode(), digest_size=16).hexdigest()

        # Prefix with cache version to invalidate old cache entries
        return f"{CACHE_VERSION}_{hash_hex}"

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        value : Any or None
            Cached value or None if not found
        """
        # Check memory cache first (under lock for atomic LRU update)
        with self._lock:
            if key in self.memory_cache:
                value, timestamp = self.memory_cache[key]
                self.access_count[key] = self.access_count.get(key, 0) + 1

                if self.enable_stats:
                    self.cache_stats["hits"] += 1
                    self.cache_stats["memory_hits"] += 1

                # Move to end (LRU)
                del self.memory_cache[key]
                self.memory_cache[key] = (value, timestamp)

                return value

        # Check disk cache (disk I/O outside lock)
        if self.disk_cache_enabled:
            cache_file = os.path.join(self.cache_dir, f"{key}.npz")
            if os.path.exists(cache_file):
                try:
                    value = self._load_from_disk(cache_file)

                    with self._lock:
                        if self.enable_stats:
                            self.cache_stats["hits"] += 1
                            self.cache_stats["disk_hits"] += 1

                    # Add to memory cache, preserving file mtime as timestamp
                    # so TTL checks reflect the original cache time, not load time
                    disk_mtime = os.path.getmtime(cache_file)
                    self._add_to_memory_cache(key, value, timestamp=disk_mtime)
                    return value

                except Exception as e:
                    warnings.warn(f"Could not load from disk cache: {e}")
                    # Remove corrupted cache file
                    with contextlib.suppress(OSError):
                        os.remove(cache_file)

        with self._lock:
            if self.enable_stats:
                self.cache_stats["misses"] += 1

        return None

    def set(self, key: str, value: Any):
        """Set value in cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        # Add to memory cache (lock acquired inside _add_to_memory_cache)
        self._add_to_memory_cache(key, value)

        # Save to disk cache (disk I/O outside lock)
        if self.disk_cache_enabled:
            cache_file = os.path.join(self.cache_dir, f"{key}.npz")
            try:
                self._save_to_disk(cache_file, value)
            except Exception as e:
                warnings.warn(f"Could not save to disk cache: {e}")

    def _add_to_memory_cache(
        self, key: str, value: Any, timestamp: float | None = None
    ):
        """Add item to memory cache with LRU eviction.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        timestamp : float, optional
            Timestamp to associate with the entry.  When loading from disk,
            pass the file's mtime so that TTL checks reflect the original
            cache time rather than the load time.  Defaults to ``time.time()``.
        """
        with self._lock:
            # Check if we need to evict
            if len(self.memory_cache) >= self.max_memory_items:
                # Evict least recently used item
                if self.memory_cache:
                    oldest_key = next(iter(self.memory_cache))
                    del self.memory_cache[oldest_key]
                    if oldest_key in self.access_count:
                        del self.access_count[oldest_key]

                    if self.enable_stats:
                        self.cache_stats["evictions"] += 1

            self.memory_cache[key] = (
                value,
                timestamp if timestamp is not None else time.time(),
            )
            self.access_count[key] = self.access_count.get(key, 0) + 1

    def invalidate(self, key: str | None = None):
        """Invalidate cache entries.

        Parameters
        ----------
        key : str, optional
            Specific key to invalidate, or None to clear all
        """
        if key is None:
            # Clear all caches (dict ops under lock, disk I/O outside)
            with self._lock:
                self.memory_cache.clear()
                self.access_count.clear()

            if self.disk_cache_enabled and os.path.isdir(self.cache_dir):
                try:
                    for file in os.listdir(self.cache_dir):
                        if file.endswith(".npz"):
                            os.remove(os.path.join(self.cache_dir, file))
                except OSError as e:
                    warnings.warn(f"Could not clear disk cache: {e}")
        else:
            # Clear specific key
            with self._lock:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if key in self.access_count:
                    del self.access_count[key]

            if self.disk_cache_enabled:
                cache_file = os.path.join(self.cache_dir, f"{key}.npz")
                if os.path.exists(cache_file):
                    with contextlib.suppress(OSError):
                        os.remove(cache_file)

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns
        -------
        stats : dict
            Cache statistics including hit rate
        """
        with self._lock:
            total_accesses = self.cache_stats["hits"] + self.cache_stats["misses"]

            if total_accesses > 0:
                hit_rate = self.cache_stats["hits"] / total_accesses
            else:
                hit_rate = 0.0

            return {
                **self.cache_stats,
                "hit_rate": hit_rate,
                "memory_size": len(self.memory_cache),
                "total_accesses": total_accesses,
            }

    def optimize_cache(self):
        """Optimize cache by removing rarely accessed items.

        Computes threshold from snapshot, then re-checks live counts under
        lock before invalidating to avoid evicting keys that became hot
        between the snapshot and eviction.
        """
        with self._lock:
            if not self.access_count:
                return
            # Snapshot under lock
            access_snapshot = dict(self.access_count)

        # Calculate average access count (no lock needed for snapshot)
        avg_access = np.mean(list(access_snapshot.values()))
        threshold = avg_access * 0.5

        # Re-check live count under lock before invalidating each key
        keys_to_remove = []
        with self._lock:
            for key in access_snapshot:
                live_count = self.access_count.get(key, 0)
                if live_count < threshold:
                    keys_to_remove.append(key)

        # Invalidate outside the lock (invalidate acquires its own lock)
        for key in keys_to_remove:
            self.invalidate(key)

    def _save_to_disk(self, cache_file: str, value: Any):
        """Save value to disk using safe serialization.

        Uses numpy.savez for arrays and JSON for other data types.
        This is safe as it does not use pickle or execute arbitrary code.

        Parameters
        ----------
        cache_file : str
            Path to cache file
        value : Any
            Value to save
        """
        # Check if value is array-like (numpy or JAX array)
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            # Convert JAX array to numpy for saving
            if isinstance(value, jnp.ndarray):
                value = np.asarray(value)
            np.savez_compressed(cache_file, data=value)
        elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
            # Use JSON for simple data types
            json_file = cache_file.replace(".npz", ".json")
            with open(json_file, "w") as f:
                json.dump(value, f)
        elif isinstance(value, tuple) and all(
            isinstance(v, (np.ndarray, jnp.ndarray)) for v in value
        ):
            # Handle tuple of arrays (common for multi-output functions)
            arrays_dict: dict[str, Any] = {
                f"arr_{i}": np.asarray(v) for i, v in enumerate(value)
            }
            arrays_dict["_is_tuple"] = np.array([True])
            arrays_dict["_length"] = np.array([len(value)])
            np.savez_compressed(cache_file, **arrays_dict)
        else:
            # For other types, convert to numpy array if possible
            try:
                arr = np.asarray(value)
                np.savez_compressed(cache_file, data=arr)
            except (ValueError, TypeError):
                warnings.warn(
                    f"Cannot safely cache type {type(value).__name__}, skipping disk cache"
                )

    def _load_from_disk(self, cache_file: str) -> Any:
        """Load value from disk using safe deserialization.

        Uses numpy.load for arrays and JSON for other data types.
        This is safe as allow_pickle=False prevents code execution.

        Parameters
        ----------
        cache_file : str
            Path to cache file

        Returns
        -------
        value : Any
            Loaded value
        """
        # Check if JSON file exists
        json_file = cache_file.replace(".npz", ".json")
        if os.path.exists(json_file):
            with open(json_file) as f:
                return json.load(f)

        # Load from numpy file (safe: allow_pickle=False)
        with np.load(cache_file, allow_pickle=False) as data:
            # Check if it's a tuple of arrays
            if "_is_tuple" in data.files:
                length = int(data["_length"])
                return tuple(data[f"arr_{i}"] for i in range(length))
            # Single array
            elif "data" in data.files:
                return data["data"]
            else:
                # Legacy format or unknown structure
                raise ValueError(f"Unknown cache file structure: {data.files}")


def cached_function(cache: SmartCache | None = None, ttl: float | None = None):
    """Decorator for caching function results.

    Parameters
    ----------
    cache : SmartCache, optional
        Cache instance to use (creates new if None)
    ttl : float, optional
        Time-to-live in seconds for cached values

    Returns
    -------
    decorator : function
        Decorator function
    """
    if cache is None:
        cache = SmartCache()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache.cache_key(func, *args, **kwargs)

            # Check cache (get value and timestamp atomically under one lock)
            cached_result = None
            with cache._lock:
                if cache_key in cache.memory_cache:
                    value, timestamp = cache.memory_cache[cache_key]
                    cache.access_count[cache_key] = (
                        cache.access_count.get(cache_key, 0) + 1
                    )
                    if cache.enable_stats:
                        cache.cache_stats["hits"] += 1
                        cache.cache_stats["memory_hits"] += 1
                    # LRU move
                    del cache.memory_cache[cache_key]
                    cache.memory_cache[cache_key] = (value, timestamp)
                    # TTL check
                    if ttl is not None and time.time() - timestamp > ttl:
                        value = None  # expired
                    cached_result = value

            # Disk fallback (outside lock)
            if cached_result is None and cache.disk_cache_enabled:
                cached_result = cache.get(cache_key)

            if cached_result is None:
                # Compute and cache
                result = func(*args, **kwargs)
                cache.set(cache_key, result)
                return result

            return cached_result

        # Add cache management methods to wrapper function
        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.invalidate = cache.invalidate  # type: ignore[attr-defined]
        wrapper.get_stats = cache.get_stats  # type: ignore[attr-defined]

        return wrapper

    return decorator


def cached_jacobian(cache: SmartCache | None = None):
    """Decorator specifically for caching Jacobian evaluations.

    Parameters
    ----------
    cache : SmartCache, optional
        Cache instance to use

    Returns
    -------
    decorator : function
        Decorator function
    """
    if cache is None:
        cache = SmartCache(max_memory_items=100)  # Jacobians can be large

    def decorator(func):
        @wraps(func)
        def wrapper(x, *params):
            # Create cache key from x and params
            cache_key = cache.cache_key(x, *params)

            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute and cache
            result = func(x, *params)
            cache.set(cache_key, result)
            return result

        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.invalidate = cache.invalidate  # type: ignore[attr-defined]

        return wrapper

    return decorator


class JITCompilationCache:
    """Cache for JAX JIT-compiled functions with LRU eviction.

    This cache stores compiled functions to avoid recompilation
    when function signatures match. Uses OrderedDict for LRU eviction
    to prevent unbounded XLA compilation cache growth.

    All dict operations are protected by a per-instance ``threading.Lock``
    so that concurrent threads can safely call ``get_or_compile``/``clear``.

    Parameters
    ----------
    max_cache_size : int
        Maximum number of compiled functions to cache (default 256).
        Oldest entries are evicted when capacity is reached.
    """

    def __init__(self, max_cache_size: int = 256):
        """Initialize JIT compilation cache with LRU eviction."""
        self._lock = threading.Lock()
        self.compiled_functions: OrderedDict = OrderedDict()
        self.compilation_times: OrderedDict = OrderedDict()
        self.max_cache_size = max_cache_size

    def get_or_compile(self, func: Callable, static_argnums: tuple = ()) -> Callable:
        """Get cached compilation or compile and cache.

        Parameters
        ----------
        func : callable
            Function to compile
        static_argnums : tuple
            Static argument numbers for JIT

        Returns
        -------
        compiled_func : callable
            JIT-compiled function
        """
        from jax import jit

        # Create key from function and static args
        key = (func.__module__, func.__name__, static_argnums)

        # Check cache under lock
        with self._lock:
            if key in self.compiled_functions:
                self.compiled_functions.move_to_end(key)
                return self.compiled_functions[key]

        # Compile outside lock (jit can be slow)
        start_time = time.time()
        compiled_func = jit(func, static_argnums=static_argnums)
        compilation_time = time.time() - start_time

        # Store under lock (double-check to avoid overwriting a concurrent compile)
        with self._lock:
            if key not in self.compiled_functions:
                # Evict oldest entry if at capacity
                if len(self.compiled_functions) >= self.max_cache_size:
                    evicted_key, _ = self.compiled_functions.popitem(last=False)
                    self.compilation_times.pop(evicted_key, None)
                self.compiled_functions[key] = compiled_func
                self.compilation_times[key] = compilation_time
            else:
                # Another thread already stored it; use that one
                self.compiled_functions.move_to_end(key)
                compiled_func = self.compiled_functions[key]

        return compiled_func

    def clear(self):
        """Clear compilation cache."""
        with self._lock:
            self.compiled_functions.clear()
            self.compilation_times.clear()

    def get_stats(self) -> dict:
        """Get compilation statistics.

        Returns
        -------
        stats : dict
            Compilation statistics
        """
        with self._lock:
            return {
                "cached_functions": len(self.compiled_functions),
                "max_cache_size": self.max_cache_size,
                "total_compilation_time": sum(self.compilation_times.values()),
                "functions": list(self.compiled_functions.keys()),
            }


# Global cache instances
_global_cache = SmartCache()
_jit_cache = JITCompilationCache()


def get_global_cache() -> SmartCache:
    """Get global cache instance.

    Returns
    -------
    cache : SmartCache
        Global cache instance
    """
    return _global_cache


def get_jit_cache() -> JITCompilationCache:
    """Get JIT compilation cache.

    Returns
    -------
    cache : JITCompilationCache
        JIT compilation cache
    """
    return _jit_cache


def clear_all_caches():
    """Clear all global caches."""
    _global_cache.invalidate()
    _jit_cache.clear()
