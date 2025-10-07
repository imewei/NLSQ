"""JIT compilation cache for optimization functions.

This module provides caching of compiled JAX functions to avoid
recompilation overhead.
"""

import hashlib
import warnings
from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any

import jax
import jax.numpy as jnp


class CompilationCache:
    """Cache for JIT-compiled functions.

    Caches compiled versions of functions based on their signature
    to avoid repeated JIT compilation overhead.

    Attributes
    ----------
    cache : dict
        Dictionary mapping function signatures to compiled functions
    stats : dict
        Compilation cache statistics
    """

    def __init__(self, enable_stats: bool = True):
        """Initialize compilation cache.

        Parameters
        ----------
        enable_stats : bool
            Track cache statistics
        """
        self.cache: dict[str, Callable] = {}
        self.enable_stats = enable_stats

        if enable_stats:
            self.stats = {
                "hits": 0,
                "misses": 0,
                "compilations": 0,
                "cache_size": 0,
            }

    def _get_function_signature(self, func: Callable, *args, **kwargs) -> str:
        """Generate unique signature for function and arguments.

        Parameters
        ----------
        func : callable
            Function to generate signature for
        args : tuple
            Positional arguments
        kwargs : dict
            Keyword arguments

        Returns
        -------
        signature : str
            Unique signature string
        """
        try:
            # Get function name and code
            func_name = func.__name__ if hasattr(func, "__name__") else "unknown"

            # Get argument shapes and dtypes
            arg_info = []
            for arg in args:
                if isinstance(arg, (jnp.ndarray, jax.Array)):
                    arg_info.append(f"{arg.shape}_{arg.dtype}")
                elif isinstance(arg, (int, float, str, bool)):
                    arg_info.append(f"{type(arg).__name__}_{arg}")
                else:
                    arg_info.append(str(type(arg).__name__))

            # Include static arguments from kwargs
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (int, float, str, bool)):
                    arg_info.append(f"{k}={v}")

            # Create signature
            sig_str = f"{func_name}_{'_'.join(arg_info)}"

            # Hash if too long
            if len(sig_str) > 200:
                sig_hash = hashlib.sha256(sig_str.encode()).hexdigest()[:16]
                sig_str = f"{func_name}_{sig_hash}"

            return sig_str

        except (AttributeError, TypeError) as e:
            warnings.warn(f"Could not generate function signature: {e}")
            return f"fallback_{id(func)}"

    def compile(
        self,
        func: Callable,
        static_argnums: tuple[int, ...] = (),
        donate_argnums: tuple[int, ...] = (),
    ) -> Callable:
        """Compile function with JIT and cache result.

        Parameters
        ----------
        func : callable
            Function to compile
        static_argnums : tuple of int
            Indices of static arguments
        donate_argnums : tuple of int
            Indices of arguments to donate

        Returns
        -------
        compiled_func : callable
            JIT-compiled function (may be cached)
        """
        # Create cache key based on function and compilation options
        try:
            func_code = func.__code__.co_code if hasattr(func, "__code__") else b""
            code_hash = hashlib.sha256(func_code).hexdigest()[:8]
            cache_key = (
                f"{func.__name__}_{code_hash}_s{static_argnums}_d{donate_argnums}"
            )
        except (AttributeError, TypeError):
            cache_key = f"{id(func)}_s{static_argnums}_d{donate_argnums}"

        # Check cache
        if cache_key in self.cache:
            if self.enable_stats:
                self.stats["hits"] += 1
            return self.cache[cache_key]

        # Compile function
        if self.enable_stats:
            self.stats["misses"] += 1
            self.stats["compilations"] += 1

        compiled_func = jax.jit(
            func, static_argnums=static_argnums, donate_argnums=donate_argnums
        )

        # Store in cache
        self.cache[cache_key] = compiled_func

        if self.enable_stats:
            self.stats["cache_size"] = len(self.cache)

        return compiled_func

    def get_or_compile(
        self, func: Callable, *args, static_argnums: tuple[int, ...] = (), **kwargs
    ) -> tuple[Callable, str]:
        """Get cached compiled function or compile if not cached.

        Parameters
        ----------
        func : callable
            Function to compile
        args : tuple
            Arguments to function (for signature generation)
        static_argnums : tuple of int
            Indices of static arguments
        kwargs : dict
            Keyword arguments

        Returns
        -------
        compiled_func : callable
            Compiled function
        signature : str
            Function signature
        """
        sig = self._get_function_signature(func, *args, **kwargs)
        full_key = f"{sig}_s{static_argnums}"

        if full_key in self.cache:
            if self.enable_stats:
                self.stats["hits"] += 1
            return self.cache[full_key], sig

        # Compile with signature-aware caching
        # Note: compile() already increments misses and compilations
        compiled_func = self.compile(func, static_argnums=static_argnums)

        # Store with full signature
        self.cache[full_key] = compiled_func

        if self.enable_stats:
            self.stats["cache_size"] = len(self.cache)

        return compiled_func, sig

    def clear(self):
        """Clear compilation cache."""
        self.cache.clear()

        if self.enable_stats:
            self.stats["cache_size"] = 0

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns
        -------
        stats : dict
            Cache hit rate and other statistics
        """
        if not self.enable_stats:
            return {"enabled": False}

        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False


# Global compilation cache
_global_compilation_cache: CompilationCache | None = None


def get_global_compilation_cache() -> CompilationCache:
    """Get or create global compilation cache.

    Returns
    -------
    cache : CompilationCache
        Global compilation cache instance
    """
    global _global_compilation_cache  # noqa: PLW0603
    if _global_compilation_cache is None:
        _global_compilation_cache = CompilationCache(enable_stats=True)
    return _global_compilation_cache


def cached_jit(
    func: Callable | None = None,
    static_argnums: tuple[int, ...] = (),
    donate_argnums: tuple[int, ...] = (),
) -> Callable:
    """Decorator for caching JIT-compiled functions.

    Parameters
    ----------
    func : callable, optional
        Function to decorate
    static_argnums : tuple of int
        Indices of static arguments
    donate_argnums : tuple of int
        Indices of arguments to donate

    Returns
    -------
    decorated : callable
        Decorated function with cached compilation

    Examples
    --------
    >>> @cached_jit
    ... def my_function(x):
    ...     return x ** 2

    >>> @cached_jit(static_argnums=(1,))
    ... def my_function_with_static(x, n):
    ...     return x ** n
    """

    def decorator(f):
        cache = get_global_compilation_cache()

        @wraps(f)
        def wrapper(*args, **kwargs):
            compiled_func, _ = cache.get_or_compile(
                f, *args, static_argnums=static_argnums, **kwargs
            )
            return compiled_func(*args, **kwargs)

        return wrapper

    if func is None:
        # Called with arguments: @cached_jit(static_argnums=(1,))
        return decorator
    else:
        # Called without arguments: @cached_jit
        return decorator(func)


def clear_compilation_cache():
    """Clear the global compilation cache."""
    global _global_compilation_cache  # noqa: PLW0602
    if _global_compilation_cache is not None:
        _global_compilation_cache.clear()
