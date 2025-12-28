"""Function caching mechanism for NLSQ to avoid JAX recompilation."""

import hashlib
import inspect
import logging
import weakref
from collections.abc import Callable
from functools import cache, wraps
from typing import Any

# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax

logger = logging.getLogger(__name__)


class FunctionCache:
    """Cache for compiled JAX functions.

    This cache helps avoid recompilation of JAX functions when the same
    function is used multiple times, which can significantly improve performance.
    """

    def __init__(self, maxsize: int = 128):
        """Initialize the function cache.

        Parameters
        ----------
        maxsize : int
            Maximum number of functions to cache
        """
        self.maxsize = maxsize
        self._cache: dict[str, Any] = {}
        self._compiled_funcs: dict[str, Callable] = {}
        self._func_refs: dict[str, weakref.ref] = {}
        self._hit_count = 0
        self._miss_count = 0

    def get_function_hash(self, func: Callable) -> str:
        """Generate stable hash for a function.

        Parameters
        ----------
        func : Callable
            Function to hash

        Returns
        -------
        str
            Hexadecimal hash string
        """
        try:
            # Try to get source code for regular functions
            source = inspect.getsource(func)
            signature = str(inspect.signature(func))
            module = getattr(func, "__module__", "unknown")
            name = getattr(func, "__name__", "unknown")

            # Combine all identifying information
            combined = f"{module}.{name}:{signature}\n{source}"
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
        except (OSError, TypeError):
            # Fallback for built-in functions, lambdas, or C functions
            # Use function identity and string representation
            try:
                # For lambdas and simple functions, use their code object
                if hasattr(func, "__code__"):
                    code = func.__code__
                    code_hash = hashlib.sha256(code.co_code).hexdigest()[:8]
                    return f"code_{code_hash}_{code.co_argcount}"
                else:
                    # Last resort: use object ID (not ideal for persistence)
                    return f"id_{id(func)}"
            except (AttributeError, TypeError, ValueError):
                return f"id_{id(func)}"

    @cache
    def get_compiled_function(
        self,
        func_hash: str,
        static_argnums: tuple[int, ...] = (),
        static_argnames: tuple[str, ...] = (),
    ) -> Callable:
        """Get or create compiled version of function.

        Parameters
        ----------
        func_hash : str
            Hash of the function
        static_argnums : Tuple[int, ...]
            Indices of static arguments for JIT compilation
        static_argnames : Tuple[str, ...]
            Names of static arguments for JIT compilation

        Returns
        -------
        Callable
            JIT-compiled function
        """
        cache_key = f"{func_hash}_{static_argnums}_{static_argnames}"

        if cache_key in self._compiled_funcs:
            self._hit_count += 1
            logger.debug(f"Cache hit for function {func_hash[:8]}")
            return self._compiled_funcs[cache_key]

        # Need to get the actual function from weak reference
        if func_hash in self._func_refs:
            func_ref = self._func_refs[func_hash]
            func = func_ref()
            if func is not None:
                self._miss_count += 1
                logger.debug(f"Cache miss, compiling function {func_hash[:8]}")

                # Compile the function
                compiled = jax.jit(
                    func, static_argnums=static_argnums, static_argnames=static_argnames
                )
                self._compiled_funcs[cache_key] = compiled
                return compiled

        raise ValueError(f"Function with hash {func_hash} not found in cache")

    def cache_function(self, func: Callable) -> str:
        """Add a function to the cache.

        Parameters
        ----------
        func : Callable
            Function to cache

        Returns
        -------
        str
            Hash of the cached function
        """
        func_hash = self.get_function_hash(func)

        # Store weak reference to avoid memory leaks
        self._func_refs[func_hash] = weakref.ref(func)

        # Clean up if cache is too large
        if len(self._func_refs) > self.maxsize:
            # Remove oldest entries (simple FIFO for now)
            oldest = next(iter(self._func_refs.keys()))
            del self._func_refs[oldest]
            # Also remove compiled versions
            keys_to_remove = [k for k in self._compiled_funcs if k.startswith(oldest)]
            for key in keys_to_remove:
                del self._compiled_funcs[key]

        return func_hash

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._compiled_funcs.clear()
        self._func_refs.clear()
        self.get_compiled_function.cache_clear()
        self._hit_count = 0
        self._miss_count = 0
        logger.info("Function cache cleared")

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary with cache statistics
        """
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": self.hit_rate,
            "cached_functions": len(self._func_refs),
            "compiled_versions": len(self._compiled_funcs),
        }


# Global cache instance
_function_cache = FunctionCache()


def get_cached_jit(
    func: Callable,
    static_argnums: tuple[int, ...] = (),
    static_argnames: tuple[str, ...] = (),
) -> Callable:
    """Get cached JIT-compiled version of function.

    This is a convenience function that uses the global cache to get or create
    a JIT-compiled version of the given function.

    Parameters
    ----------
    func : Callable
        Function to compile
    static_argnums : Tuple[int, ...]
        Indices of static arguments
    static_argnames : Tuple[str, ...]
        Names of static arguments

    Returns
    -------
    Callable
        JIT-compiled function
    """
    func_hash = _function_cache.cache_function(func)
    return _function_cache.get_compiled_function(
        func_hash, static_argnums, static_argnames
    )


def cached_jit(
    static_argnums: tuple[int, ...] = (), static_argnames: tuple[str, ...] = ()
):
    """Decorator for cached JIT compilation.

    This decorator can be used to automatically cache JIT-compiled versions
    of functions.

    Parameters
    ----------
    static_argnums : Tuple[int, ...]
        Indices of static arguments
    static_argnames : Tuple[str, ...]
        Names of static arguments

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @cached_jit(static_argnums=(1,))
    ... def my_function(x, static_param):
    ...     return x * static_param
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            compiled = get_cached_jit(func, static_argnums, static_argnames)
            return compiled(*args, **kwargs)

        # Store reference to original function
        wrapper.__wrapped__ = func
        return wrapper

    return decorator


def clear_cache():
    """Clear the global function cache."""
    _function_cache.clear()


def get_cache_stats() -> dict[str, Any]:
    """Get statistics from the global cache.

    Returns
    -------
    Dict[str, Any]
        Cache statistics
    """
    return _function_cache.get_stats()


def compare_functions(func1: Callable, func2: Callable) -> bool:
    """Compare two functions for equality.

    This function compares two functions to determine if they are the same,
    which is useful for determining if a function needs to be recompiled.

    Parameters
    ----------
    func1 : Callable
        First function
    func2 : Callable
        Second function

    Returns
    -------
    bool
        True if functions are considered equal
    """
    # Quick checks first
    if func1 is func2:
        return True

    # Compare by hash
    hash1 = _function_cache.get_function_hash(func1)
    hash2 = _function_cache.get_function_hash(func2)

    return hash1 == hash2


# Usage examples - see module docstring for details
