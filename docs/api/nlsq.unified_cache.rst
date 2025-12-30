nlsq.unified\_cache module
============================

.. automodule:: nlsq.caching.unified_cache
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``unified_cache`` module consolidates three independent cache implementations into a single unified cache with shape-relaxed keys, comprehensive statistics tracking, and optimized memory management.

Key Features
------------

- **Shape-relaxed cache keys**: Cache based on ``(func_hash, dtype, rank)`` instead of full shapes
- **Comprehensive statistics**: Track hits, misses, compile time, hit rate
- **LRU eviction**: Configurable maxsize with automatic eviction
- **Two-tier caching**: Optional memory + disk caching
- **Weak references**: Prevents memory leaks
- **Thread-safe operations**: Safe for concurrent use

Performance Goals
-----------------

- 80%+ cache hit rate on typical batch fitting workflows
- 2-5x reduction in cold-start compile time through better cache reuse
- Backward compatibility with existing cache APIs
- Zero breaking changes to curve_fit API

Classes
-------

.. autoclass:: nlsq.unified_cache.UnifiedCache
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from nlsq.unified_cache import UnifiedCache
   import jax.numpy as jnp

   # Create cache with statistics tracking
   cache = UnifiedCache(maxsize=128, enable_stats=True)


   # Define function to cache
   def my_func(x, a):
       return a * x**2


   # Use cache for JIT compilation
   x = jnp.array([1.0, 2.0, 3.0])
   compiled = cache.get_or_compile(my_func, (x, 1.0), {}, static_argnums=(1,))
   result = compiled(x, 1.0)

   # Check cache statistics
   stats = cache.get_stats()
   print(f"Hit rate: {stats['hit_rate']:.2%}")
   print(f"Cache size: {stats['cache_size']}")

See Also
--------

- :doc:`nlsq.compilation_cache` - Legacy compilation cache
- :doc:`nlsq.caching` - General caching utilities
- :doc:`nlsq.smart_cache` - Smart cache with adaptive features
