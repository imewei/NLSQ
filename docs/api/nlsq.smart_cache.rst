nlsq.smart\_cache module
=========================

.. automodule:: nlsq.caching.smart_cache
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``smart_cache`` module provides intelligent caching with adaptive features for JIT compilation and function evaluation.

Key Features
------------

- **Adaptive cache sizing** based on usage patterns
- **Memory-aware eviction policies** with size limits
- **Function evaluation caching** for repeated calls
- **Jacobian caching** with automatic invalidation
- **Statistics tracking** for cache hit rate monitoring

Classes
-------

.. autoclass:: nlsq.smart_cache.SmartCache
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: nlsq.smart_cache.cached_function
   :noindex:

Example Usage
-------------

.. code-block:: python

   from nlsq.smart_cache import SmartCache, cached_function
   from nlsq import curve_fit
   import jax.numpy as jnp

   # Configure caching
   cache = SmartCache(max_memory_items=1000, disk_cache_enabled=True)


   # Define fit function
   @cached_function(cache=cache)
   def exponential(x, a, b):
       return a * jnp.exp(-b * x)


   # First fit - compiles function
   x1 = jnp.linspace(0, 10, 100)
   y1 = 2.5 * jnp.exp(-0.5 * x1) + 0.1 * jnp.random.randn(100)
   popt1, pcov1 = curve_fit(exponential, x1, y1, p0=[1.0, 0.1])

   # Second fit - reuses JIT compilation from first fit
   x2 = jnp.linspace(0, 10, 100)
   y2 = 3.0 * jnp.exp(-0.7 * x2) + 0.1 * jnp.random.randn(100)
   popt2, pcov2 = curve_fit(exponential, x2, y2, p0=[1.2, 0.15])

   # Check cache statistics
   stats = cache.get_stats()
   print(f"Cache hit rate: {stats['hit_rate']:.1%}")
   print(f"Total compilations: {stats['compilations']}")

Cache Management
----------------

.. code-block:: python

   # Create cache with custom settings
   cache = SmartCache(
       max_memory_items=500,
       disk_cache_enabled=True,
       eviction_policy="lru",  # Least Recently Used
       max_disk_size_mb=100,
   )

   # Manual cache control
   cache.clear()  # Clear all cached items
   cache.evict_lru()  # Evict least recently used items

   # Get detailed statistics
   stats = cache.get_detailed_stats()
   print(f"Memory usage: {stats['memory_mb']:.2f} MB")
   print(f"Disk usage: {stats['disk_mb']:.2f} MB")
   print(f"Hit rate: {stats['hit_rate']:.2%}")

See Also
--------

- :doc:`nlsq.unified_cache` - Unified compilation cache
- :doc:`nlsq.caching` - General caching utilities
- :doc:`nlsq.compilation_cache` - Legacy compilation cache
