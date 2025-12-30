nlsq.compilation\_cache module
===============================

.. automodule:: nlsq.caching.compilation_cache
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``compilation_cache`` module provides JIT compilation caching to avoid recompilation overhead.

.. note::
   This module is being consolidated into :doc:`nlsq.unified_cache`. For new code, use the unified cache for better performance and features.

Key Features
------------

- **JIT compilation caching** for function reuse
- **Hash-based cache keys** for function identification
- **Automatic cache invalidation** when functions change
- **Memory-efficient storage** with weak references

Classes
-------

.. autoclass:: nlsq.compilation_cache.CompilationCache
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from nlsq.compilation_cache import CompilationCache

   cache = CompilationCache(maxsize=100)

   # Cache will automatically be used by curve_fit
   # to avoid recompiling the same function

See Also
--------

- :doc:`nlsq.unified_cache` - Unified cache (recommended)
- :doc:`nlsq.smart_cache` - Smart cache with adaptive features
