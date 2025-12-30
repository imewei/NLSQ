nlsq.memory\_pool module
=========================

.. automodule:: nlsq.caching.memory_pool
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``memory_pool`` module provides memory pool allocation to reduce overhead from repeated array allocations in optimization loops.

Key Features (Task Group 5)
----------------------------

- **Size-class bucketing**: Round shapes to nearest 1KB/10KB/100KB for 5x reuse increase
- **Reuse statistics tracking**: Monitor ``reuse_rate = reused_allocations / total_allocations``
- **Adaptive sizing**: Small arrays (1KB buckets), medium (10KB), large (100KB)
- **Memory efficiency**: Pre-allocate buffers for common array shapes

Performance Targets
-------------------

- **Peak memory reduction**: 12.5% reduction in memory usage
- **Reuse rate**: 90% reuse rate via size-class bucketing
- **Allocation overhead**: Minimize malloc/free calls in hot paths

Classes
-------

.. autoclass:: nlsq.memory_pool.MemoryPool
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: nlsq.memory_pool.round_to_bucket
   :noindex:

Example Usage
-------------

.. code-block:: python

   from nlsq.memory_pool import MemoryPool
   import jax.numpy as jnp

   # Create memory pool with statistics tracking
   pool = MemoryPool(max_pool_size=10, enable_stats=True, enable_bucketing=True)

   # Allocate array from pool
   shape = (100, 10)
   dtype = jnp.float32
   array = pool.allocate(shape, dtype)

   # Use array in computation
   result = array * 2.0

   # Return to pool for reuse
   pool.release(array)

   # Check pool statistics
   stats = pool.get_stats()
   print(f"Reuse rate: {stats['reuse_rate']:.2%}")
   print(f"Total allocations: {stats['total_allocations']}")

Bucketing Strategy
------------------

The memory pool uses tiered bucketing for better reuse:

- **Small arrays** (<10KB): Round to nearest 1KB
- **Medium arrays** (10KB-100KB): Round to nearest 10KB
- **Large arrays** (>100KB): Round to nearest 100KB

.. code-block:: python

   from nlsq.memory_pool import round_to_bucket

   print(round_to_bucket(800))  # Small: 1024 (1KB)
   print(round_to_bucket(8500))  # Medium: 10240 (10KB)
   print(round_to_bucket(85000))  # Large: 102400 (100KB)

See Also
--------

- :doc:`nlsq.memory_manager` - High-level memory management
- :doc:`nlsq.config` - Memory configuration options
