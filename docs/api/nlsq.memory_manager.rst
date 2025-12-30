nlsq.memory\_manager module
=============================

.. currentmodule:: nlsq.caching.memory_manager

.. automodule:: nlsq.caching.memory_manager
   :noindex:

Overview
--------

The ``nlsq.memory_manager`` module provides intelligent memory management capabilities for
optimization algorithms. It includes memory usage monitoring, prediction, array pooling, and
automatic garbage collection to handle large-scale curve fitting problems efficiently.

Key Features
------------

- **Memory usage monitoring** with psutil integration
- **Memory requirement prediction** for different algorithms
- **Array pooling** to reduce allocation overhead
- **Automatic garbage collection** triggers
- **Memory-safe context managers** for operations
- **Chunking strategy estimation** for large datasets
- **Allocation history tracking** and statistics

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   MemoryManager

Functions
---------

.. autosummary::
   :toctree: generated/

   get_memory_manager
   clear_memory_pool
   get_memory_stats

Usage Examples
--------------

Basic Memory Management
~~~~~~~~~~~~~~~~~~~~~~~

Monitor and manage memory during optimization:

.. code-block:: python

    from nlsq.caching.memory_manager import MemoryManager
    import numpy as np

    # Create memory manager
    mm = MemoryManager(gc_threshold=0.8, safety_factor=1.2)

    # Check available memory
    available = mm.get_available_memory()
    print(f"Available memory: {available / 1e9:.2f} GB")

    # Predict memory requirements
    n_points = 1_000_000
    n_params = 10
    bytes_needed = mm.predict_memory_requirement(n_points, n_params, algorithm="trf")
    print(f"Memory needed: {bytes_needed / 1e9:.2f} GB")

    # Check if enough memory is available
    is_available, message = mm.check_memory_availability(bytes_needed)
    if is_available:
        print("Sufficient memory available")
    else:
        print(f"Warning: {message}")

Memory-Safe Operations
~~~~~~~~~~~~~~~~~~~~~~~

Use context managers to ensure memory availability:

.. code-block:: python

    from nlsq.caching.memory_manager import MemoryManager
    import numpy as np

    mm = MemoryManager()

    # Estimate memory needed
    n_points = 5_000_000
    n_params = 5
    bytes_needed = mm.predict_memory_requirement(n_points, n_params, "trf")

    # Use memory guard to ensure availability
    try:
        with mm.memory_guard(bytes_needed):
            # Perform memory-intensive operation
            x = np.random.randn(n_points)
            y = 2 * x + 1 + np.random.randn(n_points) * 0.1

            # Your optimization here
            result = optimize(x, y)

    except MemoryError as e:
        print(f"Insufficient memory: {e}")
        # Fall back to chunked processing

Array Pooling
~~~~~~~~~~~~~

Reuse arrays to reduce allocation overhead:

.. code-block:: python

    from nlsq.caching.memory_manager import MemoryManager
    import numpy as np

    mm = MemoryManager()

    # Allocate arrays from pool
    jacobian = mm.allocate_array((1000, 10), dtype=np.float64, zero=True)
    residuals = mm.allocate_array((1000,), dtype=np.float64, zero=True)

    # Use arrays
    jacobian[:] = compute_jacobian()
    residuals[:] = compute_residuals()

    # Return arrays to pool when done
    mm.free_array(jacobian)
    mm.free_array(residuals)

    # Arrays will be reused on next allocation
    jacobian2 = mm.allocate_array((1000, 10), dtype=np.float64)  # Reuses pooled array

Temporary Allocations
~~~~~~~~~~~~~~~~~~~~~

Use context manager for temporary arrays:

.. code-block:: python

    from nlsq.caching.memory_manager import MemoryManager

    mm = MemoryManager()

    # Temporary array automatically returned to pool
    with mm.temporary_allocation((1000, 50), dtype=np.float64) as temp_array:
        # Use temporary array
        temp_array[:] = compute_intermediate_result()
        final_result = process(temp_array)

    # temp_array is now back in the pool for reuse

Chunking Strategy Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimate optimal chunk sizes for large datasets:

.. code-block:: python

    from nlsq.caching.memory_manager import MemoryManager

    mm = MemoryManager()

    # Estimate chunking for 100M points
    n_points = 100_000_000
    n_params = 5
    memory_limit_gb = 4.0

    strategy = mm.estimate_chunking_strategy(
        n_points, n_params, algorithm="trf", memory_limit_gb=memory_limit_gb
    )

    if strategy["needs_chunking"]:
        print(f"Chunking required:")
        print(f"  Chunk size: {strategy['chunk_size']:,} points")
        print(f"  Number of chunks: {strategy['n_chunks']}")
        print(f"  Memory per chunk: {strategy['memory_per_chunk_gb']:.2f} GB")
    else:
        print("No chunking needed - dataset fits in memory")

Global Memory Manager
~~~~~~~~~~~~~~~~~~~~~

Use the global memory manager instance:

.. code-block:: python

    from nlsq.caching.memory_manager import (
        get_memory_manager,
        get_memory_stats,
        clear_memory_pool,
    )

    # Get global instance
    mm = get_memory_manager()

    # Use it
    arr = mm.allocate_array((1000, 100))

    # Get memory statistics
    stats = get_memory_stats()
    print(f"Current usage: {stats['current_usage_gb']:.2f} GB")
    print(f"Peak usage: {stats['peak_usage_gb']:.2f} GB")
    print(f"Pool size: {stats['pool_arrays']} arrays")
    print(f"Allocation efficiency: {stats.get('efficiency', 1.0):.1%}")

    # Clear pool when done
    clear_memory_pool()

Integration with Large Dataset Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory manager integrates with large dataset tools:

.. code-block:: python

    from nlsq import curve_fit_large
    from nlsq.caching.memory_manager import MemoryManager
    import jax.numpy as jnp
    import numpy as np

    # Set up memory manager
    mm = MemoryManager(gc_threshold=0.75)

    # Check memory requirements
    n_points = 50_000_000
    n_params = 3
    bytes_needed = mm.predict_memory_requirement(n_points, n_params, "trf")

    is_available, message = mm.check_memory_availability(bytes_needed)

    if not is_available:
        # Use chunking
        strategy = mm.estimate_chunking_strategy(n_points, n_params, memory_limit_gb=4.0)
        print(f"Using {strategy['n_chunks']} chunks")


    # Fit with automatic memory management
    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c


    x = np.linspace(0, 10, n_points)
    y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, n_points)

    with mm.memory_guard(bytes_needed):
        popt, pcov = curve_fit_large(
            exponential, x, y, p0=[2.5, 0.6, 0.2], memory_limit_gb=4.0
        )

Memory Statistics and Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track memory usage over time:

.. code-block:: python

    from nlsq.caching.memory_manager import MemoryManager

    mm = MemoryManager()

    # Perform operations
    for i in range(10):
        arr = mm.allocate_array((10000, 100))
        # ... use array ...
        mm.free_array(arr)

    # Get detailed statistics
    stats = mm.get_memory_stats()

    print("Memory Statistics:")
    print(f"  Current usage: {stats['current_usage_gb']:.2f} GB")
    print(f"  Available: {stats['available_gb']:.2f} GB")
    print(f"  Peak usage: {stats['peak_usage_gb']:.2f} GB")
    print(f"  Usage fraction: {stats['usage_fraction']:.1%}")
    print(f"  Pool memory: {stats['pool_memory_gb']:.3f} GB")
    print(f"  Pool arrays: {stats['pool_arrays']}")
    print(f"  Total allocations: {stats['allocations']}")

    if "efficiency" in stats:
        print(f"  Allocation efficiency: {stats['efficiency']:.1%}")
        print(f"  Total requested: {stats['total_requested_gb']:.2f} GB")
        print(f"  Total used: {stats['total_used_gb']:.2f} GB")

    # Optimize pool if it grows too large
    mm.optimize_memory_pool(max_arrays=50)

Memory Prediction for Different Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare memory requirements across algorithms:

.. code-block:: python

    from nlsq.caching.memory_manager import MemoryManager

    mm = MemoryManager()

    n_points = 10_000
    n_params = 20

    algorithms = ["trf", "lm", "dogbox"]
    for algo in algorithms:
        bytes_needed = mm.predict_memory_requirement(n_points, n_params, algo)
        print(f"{algo:8s}: {bytes_needed / 1e6:.2f} MB")

    # Output:
    # trf     : 12.34 MB
    # lm      : 8.76 MB
    # dogbox  : 13.45 MB

Configuration
-------------

Memory manager can be configured at initialization:

.. code-block:: python

    from nlsq.caching.memory_manager import MemoryManager

    # Conservative settings (more GC, larger safety margin)
    conservative_mm = MemoryManager(gc_threshold=0.7, safety_factor=1.5)

    # Aggressive settings (less GC, smaller safety margin)
    aggressive_mm = MemoryManager(gc_threshold=0.9, safety_factor=1.1)

    # Custom settings for specific use case
    custom_mm = MemoryManager(
        gc_threshold=0.8,  # Trigger GC at 80% memory usage
        safety_factor=1.3,  # Add 30% safety margin to predictions
    )

Performance Considerations
--------------------------

**Array pooling benefits**:

- Reduces allocation overhead by 50-80% for repeated operations
- Most beneficial for medium-sized arrays (10 KB - 10 MB)
- Pool automatically optimized to keep largest arrays

**Memory prediction accuracy**:

- Predictions are conservative (safety_factor=1.2 by default)
- Accuracy: ±10-20% for standard algorithms
- Algorithm-specific formulas based on known memory patterns

**Garbage collection**:

- Triggered automatically when usage exceeds gc_threshold
- Manual collection available via ``clear_pool()``
- Recommended to clear pool between independent fitting operations

**psutil dependency**:

- Optional but recommended for accurate memory monitoring
- Falls back to conservative estimates if unavailable
- Install with: ``pip install psutil``

Memory Efficiency Tips
----------------------

1. **Reuse arrays**: Use array pooling for repeated allocations
2. **Clear pool periodically**: Especially between independent tasks
3. **Estimate before allocating**: Check availability with ``memory_guard()``
4. **Use chunking**: For datasets that don't fit in memory
5. **Monitor statistics**: Track efficiency with ``get_memory_stats()``
6. **Optimize pool size**: Keep pool under 100 arrays for best performance

See Also
--------

- :doc:`nlsq.large_dataset` : Large dataset handling with automatic chunking
- :doc:`nlsq.streaming_optimizer` : Streaming optimization for unlimited data
- :doc:`../howto/handle_large_data` : Large dataset guide
- :doc:`../howto/optimize_performance` : Performance optimization guide
