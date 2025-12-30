nlsq.core.profiler
==================

Performance profiling for Trust Region Reflective optimization.

.. versionadded:: 1.2.0
   Extracted from ``nlsq.core.trf`` for modularity.

This module provides profiling infrastructure for the TRF optimizer,
enabling detailed performance analysis of optimization runs.

Classes
-------

TRFProfiler
~~~~~~~~~~~

A profiler that records timing information for each phase of TRF optimization:

- Jacobian computation time
- SVD decomposition time
- Trust region step computation
- Function evaluation time
- Total iteration time

NullProfiler
~~~~~~~~~~~~

A no-op profiler implementation for production use when profiling overhead
is not desired. Implements the same interface as TRFProfiler but does nothing.

Module Contents
---------------

.. automodule:: nlsq.core.profiler
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from nlsq.core.profiler import TRFProfiler, NullProfiler
   from nlsq.core.trf import TrustRegionReflective

   # Enable profiling
   profiler = TRFProfiler()
   optimizer = TrustRegionReflective(profiler=profiler)

   # Run optimization
   result = optimizer.optimize(...)

   # Get timing statistics
   stats = profiler.get_stats()
   print(f"Jacobian time: {stats['jacobian_time']:.3f}s")
   print(f"SVD time: {stats['svd_time']:.3f}s")

See Also
--------

- :doc:`nlsq.trf` - Main Trust Region Reflective optimizer
- :doc:`nlsq.profiling` - General profiling utilities
