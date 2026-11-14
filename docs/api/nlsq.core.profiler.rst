nlsq.core.profiler
==================

Performance profiling for Trust Region Reflective optimization.

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

   import jax.numpy as jnp
   from nlsq.core.profiler import TRFProfiler, NullProfiler

   # TRFProfiler records one timing per synchronized JAX operation
   profiler = TRFProfiler()
   J = jnp.ones((1000, 3))
   profiler.time_operation("jac", J)
   profiler.time_operation("svd", jnp.linalg.svd(J, full_matrices=False)[1])

   # Get timing data (lists of seconds, keyed by operation)
   timings = profiler.get_timing_data()
   print(f"Jacobian time: {sum(timings['jtimes']):.6f}s")
   print(f"SVD time: {sum(timings['svd_times']):.6f}s")

   # NullProfiler has the same interface and records nothing
   assert NullProfiler().get_timing_data()["jtimes"] == []

``TrustRegionReflective.trf_no_bounds`` accepts a ``profiler`` argument and
defaults to ``NullProfiler``; ``curve_fit`` does not expose it.

See Also
--------

- :doc:`nlsq.trf` - Main Trust Region Reflective optimizer
- :doc:`nlsq.profiling` - General profiling utilities
