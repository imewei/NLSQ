nlsq.profiling module
=====================

.. automodule:: nlsq.utils.profiling
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``profiling`` module provides JAX profiler integration for monitoring host-device transfers in the TRF solver, enabling measurement of transfer bytes and counts per iteration.

Performance Targets (Task Group 2)
-----------------------------------

- **Host-device transfer bytes**: 80% reduction (current ~80KB → <16KB per iteration)
- **Transfer count**: Reduce from 24+ → <5 per iteration
- **GPU iteration time**: 5-15% reduction through transfer optimization

Classes
-------

.. autoclass:: nlsq.profiling.TransferProfiler
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from nlsq.profiling import TransferProfiler

   # Create profiler (requires JAX profiler)
   profiler = TransferProfiler(enable=True)

   # Profile TRF iteration
   with profiler.profile_iteration(iteration=0):
       # TRF solver iteration code
       pass

   # Get diagnostics
   diagnostics = profiler.get_diagnostics()
   print(f"Transfer bytes: {diagnostics['transfer_bytes']}")
   print(f"Transfer count: {diagnostics['transfer_count']}")
   print(f"Avg per iteration: {diagnostics['avg_bytes_per_iter']:.2f} bytes")

Chrome Trace Visualization
---------------------------

The profiler can generate Chrome trace files for visualization:

.. code-block:: python

   # Profile with trace output
   profiler = TransferProfiler(enable=True)

   with profiler.profile_iteration(iteration=0):
       # Code to profile
       pass

   # View at chrome://tracing
   profiler.save_trace("trace.json")

Requirements
------------

The profiler requires the JAX profiler optional dependency:

.. code-block:: bash

   pip install jax[profiler]

Notes
-----

- Profiling overhead is minimal (~1-2% when enabled)
- Transfer bytes are estimated based on array sizes
- Chrome traces provide detailed timeline visualization
- Automatically disabled if JAX profiler not available

See Also
--------

- :doc:`nlsq.trf` - Trust Region Reflective algorithm
- :doc:`nlsq.diagnostics` - General optimization diagnostics
- :doc:`../howto/optimize_performance` - Performance optimization guide
