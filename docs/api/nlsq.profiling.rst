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

Performance Targets
-------------------

- **Host-device transfer bytes**: 80% reduction (current ~80KB → <16KB per iteration)
- **Transfer count**: Reduce from 24+ → <5 per iteration
- **GPU iteration time**: 5-15% reduction through transfer optimization

Classes
-------

.. autoclass:: nlsq.utils.profiling.PerformanceMetrics
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: nlsq.utils.profiling.profile_optimization
   :noindex:
.. autofunction:: nlsq.utils.profiling.profile_transfers_runtime
   :noindex:
.. autofunction:: nlsq.utils.profiling.analyze_source_transfers
   :noindex:
.. autofunction:: nlsq.utils.profiling.compare_transfer_reduction
   :noindex:

Example Usage
-------------

.. code-block:: python

   import jax.numpy as jnp

   from nlsq import curve_fit
   from nlsq.utils.profiling import (
       analyze_source_transfers,
       profile_optimization,
       profile_transfers_runtime,
   )


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   x = jnp.linspace(0, 10, 1000)
   y = 2.5 * jnp.exp(-0.5 * x)

   # Wall-clock timing of an optimization
   with profile_optimization() as metrics:
       popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.1])
   print(f"Total time: {metrics.total_time_sec:.3f} s")

   # Static analysis of transfer-inducing patterns in source code
   stats = analyze_source_transfers("y = np.asarray(x)\nz = y.block_until_ready()")
   print(stats["total_potential_transfers"])

   # Runtime transfer profiling with the JAX profiler (most useful on GPU)
   (popt, pcov), transfer_stats = profile_transfers_runtime(
       curve_fit, model, x, y, p0=[1.0, 0.1]
   )
   print(transfer_stats["transfer_count"], transfer_stats["transfer_bytes"])

Requirements
------------

``profile_transfers_runtime`` uses the JAX profiler; the traces it writes
(``trace_dir``, default: system temp directory) can be inspected in TensorBoard.
``transfer_stats["profiler_available"]`` reports whether the profiler was usable.

Notes
-----

- ``analyze_source_transfers`` is static: it counts ``np.array``, ``np.asarray``
  and ``.block_until_ready`` patterns in source text
- On CPU, host-device transfers are minimal; runtime profiling is most useful on GPU

See Also
--------

- :doc:`nlsq.trf` - Trust Region Reflective algorithm
- :doc:`nlsq.diagnostics` - General optimization diagnostics
- :doc:`../howto/optimize_performance` - Performance optimization guide
