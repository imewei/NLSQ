nlsq.profiler module
====================

.. automodule:: nlsq.utils.profiler
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``profiler`` module provides performance profiling utilities for NLSQ optimization workflows.

Key Features
------------

- **Function-level profiling** for optimization steps
- **Memory profiling** to track allocation patterns
- **Timing statistics** for performance analysis
- **Integration with JAX profiler**

Classes
-------

.. autoclass:: nlsq.utils.profiler.PerformanceProfiler
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   import jax.numpy as jnp

   from nlsq import curve_fit
   from nlsq.utils.profiler import PerformanceProfiler


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   x = jnp.linspace(0, 10, 1000)
   y = 2.5 * jnp.exp(-0.5 * x)

   profiler = PerformanceProfiler()

   with profiler.profile("curve_fit"):
       popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.1])
       # Fill in run details that the profiler cannot observe itself
       profiler.update_current(n_data_points=len(x), n_parameters=2, success=True)

   summary = profiler.get_summary("curve_fit")
   print(summary)

Comparing Runs
--------------

.. code-block:: python

   for _ in range(3):
       with profiler.profile("optimization"):
           curve_fit(model, x, y, p0=[1.0, 0.1])

   # One ``ProfileMetrics`` per run; ``to_dict()`` gives plain values
   runs = profiler.get_metrics("optimization")
   print(len(runs), runs[0].to_dict()["total_time"])

   # Text report with per-run details
   print(profiler.get_report("optimization", detailed=True))

See Also
--------

- :doc:`nlsq.profiling` - Transfer profiling utilities
- :doc:`nlsq.diagnostics` - Optimization diagnostics
- :doc:`../developer/performance_tuning_guide` - Performance tuning guide
