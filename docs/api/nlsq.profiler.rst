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

.. autoclass:: nlsq.profiler.Profiler
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from nlsq.profiler import Profiler
   from nlsq import curve_fit
   import jax.numpy as jnp

   # Create profiler
   profiler = Profiler(enable=True)


   # Profile curve fitting
   def model(x, a, b):
       return a * jnp.exp(-b * x)


   x = jnp.linspace(0, 10, 1000)
   y = 2.5 * jnp.exp(-0.5 * x)

   with profiler.profile("curve_fit"):
       popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.1])

   # Get profiling results
   results = profiler.get_results()
   print(f"Total time: {results['curve_fit']['total_time']:.3f} s")
   print(f"Call count: {results['curve_fit']['call_count']}")

Performance Analysis
--------------------

.. code-block:: python

   # Detailed profiling with memory tracking
   profiler = Profiler(enable=True, track_memory=True)

   with profiler.profile("optimization"):
       result = expensive_operation()

   # Get detailed statistics
   stats = profiler.get_detailed_stats()
   print(f"Peak memory: {stats['peak_memory_mb']:.2f} MB")
   print(f"Average time: {stats['avg_time']:.3f} s")

See Also
--------

- :doc:`nlsq.profiling` - Transfer profiling utilities
- :doc:`nlsq.diagnostics` - Optimization diagnostics
- :doc:`../developer/performance_tuning_guide` - Performance tuning guide
