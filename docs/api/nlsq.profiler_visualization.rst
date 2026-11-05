nlsq.profiler\_visualization module
====================================

.. automodule:: nlsq.utils.profiler_visualization
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``profiler_visualization`` module provides visualization tools for profiling data.

Key Features
------------

- **Flame graphs** for call stack visualization
- **Timeline plots** for execution traces
- **Memory usage plots** over time
- **Interactive HTML reports**

Classes
-------

.. autoclass:: nlsq.utils.profiler_visualization.ProfilerVisualization
   :members:
   :noindex:
   :show-inheritance:

.. autoclass:: nlsq.utils.profiler_visualization.ProfilingDashboard
   :members:
   :noindex:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   import jax.numpy as jnp

   from nlsq import curve_fit
   from nlsq.utils.profiler import PerformanceProfiler
   from nlsq.utils.profiler_visualization import (
       ProfilerVisualization,
       ProfilingDashboard,
   )


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   x = jnp.linspace(0, 10, 1000)
   y = 2.5 * jnp.exp(-0.5 * x)

   # Profile operations
   profiler = PerformanceProfiler()
   for name in ("run_a", "run_b"):
       for _ in range(3):
           with profiler.profile(name):
               curve_fit(model, x, y, p0=[1.0, 0.1])

   # Plots (matplotlib Figure, or None if matplotlib is unavailable)
   viz = ProfilerVisualization(profiler)
   fig = viz.plot_timing_series("run_a")

   # HTML report and JSON export
   viz.generate_html_report("profile_report.html")
   viz.export_json("profile.json")

   # Side-by-side comparison of several profiles
   dashboard = ProfilingDashboard(profiler)
   dashboard.add_profile("run_a")
   dashboard.add_profile("run_b")
   print(dashboard.generate_comparison_report())

See Also
--------

- :doc:`nlsq.profiler` - Profiling utilities
- :doc:`nlsq.profiling` - Transfer profiling
