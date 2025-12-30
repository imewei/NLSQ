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

Functions
---------

.. autofunction:: nlsq.profiler_visualization.plot_timeline
   :noindex:
.. autofunction:: nlsq.profiler_visualization.generate_flame_graph
   :noindex:
.. autofunction:: nlsq.profiler_visualization.create_html_report
   :noindex:

Example Usage
-------------

.. code-block:: python

   from nlsq.profiler import Profiler
   from nlsq.profiler_visualization import plot_timeline, create_html_report

   # Profile operations
   profiler = Profiler(enable=True)
   # ... run profiled code ...

   # Visualize results
   plot_timeline(profiler.get_results())

   # Generate HTML report
   create_html_report(profiler.get_results(), output_file="profile_report.html")

See Also
--------

- :doc:`nlsq.profiler` - Profiling utilities
- :doc:`nlsq.profiling` - Transfer profiling
