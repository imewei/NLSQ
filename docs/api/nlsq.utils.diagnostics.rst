nlsq.utils.diagnostics module
==============================

.. automodule:: nlsq.utils.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``diagnostics`` module provides real-time monitoring, convergence detection,
and diagnostic reporting for optimization processes.

Key Features
------------

- **Convergence monitoring** with sliding-window pattern detection
- **Oscillation, stagnation, and divergence detection**
- **Memory usage tracking** (via psutil when available)
- **Iteration-level diagnostic reporting** with configurable verbosity

Classes
-------

.. autoclass:: nlsq.utils.diagnostics.ConvergenceMonitor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nlsq.utils.diagnostics.OptimizationDiagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: nlsq.utils.diagnostics.get_diagnostics

.. autofunction:: nlsq.utils.diagnostics.reset_diagnostics

Example Usage
-------------

.. code-block:: python

   from nlsq.utils.diagnostics import get_diagnostics, reset_diagnostics

   # Get a diagnostics instance with verbosity level 2
   diag = get_diagnostics(verbosity=2)

   # Reset diagnostics for a new optimization run
   reset_diagnostics(verbosity=1)

See Also
--------

- :doc:`nlsq.stability.guard` - Numerical stability monitoring
- :doc:`nlsq.callbacks` - Callback system for optimization events
