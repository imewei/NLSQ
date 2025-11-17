nlsq.constants module
=====================

.. automodule:: nlsq.constants
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``constants`` module defines numerical constants and thresholds used throughout NLSQ.

Constants
---------

This module provides:

- **Machine epsilon** values for different dtypes
- **Convergence tolerances** for optimization algorithms
- **Numerical stability thresholds**
- **Default parameter values**

Example Usage
-------------

.. code-block:: python

   from nlsq.constants import MACHINE_EPSILON, DEFAULT_XTOL, DEFAULT_FTOL

   # Use in convergence checks
   if abs(f_new - f_old) < DEFAULT_FTOL:
       print("Converged based on function tolerance")

   # Machine precision aware computations
   safe_denominator = max(value, MACHINE_EPSILON)

See Also
--------

- :doc:`nlsq.config` - Configuration management
- :doc:`nlsq.least_squares` - Uses constants for convergence criteria
