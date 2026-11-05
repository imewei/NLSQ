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

- **Convergence tolerances** (``DEFAULT_FTOL``, ``DEFAULT_XTOL``, ``DEFAULT_GTOL``) and termination codes
- **Trust-region parameters** (initial, minimum and maximum radius and update factors)
- **Numerical stability thresholds** (``MIN_POSITIVE_VALUE``, ``MAX_CONDITION_NUMBER``)
- **Dataset-size, chunking and memory defaults**

Example Usage
-------------

.. code-block:: python

   from nlsq.constants import (
       DEFAULT_FTOL,
       DEFAULT_XTOL,
       MAX_CONDITION_NUMBER,
       MIN_POSITIVE_VALUE,
   )

   # Use in convergence checks
   f_old, f_new = 1.0, 1.0 + 1e-10
   if abs(f_new - f_old) < DEFAULT_FTOL:
       print("Converged based on function tolerance")

   # Guard against division by (near) zero
   value = 0.0
   safe_denominator = max(value, MIN_POSITIVE_VALUE)

See Also
--------

- :doc:`nlsq.config` - Configuration management
- :doc:`nlsq.least_squares` - Uses constants for convergence criteria
