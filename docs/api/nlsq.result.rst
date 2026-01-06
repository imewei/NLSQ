nlsq.result module
==================

Result types for NLSQ optimization operations.

This module provides consolidated result types that were previously scattered
across the codebase. As of v0.4.3, ``OptimizeResult`` and ``OptimizeWarning``
have been moved here from ``nlsq.core._optimize``.

.. automodule:: nlsq.result
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Classes
-------

OptimizeResult
~~~~~~~~~~~~~~

.. autoclass:: nlsq.result.OptimizeResult
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

OptimizeWarning
~~~~~~~~~~~~~~~

.. autoclass:: nlsq.result.OptimizeWarning
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from nlsq import curve_fit

   # Perform fit
   popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.1], full_output=True)

   # Access additional result information
   # (when using LeastSquares directly)
   from nlsq import LeastSquares

   ls = LeastSquares()
   result = ls.least_squares(residuals, p0)

   print(f"Success: {result.success}")
   print(f"Message: {result.message}")
   print(f"Number of iterations: {result.nit}")
   print(f"Final cost: {result.cost}")

Result Attributes
-----------------

The OptimizeResult object contains:

- **x**: Solution parameters
- **success**: Whether optimization succeeded
- **status**: Termination status code
- **message**: Description of termination
- **fun**: Final residuals
- **jac**: Final Jacobian matrix
- **cost**: Final cost value
- **nfev**: Number of function evaluations
- **njev**: Number of Jacobian evaluations
- **nit**: Number of iterations

Migration Notes
---------------

As of v0.4.3, ``OptimizeResult`` and ``OptimizeWarning`` have been moved from
``nlsq.core._optimize`` to ``nlsq.result``. The old import paths continue to
work with deprecation warnings during the 12-month transition period:

.. code-block:: python

   # Old (deprecated)
   from nlsq.core._optimize import OptimizeResult

   # New (recommended)
   from nlsq.result import OptimizeResult

   # or
   from nlsq import OptimizeResult

See Also
--------

- :doc:`nlsq.least_squares` - Least squares solver
- :doc:`nlsq.minpack` - curve_fit interface
