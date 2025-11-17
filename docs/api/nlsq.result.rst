nlsq.result module
==================

.. automodule:: nlsq.result
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``result`` module defines result containers for optimization outputs.

Classes
-------

.. autoclass:: nlsq.result.OptimizeResult
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
   # (when using least_squares directly)
   from nlsq import least_squares

   result = least_squares(residuals, p0)

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

See Also
--------

- :doc:`nlsq.least_squares` - Least squares solver
- :doc:`nlsq.minpack` - curve_fit interface
