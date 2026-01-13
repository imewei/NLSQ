Result Types
============

NLSQ uses specific result types for optimization outputs.

OptimizeResult
--------------

The main result container for optimization:

.. code-block:: python

   from nlsq import OptimizeResult

   # Returned by least_squares
   result = optimizer.least_squares(...)

   # Attributes
   result.x  # Optimal parameters (array)
   result.cost  # Final cost: 0.5 * sum(residuals²)
   result.fun  # Residuals at solution
   result.jac  # Jacobian at solution
   result.grad  # Gradient at solution
   result.optimality  # Optimality measure
   result.active_mask  # Which bounds are active
   result.nfev  # Number of function evaluations
   result.njev  # Number of Jacobian evaluations
   result.status  # Exit status code
   result.message  # Human-readable status
   result.success  # True if converged

Accessing Results
-----------------

.. code-block:: python

   from nlsq.core.least_squares import LeastSquares

   optimizer = LeastSquares()
   result = optimizer.least_squares(fun=residuals, x0=x0)

   # Parameters
   popt = result.x
   print(f"Parameters: {popt}")

   # Convergence info
   print(f"Cost: {result.cost:.6e}")
   print(f"Evaluations: {result.nfev}")
   print(f"Status: {result.status} - {result.message}")

   # Check success
   if result.success:
       print("Optimization converged!")
   else:
       print(f"Warning: {result.message}")

Status Codes
------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Status
     - Description
   * - 1
     - Function tolerance convergence (cost change < ftol)
   * - 2
     - Parameter tolerance convergence (step size < xtol)
   * - 3
     - Gradient tolerance convergence (gradient norm < gtol)
   * - 0
     - Maximum function evaluations reached
   * - -1
     - Improper input parameters

OptimizeWarning
---------------

Warning type for optimization issues:

.. code-block:: python

   from nlsq import OptimizeWarning
   import warnings

   # Catch optimization warnings
   with warnings.catch_warnings(record=True) as w:
       warnings.simplefilter("always")
       popt, pcov = fit(model, x, y, p0=[...])

       for warning in w:
           if issubclass(warning.category, OptimizeWarning):
               print(f"Warning: {warning.message}")

Common Warnings
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Warning
     - Cause
   * - "Covariance could not be estimated"
     - Singular Jacobian or ill-conditioning
   * - "Parameter at bound"
     - Solution constrained by bounds
   * - "Maximum iterations reached"
     - Didn't converge within max_nfev
   * - "Ill-conditioned Jacobian"
     - Numerical stability issues

Full Output Mode
----------------

For ``curve_fit`` with ``full_output=True``:

.. code-block:: python

   popt, pcov, info = curve_fit(model, x, y, p0=[...], full_output=True)

   # info contains:
   # - infodict: convergence information
   # - mesg: status message
   # - ier: integer status code

Creating Results
----------------

For custom optimizers, create OptimizeResult:

.. code-block:: python

   from nlsq import OptimizeResult

   result = OptimizeResult(
       x=optimal_params,
       cost=final_cost,
       fun=final_residuals,
       jac=final_jacobian,
       grad=final_gradient,
       optimality=optimality_measure,
       active_mask=bound_mask,
       nfev=n_function_evals,
       njev=n_jacobian_evals,
       status=1,
       message="Optimization converged",
       success=True,
   )

Serialization
-------------

Results can be saved:

.. code-block:: python

   import numpy as np

   # Save to file
   np.savez("result.npz", x=result.x, cost=result.cost, nfev=result.nfev)

   # Load
   data = np.load("result.npz")
   popt = data["x"]

Next Steps
----------

- :doc:`../factories_di/index` - Factory patterns
- :doc:`../orchestration/index` - Component-based design
