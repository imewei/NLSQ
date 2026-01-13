Core APIs
=========

This chapter covers NLSQ's core API classes for direct control over optimization.

.. toctree::
   :maxdepth: 1

   curve_fit_class
   least_squares
   trf_optimizer
   result_types

Chapter Overview
----------------

**CurveFit Class** (15 min)
   Reusable curve fitting with JIT caching.

**LeastSquares** (15 min)
   Direct optimizer control.

**TRF Optimizer** (10 min)
   Trust Region Reflective algorithm details.

**Result Types** (5 min)
   OptimizeResult and OptimizeWarning.

API Level Comparison
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - API
     - Control Level
     - Use Case
   * - ``fit()``
     - Automatic
     - Simple, workflow-based fitting
   * - ``CurveFit``
     - Medium
     - Reusable fitting, batch processing
   * - ``LeastSquares``
     - High
     - Custom residual functions, diagnostics
   * - ``TRF``
     - Full
     - Algorithm customization, research

Quick Examples
--------------

.. code-block:: python

   from nlsq import fit, CurveFit
   from nlsq.core.least_squares import LeastSquares

   # Simple fit
   popt, pcov = fit(model, x, y, p0=[...])

   # Reusable CurveFit
   fitter = CurveFit()
   popt1, pcov1 = fitter.curve_fit(model, x1, y1, p0=[...])
   popt2, pcov2 = fitter.curve_fit(model, x2, y2, p0=[...])

   # Direct LeastSquares
   optimizer = LeastSquares()
   result = optimizer.least_squares(
       fun=residual_func, x0=initial_params, bounds=(-np.inf, np.inf)
   )
