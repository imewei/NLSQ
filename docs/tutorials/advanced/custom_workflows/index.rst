Custom Workflows
================

This chapter covers building your own optimization pipelines using NLSQ's
components.

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from nlsq.core.least_squares import LeastSquares


   class CustomPipeline:
       def __init__(self):
           self.optimizer = LeastSquares()

       def fit(self, model, x, y, p0, bounds=(-np.inf, np.inf)):
           # Custom preprocessing (drop non-finite points)
           finite = np.isfinite(x) & np.isfinite(y)
           x, y = x[finite], y[finite]

           def residuals(params):
               return model(x, *params) - y

           result = self.optimizer.least_squares(fun=residuals, x0=p0, bounds=bounds)

           return result.x, None  # popt, pcov


   pipeline = CustomPipeline()
   popt, _ = pipeline.fit(model, x, y, p0=[1, 0.5])
