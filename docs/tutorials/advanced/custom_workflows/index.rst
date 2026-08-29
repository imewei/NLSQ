Custom Workflows
================

This chapter covers building your own optimization pipelines using NLSQ's
components.

.. toctree::
   :maxdepth: 1

   custom_optimizer
   custom_preprocessing
   two_stage_optimization
   integration_patterns

Chapter Overview
----------------

**Custom Optimizer** (15 min)
   Implement your own optimizer using NLSQ protocols.

**Custom Preprocessing** (10 min)
   Create specialized data preprocessing pipelines.

**Two-Stage Optimization** (15 min)
   Combine global search with local refinement.

**Integration Patterns** (10 min)
   Integrate NLSQ with external tools and frameworks.

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from nlsq.core.least_squares import LeastSquares


   class CustomPipeline:
       def __init__(self):
           self.optimizer = LeastSquares()

       def fit(self, model, x, y, p0, bounds=None):
           # Custom preprocessing (drop non-finite points)
           finite = np.isfinite(x) & np.isfinite(y)
           x, y = x[finite], y[finite]

           def residuals(params):
               return model(x, *params) - y

           result = self.optimizer.least_squares(fun=residuals, x0=p0, bounds=bounds)

           return result.x, None  # popt, pcov


   pipeline = CustomPipeline()
   popt, _ = pipeline.fit(model, x, y, p0=[1, 0.5])
