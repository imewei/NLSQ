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

   from nlsq.core.orchestration import DataPreprocessor, OptimizationSelector
   from nlsq.core.least_squares import LeastSquares


   class CustomPipeline:
       def __init__(self):
           self.preprocessor = DataPreprocessor()
           self.selector = OptimizationSelector()
           self.optimizer = LeastSquares()

       def fit(self, model, x, y, p0, **kwargs):
           # Custom preprocessing
           preprocessed = self.preprocessor.preprocess(f=model, xdata=x, ydata=y)

           # Custom configuration
           config = self.selector.select(
               f=model, xdata=preprocessed.xdata, ydata=preprocessed.ydata, p0=p0
           )

           # Run optimization
           def residuals(params):
               return model(preprocessed.xdata, *params) - preprocessed.ydata

           result = self.optimizer.least_squares(
               fun=residuals, x0=config.p0, bounds=config.bounds
           )

           return result.x, None  # popt, pcov


   pipeline = CustomPipeline()
   popt, _ = pipeline.fit(model, x, y, p0=[1, 0.5])
