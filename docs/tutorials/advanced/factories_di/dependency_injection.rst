Dependency Injection
====================

Dependency injection (DI) enables flexible, testable component composition.

What Is Dependency Injection?
-----------------------------

Instead of creating dependencies internally, they are "injected" from outside:

.. code-block:: python

   # Without DI - hard-coded dependency
   class Fitter:
       def __init__(self):
           self.cache = GlobalCache()  # Hard-coded


   # With DI - injected dependency
   class Fitter:
       def __init__(self, cache: CacheProtocol = None):
           self.cache = cache or GlobalCache()  # Injected

Constructor Injection
---------------------

Pass dependencies via constructor:

.. code-block:: python

   from nlsq.interfaces.cache_protocol import CacheProtocol
   from nlsq.interfaces.optimizer_protocol import CurveFitProtocol


   class AnalysisPipeline:
       def __init__(self, fitter: CurveFitProtocol, cache: CacheProtocol | None = None):
           self.fitter = fitter
           self.cache = cache

       def run(self, model, datasets):
           results = []
           for x, y in datasets:
               cache_key = f"{hash((tuple(x), tuple(y)))}"

               # Check cache first
               if self.cache and (cached := self.cache.get(cache_key)):
                   results.append(cached)
                   continue

               # Fit and cache
               popt, pcov = self.fitter.curve_fit(model, x, y)
               if self.cache:
                   self.cache.set(cache_key, (popt, pcov))
               results.append((popt, pcov))

           return results


   # Usage with custom components
   from nlsq import CurveFit
   from nlsq.caching.smart_cache import SmartCache

   pipeline = AnalysisPipeline(fitter=CurveFit(), cache=SmartCache(max_size=100))

Method Injection
----------------

Pass dependencies per-method call:

.. code-block:: python

   class FlexibleFitter:
       def fit_with(self, optimizer: OptimizerProtocol, model, x, y, **kwargs):
           return optimizer.optimize(
               lambda p: model(x, *p) - y, x0=kwargs.get("p0", [1.0, 1.0])
           )


   # Different optimizers for different calls
   fitter = FlexibleFitter()
   result1 = fitter.fit_with(LocalOptimizer(), model, x1, y1)
   result2 = fitter.fit_with(GlobalOptimizer(), model, x2, y2)

Interface Segregation
---------------------

Use narrow interfaces:

.. code-block:: python

   from typing import Protocol


   # Narrow interface for preprocessing
   class PreprocessorProtocol(Protocol):
       def preprocess(self, x, y): ...


   # Narrow interface for optimization
   class SolverProtocol(Protocol):
       def solve(self, residuals, x0): ...


   # Compose with narrow dependencies
   class ModularFitter:
       def __init__(self, preprocessor: PreprocessorProtocol, solver: SolverProtocol):
           self.preprocessor = preprocessor
           self.solver = solver

       def fit(self, model, x, y, p0):
           x_clean, y_clean = self.preprocessor.preprocess(x, y)
           residuals = lambda p: model(x_clean, *p) - y_clean
           return self.solver.solve(residuals, p0)

Testing with DI
---------------

**Mock dependencies for testing:**

.. code-block:: python

   class MockFitter:
       def curve_fit(self, f, xdata, ydata, **kwargs):
           return np.array([1.0, 2.0]), np.eye(2)


   class MockCache:
       def __init__(self):
           self.data = {}

       def get(self, key, default=None):
           return self.data.get(key, default)

       def set(self, key, value):
           self.data[key] = value

       def clear(self):
           self.data.clear()


   # Test pipeline with mocks
   def test_pipeline():
       pipeline = AnalysisPipeline(fitter=MockFitter(), cache=MockCache())

       results = pipeline.run(dummy_model, [(x, y)])
       assert len(results) == 1
       assert results[0][0].shape == (2,)

Complete DI Example
-------------------

.. code-block:: python

   from nlsq import CurveFit
   from nlsq.core.orchestration import DataPreprocessor, CovarianceComputer
   from nlsq.caching.smart_cache import SmartCache
   from nlsq.stability.guard import NumericalStabilityGuard


   class ProductionPipeline:
       """Production pipeline with injected dependencies."""

       def __init__(
           self,
           fitter=None,
           preprocessor=None,
           covariance_computer=None,
           cache=None,
           stability_guard=None,
       ):
           self.fitter = fitter or CurveFit()
           self.preprocessor = preprocessor or DataPreprocessor()
           self.covariance = covariance_computer or CovarianceComputer()
           self.cache = cache or SmartCache()
           self.stability = stability_guard or NumericalStabilityGuard()

       def fit(self, model, x, y, p0, **kwargs):
           # Preprocess
           preprocessed = self.preprocessor.preprocess(f=model, xdata=x, ydata=y)

           # Fit
           popt, pcov = self.fitter.curve_fit(
               model, preprocessed.xdata, preprocessed.ydata, p0=p0, **kwargs
           )

           return popt, pcov


   # Production configuration
   prod_pipeline = ProductionPipeline(
       fitter=CurveFit(enable_diagnostics=True),
       cache=SmartCache(max_size=1000),
       stability_guard=NumericalStabilityGuard(strict=True),
   )

   # Test configuration
   test_pipeline = ProductionPipeline(fitter=MockFitter(), cache=MockCache())

Next Steps
----------

- :doc:`../orchestration/index` - NLSQ's component system
- :doc:`../custom_workflows/index` - Building custom pipelines
