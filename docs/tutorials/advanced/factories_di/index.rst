Factories and Dependency Injection
===================================

This chapter covers NLSQ's factory functions and protocol-based dependency
injection patterns for composing custom optimization pipelines.

.. toctree::
   :maxdepth: 1

   factory_functions
   protocols
   adapters
   dependency_injection

Chapter Overview
----------------

**Factory Functions** (10 min)
   ``create_optimizer()`` and ``configure_curve_fit()`` for runtime composition.

**Protocols** (10 min)
   Interface contracts for loose coupling.

**Adapters** (5 min)
   Protocol implementations that bridge components.

**Dependency Injection** (10 min)
   Composing pipelines with injected dependencies.

Why Factories and DI?
---------------------

1. **Flexibility**: Configure behavior at runtime
2. **Testing**: Inject mocks for testing
3. **Extension**: Add custom components without modifying core
4. **Decoupling**: Components don't know each other's implementations

Quick Examples
--------------

.. code-block:: python

   from nlsq.core.factories import create_optimizer, configure_curve_fit

   # Factory function creates configured optimizer
   optimizer = create_optimizer(global_optimization=True, diagnostics=True)
   popt, pcov = optimizer.fit(model, x, y)

   # Configured fit function with preset defaults
   my_fit = configure_curve_fit(ftol=1e-10, xtol=1e-10)
   popt, pcov = my_fit(model, x, y)

   # Protocol-based injection
   from nlsq.interfaces import CurveFitProtocol


   class MyCustomFitter:
       def curve_fit(self, f, xdata, ydata, **kwargs):
           # Custom implementation
           pass


   fitter: CurveFitProtocol = MyCustomFitter()
